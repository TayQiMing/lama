import logging

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from saicinpainting.training.data.datasets import make_constant_area_crop_params
from saicinpainting.training.losses.distance_weighting import make_mask_distance_weighter
from saicinpainting.training.losses.feature_matching import feature_matching_loss, masked_l1_loss
from saicinpainting.training.modules.fake_fakes import FakeFakesGenerator
from saicinpainting.training.trainers.base import BaseInpaintingTrainingModule, make_multiscale_noise
from saicinpainting.utils import add_prefix_to_keys, get_ramp

LOGGER = logging.getLogger(__name__)


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(img_height=batch['image'].shape[2],
                                                                             img_width=batch['image'].shape[3],
                                                                             **kwargs)
    batch['image'] = batch['image'][:, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    batch['mask'] = batch['mask'][:, :, crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]
    return batch


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, concat_mask=True, rescale_scheduler_kwargs=None, image_to_discriminator='predicted_image',
                 add_noise_kwargs=None, noise_fill_hole=False, const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0, fake_fakes_generator_kwargs=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask
        self.rescale_size_getter = get_ramp(**rescale_scheduler_kwargs) if rescale_scheduler_kwargs is not None else None
        self.image_to_discriminator = image_to_discriminator
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole
        self.const_area_crop_kwargs = const_area_crop_kwargs
        self.refine_mask_for_losses = make_mask_distance_weighter(**distance_weighter_kwargs) \
            if distance_weighter_kwargs is not None else None
        self.distance_weighted_mask_for_discr = distance_weighted_mask_for_discr
#         print("====== ！@#￥%……& ======== The image_to_discriminator is: ", image_to_discriminator)
        self.fake_fakes_proba = fake_fakes_proba
        if self.fake_fakes_proba > 1e-3:
            self.fake_fakes_gen = FakeFakesGenerator(**(fake_fakes_generator_kwargs or {}))

    def forward(self, batch):
        if self.training and self.rescale_size_getter is not None:
            cur_size = self.rescale_size_getter(self.global_step)
            batch['image'] = F.interpolate(batch['image'], size=cur_size, mode='bilinear', align_corners=False)
            batch['mask'] = F.interpolate(batch['mask'], size=cur_size, mode='nearest')

        if self.training and self.const_area_crop_kwargs is not None:
            batch = make_constant_area_crop_batch(batch, **self.const_area_crop_kwargs)

        img = batch['image']
        mask = batch['mask']

        masked_img = img * (1 - mask)

        if self.add_noise_kwargs is not None:
            noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
            masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        batch['predicted_image'] = self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
#         print("￥￥￥￥￥ PRODUCE PREDICTED IMAGE ￥￥￥￥￥"), print("====== ！@#￥%……& ======== The batch['predicted_image'] is: ", batch['predicted_image'])
        if self.fake_fakes_proba > 1e-3:
            if self.training and torch.rand(1).item() < self.fake_fakes_proba:
                batch['fake_fakes'], batch['fake_fakes_masks'] = self.fake_fakes_gen(img, mask)
                batch['use_fake_fakes'] = True
            else:
                batch['fake_fakes'] = torch.zeros_like(img)
                batch['fake_fakes_masks'] = torch.zeros_like(mask)
                batch['use_fake_fakes'] = False

        batch['mask_for_losses'] = self.refine_mask_for_losses(img, batch['predicted_image'], mask) \
            if self.refine_mask_for_losses is not None and self.training \
            else mask

        return batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']
#         print("￥￥￥￥￥ USE PREDICTED IMAGE ￥￥￥￥￥"),print("====== ！@#￥%……& ======== The predicted_img is: ", predicted_img)
        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses.l1.weight_known,
                                  self.config.losses.l1.weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # vgg-based perceptual loss
        if self.config.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, img, mask=supervised_mask).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = supervised_mask if self.distance_weighted_mask_for_discr else original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.config.losses.feature_matching.weight > 0:
            need_mask_in_fm = OmegaConf.to_container(self.config.losses.feature_matching).get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value

        ##################################   ADDING HERE    ##################################
        
        # total variation loss
        if self.config.losses.total_variational.weight > 0:
            tv_value = self.loss_tv(predicted_img) * self.config.losses.total_variational.weight
            total_loss = total_loss + tv_value
            metrics['gen_tv'] = tv_value

        # style loss
        if self.config.losses.style.weight > 0:
            style_value = self.loss_styl(predicted_img, img, mask=supervised_mask) * self.config.losses.style.weight
            total_loss += style_value
            metrics['gen_style'] = style_value

        # structure loss
        if self.config.losses.structure.weight > 0:
            st_value = self.loss_strl(predicted_img, img, mask=supervised_mask) * self.config.losses.structure.weight  # supervised_mask for image inpainting task
#             st_value = self.loss_strl(predicted_img, img, mask=original_mask) * self.config.losses.structure.weight  #original_mask then those not inpaitning area also consider， normally for generate whole new image then use this
            total_loss = total_loss + st_value
            metrics['gen_st'] = st_value

        # GAN based reconstruction loss
        if self.config.losses.gan_reconstruction.weight > 0:
            rec_loss = self.loss_gr(predicted_img, img, supervised_mask) * self.config.losses.gan_reconstruction.weight
            total_loss = total_loss + rec_loss
            metrics['gen_rec'] = rec_loss

        # HAD loss
        if self.config.losses.HAD.weight > 0:
            had_loss = self.loss_had(img, predicted_img, self.discriminator, original_mask, supervised_mask) * self.config.losses.HAD.weight
            total_loss = total_loss + had_loss
            metrics['gen_had'] = had_loss

        #  contextual loss
        if self.config.losses.contextual.weight > 0:
            ctxt_loss = self.loss_ctxt(predicted_img, img, supervised_mask) * self.config.losses.contextual.weight
            total_loss = total_loss + ctxt_loss
            metrics['gen_ctxt'] = ctxt_loss

        #  edges loss
        if self.config.losses.edges.weight > 0:
            edges_loss = self.loss_edges(predicted_img, img, supervised_mask) * self.config.losses.edges.weight
            total_loss = total_loss + edges_loss
            metrics['gen_edges'] = edges_loss

        #  gradient different loss
        if self.config.losses.gradient_different.weight > 0:
            gd_loss = self.loss_gd(predicted_img, img, supervised_mask) * self.config.losses.gradient_different.weight
            total_loss = total_loss + gd_loss
            metrics['gen_gd'] = gd_loss

        #  MSSSIM loss
        if self.config.losses.MSSSIM.weight > 0:
            msssim_loss = self.loss_msssim(img, predicted_img) * self.config.losses.MSSSIM.weight
            total_loss = total_loss + msssim_loss
            metrics['gen_msssim'] = msssim_loss
            
        #  AA loss
        if self.config.losses.AA.weight > 0:
            aa_loss = self.loss_aa(self.generator, self.discriminator,predicted_img, img, supervised_mask) * self.config.losses.AA.weight
            total_loss = total_loss + aa_loss
            metrics['gen_aa'] = aa_loss
            
        #  DFCP loss
        if self.config.losses.DFCP.weight > 0:
            dfcp_loss = self.loss_dfcp(predicted_img, img, supervised_mask) * self.config.losses.DFCP.weight
            total_loss = total_loss + dfcp_loss
            metrics['gen_dfcp'] = dfcp_loss
            
        #  PP loss
        if self.config.losses.PP.weight > 0:
            pp_loss = self.loss_pp(predicted_img, img, supervised_mask) * self.config.losses.PP.weight
            total_loss = total_loss + pp_loss
            metrics['gen_pp'] = pp_loss
            
        #  patchgan loss
        if self.config.losses.patchgan.weight > 0:
            patchgan_loss = self.loss_patchgan(discr_real_pred, discr_fake_pred) * self.config.losses.patchgan.weight
            total_loss = total_loss + patchgan_loss
            metrics['gen_patchgan'] = patchgan_loss
            
        #  FCPL loss
        if self.config.losses.FCPL.weight > 0:
            fcpl_loss = self.loss_fcpl(predicted_img, img, supervised_mask) * self.config.losses.FCPL.weight
            total_loss = total_loss + fcpl_loss
            metrics['gen_fcpl'] = fcpl_loss
            
        ######################################################################################
            
            
            
        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=batch['mask'])
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))


        if batch.get('use_fake_fakes', False):
            fake_fakes = batch['fake_fakes']
            self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=fake_fakes,
                                                         generator=self.generator, discriminator=self.discriminator)
            discr_fake_fakes_pred, _ = self.discriminator(fake_fakes)
            fake_fakes_adv_discr_loss, fake_fakes_adv_metrics = self.adversarial_loss.discriminator_loss(
                real_batch=batch['image'],
                fake_batch=fake_fakes,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_fakes_pred,
                mask=batch['mask']
            )
            total_loss = total_loss + fake_fakes_adv_discr_loss
            metrics['discr_adv_fake_fakes'] = fake_fakes_adv_discr_loss
            metrics.update(add_prefix_to_keys(fake_fakes_adv_metrics, 'adv_'))

        return total_loss, metrics
