# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class HoleAdvantageousDiscriminativeLoss(nn.Module):
#     def __init__(self):
#         super(HoleAdvantageousDiscriminativeLoss, self).__init__()
        

#     def forward(self, discriminator, generator_output, target_image, mask):
#         # Generate inpainted images
#         inpainted_images = generator_output

#         # Discriminate real and inpainted images
#         real_preds = discriminator(target_image)
#         inpainted_preds = discriminator(inpainted_images)[0]

#         # Calculate adversarial loss
#         adversarial_loss = -torch.mean(torch.log(inpainted_preds))

#         # Calculate discriminative loss
#         discriminative_loss = F.mse_loss(inpainted_preds, real_preds)

#         # Apply mask to focus on the inpainted regions
#         masked_inpainted_images = inpainted_images * mask
#         masked_target_image = target_image * mask

#         # Calculate advantageous loss
#         advantageous_loss = F.l1_loss(masked_inpainted_images, masked_target_image)

#         # Calculate the total loss
#         total_loss = adversarial_loss + discriminative_loss + advantageous_loss

#         return total_loss







import torch
import torch.nn.functional as F

class HADLoss(torch.nn.Module):
    def __init__(self):
        super(HADLoss, self).__init__()
        self.mask = None


    def forward(self,feature_i,feature_g, img, predicted_img,discrim, ori_mask, supervised_mask):
        device = predicted_img.device
        print(feature_i[0])
        print("AND " , feature_g[0])
        # Compute features for inpainted and ground truth images
#         _, feat_i = discrim(predicted_img)
#         _, feat_g = discrim(img)
#         feat_i = torch.stack(feature_i)
        feat_i = torch.nn.functional.interpolate(feature_i, size=(feature_g.size(2), feature_g.size(3)), mode='bilinear', align_corners=False)
        feat_g = torch.stack(feature_g)
        
        # Convert the mask to a bool type
        
        ori_mask = ori_mask.bool()
        supervised_mask = supervised_mask.bool()
        
        # Compute masked regions
        masked_i = predicted_img.masked_select(ori_mask)
        masked_g = img.masked_select(ori_mask)
        
#         if not isinstance(supervised_mask, torch.Tensor):
#             supervised_mask = torch.tensor(supervised_mask).to(device)
            
#         masked_feat_i = feat_i.masked_select(supervised_mask)
#         masked_feat_g = feat_g.masked_select(supervised_mask)
        masked_feat_i = feat_i.masked_select(supervised_mask.view(supervised_mask.size(0), supervised_mask.size(1), 1, 1)) # if cannot then see the shape of
        masked_feat_g = feat_g.masked_select(supervised_mask.view(supervised_mask.size(0), supervised_mask.size(1), 1, 1)) # feat_i and spv_mask, or channel

        # Compute HAD loss
        dist_feat = F.pairwise_distance(masked_feat_i, masked_feat_g, p=2)
        dist_pixel = torch.mean(torch.abs(masked_i - masked_g))
        had_loss = torch.mean(dist_feat) - torch.mean(dist_pixel)

        return had_loss
