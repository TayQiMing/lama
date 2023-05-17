import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionalAdversarialLoss(nn.Module):
    def __init__(self):
        super(AttentionalAdversarialLoss, self).__init__()
        
        # Load VGG19 model to extract feature maps
        self.vgg = models.vgg19(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Define the adversarial loss criterion
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Attention mechanism
        self.att_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        
    def forward(self, generator, discriminator, predicted_img, real_img, mask):
        device = predicted_img.device
        
        # Generate a fake image by filling in the masked region of the real image
        fake_img = predicted_img * mask + real_img * (1 - mask)
        
        # Compute the attention maps for the real and fake images
        real_atts = self.calc_attention_maps(real_img)
        fake_atts = self.calc_attention_maps(fake_img)
        
        # Compute the feature maps for the real and fake images
        real_features = self.vgg(real_img)
        fake_features = self.vgg(fake_img)
        
        # Compute the adversarial loss for the fake image
        fake_preds, _ = discriminator(fake_img)
        adv_loss = self.criterion(fake_preds, torch.ones_like(fake_preds).to(device))
        
        # Compute the attentional loss
        att_loss = F.l1_loss(real_atts * fake_features, fake_atts * real_features)
        
        # Combine the adversarial and attentional losses using a weight factor of 0.001
        loss = adv_loss + 0.001 * att_loss
        
        return loss
    
    def calc_attention_maps(self, img):
        att_maps = self.att_conv(img)
        att_maps = torch.sigmoid(att_maps)
        
        return att_maps



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class AttentionalAdversarialLoss(nn.Module):
#     def __init__(self):
#         super(AttentionalAdversarialLoss, self).__init__()
        

#     def forward(self,discriminator, generator_output, target_image, attention_mask):
#         # Generate fake images
#         fake_images = generator_output

#         # Discriminate real and fake images
#         real_preds = discriminator(target_image)
#         fake_preds = discriminator(fake_images)[0]

#         # Calculate adversarial loss
#         adversarial_loss = -torch.mean(torch.log(fake_preds))

#         # Apply attention mask to guide the generation process
#         guided_fake_images = fake_images * attention_mask
#         guided_target_image = target_image * attention_mask

#         # Calculate attentional loss
#         attentional_loss = F.mse_loss(guided_fake_images, guided_target_image)

#         # Calculate the total loss
#         total_loss = adversarial_loss + attentional_loss

#         return total_loss



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models


# class AttentionalAdversarialLoss(nn.Module):
#     def __init__(self):
#         super(AttentionalAdversarialLoss, self).__init__()
        
#         # Load VGG19 model to extract feature maps
#         self.vgg = models.vgg19(pretrained=True).features.eval()
#         for param in self.vgg.parameters():
#             param.requires_grad = False
        
#         # Define the adversarial loss criterion
#         self.criterion = nn.BCEWithLogitsLoss()
        
        
#     def forward(self, generator, discriminator, predicted_img, real_img, mask):
#         device = predicted_img.device
#         # Generate a fake image by filling in the masked region of the real image
#         fake_img = predicted_img
        
#         # Compute the attention maps for the real and fake images
#         real_atts = self.calc_attention_maps(real_img)
#         fake_atts = self.calc_attention_maps(fake_img)
        
#         # Compute the feature maps for the real and fake images
#         real_features = self.vgg(real_img)
#         fake_features = self.vgg(fake_img)
        
#         # Compute the adversarial loss for the fake image
#         fake_preds = discriminator(fake_img)
#         adv_loss = self.criterion(fake_preds, torch.ones_like(fake_preds).to(device))
        
#         # Compute the attentional loss
#         att_loss = F.l1_loss(real_atts * fake_features, fake_atts * real_features)
        
#         # Combine the adversarial and attentional losses using a weight factor of 0.001
#         loss = adv_loss + 0.001 * att_loss
        
#         return loss
    
#     def calc_attention_maps(self, img):
#         # Compute the gradients of the image with respect to the pixel values
#         img.requires_grad = True
#         img_grads = torch.autograd.grad(outputs=img.sum(), inputs=img, create_graph=True)[0]

#         # Compute the attention maps
#         att_maps = torch.abs(img_grads)
#         att_maps = torch.mean(att_maps, dim=1, keepdim=True)
#         att_maps = F.interpolate(att_maps, size=img.shape[2:], mode='bilinear', align_corners=False)
#         att_maps = F.normalize(att_maps, p=1, dim=2)

#         return att_maps
