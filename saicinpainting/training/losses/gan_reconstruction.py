import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def masked_l1_loss(predicted_img, img, mask, weight_known=1.0, weight_missing=0.0):
    diff = predicted_img - img
    diff = diff * mask
    weighted_diff = weight_known * torch.abs(diff) + weight_missing * torch.abs(diff) * (1 - mask)
    loss = torch.sum(weighted_diff) / torch.sum(mask)
    return loss


class GANReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super(GANReconstructionLoss, self).__init__()

    def forward(self, predicted_img, img, discr_fake_pred):
        gan_loss = F.binary_cross_entropy_with_logits(discr_fake_pred, torch.ones_like(discr_fake_pred))
        l1_loss = masked_l1_loss(predicted_img, img, mask=discr_fake_pred, weight_known=10, weight_missing=0)
        return gan_loss + l1_loss
    
    
    
    
    
    
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch
# import torch.nn.functional as F

# def masked_l1_loss(pred, target, mask, weight_known, weight_missing):
#     per_pixel_l1 = F.l1_loss(pred, target, reduction='none')
#     pixel_weights = mask * weight_missing + (1 - mask) * weight_known
#     return (pixel_weights * per_pixel_l1).mean()


# class GANReconstructionLoss(torch.nn.Module):
#     def __init__(self):
#         super(GANReconstructionLoss, self).__init__()

#     def forward(self, predicted_img, img, discr_fake_pred):
#         gan_loss = F.binary_cross_entropy_with_logits(discr_fake_pred, torch.ones_like(discr_fake_pred))
#         l1_loss = masked_l1_loss(predicted_img, img, mask=None, weight_known=1.0, weight_missing=0.0)
#         return gan_loss + l1_loss
      
      
