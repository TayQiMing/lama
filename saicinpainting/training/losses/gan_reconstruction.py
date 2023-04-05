import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def masked_l1_loss(predicted_img, img, mask, weight_known=1.0, weight_missing=0.0):
    """
    Computes the L1 loss between two images, but ignores the pixels specified by the mask.

    Args:
        predicted_img (torch.Tensor): The predicted image.
        img (torch.Tensor): The ground truth image.
        mask (torch.Tensor): A binary mask indicating which pixels to ignore.
        weight_known (float, optional): The weight to give to known pixels (i.e., where mask == 1).
        weight_missing (float, optional): The weight to give to missing pixels (i.e., where mask == 0).

    Returns:
        torch.Tensor: The masked L1 loss between the two images.
    """
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
        l1_loss = masked_l1_loss(predicted_img, img, mask=None, weight_known=1.0, weight_missing=0.0)
        return gan_loss + l1_loss
      
