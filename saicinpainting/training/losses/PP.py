import torch
import torch.nn as nn
import torch.nn.functional as F

class PerPixelLoss(nn.Module):
    def __init__(self):
        super(PerPixelLoss, self).__init__()
        
    def forward(self, predicted, target, mask):
        # Apply the mask to the predicted and target images
        masked_predicted = predicted * mask
        masked_target = target * mask
        
        # Compute the per-pixel loss using mean absolute error (MAE)
        loss = F.l1_loss(masked_predicted, masked_target)
        
        return loss
