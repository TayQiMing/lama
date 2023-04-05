import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TotalVariationLoss(torch.nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        # Compute the total variation loss
        tv_loss = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
                  torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return tv_loss

