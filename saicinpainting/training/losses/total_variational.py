import torch
import torch.nn.functional as F

class TotalVariationLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
        self.conv_h = torch.nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=0, bias=False)
        self.conv_v = torch.nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=0, bias=False)
        self.conv_h.weight.data = torch.tensor([[[[1, -1]]]])
        self.conv_v.weight.data = torch.tensor([[[[1], [-1]]]])
        for param in self.conv_h.parameters():
            param.requires_grad = False
        for param in self.conv_v.parameters():
            param.requires_grad = False

    def forward(self, img):
        device = predicted_img.device
        img = img.mean(dim=1, keepdim=True)
        img_h = self.conv_h(img.to(device))
        img_v = self.conv_v(img.to(device))
        loss = torch.mean(torch.abs(img_h)) + torch.mean(torch.abs(img_v))
        return loss








################ OLD VERSION ###############


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision


# class TotalVariationLoss(torch.nn.Module):
#     def __init__(self):
#         super(TotalVariationLoss, self).__init__()

#     def forward(self, x):
#         # Compute the total variation loss
#         tv_loss = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
#                   torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
#         return tv_loss

