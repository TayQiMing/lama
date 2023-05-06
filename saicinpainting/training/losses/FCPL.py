import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCPLLoss(nn.Module):
    def __init__(self):
        super(FCPLLoss, self).__init__()

        # Load VGG19 model to extract feature maps
        self.vgg = models.vgg19(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, predicted, target, mask):
        device = predicted.device

        # Extract features from the predicted and target images
        pred_features = self.vgg(predicted)
        target_features = self.vgg(target)

        # Compute the deep feature consistent perceptual loss
        fcpl_loss = 0.0
        for pred_feature, target_feature in zip(pred_features, target_features):
            # Compute the channel-wise L1 distance between the predicted and target feature maps
            fcpl_loss += F.l1_loss(pred_feature, target_feature)

        # Apply the mask to the loss
        fcpl_loss = torch.mean(fcpl_loss * mask.to(device))

        return fcpl_loss
