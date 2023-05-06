import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DFCPLoss(nn.Module):
    def __init__(self):
        super(DFCPLoss, self).__init__()
        
        # Load VGG19 model to extract feature maps
        self.vgg = models.vgg19(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        
    def forward(self, predicted, target, mask):
        device = predicted.device
        # Extract features from the predicted and target images
        pred_features = self.vgg(predicted)
        target_features = self.vgg(target)
        
        # Compute the mean squared error (MSE) between the predicted and target feature maps
        mse_loss = F.mse_loss(pred_features, target_features)
        
        # Compute the deep feature consistent perceptual loss
        dfcpl_loss = 0.0
        for pred_feature, target_feature in zip(pred_features, target_features):
            # Compute the Gram matrix of the predicted and target feature maps
            pred_gram = self.calc_gram_matrix(pred_feature)
            target_gram = self.calc_gram_matrix(target_feature)
            
            # Compute the MSE between the Gram matrices
            dfcpl_loss += F.mse_loss(pred_gram, target_gram)
        
        # Combine the MSE and deep feature consistent perceptual losses using a weight factor of 0.1
        loss = mse_loss + 0.1 * dfcpl_loss
        
        # Apply the mask to the loss
        loss = torch.mean(loss * mask.to(device))
        
        return loss
    
    def calc_gram_matrix(self, feature_map):
        batch_size, num_channels, height, width = feature_map.size()
        
        # Reshape the feature map to a 2D matrix
        reshaped_feature_map = feature_map.view(batch_size, num_channels, height * width)
        
        # Compute the Gram matrix
        gram_matrix = torch.bmm(reshaped_feature_map, reshaped_feature_map.transpose(1, 2))
        
        # Normalize the Gram matrix
        gram_matrix /= (num_channels * height * width)
        
        return gram_matrix
