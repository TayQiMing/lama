import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Load VGG19 model to extract feature maps
        self.vgg = models.vgg19(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        

        
    def forward(self, predicted, target, mask):
        device = predicted.device

        # Calculate the Canny edge maps
        pred_edges = self.calc_canny_edge_maps(predicted)
        target_edges = self.calc_canny_edge_maps(target)
        
        # Extract features from the predicted and target images
        pred_features = self.vgg(predicted)
        target_features = self.vgg(target)
        
        # Compute the mean squared error (MSE) between the predicted and target feature maps
        feature_loss = F.mse_loss(pred_features, target_features)
        
        # Compute the MSE between the predicted and target edge maps
        edge_loss = F.mse_loss(pred_edges, target_edges)
        
        # Combine the feature and edge losses using a weight factor of 0.5
        loss = 0.5 * feature_loss + 0.5 * edge_loss
        
        # Apply the mask to the loss
        loss = torch.mean(loss * mask.to(device))
        
        return loss
    
    def calc_canny_edge_maps(self, img, sigma=1.0):
        # Convert the image to grayscale
        gray_img = F.rgb_to_grayscale(img)
        
        # Compute the Canny edge map
        edges = F.canny(gray_img, sigma=sigma)
        
        # Expand the edge map to match the dimensions of the input image
        edges = edges.expand_as(img)
        
        return edges
