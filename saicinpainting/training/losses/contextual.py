import torch.nn.functional as F
import torch
import torchvision.models as models

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

class ContextualLoss(torch.nn.Module):
    def __init__(self, normalize_inputs=True, alpha=0.85, reduction='mean'):
        super(ContextualLoss, self).__init__()
        
        self.normalize_inputs = normalize_inputs
        self.alpha = alpha
        self.reduction = reduction
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD
        
        self.vgg = models.vgg19(pretrained=True).features[:35].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)
        
    def forward(self, predicted_img, img, mask):
        device = predicted_img.device
        
        #no need resize to 224x224 as the vgg will resize internally
        # preprocess images
        if self.normalize_inputs:
            predicted_img = self.do_normalize_inputs(predicted_img)
            img = self.do_normalize_inputs(img)
        
        # extract features using vgg
        features_predicted = self.vgg(predicted_img)
        features_img = self.vgg(img)
        
        # calculate the element-wise distance between features_predicted and features_img
        dist_matrix = torch.cdist(features_predicted, features_img)
        
        # calculate the minimum distance for each pixel
        min_dist, _ = torch.min(dist_matrix, dim=1, keepdim=True)
#         print('HERE HERE min dist -> ',min_dist)
        # calculate the mask for valid pixels
        valid_pixels_mask = F.interpolate(mask, size=min_dist.shape[-2:], mode='nearest').expand_as(min_dist)
        valid_pixels_mask[min_dist > self.alpha] = 0
        
        # calculate the contextual loss using the valid pixels mask
        contextual_loss = F.mse_loss(features_predicted * valid_pixels_mask, features_img * valid_pixels_mask, reduction=self.reduction)
        
        return contextual_loss
