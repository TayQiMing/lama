import torch.nn.functional as F
import torch

class GradientDifferenceLoss(torch.nn.Module):
    def __init__(self, alpha=1):
        super(GradientDifferenceLoss, self).__init__()
        
        self.alpha = alpha
    
    def forward(self, predicted_img, img, mask):
        device = predicted_img.device
        
        # compute the gradients of the predicted and target images
        grad_predicted = torch.abs(predicted_img[:, :, :, :-1] - predicted_img[:, :, :, 1:])
        grad_target = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])

        # calculate the gradient difference loss using mean squared error (MSE)
        loss = F.mse_loss(grad_predicted, grad_target)

        # apply the mask to the loss
        loss = torch.mean(loss * mask.to(device))

        # multiply the loss by alpha
        loss *= self.alpha

        return loss
