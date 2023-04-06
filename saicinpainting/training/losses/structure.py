import torch.nn.functional as F
import torch
import torchvision.models as models

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

class StructureLoss(torch.nn.Module):
    def __init__(self, normalize_inputs=True):
        super(StructureLoss, self).__init__()
        
        self.normalize_inputs = normalize_inputs
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

        # calculate the structure loss using mean squared error (MSE)
        loss = F.mse_loss(features_predicted, features_img)

        # apply the mask to the loss
        loss = torch.mean(loss * mask.to(device))

        return loss




############################### BELOW IS SAMPLE FOR USING GRADIENT OF GRAYIMAGE ###############################
# import torch.nn.functional as F
# import torch

# class StructureLoss(torch.nn.Module):
#     def __init__(self):
#         super(StructureLoss, self).__init__()

#     def forward(self, predicted_img, img, mask):
#         # convert images to grayscale
#         gray_predicted_img = torch.mean(predicted_img, dim=1)
#         gray_img = torch.mean(img, dim=1)

#         # calculate gradients
#         gradient_x_predicted, gradient_y_predicted = torch.gradient(gray_predicted_img)
#         gradient_x_img, gradient_y_img = torch.gradient(gray_img)

#         # calculate the structure loss using mean squared error (MSE)
#         loss = F.mse_loss(gradient_x_predicted, gradient_x_img) + F.mse_loss(gradient_y_predicted, gradient_y_img)

#         # apply the mask to the loss
#         loss = torch.mean(loss * mask)

#         return loss

################################################################################################################

