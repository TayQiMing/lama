import torch.nn.functional as F
import torch
import torchvision.models as models

class StructureLoss(torch.nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:35].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, predicted_img, img, mask):
        device = predicted_img.device
        
        #no need resize to 224x224 as the vgg will resize internally
        # normalize
        transform = torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # preprocess images
        predicted_img = transform(predicted_img).to(device)
        img = transform(img).to(device)

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

