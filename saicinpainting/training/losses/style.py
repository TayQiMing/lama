import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class StyleLoss(torch.nn.Module):
    def __init__(self, normalize_inputs=True):
        super(StyleLoss, self).__init__()
        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD

        vgg = torchvision.models.vgg19(pretrained=True).features
        vgg_avg_pooling = []

        for weights in vgg.parameters():
            weights.requires_grad = False

        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)

        self.vgg = nn.Sequential(*vgg_avg_pooling)

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def get_style_features(self, x):
        if self.normalize_inputs:
            x = self.do_normalize_inputs(x)
        features = self.vgg(x)
        gram_matrices = []
        for f in features:
            N = f.size(0)
            C = f.size(1)
            spatial_dim = f.dim() - 2
            H = f.size(-2)
            W = f.size(-1)
#             N, C, H, W = f.size()
            f = F.normalize(f.view(N, C, -1), dim=2)
            gram_matrices.append(torch.bmm(f, f.transpose(1, 2)) / (C * H * W))
        return gram_matrices

    def forward(self, predicted_img, img, mask=None):
        if mask is not None:
            predicted_img = predicted_img * mask
            img = img * mask

        predicted_grams = self.get_style_features(predicted_img)
        img_grams = self.get_style_features(img)

        style_loss = 0
        for predicted_gram, img_gram in zip(predicted_grams, img_grams):
            style_loss += F.mse_loss(predicted_gram, img_gram)

        return style_loss


# import torch.nn.functional as F
# import torch
# import torchvision.models as models

# IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
# IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

# class StyleLoss(torch.nn.Module):
#     def __init__(self, normalize_inputs=True):
#         super(StyleLoss, self).__init__()
        
#         self.normalize_inputs = normalize_inputs
#         self.mean_ = IMAGENET_MEAN
#         self.std_ = IMAGENET_STD
        
#         self.vgg = models.vgg19(pretrained=True).features[:35].eval()
#         for param in self.vgg.parameters():
#             param.requires_grad = False

#     def do_normalize_inputs(self, x):
#         return (x - self.mean_.to(x.device)) / self.std_.to(x.device)
            
#     def gram_matrix(self, input):
# #         a, b, c, d = input.size()  # a=batch size(=1)
        
#         a = input.size(0)
#         b = input.size(1)
#         spatial_dim = input.dim() - 2
#         c = input.size(-2)
#         d = input.size(-1)
        
# #         features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
#         features = input.view(a, b, c*d) 
# #         G = torch.mm(features, features.t())  # compute the gram product
#         G = torch.bmm(features, features.transpose(1,2)) # compute the gram product
#         return G.div(a * b * c * d)

#     def forward(self, predicted_img, img, mask):
#         device = predicted_img.device
#         # resize images to (224, 224) and normalize  - NOTE: resize to 224 is necessary because the Gram matrix calculation requires a fixed input size
#         transform = torch.nn.Sequential(
#             torch.nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True),
#         )

#         # preprocess images
#         if self.normalize_inputs:
#             predicted_img = self.do_normalize_inputs(predicted_img)
#             img = self.do_normalize_inputs(img)

#         predicted_img = transform(predicted_img).to(device)
#         img = transform(img).to(device)

#         # extract features using vgg
#         features_predicted = self.vgg(predicted_img)
#         features_img = self.vgg(img)

#         # calculate the style loss using mean squared error (MSE)
#         loss = 0
#         for feat_predicted, feat_img in zip(features_predicted, features_img):
#             gram_predicted = self.gram_matrix(feat_predicted)
#             gram_img = self.gram_matrix(feat_img)
#             loss += F.mse_loss(gram_predicted, gram_img)

#         # apply the mask to the loss
#         loss = torch.mean(loss * mask.to(device))

#         return loss




################################# OLD VERSION ####################################


####################################################################################
