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
            N, C, H, W = f.size()
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
