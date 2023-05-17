import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class HADLoss(torch.nn.Module):
    def __init__(self):
        super(HADLoss, self).__init__()

        # Load pre-trained VGG-16 model
        self.model = vgg16(pretrained=True).features.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img, predicted_img, ori_mask, supervised_mask):
        device = predicted_img.device

        # Convert the mask to a float type
        ori_mask = ori_mask.float()
        supervised_mask = supervised_mask.float()

        # Apply the mask to the images
        masked_i = img * ori_mask
        masked_g = predicted_img * ori_mask

        # Apply the mask to the supervised region
        masked_supervised = supervised_mask * ori_mask

        # Expand masked_supervised to match the number of channels expected by the model
        masked_supervised = masked_supervised.expand(-1, 3, -1, -1)

        # Resize the masked_g tensor to have the same size as masked_supervised
        masked_g_resized = F.interpolate(masked_g, size=masked_supervised.size()[2:], mode='bilinear', align_corners=False)

        # Compute feature maps
        feat_supervised = self.model(masked_supervised)
        feat_g = self.model(masked_g_resized)

        # Compute HAD loss
        dist_feat = torch.sqrt(torch.sum(torch.pow(feat_supervised - feat_g, 2), dim=1))
        dist_pixel = torch.mean(torch.abs(masked_i - masked_g))
        had_loss = torch.mean(dist_feat) - torch.mean(dist_pixel)

        return had_loss




# import torch
# import torch.nn.functional as F

# class HADLoss(torch.nn.Module):
#     def __init__(self):
#         super(HADLoss, self).__init__()

#     def forward(self, img, predicted_img, ori_mask, supervised_mask):
#         device = predicted_img.device

#         # Convert the mask to a bool type
#         ori_mask = ori_mask.bool()
#         supervised_mask = supervised_mask.bool()

#         # Apply the mask to the images
#         masked_i = img * ori_mask
#         masked_g = predicted_img * ori_mask

#         # Apply the mask to the supervised region
#         masked_supervised = supervised_mask * ori_mask

#         # Compute HAD loss
#         dist_feat = F.pairwise_distance(masked_supervised.view(masked_supervised.size(0), -1),
#                                         masked_g.view(masked_g.size(0), -1), p=2)
#         dist_pixel = torch.mean(torch.abs(masked_i - masked_g))
#         had_loss = torch.mean(dist_feat) - torch.mean(dist_pixel)

#         return had_loss


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class HoleAdvantageousDiscriminativeLoss(nn.Module):
#     def __init__(self):
#         super(HoleAdvantageousDiscriminativeLoss, self).__init__()
        

#     def forward(self, discriminator, generator_output, target_image, mask):
#         # Generate inpainted images
#         inpainted_images = generator_output

#         # Discriminate real and inpainted images
#         real_preds = discriminator(target_image)
#         inpainted_preds = discriminator(inpainted_images)[0]

#         # Calculate adversarial loss
#         adversarial_loss = -torch.mean(torch.log(inpainted_preds))

#         # Calculate discriminative loss
#         discriminative_loss = F.mse_loss(inpainted_preds, real_preds)

#         # Apply mask to focus on the inpainted regions
#         masked_inpainted_images = inpainted_images * mask
#         masked_target_image = target_image * mask

#         # Calculate advantageous loss
#         advantageous_loss = F.l1_loss(masked_inpainted_images, masked_target_image)

#         # Calculate the total loss
#         total_loss = adversarial_loss + discriminative_loss + advantageous_loss

#         return total_loss






# import torch
# import torch.nn.functional as F
# import numpy as np

# class HADLoss(torch.nn.Module):
#     def __init__(self):
#         super(HADLoss, self).__init__()

#     def forward(self, feature_i, feature_g, img, predicted_img, discrim, ori_mask, supervised_mask):
#         device = predicted_img.device
# #         aaa = np.array(feature_i)
# #         bbb = np.array(feature_g)
# #         print(aaa.shape)
# #         print("AND ", bbb.shape)

#         # Convert the mask to a bool type
#         ori_mask = ori_mask.bool()
#         supervised_mask = supervised_mask.bool()

#         # Apply the mask to the images
#         masked_i = img * ori_mask
#         masked_g = predicted_img * ori_mask

# #         # Apply the mask to the features
# #         masked_feat_i = []
# #         masked_feat_g = []
# #         for feat_i, feat_g in zip(feature_i, feature_g):
# #             masked_feat_i.append(feat_i * supervised_mask)
# #             masked_feat_g.append(feat_g * supervised_mask)

#         # Stack the masked features tensors along the channel dimension
#         masked_feat_i = torch.stack(feature_i, dim=1)
#         masked_feat_g = torch.stack(feature_g, dim=1)
        
#         # Apply the mask to the features
#         masked_feat_i = masked_feat_i * supervised_mask.unsqueeze(2).unsqueeze(3)
#         masked_feat_g = masked_feat_g * supervised_mask.unsqueeze(2).unsqueeze(3)


#         # Compute HAD loss
#         dist_feat = F.pairwise_distance(masked_feat_i, masked_feat_g, p=2)
#         dist_pixel = torch.mean(torch.abs(masked_i - masked_g))
#         had_loss = torch.mean(dist_feat) - torch.mean(dist_pixel)

#         return had_loss


# import torch
# import torch.nn.functional as F
# import numpy as np

# class HADLoss(torch.nn.Module):
#     def __init__(self):
#         super(HADLoss, self).__init__()

#     def forward(self, feature_i, feature_g, img, predicted_img, discrim, ori_mask, supervised_mask):
#         device = predicted_img.device

#         # Convert the mask to a bool type
#         ori_mask = ori_mask.bool()
#         supervised_mask = supervised_mask.bool()

#         # Apply the mask to the images
#         masked_i = img * ori_mask
#         masked_g = predicted_img * ori_mask

#         # Resize the feature tensors to have the same number of channels
#         num_channels_i = feature_i[0].size(1)  # Get the number of channels in feature_i
#         num_channels_g = feature_g[0].size(1)  # Get the number of channels in feature_g

#         feature_i_resized = []
#         feature_g_resized = []
#         for feat_i, feat_g in zip(feature_i, feature_g):
#             if feat_i.size(1) != num_channels_i:
#                 feat_i = self.resize_channels(feat_i, num_channels_i)
#             if feat_g.size(1) != num_channels_g:
#                 feat_g = self.resize_channels(feat_g, num_channels_g)
#             feature_i_resized.append(feat_i)
#             feature_g_resized.append(feat_g)

#         # Stack the masked features tensors along the channel dimension
#         masked_feat_i = torch.stack(feature_i_resized, dim=1)
#         masked_feat_g = torch.stack(feature_g_resized, dim=1)

#         # Apply the mask to the features
#         masked_feat_i = masked_feat_i * supervised_mask.unsqueeze(2).unsqueeze(3)
#         masked_feat_g = masked_feat_g * supervised_mask.unsqueeze(2).unsqueeze(3)

#         # Compute HAD loss
#         dist_feat = F.pairwise_distance(masked_feat_i, masked_feat_g, p=2)
#         dist_pixel = torch.mean(torch.abs(masked_i - masked_g))
#         had_loss = torch.mean(dist_feat) - torch.mean(dist_pixel)

#         return had_loss

#     def resize_channels(self, tensor, num_channels):
#         # Resize the tensor to have the specified number of channels
#         num_current_channels = tensor.size(1)
#         if num_current_channels == num_channels:
#             return tensor
#         elif num_current_channels > num_channels:
#             return tensor[:, :num_channels]
#         else:
#             pad_channels = num_channels - num_current_channels
#             padding = torch.zeros((tensor.size(0), pad_channels, *tensor.shape[2:]), device=tensor.device)
#             return torch.cat([tensor, padding], dim=1)


# this just 64 and 128 diff
# import torch
# import torch.nn.functional as F
# import numpy as np

# class HADLoss(torch.nn.Module):
#     def __init__(self):
#         super(HADLoss, self).__init__()

#     def forward(self, feature_i, feature_g, img, predicted_img, discrim, ori_mask, supervised_mask):
#         device = predicted_img.device

#         # Convert the mask to a bool type
#         ori_mask = ori_mask.bool()
#         supervised_mask = supervised_mask.bool()

#         # Apply the mask to the images
#         masked_i = img * ori_mask
#         masked_g = predicted_img * ori_mask

#         # Resize the feature tensors to have the same size
#         size_i = feature_i[0].size()[-2:]  # Get the spatial size of feature_i
#         size_g = feature_g[0].size()[-2:]  # Get the spatial size of feature_g

#         feature_i_resized = []
#         feature_g_resized = []
#         for feat_i, feat_g in zip(feature_i, feature_g):
#             feat_i_resized = F.interpolate(feat_i, size=size_i, mode='bilinear', align_corners=False)
#             feat_g_resized = F.interpolate(feat_g, size=size_g, mode='bilinear', align_corners=False)
#             feature_i_resized.append(feat_i_resized)
#             feature_g_resized.append(feat_g_resized)

#         # Stack the masked features tensors along the channel dimension
#         masked_feat_i = torch.stack(feature_i_resized, dim=1)
#         masked_feat_g = torch.stack(feature_g_resized, dim=1)
        
#         # Apply the mask to the features
#         masked_feat_i = masked_feat_i * supervised_mask.unsqueeze(2).unsqueeze(3)
#         masked_feat_g = masked_feat_g * supervised_mask.unsqueeze(2).unsqueeze(3)

#         # Compute HAD loss
#         dist_feat = F.pairwise_distance(masked_feat_i, masked_feat_g, p=2)
#         dist_pixel = torch.mean(torch.abs(masked_i - masked_g))
#         had_loss = torch.mean(dist_feat) - torch.mean(dist_pixel)

#         return had_loss


# import torch
# import torch.nn.functional as F

# class HADLoss(torch.nn.Module):
#     def __init__(self):
#         super(HADLoss, self).__init__()
#         self.mask = None


#     def forward(self,feature_i,feature_g, img, predicted_img,discrim, ori_mask, supervised_mask):
#         device = predicted_img.device
# # #         print(len(feature_i[0][0][0]),len(feature_i[0][0][0][0]),len(feature_i[0][0][0][0]))
# # #         print("AND " , len(feature_i[1][0][0]),len(feature_i[1][0][0][0]),len(feature_i[1][0][0][0]))
# # #         print("ALSO ",len(feature_i[0][0][0][0]),len(feature_i[1][0][0][0]))
# #         # Compute features for inpainted and ground truth images
# # #         _, feat_i = discrim(predicted_img)
# # #         _, feat_g = discrim(img)
# # #         feat_i = torch.stack(feature_i)
# #         feat_i = torch.nn.functional.interpolate(feature_i, size=(feature_g.size(2), feature_g.size(3)), mode='bilinear', align_corners=False)
# #         feat_g = torch.stack(feature_g)
        
        
        
        
#         # Convert the mask to a bool type
        
#         ori_mask = ori_mask.bool()
#         supervised_mask = supervised_mask.bool()
        
        
#         masked_feat_i = []
#         masked_feat_g = []
#         for feat_i, feat_g in zip(feature_i, feature_g):
#             masked_feat_i.append(feat_i.masked_select(supervised_mask.view(supervised_mask.size(0), supervised_mask.size(1), 1, 1)))
#             masked_feat_g.append(feat_g.masked_select(supervised_mask.view(supervised_mask.size(0), supervised_mask.size(1), 1, 1)))

#         # Convert the masked features lists to tensors
#         masked_feat_i = torch.cat(masked_feat_i)
#         masked_feat_g = torch.cat(masked_feat_g)
        
        
        
#         # Compute masked regions
#         masked_i = predicted_img.masked_select(ori_mask)
#         masked_g = img.masked_select(ori_mask)
        
# # #         if not isinstance(supervised_mask, torch.Tensor):
# # #             supervised_mask = torch.tensor(supervised_mask).to(device)
            
# # #         masked_feat_i = feat_i.masked_select(supervised_mask)
# # #         masked_feat_g = feat_g.masked_select(supervised_mask)
# #         masked_feat_i = feat_i.masked_select(supervised_mask.view(supervised_mask.size(0), supervised_mask.size(1), 1, 1)) # if cannot then see the shape of
# #         masked_feat_g = feat_g.masked_select(supervised_mask.view(supervised_mask.size(0), supervised_mask.size(1), 1, 1)) # feat_i and spv_mask, or channel

#         # Compute HAD loss
#         dist_feat = F.pairwise_distance(masked_feat_i, masked_feat_g, p=2)
#         dist_pixel = torch.mean(torch.abs(masked_i - masked_g))
#         had_loss = torch.mean(dist_feat) - torch.mean(dist_pixel)

#         return had_loss
