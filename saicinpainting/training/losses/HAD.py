import torch
import torch.nn.functional as F

class HADLoss(torch.nn.Module):
    def __init__(self):
        super(HADLoss, self).__init__()
        self.mask = None


    def forward(self, img, predicted_img,discrim, ori_mask, supervised_mask):
        device = predicted_img.device
        # Compute features for inpainted and ground truth images
        feat_i = discrim(predicted_img)
        feat_g = discrim(img)

        # Convert the mask to a bool type
        ori_mask = ori_mask.bool()
        
        # Compute masked regions
        masked_i = predicted_img.masked_select(ori_mask)
        masked_g = img.masked_select(ori_mask)
        
        if not isinstance(supervised_mask, torch.Tensor):
            supervised_mask = torch.tensor(supervised_mask).to(device)
            
        masked_feat_i = feat_i.masked_select(supervised_mask)
        masked_feat_g = feat_g.masked_select(supervised_mask)

        # Compute HAD loss
        dist_feat = F.pairwise_distance(masked_feat_i, masked_feat_g, p=2)
        dist_pixel = torch.mean(torch.abs(masked_i - masked_g))
        had_loss = torch.mean(dist_feat) - torch.mean(dist_pixel)

        return had_loss
