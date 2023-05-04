import torch
import torch.nn.functional as F

class MSSSIMLoss(torch.nn.Module):
    def __init__(self,window_size=11, size_average=True, full=False, weights=None):
        super(MSSSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.full = full
        self.weights = weights

    def ssim(self,img1, img2):
        """
        Computes the Structural Similarity Index (SSIM) between two images.
        """
        device = img1.device

        # Define window
        window = torch.tensor([[0.1520, 0.2196, 0.1520],
                               [0.2196, 0.3183, 0.2196],
                               [0.1520, 0.2196, 0.1520]], dtype=torch.float32, device=device)
        window = window.view(1, 1, self.window_size, self.window_size).repeat(1, 3, 1, 1)

        # Compute means and variances
        mu1 = F.conv2d(img1, window, stride=1, padding=self.window_size//2, groups=3)
        mu2 = F.conv2d(img2, window, stride=1, padding=self.window_size//2, groups=3)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, stride=1, padding=self.window_size//2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=self.window_size//2, groups=3) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, stride=1, padding=self.window_size//2, groups=3) - mu1_mu2

        # Compute SSIM
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


    def forward(self, img1, img2):
        """
        Computes the Multi-Scale Structural Similarity Index (MS-SSIM) loss between two images.
        img1: groundtruth image
        img2: predicted image
        """
        levels = 5
        device = img1.device

        # Define weights for each level
        if self.weights is None:
            self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=img1.device)

        msssim_val = torch.tensor(1.0, device=img1.device)
        for i in range(levels):
            ssim_map = ssim(img1, img2)
            msssim_val *= ssim_map ** self.weights[i]

            # Downsample images
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

        # Compute MS-SSIM
        msssim_val = torch.clamp(msssim_val, min=1e-5, max=1)
        if self.size_average:
            loss = -torch.log(msssim_val).mean()
        else:
            loss = -torch.log(msssim_val)

        return loss
