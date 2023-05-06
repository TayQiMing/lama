import torch
import torch.nn as nn

class PatchGANLoss(nn.Module):
    def __init__(self):
        super(PatchGANLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, real_preds, fake_preds):
        device = real_preds.device
        # Create target labels (1 for real, 0 for fake)
        real_labels = torch.ones_like(real_preds).to(device)
        fake_labels = torch.zeros_like(fake_preds).to(device)
        
        # Compute the binary cross-entropy loss for the real and fake predictions
        real_loss = self.criterion(real_preds, real_labels)
        fake_loss = self.criterion(fake_preds, fake_labels)
        
        # Combine the losses and return the average
        loss = (real_loss + fake_loss) / 2
        return loss
