import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchmetrics.image import SpectralAngleMapper

class SpectralLoss(nn.Module):
    """
    Loss = MSE + lambda_sam * SAM

    Inputs:
        pred   : [B, C, H, W]
        target : [B, C, H, W]
    """

    def __init__(self, lambda_sam=0.1):
        super().__init__()
        self.lambda_sam = lambda_sam
        self.sam = SpectralAngleMapper()

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)

        sam = self.sam(pred, target)

        loss = mse + self.lambda_sam * sam

        return loss, mse, sam