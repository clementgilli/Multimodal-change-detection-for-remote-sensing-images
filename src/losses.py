import torch
import torch.nn as nn 
import torch.nn.functional as F

class SpectralLoss(nn.Module):
    """
    Loss = MSE + lambda_sam * SAM
    reduction_mode: 'mean' -> global scalar (default)
                    'per_channel' -> returns tensors per channel
    """

    def __init__(self, lambda_sam=0.1, eps=1e-8, reduction_mode='mean'):
        super().__init__()
        self.lambda_sam = lambda_sam
        self.eps = eps
        self.reduction_mode = reduction_mode

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        
        mse_map = F.mse_loss(pred, target, reduction='none') # [B,C,H,W]
        if self.reduction_mode == 'per_channel':
            mse = mse_map.mean(dim=(0,2,3)) 
        else:
            mse = mse_map.mean()

        dot_product = torch.sum(pred * target, dim=1)
        
        norm_pred = torch.norm(pred, dim=1)
        norm_target = torch.norm(target, dim=1)
        
        cos_sim = dot_product / (norm_pred * norm_target + self.eps)
        
        cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)
        
        sam_map = torch.acos(cos_sim)
        sam = torch.mean(sam_map)
        #sam = torch.Tensor([0.]).to(pred.device)
        loss = mse + self.lambda_sam * sam

        return loss, mse, sam

