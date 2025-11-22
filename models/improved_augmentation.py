"""
Proper batch-level MixUp and CutMix implementations
"""

import torch
import torch.nn as nn
import numpy as np


class BatchMixUp(nn.Module):
    """Batch-level MixUp - one lambda for entire batch"""
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        if self.alpha > 0 and self.training:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)

            mixed_x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
        return x, y, y, 1.0


class BatchCutMix(nn.Module):
    """Batch-level CutMix - one lambda for entire batch"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        if self.alpha > 0 and self.training:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)

            # Get random box
            _, _, H, W = x.shape
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            # Apply cutmix
            mixed_x = x.clone()
            mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

            # Adjust lambda based on actual box size
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
        return x, y, y, 1.0


class MixedAugmentation(nn.Module):
    """Randomly apply MixUp or CutMix"""
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        super().__init__()
        self.mixup = BatchMixUp(mixup_alpha)
        self.cutmix = BatchCutMix(cutmix_alpha)
        self.prob = prob

    def forward(self, x, y):
        if self.training and np.random.rand() < self.prob:
            if np.random.rand() < 0.5:
                return self.mixup(x, y)
            else:
                return self.cutmix(x, y)
        return x, y, y, 1.0


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixed loss for mixup/cutmix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
