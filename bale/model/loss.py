import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class ClipLoss(nn.Module):
    def __init__( self):
        super().__init__()

    def forward(self, logits_per_image, logits_per_brain, *args):
        device = logits_per_image.device
        labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_brain, labels)
                     ) / 2

        return total_loss



class SigmoidContrastiveLoss(nn.Module):
    def __init__( self):
        super().__init__()
        self.max_tau = 5

    def forward(self, logits_per_image, logits_per_brain, tau, b):
        n = logits_per_brain.size(0)
        device = logits_per_brain.device
        labels = 2 * torch.eye(n, device=device) - 1

        logits = logits_per_brain
        loss = -torch.sum(F.logsigmoid(labels * logits)) / n

        # return invariance_loss for debug
        return loss


class MSELoss(nn.Module):
    def __init__( self):
        super().__init__()
        self.max_tau = 5

    def forward(self, logits_per_image, logits_per_brain, *args):
        return nn.MSELoss()(logits_per_image, logits_per_brain)
