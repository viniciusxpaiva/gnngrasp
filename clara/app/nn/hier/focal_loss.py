import torch.nn as nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probas = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probas, 1 - probas)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        return loss.mean() if self.reduction == "mean" else loss.sum()
