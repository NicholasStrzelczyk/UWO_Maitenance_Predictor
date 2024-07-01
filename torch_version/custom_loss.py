from torch import nn


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets)
        pt = targets * inputs + (1 - targets) * (1 - inputs)
        focal_bce = ((1.0 - pt) ** self.gamma) * bce_loss
        if self.alpha is not None:
            weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_bce = weight * focal_bce
        return focal_bce.mean()
