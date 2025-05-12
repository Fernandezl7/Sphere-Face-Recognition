import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SphereFaceLoss(nn.Module):
    def __init__(self, m=4, s=30.0):
        super(SphereFaceLoss, self).__init__()
        self.m = m  # angular margin
        self.s = s  # scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cos_theta, labels):
        # Clamp to valid acos range
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        theta = torch.acos(cos_theta)
        k = (self.m * theta / math.pi).floor()
        phi_theta = torch.cos(self.m * theta) * ((-1.0) ** k) - 2 * k

        one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).to(cos_theta.dtype)
        output = cos_theta * (1 - one_hot) + phi_theta * one_hot

        return self.ce(self.s * output, labels)
