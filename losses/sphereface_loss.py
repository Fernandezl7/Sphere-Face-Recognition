import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SphereFaceLoss(nn.Module):
    def __init__(self, m=4, s=30.0):
        """
        Angular Softmax (SphereFace) Loss

        Args:
            m (int): angular margin (default: 4)
            s (float): scale factor (default: 30.0)
        """
        super(SphereFaceLoss, self).__init__()
        self.m = m
        self.s = s
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # Normalize logits
        logits_norm = F.normalize(logits, dim=1)

        # Gather the angle for correct classes
        batch_size = logits.size(0)
        theta = torch.acos(torch.clamp(logits_norm[range(batch_size), labels], -1.0 + 1e-7, 1.0 - 1e-7))
        k = (self.m * theta / math.pi).floor()
        phi_theta = torch.cos(self.m * theta) * ((-1.0) ** k) - 2 * k

        # Create output tensor
        output = logits_norm.clone()
        output[range(batch_size), labels] = phi_theta

        return self.criterion(self.s * output, labels)
