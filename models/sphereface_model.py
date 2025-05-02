import torch
import torch.nn as nn
import torch.nn.functional as F

class SphereFaceNet(nn.Module):
    def __init__(self, embedding_size=512, num_classes=10000):
        super(SphereFaceNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x, label=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        normed_embedding = F.normalize(embedding)
        logits = self.classifier(normed_embedding)

        if self.training:
            return logits, label  # for classification
        else:
            return normed_embedding  # for inference
