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
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_size),
            nn.Dropout(p=0.4)
        )
        #self.classifier = nn.Linear(embedding_size, num_classes, bias=False)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, label=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        normed_embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize features

        logits = self.classifier(embedding)
        return logits
        '''
        if self.training:
            # L2 normalize classifier weights
            normed_weights = F.normalize(self.classifier.weight, p=2, dim=1)
            cos_theta = F.linear(normed_embedding, normed_weights)  # cosine similarity
            return cos_theta, label  # logits used by SphereFaceLoss
        else:
            return normed_embedding
        '''

