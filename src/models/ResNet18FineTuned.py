import torch
import torch.nn as nn
from torchvision import models


class ResNet18FineTuned(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        # On enlève avgpool et fc
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        self.lstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(512, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)  # (B, 512, H, W)

        B, C, H, W = x.shape

        # On considère W = temps
        x = x.mean(dim=2)        # (B, 512, W)
        x = x.permute(0, 2, 1)   # (B, W, 512)

        x, _ = self.lstm(x)

        # Attention
        weights = torch.softmax(self.attn(x), dim=1)
        x = (x * weights).sum(dim=1)

        return self.fc(x)