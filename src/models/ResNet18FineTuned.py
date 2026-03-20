import torch
import torch.nn as nn
from torchvision import models


class ResNet18FineTuned(nn.Module):
    """
    ResNet18 pré-entraîné, adapté à 5 classes.
    """

    def __init__(self, num_classes=5, freeze_backbone=True):
        super().__init__()

        # Compatibilité torchvision ancienne / récente
        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet18(pretrained=True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_last_block(self):
        """
        Dégèle layer4 + fc pour le vrai fine-tuning.
        """
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True