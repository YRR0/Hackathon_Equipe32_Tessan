import torch
import torch.nn as nn
from torchvision import models

class ResNet18FineTuned(nn.Module):
    """
    Modele hybride: embedding image (ResNet18) + embedding tabulaire, puis concat et classification.
    """

    def __init__(
        self,
        num_classes=5,
        freeze_backbone=True,
        tabular_dim=0,
        image_embed_dim=256,
        tabular_hidden_dim=128,
        dropout=0.2,
    ):
        super().__init__()

        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet18(pretrained=True)

        self.tabular_dim = int(tabular_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.image_head = nn.Sequential(
            nn.Linear(in_features, image_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        if self.tabular_dim > 0:
            self.tabular_head = nn.Sequential(
                nn.Linear(self.tabular_dim, tabular_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            classifier_in = image_embed_dim + tabular_hidden_dim
        else:
            self.tabular_head = None
            classifier_in = image_embed_dim

        self.classifier = nn.Linear(classifier_in, num_classes)

    def forward(self, x, tabular=None):
        img_feat = self.backbone(x)
        img_feat = self.image_head(img_feat)

        if self.tabular_head is not None:
            if tabular is None:
                tabular = torch.zeros(x.size(0), self.tabular_dim, device=x.device, dtype=img_feat.dtype)
            tab_feat = self.tabular_head(tabular)
            fused = torch.cat([img_feat, tab_feat], dim=1)
        else:
            fused = img_feat

        return self.classifier(fused)

    def unfreeze_last_block(self):
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.image_head.parameters():
            param.requires_grad = True
        if self.tabular_head is not None:
            for param in self.tabular_head.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True


