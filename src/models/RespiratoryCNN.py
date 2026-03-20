import torch
import torch.nn as nn

class RespiratoryCNN(nn.Module):
    def __init__(self, num_classes=5, in_channels=6):
        super().__init__()

        # Bloc 1 — détecte les patterns simples (bords, textures)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 128×259 → 64×129
            nn.Dropout2d(0.25)
        )

        # Bloc 2 — détecte les patterns plus complexes
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 64×129 → 32×64
            nn.Dropout2d(0.25)
        )

        # Bloc 3 — patterns de haut niveau
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # → 128×4×4 = 2048 features
            nn.Dropout2d(0.25)
        )

        # Classifieur final
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Vérifier les dimensions (6 canaux : mel, mfcc, centroid, bandwidth, zcr, chroma)
    model = RespiratoryCNN(num_classes=5, in_channels=6)
    dummy = torch.randn(8, 6, 128, 259)   # batch=8, canaux=6, H=128, W=259
    out = model(dummy)
    print(f"Sortie : {out.shape}")