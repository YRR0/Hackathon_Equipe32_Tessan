import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTMAttention(nn.Module):
    """
    Architecture CNN-BiLSTM-Attention pour la classification
    de sons respiratoires.

    - CNN      : extrait les patterns spatiaux du spectrogramme
    - BiLSTM   : capture les dépendances temporelles dans les deux sens
    - Attention: pondère les frames temporelles les plus discriminantes
    """

    def __init__(self, num_classes=5, in_channels=6, dropout=0.25):
        super().__init__()

        # ── Bloc CNN — extraction de features spatiales ───────────
        self.cnn = nn.Sequential(
            # Bloc 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # (6,128,259) → (32,64,129)
            nn.Dropout2d(dropout),

            # Bloc 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # (32,64,129) → (64,32,64)
            nn.Dropout2d(dropout),

            # Bloc 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),      # (64,32,64) → (128,16,64)
            nn.Dropout2d(dropout)
        )

        # ── BiLSTM — capture des patterns temporels ───────────────
        # Entrée : (batch, time=64, features=128*16=2048)
        self.lstm_input_size = 128 * 16
        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,       # → sortie : (batch, time, 256)
            dropout=dropout
        )

        # ── Attention — pondération temporelle ────────────────────
        # Apprend quelles frames sont les plus importantes
        self.attention = nn.Sequential(
            nn.Linear(256, 128),      # 256 = 128 * 2 (bidirectionnel)
            nn.Tanh(),
            nn.Linear(128, 1)         # score par frame
        )

        # ── Classifieur final ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x : (batch, in_channels, H=128, W=259)

        # ── CNN ───────────────────────────────────────────────────
        x = self.cnn(x)
        # x : (batch, 128, 16, 64)

        batch, C, H, W = x.shape

        # Reformater pour le LSTM : (batch, time=W, features=C*H)
        x = x.permute(0, 3, 1, 2)            # (batch, 64, 128, 16)
        x = x.reshape(batch, W, C * H)        # (batch, 64, 2048)

        # ── BiLSTM ────────────────────────────────────────────────
        x, _ = self.bilstm(x)
        # x : (batch, 64, 256)

        # ── Attention ─────────────────────────────────────────────
        scores  = self.attention(x)            # (batch, 64, 1)
        weights = F.softmax(scores, dim=1)     # normalisation sur time
        context = (weights * x).sum(dim=1)     # (batch, 256) — vecteur résumé

        # ── Classification ────────────────────────────────────────
        out = self.classifier(context)         # (batch, num_classes)

        return out


if __name__ == "__main__":
    model = CNNBiLSTMAttention(num_classes=5, in_channels=6)
    dummy = torch.randn(8, 6, 128, 259)
    out   = model(dummy)
    print(f"Sortie : {out.shape}")             # → torch.Size([8, 5])

    # Compter les paramètres
    total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres : {total:,}")           # → ~2.5M