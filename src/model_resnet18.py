from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models

from utils.melresnetdataset import MelResNetDataset


class ResNet18FineTuned(nn.Module):
    """
    ResNet18 pré-entraîné, adapté à 5 classes.
    """

    def __init__(self, num_classes=5, freeze_backbone=True):
        super().__init__()

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
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True


class ResNet18Trainer:
    """
    Classe dédiée à l'entraînement et à l'évaluation de ResNet18 sur Mel-spectrogrammes.
    """

    def __init__(self, batch_size=16, num_workers=0, pin_memory=True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def load_data(self):
        data_file = Path(__file__).resolve().parent.parent / "data" / "spectres.npy"
        self.spectres = np.load(data_file, allow_pickle=True).item()

        self.mels = self.spectres["mel"]
        self.labels = self.spectres["labels"]

        assert len(self.mels) == len(self.labels), "Désalignement entre mels et labels dans spectres.npy"

        self.le = LabelEncoder()
        self.y_enc = self.le.fit_transform(self.labels)

        indices = np.arange(len(self.mels))

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        self.train_idx, temp_idx = next(sss1.split(indices, self.y_enc))

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_sub_idx, test_sub_idx = next(sss2.split(temp_idx, self.y_enc[temp_idx]))

        self.val_idx = temp_idx[val_sub_idx]
        self.test_idx = temp_idx[test_sub_idx]

        print(f"Train : {len(self.train_idx)} | Val : {len(self.val_idx)} | Test : {len(self.test_idx)}")

    def build_loaders(self):
        self.full_dataset = MelResNetDataset(self.spectres, self.y_enc, target_size=(224, 224))

        self.train_loader = DataLoader(
            Subset(self.full_dataset, self.train_idx),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        self.val_loader = DataLoader(
            Subset(self.full_dataset, self.val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        self.test_loader = DataLoader(
            Subset(self.full_dataset, self.test_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def _train_epoch(self, model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        correct = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(dim=1) == y_batch).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)

    def _eval_epoch(self, model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        correct = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                out = model(X_batch)
                loss = criterion(out, y_batch)

                total_loss += loss.item()
                correct += (out.argmax(dim=1) == y_batch).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)

    def export_onnx(self, onnx_path="models/resnet18_mel_finetuned.onnx", input_shape=(1, 3, 224, 224), opset_version=18):
        if not hasattr(self, "model"):
            raise RuntimeError("Le modèle n'est pas initialisé. Lance l'entraînement ou charge un checkpoint avant export ONNX.")

        model_file = Path(onnx_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parent / model_file
        model_file.parent.mkdir(parents=True, exist_ok=True)

        device = next(self.model.parameters()).device
        dummy_input = torch.randn(*input_shape, device=device)

        self.model.eval()
        export_kwargs = dict(
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        with torch.no_grad():
            try:
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    str(model_file),
                    dynamo=False,
                    **export_kwargs,
                )
            except TypeError:
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    str(model_file),
                    **export_kwargs,
                )

        print(f"Modèle ONNX exporté dans {model_file}")

    def train(self, epochs_head=5, epochs_finetune=10, save_model=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device utilisée : {device}")

        if device.type == "cpu":
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            print(f"CPU threads: {torch.get_num_threads()}")

        self.load_data()
        self.build_loaders()

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.y_enc),
            y=self.y_enc[self.train_idx]
        )
        weights_tensor = torch.FloatTensor(class_weights).to(device)

        self.model = ResNet18FineTuned(
            num_classes=len(self.le.classes_),
            freeze_backbone=True
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        # Phase 1 : entraînement de la tête uniquement
        print("\n=== Phase 1 : entraînement de la tête ===")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        for epoch in range(epochs_head):
            train_loss, train_acc = self._train_epoch(self.model, self.train_loader, criterion, optimizer, device)
            val_loss, val_acc = self._eval_epoch(self.model, self.val_loader, criterion, device)
            scheduler.step(val_loss)

            print(
                f"[HEAD] Epoch {epoch+1:02d} | "
                f"Train loss {train_loss:.3f} acc {train_acc:.3f} | "
                f"Val loss {val_loss:.3f} acc {val_acc:.3f}"
            )

        # Phase 2 : fine-tuning de layer4 + fc
        print("\n=== Phase 2 : fine-tuning de layer4 + fc ===")
        self.model.unfreeze_last_block()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        for epoch in range(epochs_finetune):
            train_loss, train_acc = self._train_epoch(self.model, self.train_loader, criterion, optimizer, device)
            val_loss, val_acc = self._eval_epoch(self.model, self.val_loader, criterion, device)
            scheduler.step(val_loss)

            print(
                f"[FT] Epoch {epoch+1:02d} | "
                f"Train loss {train_loss:.3f} acc {train_acc:.3f} | "
                f"Val loss {val_loss:.3f} acc {val_acc:.3f}"
            )

        if save_model:
            model_file = Path(__file__).resolve().parent / "models" / "resnet18_mel_finetuned.pth"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), model_file)
            print(f"Modèle sauvegardé dans {model_file}")

            self.export_onnx(onnx_path="models/resnet18_mel_finetuned.onnx")

    def evaluate(self, model_path="models/resnet18_mel_finetuned.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not hasattr(self, "test_loader"):
            self.load_data()
            self.build_loaders()

        if not hasattr(self, "model"):
            self.model = ResNet18FineTuned(
                num_classes=len(self.le.classes_),
                freeze_backbone=False
            ).to(device)

        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parent / model_file

        try:
            state_dict = torch.load(model_file, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_file, map_location=device)

        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        all_preds, all_probs, all_true = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                out = self.model(X_batch)
                probs = torch.softmax(out, dim=1)

                all_preds.extend(out.argmax(1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)

        print(classification_report(
            all_true,
            all_preds,
            target_names=self.le.classes_,
            zero_division=0
        ))

        f1_macro = f1_score(all_true, all_preds, average="macro")
        print(f"Macro F1-score : {f1_macro:.3f}")

        auc = roc_auc_score(all_true, all_probs, multi_class="ovr", average="macro")
        print(f"Macro AUC-ROC  : {auc:.3f}")

        cm = confusion_matrix(all_true, all_preds)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.le.classes_,
            yticklabels=self.le.classes_
        )
        plt.title("Matrice de confusion — test set")
        plt.ylabel("Réel")
        plt.xlabel("Prédit")
        plt.tight_layout()
        plt.show()