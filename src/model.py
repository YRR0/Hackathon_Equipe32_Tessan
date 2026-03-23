from pathlib import Path
from itertools import product
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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
        self._cpu_threads_configured = False
        self.train_aug_params = {
            "gaussian_noise_std": 0.01,
            "time_shift_max": 12,
            "pitch_shift_max": 4,
            "num_time_masks": 1,
            "num_freq_masks": 1,
            "time_mask_max": 20,
            "freq_mask_max": 8,
        }

    @staticmethod
    def _log(msg):
        print(msg, flush=True)

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

        self._log(f"Train : {len(self.train_idx)} | Val : {len(self.val_idx)} | Test : {len(self.test_idx)}")

    def build_loaders(self):
        train_aug = dict(self.train_aug_params)
        self.train_dataset = MelResNetDataset(
            self.spectres,
            self.y_enc,
            target_size=(224, 224),
            augment=True,
            **train_aug,
        )
        self.eval_dataset = MelResNetDataset(
            self.spectres,
            self.y_enc,
            target_size=(224, 224),
            augment=False,
        )

        self.train_loader = DataLoader(
            Subset(self.train_dataset, self.train_idx),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        self.val_loader = DataLoader(
            Subset(self.eval_dataset, self.val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        self.test_loader = DataLoader(
            Subset(self.eval_dataset, self.test_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    @staticmethod
    def _set_seed(seed):
        if seed is None:
            return
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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

    def train(
        self,
        epochs_head=5,
        epochs_finetune=10,
        save_model=True,
        use_class_weights=True,
        lr_head=1e-3,
        lr_finetune=1e-4,
        weight_decay_head=0.0,
        weight_decay_finetune=0.0,
        scheduler_patience=3,
        scheduler_factor=0.5,
        log_prefix="",
        train_indices=None,
        val_indices=None,
        seed=None,
        verbose_epochs=False,
    ):
        self._set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._log(f"{log_prefix}Device utilisée : {device}")

        if device.type == "cpu" and not self._cpu_threads_configured:
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            self._cpu_threads_configured = True
            self._log(f"{log_prefix}CPU threads: {torch.get_num_threads()}")

        self.load_data()

        if train_indices is not None and val_indices is not None:
            self.train_idx = np.asarray(train_indices)
            self.val_idx = np.asarray(val_indices)

        self.build_loaders()

        weights_tensor = None
        if use_class_weights:
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(self.y_enc),
                y=self.y_enc[self.train_idx]
            )
            weights_tensor = torch.FloatTensor(class_weights).to(device)
            self._log(f"{log_prefix}Class weights: {np.round(class_weights, 3)}")

        self.model = ResNet18FineTuned(
            num_classes=len(self.le.classes_),
            freeze_backbone=True
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        best_val_acc = 0.0
        best_val_loss = float("inf")
        history = []

        # Phase 1 : entraînement de la tête uniquement
        if verbose_epochs:
            self._log(f"\n{log_prefix}=== Phase 1 : entraînement de la tête ===")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr_head,
            weight_decay=weight_decay_head,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=scheduler_patience, factor=scheduler_factor
        )

        for epoch in range(epochs_head):
            train_loss, train_acc = self._train_epoch(self.model, self.train_loader, criterion, optimizer, device)
            val_loss, val_acc = self._eval_epoch(self.model, self.val_loader, criterion, device)
            scheduler.step(val_loss)

            if verbose_epochs:
                self._log(
                    f"{log_prefix}[HEAD] Epoch {epoch+1:02d} | "
                    f"Train loss {train_loss:.3f} acc {train_acc:.3f} | "
                    f"Val loss {val_loss:.3f} acc {val_acc:.3f}"
                )
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            history.append({
                "phase": "head",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

        # Phase 2 : fine-tuning de layer4 + fc
        if verbose_epochs:
            self._log(f"\n{log_prefix}=== Phase 2 : fine-tuning de layer4 + fc ===")
        self.model.unfreeze_last_block()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr_finetune,
            weight_decay=weight_decay_finetune,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=scheduler_patience, factor=scheduler_factor
        )

        for epoch in range(epochs_finetune):
            train_loss, train_acc = self._train_epoch(self.model, self.train_loader, criterion, optimizer, device)
            val_loss, val_acc = self._eval_epoch(self.model, self.val_loader, criterion, device)
            scheduler.step(val_loss)

            if verbose_epochs:
                self._log(
                    f"{log_prefix}[FT] Epoch {epoch+1:02d} | "
                    f"Train loss {train_loss:.3f} acc {train_acc:.3f} | "
                    f"Val loss {val_loss:.3f} acc {val_acc:.3f}"
                )
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            history.append({
                "phase": "finetune",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

        if not verbose_epochs:
            self._log(
                f"{log_prefix}Résumé entraînement | "
                f"best_val_acc={best_val_acc:.4f} | best_val_loss={best_val_loss:.4f}"
            )

        if save_model:
            model_file = Path(__file__).resolve().parent / "models" / "resnet18_mel_finetuned.pth"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), model_file)
            self._log(f"{log_prefix}Modèle sauvegardé dans {model_file}")

            self.export_onnx(onnx_path="models/resnet18_mel_finetuned.onnx")

        return {
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "last_train_acc": history[-1]["train_acc"] if history else None,
            "last_val_acc": history[-1]["val_acc"] if history else None,
            "history": history,
        }

    @staticmethod
    def _expand_param_grid(param_grid):
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def grid_search(
        self,
        param_grid,
        metric="best_val_acc",
        maximize=True,
        save_results=True,
        results_path="models/resnet18_grid_search_results.csv",
        max_trials=None,
        random_state=42,
        cv_folds=1,
        log_epochs=True,
    ):
        if not isinstance(param_grid, dict) or len(param_grid) == 0:
            raise ValueError("param_grid doit être un dict non vide de listes de valeurs.")

        for key, value in param_grid.items():
            if not isinstance(value, (list, tuple)) or len(value) == 0:
                raise ValueError(f"param_grid['{key}'] doit être une liste/tuple non vide.")

        model_dir = Path(__file__).resolve().parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        if cv_folds < 1:
            raise ValueError("cv_folds doit être >= 1.")

        cv_pool_idx = None
        cv_pool_y = None
        if cv_folds > 1:
            self.load_data()
            cv_pool_idx = np.concatenate([self.train_idx, self.val_idx])
            cv_pool_y = self.y_enc[cv_pool_idx]
            class_counts = np.bincount(cv_pool_y)
            min_count = int(class_counts.min()) if len(class_counts) > 0 else 0
            if min_count < cv_folds:
                raise ValueError(
                    f"Impossible d'utiliser cv_folds={cv_folds}: une classe n'a que {min_count} échantillon(s) dans train+val."
                )

        original_batch_size = self.batch_size
        original_aug_params = dict(self.train_aug_params)

        results = []
        best_score = -np.inf if maximize else np.inf
        best_result = None

        combinations = list(self._expand_param_grid(param_grid))
        if max_trials is not None and max_trials > 0 and len(combinations) > max_trials:
            rng = np.random.default_rng(random_state)
            sampled_idx = rng.choice(len(combinations), size=max_trials, replace=False)
            combinations = [combinations[i] for i in sampled_idx]
            self._log(f"Sous-échantillonnage grid search: {max_trials} combinaisons retenues")
        self._log(f"\nGrid search: {len(combinations)} combinaisons à tester")

        for idx, params in enumerate(combinations, start=1):
            self._log(f"\n--- Combo {idx}/{len(combinations)} ---")
            self._log(str(params))

            train_kwargs = dict(params)

            if "batch_size" in train_kwargs:
                self.batch_size = train_kwargs.pop("batch_size")

            aug_keys = {
                "gaussian_noise_std",
                "time_shift_max",
                "pitch_shift_max",
                "num_time_masks",
                "num_freq_masks",
                "time_mask_max",
                "freq_mask_max",
            }
            current_aug = dict(original_aug_params)
            for key in list(train_kwargs.keys()):
                if key in aug_keys:
                    current_aug[key] = train_kwargs.pop(key)
            self.train_aug_params = current_aug

            train_kwargs["save_model"] = False

            try:
                if cv_folds > 1:
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                    fold_metrics = []
                    fold_scores = []

                    for fold, (train_rel, val_rel) in enumerate(skf.split(cv_pool_idx, cv_pool_y), start=1):
                        fold_train_idx = cv_pool_idx[train_rel]
                        fold_val_idx = cv_pool_idx[val_rel]
                        run_metrics = self.train(
                            **train_kwargs,
                            log_prefix=f"[GS {idx}/{len(combinations)}][Fold {fold}/{cv_folds}] ",
                            train_indices=fold_train_idx,
                            val_indices=fold_val_idx,
                            seed=random_state + idx * 100 + fold,
                            verbose_epochs=log_epochs,
                        )
                        fold_metrics.append(run_metrics)
                        fold_score = run_metrics.get(metric)
                        if fold_score is not None:
                            fold_scores.append(float(fold_score))

                        self._log(
                            f"[GS {idx}/{len(combinations)}] Fold {fold}/{cv_folds} terminé | "
                            f"best_val_acc={run_metrics.get('best_val_acc', float('nan')):.4f} | "
                            f"best_val_loss={run_metrics.get('best_val_loss', float('nan')):.4f}"
                        )

                    score = float(np.mean(fold_scores)) if len(fold_scores) > 0 else None
                    score_std = float(np.std(fold_scores)) if len(fold_scores) > 0 else None
                    row = {
                        **params,
                        "cv_folds": cv_folds,
                        "cv_metric": metric,
                        "cv_score_mean": score,
                        "cv_score_std": score_std,
                        "cv_scores": str([round(s, 6) for s in fold_scores]),
                        "score": score,
                    }
                else:
                    run_metrics = self.train(
                        **train_kwargs,
                        log_prefix=f"[GS {idx}/{len(combinations)}] ",
                        seed=random_state + idx,
                        verbose_epochs=log_epochs,
                    )
                    score = run_metrics.get(metric)
                    row = {**params, **run_metrics, "score": score}

                results.append(row)

                if score is None:
                    self._log(f"Metric '{metric}' absente pour cette combinaison.")
                else:
                    is_better = score > best_score if maximize else score < best_score
                    if is_better:
                        best_score = score
                        best_result = row
                        self._log(f"[GS] Nouveau meilleur score {best_score:.4f} avec combo {idx}")
            except Exception as e:
                row = {**params, "error": str(e), "score": None}
                results.append(row)
                self._log(f"Erreur sur la combinaison {idx}: {e}")
            finally:
                self.batch_size = original_batch_size
                self.train_aug_params = dict(original_aug_params)

        if save_results and len(results) > 0:
            output_file = Path(results_path)
            if not output_file.is_absolute():
                output_file = Path(__file__).resolve().parent / output_file
            output_file.parent.mkdir(parents=True, exist_ok=True)

            all_keys = set()
            for row in results:
                all_keys.update(row.keys())
            columns = sorted(all_keys)

            import csv
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(results)

            self._log(f"Résultats grid search sauvegardés dans {output_file}")

        if best_result is None:
            self._log("Aucune combinaison valide trouvée.")
        else:
            self._log(f"\nMeilleure combinaison ({metric}): {best_score}")
            self._log(str(best_result))

        return best_result, results

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