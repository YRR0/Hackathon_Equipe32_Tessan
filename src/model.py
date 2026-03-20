'''
File for using models created in other .py files,
should be modulable to allow for different preprocessing hyperparameters.
Requires pre-processed data
'''

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from scipy import signal as scipy_signal
import os
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import cv2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns


from utils.multispectredataset import MultiSpectreDataset

class Model:
    '''
    Class for using the models created in other .py files.
    
    '''
    def __init__(self, batch_size=32, num_workers=4, pin_memory=True, feature_keys=None, verbose = False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.feature_keys = feature_keys
        self.verbose = verbose

        self.criterion = nn.CrossEntropyLoss
        self.optimizer = torch.optim.Adam

    def load_data(self):
        # Chargement des données depuis le fichier unifié
        data_file = Path(__file__).resolve().parent.parent / "data" / "spectres.npy"
        self.spectres = np.load(data_file, allow_pickle=True).item() # MAYBE MODIF AVC NEW FORMAT
        self.mels = self.spectres["mel"]
        self.labels = self.spectres["labels"]

        assert len(self.mels) == len(self.labels), "Désalignement entre mels et labels dans spectres.npy"
        # Encodage des labels en entiers
        self.le = LabelEncoder()
        self.y_enc = self.le.fit_transform(self.labels)  # ex: asthma->0, bronchial->1...
        
        # # Print label -> code mapping
        # print("\nLabel -> Code mapping:")
        # for i, label in enumerate(self.le.classes_):
        #     print(f"  {label} -> {i}")
        # print()

        indices = np.arange(len(self.mels))
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        self.train_idx, temp_idx = next(sss1.split(indices, self.y_enc))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        self.val_idx, self.test_idx = next(sss2.split(temp_idx, self.y_enc[temp_idx]))
        self.val_idx = temp_idx[self.val_idx]
        self.test_idx = temp_idx[self.test_idx]

        if self.verbose:
            print(f"Train : {len(self.train_idx)} | Val : {len(self.val_idx)} | Test : {len(self.test_idx)}")



    def data_loader(self, batch_size=None, num_workers=None, pin_memory=None, feature_keys=None):
        '''
        feature_keys donne la liste des canaux a utiliser
        feature_keys [list] : "mel", "mfcc", "centroid", "bandwidth", "zcr", "chroma"
        '''
        self.full_dataset = MultiSpectreDataset(self.spectres, self.y_enc, feature_keys=feature_keys)
        if self.verbose:
            print(f"Canaux utilisés : {self.full_dataset.feature_keys}")
            print(f"Nombre de canaux : {self.full_dataset.num_channels}")

        self.train_loader = DataLoader(Subset(self.full_dataset, self.train_idx), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.val_loader   = DataLoader(Subset(self.full_dataset, self.val_idx), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.test_loader  = DataLoader(Subset(self.full_dataset, self.test_idx), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    
    def train(
        self,
        CNN_model,
        epochs=50,
        save_model=True,
        criterion_cls=None,
        optimizer_cls=None,
        lr=1e-3,
        optimizer_kwargs=None,
        feature_keys=None,
        model_kwargs=None,
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Device utilisee : {device}")

        # Load des data
        self.load_data()
        self.data_loader(feature_keys=feature_keys)

        # Calcul des poids pour compenser le déséquilibre (Q2)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y_enc),
            y=self.y_enc[self.train_idx]        # calculé sur le train seulement
        )
        weights_tensor = torch.FloatTensor(class_weights)

        self.model = CNN_model(
            num_classes=len(self.le.classes_),
            in_channels=self.full_dataset.num_channels,
            **(model_kwargs or {})
        ).to(device)
        weights_tensor = weights_tensor.to(device)
        criterion_cls = criterion_cls or self.criterion
        optimizer_cls = optimizer_cls or self.optimizer
        optimizer_kwargs = optimizer_kwargs or {}

        criterion = criterion_cls(weight=weights_tensor)
        optimizer = optimizer_cls(self.model.parameters(), lr=lr, **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        def train_epoch(model, loader, criterion, optimizer):
            model.train()
            total_loss, correct = 0, 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (out.argmax(1) == y_batch).sum().item()
            return total_loss / len(loader), correct / len(loader.dataset)

        def eval_epoch(model, loader, criterion):
            model.eval()
            total_loss, correct = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    out = model(X_batch)
                    loss = criterion(out, y_batch)
                    total_loss += loss.item()
                    correct += (out.argmax(1) == y_batch).sum().item()
            return total_loss / len(loader), correct / len(loader.dataset)

        # Entraînement
        EPOCHS = epochs
        best_val_acc = 0.0
        best_val_loss = float("inf")
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(self.model, self.train_loader, criterion, optimizer)
            val_loss, val_acc = eval_epoch(self.model, self.val_loader, criterion)
            scheduler.step(val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)

            if (epoch + 1) % 5 == 0:
                if self.verbose:
                    print(
                        f"Epoch {epoch+1:3d} | "
                        f"Train loss {train_loss:.3f} acc {train_acc:.3f} | "
                        f"Val loss {val_loss:.3f} acc {val_acc:.3f}"
                    )

        if save_model:
            model_name = self.model.__class__.__name__
            model_file = Path(__file__).resolve().parent / "models" / f"{model_name}.pth"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), model_file)
            self.model_file = model_file

        return {
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "criterion": criterion_cls.__name__ if hasattr(criterion_cls, "__name__") else str(criterion_cls),
            "optimizer": optimizer_cls.__name__ if hasattr(optimizer_cls, "__name__") else str(optimizer_cls),
            "lr": lr,
            "optimizer_kwargs": optimizer_kwargs,
            "feature_keys": feature_keys,
            "model_kwargs": model_kwargs or {},
        }
    
    # add a save fct for grid_search?

    def evaluate(self, model_path=None, short=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is not None:
            model_file = Path(model_path)
            if not model_file.is_absolute():
                model_file = Path(__file__).resolve().parent / model_file
            try:
                state_dict = torch.load(model_file, map_location=device, weights_only=True)
            except TypeError:
                state_dict = torch.load(model_file, map_location=device)
            self.model.load_state_dict(state_dict)
        self.model.eval()

        all_preds, all_probs, all_true = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = self.model(X_batch)
                probs = torch.softmax(out, dim=1)
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)
        test_acc = (all_preds == all_true).mean()

        if short:
            print(f"Test Accuracy : {test_acc:.3f}")
            return float(test_acc)

        print(classification_report(all_true, all_preds,
                                    target_names=self.le.classes_,
                                    zero_division=0))

        f1_macro = f1_score(all_true, all_preds, average='macro')
        print(f"Macro F1-score : {f1_macro:.3f}")

        auc = roc_auc_score(all_true, all_probs, multi_class='ovr', average='macro')
        print(f"Macro AUC-ROC  : {auc:.3f}")

        cm = confusion_matrix(all_true, all_preds)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.le.classes_,
                    yticklabels=self.le.classes_)
        plt.title("Matrice de confusion — test set")
        plt.ylabel("Réel")
        plt.xlabel("Prédit")
        plt.tight_layout()
        # plt.show()

    def grid_search(
        self,
        CNN_model,
        lr_values,
        optimizers,
        criterions,
        epochs,
        optimizer_kwargs_list=None,
        feature_sets=None,
        dropout_values=None,
        base_model_kwargs=None,
    ):
        # Recherche exhaustive simple sur les combinaisons d'hyperparamètres.
        optimizer_kwargs_list = optimizer_kwargs_list or [{}]
        feature_sets = feature_sets or [None]
        dropout_values = dropout_values or [None]
        base_model_kwargs = base_model_kwargs or {}
        best_result = None
        all_results = []

        for crit in criterions:
            for opt in optimizers:
                for lr_val in lr_values:
                    for opt_kwargs in optimizer_kwargs_list:
                        for feature_keys in feature_sets:
                            for dropout in dropout_values:
                                run_model_kwargs = dict(base_model_kwargs)
                                if dropout is not None:
                                    run_model_kwargs["dropout"] = dropout

                                print(
                                    f"Testing criterion={crit.__name__}, optimizer={opt.__name__}, "
                                    f"lr={lr_val}, opt_kwargs={opt_kwargs}, "
                                    f"feature_keys={feature_keys}, dropout={dropout}"
                                )
                                train_result = self.train(
                                    CNN_model,
                                    epochs=epochs,
                                    save_model=False,
                                    criterion_cls=crit,
                                    optimizer_cls=opt,
                                    lr=lr_val,
                                    optimizer_kwargs=opt_kwargs,
                                    feature_keys=feature_keys,
                                    model_kwargs=run_model_kwargs,
                                )
                                test_acc = self.evaluate(short=True)
                                result = {
                                    **train_result,
                                    "test_acc": test_acc,
                                }
                                all_results.append(result)

                                if best_result is None or result["best_val_acc"] > best_result["best_val_acc"]:
                                    best_result = result

        print("\nBest configuration:")
        print(best_result)
        return best_result, all_results
