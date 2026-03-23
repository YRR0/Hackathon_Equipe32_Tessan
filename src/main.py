"""
Main dédié à la version ResNet18 fine-tunée sur Mel-spectrogrammes.
Ne modifie pas le pipeline existant.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from preprocessing import Preprocessor
from model import ResNet18Trainer, ResNet18FineTuned


class MainResNet18:
    def __init__(self, data_root="../data/"):
        base_dir = Path(__file__).resolve().parent
        data_path = Path(data_root)
        self.data_root = data_path if data_path.is_absolute() else (base_dir / data_path)

    def preprocess(self):
        preprocessor = Preprocessor(22050, 6, self.data_root, verbose=True)
        preprocessor.spectres_creation_and_save()

    def training(self, batch_size=16, epochs_head=5, epochs_finetune=10):
        trainer = ResNet18Trainer(batch_size=batch_size, num_workers=0, pin_memory=True)
        trainer.train(epochs_head=epochs_head, epochs_finetune=epochs_finetune, save_model=True)
        print("Evaluation")
        trainer.evaluate(model_path="models/resnet18_mel_finetuned.pth")

    def predict_file(self, file_path, model_path="models/resnet18_mel_finetuned.pth", top_k=3):
        wav_file = Path(file_path)
        if not wav_file.is_absolute():
            wav_file = (Path(__file__).resolve().parent / wav_file).resolve()
        if not wav_file.exists():
            raise FileNotFoundError(f"Fichier introuvable: {wav_file}")

        trainer = ResNet18Trainer(batch_size=1, num_workers=0, pin_memory=False)
        trainer.load_data()
        class_names = list(trainer.le.classes_)

        preprocessor = Preprocessor(22050, 6, self.data_root, verbose=False)
        y = preprocessor.preprocess_audio_file(wav_file, target_sr=22050, target_duration_sec=6)
        y = preprocessor.apply_bandpass_filter(y, sr=22050)
        mel = preprocessor.compute_mel_spectrogram(y, sr=22050)

        mel = np.asarray(mel, dtype=np.float32)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.squeeze(0).repeat(3, 1, 1).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet18FineTuned(num_classes=len(class_names), freeze_backbone=False).to(device)

        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parent / model_file

        try:
            state_dict = torch.load(model_file, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_file, map_location=device)

        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        pred_conf = float(probs[pred_idx])

        top_k = max(1, min(top_k, len(class_names)))
        top_idx = np.argsort(-probs)[:top_k]

        print(f"Prediction pour {wav_file.name}: {pred_label} ({pred_conf:.3f})")
        print("Top probabilités:")
        for i in top_idx:
            print(f"- {class_names[int(i)]}: {float(probs[int(i)]):.3f}")

        return {
            "file": str(wav_file),
            "predicted_label": pred_label,
            "confidence": pred_conf,
            "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        }

    def grid_search(self, cv_folds=5):
        trainer = ResNet18Trainer(batch_size=32, num_workers=0, pin_memory=True)

        param_grid = {
            "batch_size": [16, 32],
            "epochs_head": [5, 8],
            "epochs_finetune": [10, 15],
            "lr_head": [1e-3, 5e-4],
            "lr_finetune": [1e-4, 5e-5],
            "weight_decay_head": [0.0, 1e-4],
            "weight_decay_finetune": [0.0, 1e-4],
            "use_class_weights": [True],
            "gaussian_noise_std": [0.0, 0.01],
            "time_shift_max": [8, 12],
            "pitch_shift_max": [2, 4],
            "num_time_masks": [1, 2],
            "num_freq_masks": [1, 2],
            "time_mask_max": [15, 25],
            "freq_mask_max": [6, 10],
        }

        best, _ = trainer.grid_search(
            param_grid=param_grid,
            metric="best_val_acc",
            maximize=True,
            save_results=True,
            results_path="models/resnet18_grid_search_results.csv",
            max_trials=24,
            random_state=42,
            cv_folds=cv_folds,
        )
        print("Best configuration:")
        print(best)

    def export_model_to_onnx(
        self,
        model_path="models/resnet18_mel_finetuned.pth",
        onnx_path="models/resnet18_mel_finetuned.onnx",
    ):
        """
        Exporte un checkpoint .pth vers ONNX.
        """
        trainer = ResNet18Trainer(batch_size=1, num_workers=0, pin_memory=False)
        trainer.load_data()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet18FineTuned(num_classes=len(trainer.le.classes_), freeze_backbone=False).to(device)

        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parent / model_file

        try:
            state_dict = torch.load(model_file, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_file, map_location=device)

        model.load_state_dict(state_dict)
        trainer.model = model
        trainer.export_onnx(onnx_path=onnx_path)

        print(f"Export ONNX terminé: {onnx_path}")

    def train_fixed_params_5fold(self):
        """
        Lance une CV 5-fold avec un set figé, puis un entraînement final
        et une évaluation sur le test set.
        """
        fixed_params = {
            "batch_size": 16,
            "epochs_head": 5,
            "epochs_finetune": 15,
            "lr_head": 0.001,
            "lr_finetune": 0.0001,
            "weight_decay_head": 0.0,
            "weight_decay_finetune": 0.0,
            "use_class_weights": True,
            "gaussian_noise_std": 0.0,
            "time_shift_max": 12,
            "pitch_shift_max": 4,
            "num_time_masks": 1,
            "num_freq_masks": 1,
            "time_mask_max": 25,
            "freq_mask_max": 6,
            "cv_folds": 5,
        }

        trainer = ResNet18Trainer(
            batch_size=fixed_params["batch_size"],
            num_workers=0,
            pin_memory=True,
        )

        cv_folds = fixed_params["cv_folds"]
        train_params = {k: v for k, v in fixed_params.items() if k != "cv_folds"}
        param_grid = {k: [v] for k, v in train_params.items()}

        best, _ = trainer.grid_search(
            param_grid=param_grid,
            metric="best_val_acc",
            maximize=True,
            save_results=True,
            results_path="models/resnet18_fixed_params_5fold_results.csv",
            max_trials=1,
            random_state=42,
            cv_folds=cv_folds,
            log_epochs=False,
        )

        print("Best configuration (fixed 5-fold):")
        print(best)

        print("\nEntraînement final avec les mêmes paramètres...")
        final_trainer = ResNet18Trainer(
            batch_size=fixed_params["batch_size"],
            num_workers=0,
            pin_memory=True,
        )
        final_trainer.train_aug_params.update({
            "gaussian_noise_std": fixed_params["gaussian_noise_std"],
            "time_shift_max": fixed_params["time_shift_max"],
            "pitch_shift_max": fixed_params["pitch_shift_max"],
            "num_time_masks": fixed_params["num_time_masks"],
            "num_freq_masks": fixed_params["num_freq_masks"],
            "time_mask_max": fixed_params["time_mask_max"],
            "freq_mask_max": fixed_params["freq_mask_max"],
        })

        final_train_metrics = final_trainer.train(
            epochs_head=fixed_params["epochs_head"],
            epochs_finetune=fixed_params["epochs_finetune"],
            save_model=True,
            use_class_weights=fixed_params["use_class_weights"],
            lr_head=fixed_params["lr_head"],
            lr_finetune=fixed_params["lr_finetune"],
            weight_decay_head=fixed_params["weight_decay_head"],
            weight_decay_finetune=fixed_params["weight_decay_finetune"],
            verbose_epochs=False,
        )

        self.export_model_to_onnx(
            model_path="models/resnet18_mel_finetuned.pth",
            onnx_path="models/resnet18_mel_finetuned_final.onnx",
        )

        print("\nÉvaluation finale sur test set:")
        final_trainer.evaluate(model_path="models/resnet18_mel_finetuned.pth")

        return {
            "cv_best": best,
            "final_train": final_train_metrics,
        }


def main():
    app = MainResNet18()

    # À lancer une première fois si spectres.npy n'existe pas encore
    # app.preprocess()

    # app.training(batch_size=32, epochs_head=5, epochs_finetune=10)
    # app.grid_search(cv_folds=5)
    app.train_fixed_params_5fold()
    # app.predict_file("../data/data_updated/Bronchial/P1BronchialSc_2.wav")


if __name__ == "__main__":
    main()


    # pas mal :
    # {'batch_size': 16, 'epochs_head': 8, 'epochs_finetune': 15, 'lr_head': 0.0005, 'lr_finetune': 0.0001, 'weight_decay_head': 0.0, 'weight_decay_finetune': 0.0, 'use_class_weights': True, 'gaussian_noise_std': 0.0, 'time_shift_max': 8, 'pitch_shift_max': 2, 'num_time_masks': 2, 'num_freq_masks': 2, 'time_mask_max': 15, 'freq_mask_max': 10}

    # hit 90 :
    # {'batch_size': 16, 'epochs_head': 5, 'epochs_finetune': 15, 'lr_head': 0.001, 'lr_finetune': 0.0001, 'weight_decay_head': 0.0, 'weight_decay_finetune': 0.0, 'use_class_weights': True, 'gaussian_noise_std': 0.0, 'time_shift_max': 12, 'pitch_shift_max': 4, 'num_time_masks': 1, 'num_freq_masks': 1, 'time_mask_max': 25, 'freq_mask_max': 6}
    