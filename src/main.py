"""
Main dédié à la version ResNet18 fine-tunée sur Mel-spectrogrammes.
Ne modifie pas le pipeline existant.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from preprocessing import Preprocessor
from model_resnet18 import ResNet18Trainer, ResNet18FineTuned


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

    def grid_search(self):
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
        )
        print("Best configuration:")
        print(best)


def main():
    app = MainResNet18()

    # À lancer une première fois si spectres.npy n'existe pas encore
    # app.preprocess()

    # app.training(batch_size=32, epochs_head=8, epochs_finetune=15)
    # app.grid_search()
    app.predict_file("../data/data_updated/Bronchial/P1BronchialSc_2.wav")


if __name__ == "__main__":
    main()