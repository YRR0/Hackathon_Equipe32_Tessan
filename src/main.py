"""
Main d'execution pour preprocess, entrainement/evaluation et prediction.
La logique d'apprentissage est centralisee dans ResNet18Trainer.
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

    def evaluate_on_unseen_test(self, model_path="models/resnet18_mel_finetuned.pth"):
        """Evaluation sur le split test (15%) jamais vu en entrainement."""
        print("\n[EVAL] Evaluation sur test set jamais vu (15%)")
        trainer = ResNet18Trainer(batch_size=16, num_workers=0, pin_memory=True)
        trainer.load_data()
        trainer.build_loaders()
        trainer.evaluate(model_path=model_path)

    def run_full_pipeline(
        self,
        mode="fixed_5fold",
        preproc=False,
        batch_size=16,
        epochs_head=5,
        epochs_finetune=10,
        export_final_onnx=True,
    ):
        """
        Pipeline complete en une commande:
        1) preprocess (optionnel)
        2) train + eval
        3) export ONNX (optionnel)
        """
        if preproc:
            print("\n[PIPELINE] Step 1/3 - Preprocessing")
            self.preprocess()

        print("\n[PIPELINE] Step 2/3 - Training + Evaluation")
        trainer = ResNet18Trainer(batch_size=batch_size, num_workers=0, pin_memory=True)

        if mode == "fixed_5fold":
            result = trainer.train_fixed_params_5fold(verbose_final_epochs=True)
        elif mode == "standard":
            result = trainer.train_and_evaluate(
                epochs_head=epochs_head,
                epochs_finetune=epochs_finetune,
                verbose_epochs=True,
            )
        else:
            raise ValueError("mode doit etre 'fixed_5fold' ou 'standard'")

        if export_final_onnx:
            print("\n[PIPELINE] Step 3/3 - ONNX export")
            exporter = ResNet18Trainer(batch_size=1, num_workers=0, pin_memory=False)
            exporter.export_checkpoint_to_onnx(
                model_path="models/resnet18_mel_finetuned.pth",
                onnx_path="models/resnet18_mel_finetuned.onnx",
            )

        return result

    def predict_file(self, file_path, model_path="models/resnet18_mel_finetuned.pth", top_k=3):
        wav_file = Path(file_path)
        if not wav_file.is_absolute():
            wav_file = (Path(__file__).resolve().parent / wav_file).resolve()
        if not wav_file.exists():
            raise FileNotFoundError(f"Fichier introuvable: {wav_file}")

        trainer = ResNet18Trainer(batch_size=1, num_workers=0, pin_memory=False)
        trainer.load_data()
        trainer.build_loaders()
        class_names = list(trainer.le.classes_)

        preprocessor = Preprocessor(22050, 6, self.data_root, verbose=False)
        y = preprocessor.preprocess_audio_file(wav_file, target_sr=22050, target_duration_sec=6)
        y = preprocessor.apply_bandpass_filter(y, sr=22050)
        mel = preprocessor.compute_mel_spectrogram(y, sr=22050)

        tab_tensor = None
        if trainer.tabular_dim > 0 and len(trainer.tabular_feature_names) > 0:
            feats = preprocessor.extract_all_features(y, sr=22050)
            tab_vec = np.array([float(feats[name]) for name in trainer.tabular_feature_names], dtype=np.float32)
            tab_vec = (tab_vec - trainer.tabular_mean) / trainer.tabular_std
            tab_tensor = torch.tensor(tab_vec, dtype=torch.float32).unsqueeze(0)

        mel = np.asarray(mel, dtype=np.float32)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.squeeze(0).repeat(3, 1, 1).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet18FineTuned(
            num_classes=len(class_names),
            freeze_backbone=False,
            tabular_dim=trainer.tabular_dim,
        ).to(device)

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
            if tab_tensor is not None:
                logits = model(x.to(device), tab_tensor.to(device))
            else:
                logits = model(x.to(device), None)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        pred_conf = float(probs[pred_idx])

        top_k = max(1, min(top_k, len(class_names)))
        top_idx = np.argsort(-probs)[:top_k]

        print(f"Prediction pour {wav_file.name}: {pred_label} ({pred_conf:.3f})")
        print("Top probabilites:")
        for i in top_idx:
            print(f"- {class_names[int(i)]}: {float(probs[int(i)]):.3f}")

        return {
            "file": str(wav_file),
            "predicted_label": pred_label,
            "confidence": pred_conf,
            "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        }


def main():
    app = MainResNet18()

    app.run_full_pipeline(mode="fixed_5fold", preproc=True, export_final_onnx=True)
    # app.evaluate_on_unseen_test(model_path="models/resnet18_mel_finetuned.pth")
    # app.run_full_pipeline(mode="standard", batch_size=32, epochs_head=5, epochs_finetune=10)
    # app.predict_file("../data/data_updated/Bronchial/P1BronchialSc_2.wav")


if __name__ == "__main__":
    main()
