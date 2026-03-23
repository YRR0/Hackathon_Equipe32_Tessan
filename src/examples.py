"""
Visualize false positives from the ResNet18 model on the test split.

Usage examples:
  python src/examples.py
  python src/examples.py --num 8
  python src/examples.py --pred-class asthma --num 6
"""

from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import ResNet18Trainer, ResNet18FineTuned


def load_model(model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = ResNet18FineTuned(num_classes=num_classes, freeze_backbone=False).to(device)

    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = Path(__file__).resolve().parent / model_file

    try:
        state_dict = torch.load(model_file, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_file, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_false_positives(model_path: str = "models/resnet18_mel_finetuned.pth", pred_class: str | None = None):
    trainer = ResNet18Trainer(batch_size=32, num_workers=0, pin_memory=False)
    trainer.load_data()
    trainer.build_loaders()

    class_names = list(trainer.le.classes_)
    if pred_class is not None and pred_class not in class_names:
        raise ValueError(f"Classe inconnue '{pred_class}'. Classes disponibles: {class_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=len(class_names), device=device)

    false_positives = []

    with torch.no_grad():
        for dataset_index in trainer.test_idx:
            x, y_true = trainer.eval_dataset[int(dataset_index)]
            logits = model(x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            y_pred = int(np.argmax(probs))
            y_true = int(y_true)

            if y_pred == y_true:
                continue

            pred_label = class_names[y_pred]
            true_label = class_names[y_true]

            # False positive for a class = predicted as that class while true label is different.
            if pred_class is not None and pred_label != pred_class:
                continue

            false_positives.append(
                {
                    "dataset_index": int(dataset_index),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": float(probs[y_pred]),
                    "mel": np.asarray(trainer.spectres["mel"][int(dataset_index)], dtype=np.float32),
                }
            )

    false_positives.sort(key=lambda d: d["confidence"], reverse=True)
    return false_positives, class_names


def plot_false_positives(false_positives, max_examples: int = 6):
    if len(false_positives) == 0:
        print("Aucun faux positif trouvé avec les filtres actuels.")
        return

    shown = false_positives[:max_examples]
    cols = min(3, len(shown))
    rows = int(np.ceil(len(shown) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, sample in zip(axes, shown):
        mel = sample["mel"]
        ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(
            f"idx={sample['dataset_index']} | {sample['true_label']} -> {sample['pred_label']}\n"
            f"conf={sample['confidence']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel bins")

    for ax in axes[len(shown):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Show false positives with MEL spectrograms.")
    parser.add_argument("--model-path", default="models/resnet18_mel_finetuned.pth", help="Path to .pth model")
    parser.add_argument("--pred-class", default=None, help="Optional predicted class filter")
    parser.add_argument("--num", type=int, default=6, help="Number of examples to display")
    args = parser.parse_args()

    false_positives, class_names = collect_false_positives(
        model_path=args.model_path,
        pred_class=args.pred_class,
    )

    print(f"Classes: {class_names}")
    print(f"False positives found: {len(false_positives)}")
    for sample in false_positives[: min(10, len(false_positives))]:
        print(
            f"- idx={sample['dataset_index']} | true={sample['true_label']} | "
            f"pred={sample['pred_label']} | conf={sample['confidence']:.3f}"
        )

    plot_false_positives(false_positives, max_examples=max(1, args.num))


if __name__ == "__main__":
    main()
