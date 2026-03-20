"""
Main dédié à la version ResNet18 fine-tunée sur Mel-spectrogrammes.
Ne modifie pas le pipeline existant.
"""

from pathlib import Path
from preprocessing import Preprocessor
from model_resnet18 import ResNet18Trainer


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


def main():
    app = MainResNet18()

    # À lancer une première fois si spectres.npy n'existe pas encore
    # app.preprocess()

    app.training(batch_size=16, epochs_head=5, epochs_finetune=10)


if __name__ == "__main__":
    main()