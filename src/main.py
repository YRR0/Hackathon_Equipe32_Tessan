'''
Main class for testing preprocessing and model pipeline
'''

import numpy as np
import torch
from pathlib import Path
from preprocessing import Preprocessor
from model import Model, MultiSpectreDataset
from models.RespiratoryCNN import RespiratoryCNN
from models.CNNBILSTMANAttention import CNNBiLSTMAttention
from models.TestModel import TestModel
import torch.nn as nn

class Main:
    """
    Main class
    """
    
    def __init__(self, data_root="../data/"):
        base_dir = Path(__file__).resolve().parent
        data_path = Path(data_root)
        self.data_root = data_path if data_path.is_absolute() else (base_dir / data_path)

    def preprocess(self):
        preprocessor = Preprocessor(22050, 6, self.data_root, verbose=True)
        preprocessor.spectres_creation_and_save()

    def preprocess_with_params(self, preprocessing_params=None):
        preprocessing_params = preprocessing_params or {}
        preprocessor = Preprocessor(
            22050,
            6,
            self.data_root,
            verbose=True,
            **preprocessing_params,
        )
        preprocessor.spectres_creation_and_save()

    def training(self, model):
        model_class = Model(batch_size=16, num_workers=0, pin_memory=True)
        model_class.train(model, epochs=50, save_model=True)
        print("Evaluation")
        model_class.evaluate()        



def main():
    main = Main()

    # main.preprocess()

    model = CNNBiLSTMAttention
    # main.training(model)
    
    # gridsearch :

    model_class = Model(batch_size=16, num_workers=0, pin_memory=True)
    best_result, all_results = model_class.grid_search(
        model,
        lr_values=[1e-3, 1e-4],
        optimizers=[torch.optim.AdamW],
        criterions=[nn.CrossEntropyLoss],
        epochs=20,
        preprocessing_params_list=[
            {"n_fft": 1024, "hop_length": 256, "n_mels": 128, "n_mfcc": 20},
            {"n_fft": 2048, "hop_length": 512, "n_mels": 128, "n_mfcc": 20},
            {"n_fft": 2048, "hop_length": 256, "n_mels": 64, "n_mfcc": 13},
        ],
        preprocess_fn=main.preprocess_with_params,
        feature_sets=[
            ["mel"],
            ["mel", "mfcc"],
            ["mel", "mfcc", "chroma"],
        ],
        dropout_values=[0.2, 0.3, 0.5],
        optimizer_kwargs_list=[{}, {"weight_decay": 1e-4}],
    )

    print("Grid search finished")
    print("Best result:", best_result)
    print("Total runs:", len(all_results))


if __name__ == "__main__":
    main()

# Base model
# Best configuration:
# {'best_val_acc': 0.7417582417582418, 'best_val_loss': 0.7318346525231997, 'criterion': 'CrossEntropyLoss', 'optimizer': 'AdamW', 'lr': 0.001, 'optimizer_kwargs': {'weight_decay': 0.0001}, 'feature_keys': ['mel', 'mfcc', 'chroma'], 'model_kwargs': {'dropout': 0.2}, 
# 'test_acc': 0.7692307692307693}

# Tsiory :
# Best configuration:
# {'best_val_acc': 0.8186813186813187, 'best_val_loss': 0.5334082990884781, 'criterion': 'CrossEntropyLoss', 'optimizer': 'AdamW', 'lr': 0.001, 'optimizer_kwargs': {'weight_decay': 0.0001}, 'feature_keys': ['mel', 'mfcc', 'chroma'], 'model_kwargs': {'dropout': 0.2},
# 'test_acc': 0.7802197802197802}