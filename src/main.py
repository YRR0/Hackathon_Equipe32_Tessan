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

    def training(self, model):
        model_class = Model(batch_size=1, num_workers=0, pin_memory=True)
        model_class.train(model, epochs=2, save_model=True)
        print("Evaluation")
        model_class.evaluate()



def main():
    main = Main()
    # main.preprocess()

    model = CNNBiLSTMAttention
    main.training(model)

if __name__ == "__main__":
    main()