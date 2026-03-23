import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MelResNetDataset(Dataset):
    """
    Dataset qui utilise uniquement les Mel-spectrogrammes
    et les convertit en pseudo-images 3 canaux pour ResNet18.
    """

    def __init__(self, spectres, labels_encoded, target_size=(224, 224)):
        self.mels = spectres["mel"]
        self.y = torch.LongTensor(labels_encoded)
        self.target_size = target_size

    def __len__(self):
        return len(self.y)

    @staticmethod
    def _to_2d(arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr

    @staticmethod
    def _normalize(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        mel = self._to_2d(self.mels[idx])
        mel = self._normalize(mel)

        # (H, W) -> (1, 1, H, W)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Resize en 224x224 pour ResNet18
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)

        # (1, 1, 224, 224) -> (1, 224, 224)
        x = x.squeeze(0)

        # Répéter en 3 canaux
        x = x.repeat(3, 1, 1)  # (3, 224, 224)

        return x, self.y[idx]