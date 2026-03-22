import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MelResNetDataset(Dataset):
    """
    Dataset qui utilise uniquement les Mel-spectrogrammes
    et les convertit en pseudo-images 3 canaux pour ResNet18.
    """

    def __init__(
        self,
        spectres,
        labels_encoded,
        target_size=(224, 224),
        augment=False,
        gaussian_noise_std=0.01,
        time_shift_max=12,
        pitch_shift_max=4,
        num_time_masks=1,
        num_freq_masks=1,
        time_mask_max=20,
        freq_mask_max=8,
    ):
        self.mels = spectres["mel"]
        self.y = torch.LongTensor(labels_encoded)
        self.target_size = target_size
        self.augment = augment

        self.gaussian_noise_std = gaussian_noise_std
        self.time_shift_max = time_shift_max
        self.pitch_shift_max = pitch_shift_max
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max

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

    def _apply_time_shift(self, mel):
        if self.time_shift_max <= 0:
            return mel
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        if shift == 0:
            return mel
        return np.roll(mel, shift=shift, axis=1)

    def _apply_pitch_shift(self, mel):
        if self.pitch_shift_max <= 0:
            return mel
        shift = np.random.randint(-self.pitch_shift_max, self.pitch_shift_max + 1)
        if shift == 0:
            return mel
        return np.roll(mel, shift=shift, axis=0)

    def _apply_gaussian_noise(self, mel):
        if self.gaussian_noise_std <= 0:
            return mel
        noise = np.random.normal(0.0, self.gaussian_noise_std, size=mel.shape).astype(np.float32)
        mel_noisy = mel + noise
        return np.clip(mel_noisy, 0.0, 1.0)

    def _apply_specaugment(self, mel):
        out = mel.copy()
        freq_bins, time_steps = out.shape

        for _ in range(self.num_time_masks):
            if self.time_mask_max <= 0 or time_steps <= 1:
                break
            t = np.random.randint(1, min(self.time_mask_max, time_steps) + 1)
            t0 = np.random.randint(0, max(1, time_steps - t + 1))
            out[:, t0:t0 + t] = 0.0

        for _ in range(self.num_freq_masks):
            if self.freq_mask_max <= 0 or freq_bins <= 1:
                break
            f = np.random.randint(1, min(self.freq_mask_max, freq_bins) + 1)
            f0 = np.random.randint(0, max(1, freq_bins - f + 1))
            out[f0:f0 + f, :] = 0.0

        return out

    def _augment_mel(self, mel):
        mel = self._apply_time_shift(mel)
        mel = self._apply_pitch_shift(mel)
        mel = self._apply_gaussian_noise(mel)
        mel = self._apply_specaugment(mel)
        return mel

    def __getitem__(self, idx):
        mel = self._to_2d(self.mels[idx])
        mel = self._normalize(mel)

        if self.augment:
            mel = self._augment_mel(mel)

        # (H, W) -> (1, 1, H, W)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Resize en 224x224 pour ResNet18
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)

        # (1, 1, 224, 224) -> (1, 224, 224)
        x = x.squeeze(0)

        # Répéter en 3 canaux
        x = x.repeat(3, 1, 1)  # (3, 224, 224)

        return x, self.y[idx]