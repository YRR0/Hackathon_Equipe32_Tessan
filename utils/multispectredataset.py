import torch
import torch.nn.functional as F
import cv2

class MultiSpectreDataset(Dataset):
    def __init__(self, spectres, labels, feature_keys=None):
        self.spectres = spectres
        self.y = torch.LongTensor(labels)
        self.feature_keys = feature_keys or ["mel", "mfcc", "centroid", "bandwidth", "zcr", "chroma"]
        self.num_channels = len(self.feature_keys)

    def __len__(self):
        return len(self.y)

    @staticmethod
    def _to_2d(arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return arr[np.newaxis, :]
        return arr

    # @staticmethod
    # def _resize_2d(arr_2d, target_h, target_w):
    #     x = torch.tensor(arr_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    #     x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    #     return x.squeeze(0).squeeze(0).numpy()
    

    @staticmethod
    def _resize_2d(arr_2d, target_h, target_w):
        return cv2.resize(arr_2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, idx):
        mel = self._to_2d(self.spectres["mel"][idx])
        target_h, target_w = mel.shape

        channels = []
        for key in self.feature_keys:
            feat = self._to_2d(self.spectres[key][idx])
            if feat.shape != (target_h, target_w):
                feat = self._resize_2d(feat, target_h, target_w)
            channels.append(feat)

        x = np.stack(channels, axis=0).astype(np.float32)  # (C, H, W)
        return torch.from_numpy(x), self.y[idx]

