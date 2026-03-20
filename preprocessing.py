'''
File for preprocessing the data,
should be modulable to allow for different preprocessing hyperparameters
'''

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from scipy import signal as scipy_signal
import os



class Preprocessor:
    '''
    Class for preprocessing the audio data. Feature creation and saving of the spectrograms.

    Need to call function spectres_creation_and_save().
    '''
    def __init__(self,target_sr=22050,target_duration_sec=6,input_root="data",
                    n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
        
        self.target_sr = target_sr
        self.target_duration_sec = target_duration_sec
        self.input_root = input_root

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc


    def preprocess_audio_dataset(self, target_sr: int, target_duration_sec: float, input_root: str = "data"):
        input_root = Path(input_root)
        output_root = Path(f"data_updated")
        output_root.mkdir(parents=True, exist_ok=True)

        target_samples = int(target_sr * target_duration_sec)
        rows = []

        for src_wav in input_root.rglob("*.wav"):
            rel_path = src_wav.relative_to(input_root)
            dst_wav = output_root / rel_path
            dst_wav.parent.mkdir(parents=True, exist_ok=True)

            y, _ = librosa.load(str(src_wav), sr=target_sr, mono=True)
            original_samples = len(y)

            if original_samples == 0:
                y_fixed = np.zeros(target_samples, dtype=np.float32)
                action = "silence_fill"
            elif original_samples < target_samples:
                repeats = int(np.ceil(target_samples / original_samples))
                y_fixed = np.tile(y, repeats)[:target_samples]
                action = "looped"
            elif original_samples > target_samples:
                y_fixed = y[:target_samples]
                action = "trimmed"
            else:
                y_fixed = y
                action = "unchanged"

            sf.write(str(dst_wav), y_fixed, target_sr)
            rows.append(
                {
                    "file": str(rel_path).replace("\\", "/"),
                    "orig_duration_sec": round(original_samples / target_sr, 4),
                    "new_duration_sec": round(len(y_fixed) / target_sr, 4),
                    "action": action,
                }
            )

        fix_duration_df = pd.DataFrame(rows)
        return fix_duration_df, output_root

    def compute_mel_spectrogram(self, y, sr=22050):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,      # nombre de bandes de fréquences → hauteur de l'image
            n_fft=self.n_fft,      # taille de la fenêtre FFT → résolution fréquentielle
            hop_length=self.hop_length,  # pas entre chaque fenêtre → résolution temporelle
            # fmax=4000        # fréquence max utile pour sons respiratoires
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) # Normalisation
        return mel_norm  # shape : (128, 259) → image 128×259 pixels
    
    def compute_mfcc_spectrogram(self, y, sr=22050, n_mfcc=20):
        """MFCC - Mel-Frequency Cepstral Coefficients (20 coeffs)"""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_db = librosa.power_to_db(mfcc, ref=np.max)
        mfcc_norm = (mfcc_db - mfcc_db.min()) / (mfcc_db.max() - mfcc_db.min() + 1e-8)
        return mfcc_norm  # shape : (13, temps)

    def compute_spectral_centroid_spectrogram(self, y, sr=22050):
        """Spectral Centroid - centre de gravité fréquentiel"""
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_norm = (centroid - centroid.min()) / (centroid.max() - centroid.min() + 1e-8)
        return centroid_norm  # shape : (1, temps)

    def compute_spectral_bandwidth_spectrogram(self, y, sr=22050):
        """Spectral Bandwidth - largeur du spectre"""
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_norm = (bandwidth - bandwidth.min()) / (bandwidth.max() - bandwidth.min() + 1e-8)
        return bandwidth_norm  # shape : (1, temps)

    def compute_zcr_spectrogram(self, y, sr=22050):
        """Zero-Crossing Rate - nb de changement de signe"""
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_norm = (zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-8)
        return zcr_norm  # shape : (1, temps)
    
    def compute_chroma_spectrogram(self, y, sr=22050):
        """Chroma - énergie par note musicale (12 notes)"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_norm = (chroma - chroma.min()) / (chroma.max() - chroma.min() + 1e-8)
        return chroma_norm  # shape : (12, temps)

    def apply_bandpass_filter(self, y, sr=22050, lowcut=100, highcut=2000):
        # Filtre Butterworth ordre 4 — bon compromis précision/stabilité
        nyquist = sr / 2
        low  = lowcut  / nyquist
        high = highcut / nyquist

        b, a = scipy_signal.butter(N=4, Wn=[low, high], btype='band')
        y_filtered = scipy_signal.filtfilt(b, a, y)  # filtfilt = sans déphasage OU .trim

        return y_filtered


    def extract_all_features(y, sr=22050):
        features = {}
        # MFCC — 13 coefficients qui résument l'enveloppe spectrale
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(mfcc[i].mean())
            features[f'mfcc_{i}_std']  = float(mfcc[i].std())
        # Spectral Centroid — "centre de gravité" fréquentiel
        # élevé = sons aigus (sifflements asthme), bas = sons graves (ronchi BPCO)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['centroid_mean'] = float(centroid.mean())
        features['centroid_std']  = float(centroid.std())
        # Spectral Bandwidth — largeur du spectre autour du centroid
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['bandwidth_mean'] = float(bandwidth.mean())
        # Zero Crossing Rate — nb de fois que le signal change de signe
        # élevé = sons bruités (crépitements pneumonie)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(zcr.mean())
        features['zcr_std']  = float(zcr.std())
        # Chroma — énergie par note musicale (12 notes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(chroma.mean())
        # RMS Energy — énergie moyenne du signal
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(rms.mean())
        features['rms_std']  = float(rms.std())
        return features  # dict de ~32 valeurs numériques
    
        
    def spectres_creation_and_save(self, target_sr=22050, target_duration_sec=6, input_root=None):
        all_features = []
        all_mels = []
        all_mfccs = []
        all_centroids = []
        all_bandwidths = []
        all_zcrs = []
        all_chromas = []
        all_labels = []
        for label in os.listdir(input_root):
            folder = os.path.join(input_root, label)
            if not os.path.isdir(folder):
                continue

            for wav_file in os.listdir(folder):
                if not wav_file.endswith(".wav"):
                    continue

                filepath = os.path.join(folder, wav_file)

                try:
                    y = self.preprocess_audio_dataset(target_sr, target_duration_sec, filepath)
                    y_clean = self.apply_bandpass_filter(y, sr=target_sr)

                    mel = self.compute_mel_spectrogram(y_clean)
                    all_mels.append(mel)
                    mfcc = self.compute_mfcc_spectrogram(y_clean)
                    all_mfccs.append(mfcc)
                    centroid = self.compute_spectral_centroid_spectrogram(y_clean)
                    all_centroids.append(centroid)
                    bandwidth = self.compute_spectral_bandwidth_spectrogram(y_clean)
                    all_bandwidths.append(bandwidth)
                    zcr = self.compute_zcr_spectrogram(y_clean)
                    all_zcrs.append(zcr)
                    chroma = self.compute_chroma_spectrogram(y_clean)
                    all_chromas.append(chroma)

                    feats = self.extract_all_features(y_clean)
                    feats['label'] = label
                    feats['filename'] = wav_file
                    all_features.append(feats)

                    all_labels.append(label)

                except Exception as e:
                    print(f"Erreur sur {wav_file} : {e}")