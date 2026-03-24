import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
import librosa
from scipy import signal as scipy_signal
from scipy.ndimage import zoom

def test_onnx_local(wav_path, onnx_path, scaler_path):
    print(f"--- Lancement du test local ONNX ---")
    print(f"Fichier audio : {wav_path}")
    print(f"Modèle ONNX   : {onnx_path}")
    print(f"Scaler JSON   : {scaler_path}")

    # 1. Lecture audio (équivalent extract_mel_librosa)
    TARGET_SR = 22050
    DURATION = 6.0
    TARGET_LEN = int(TARGET_SR * DURATION)
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048
    FRAME_LEN = 2048

    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    
    if len(y) == 0:
        y = np.zeros(TARGET_LEN, dtype=np.float32)
    elif len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    elif len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]

    nyquist = TARGET_SR / 2
    b, a = scipy_signal.butter(4, [100 / nyquist, 2000 / nyquist], btype='band')
    y_filtered = scipy_signal.filtfilt(b, a, y).astype(np.float32)

    S = np.abs(librosa.stft(y=y_filtered, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    mel_spec = librosa.feature.melspectrogram(S=S, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    vmin, vmax = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - vmin) / (vmax - vmin + 1e-8)
    mel_norm = mel_norm.astype(np.float32)

    # Variables
    features = {}

    y_feat = y_filtered

    mfcc_13 = librosa.feature.mfcc(y=y_feat, sr=TARGET_SR, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = float(mfcc_13[i].mean())
        features[f'mfcc_{i}_std']  = float(mfcc_13[i].std())

    centroid = librosa.feature.spectral_centroid(y=y_feat, sr=TARGET_SR)
    features['centroid_mean'] = float(centroid.mean())
    features['centroid_std']  = float(centroid.std())

    bandwidth = librosa.feature.spectral_bandwidth(y=y_feat, sr=TARGET_SR)
    features['bandwidth_mean'] = float(bandwidth.mean())

    # ZCR manuel pour éviter Numba mais se rapprocher au mieux de librosa ZCR
    # Librosa ZCR = np.abs(np.diff(np.signbit(y)))  en gros
    zcr = librosa.feature.zero_crossing_rate(y_feat)
    features['zcr_mean'] = float(zcr.mean())
    features['zcr_std']  = float(zcr.std())

    chroma_stft = librosa.feature.chroma_stft(y=y_feat, sr=TARGET_SR)
    features['chroma_mean'] = float(chroma_stft.mean())

    rms_feat = librosa.feature.rms(y=y_feat)
    features['rms_mean'] = float(rms_feat.mean())
    features['rms_std']  = float(rms_feat.std())

    threshold = 1e-3
    rolloff = librosa.feature.spectral_rolloff(y=y_feat, sr=TARGET_SR)
    features["rolloff_mean"] = float(rolloff.mean())
    features["rolloff_std"] = float(rolloff.std())

    contrast = librosa.feature.spectral_contrast(y=y_feat, sr=TARGET_SR)
    for i in range(contrast.shape[0]):
        features[f"contrast_{i}_mean"] = float(contrast[i].mean())
        features[f"contrast_{i}_std"] = float(contrast[i].std())

    tonnetz_feat = librosa.feature.tonnetz(y=y_feat, sr=TARGET_SR)
    for i in range(tonnetz_feat.shape[0]):
        features[f"tonnetz_{i}_mean"] = float(tonnetz_feat[i].mean())
        features[f"tonnetz_{i}_std"] = float(tonnetz_feat[i].std())

    mfcc_full = librosa.feature.mfcc(y=y_feat, sr=TARGET_SR)
    for i in range(mfcc_full.shape[0]):
        features[f"mfcc_full_{i}_mean"] = float(mfcc_full[i].mean())
        features[f"mfcc_full_{i}_std"] = float(mfcc_full[i].std())

    delta = librosa.feature.delta(mfcc_full)
    for i in range(delta.shape[0]):
        features[f"delta_{i}_mean"] = float(delta[i].mean())
        features[f"delta_{i}_std"] = float(delta[i].std())

    delta2 = librosa.feature.delta(mfcc_full, order=2)
    for i in range(delta2.shape[0]):
        features[f"delta2_{i}_mean"] = float(delta2[i].mean())
        features[f"delta2_{i}_std"] = float(delta2[i].std())

    features["rms_var"] = float(np.var(rms_feat))
    features["silence_ratio"] = float(np.mean(np.abs(y_filtered) < threshold))

    fft_vals = np.abs(np.fft.rfft(y_filtered))
    peak_freq = int(np.argmax(fft_vals))
    features["peak_freq_bin"] = float(peak_freq)
    features["peak_freq_hz"] = float(peak_freq * TARGET_SR / max(1, len(y_filtered)))

    # Ordre alphabétique pour le scaler !
    sorted_keys = sorted(list(features.keys()))
    raw_tabular = np.array([features[k] for k in sorted_keys], dtype=np.float32)

    # 2. Prédiction ONNX (équivalent predict_from_mel)
    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)
    t_mean = np.array(scaler_data["mean"], dtype=np.float32)
    t_std = np.array(scaler_data["std"], dtype=np.float32)

    # Application du Scaler
    tab_scaled = (raw_tabular - t_mean) / (t_std + 1e-8)
    
    sess = ort.InferenceSession(onnx_path)
    
    zh = 224 / mel_norm.shape[0]
    zw = 224 / mel_norm.shape[1]
    mel_resized = zoom(mel_norm, (zh, zw), order=1).astype(np.float32)
    tensor_img = np.stack([mel_resized, mel_resized, mel_resized], axis=0)[np.newaxis]
    tensor_tab = tab_scaled.reshape(1, -1)

    input_feed = {}
    for inp in sess.get_inputs():
        if "tabular" in inp.name or "tab" in inp.name or inp.shape[1] == 186:
            input_feed[inp.name] = tensor_tab
        elif "mel" in inp.name or "img" in inp.name or inp.shape[1] == 3:
            input_feed[inp.name] = tensor_img
        else:
            input_feed[inp.name] = tensor_img

    scores = sess.run(None, input_feed)[0]
    e = np.exp(scores - np.max(scores))
    probs = (e / np.sum(e)).squeeze()

    CLASSES = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']
    sorted_idx = probs.argsort()[::-1]

    print(f"\n--- RÉSULTATS DE LA PRÉDICTION ONNX ---")
    print(f"Classe prédite : {CLASSES[int(sorted_idx[0])]}  ({probs[sorted_idx[0]]:.4f})")
    print("Top classes:")
    for i in sorted_idx:
        print(f" - {CLASSES[int(i)]}: {probs[int(i)]:.4f}")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    # Détermine le dossier src/ où se trouve ce script
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    parser = argparse.ArgumentParser(description="Test local ONNX model")
    # Valeurs par défaut en chemins absolus par rapport au script pour éviter les soucis de dossier courant
    parser.add_argument("--wav", default=str(SCRIPT_DIR.parent / "data" / "data_updated" / "healthy" / "P1Healthy29S.wav"), help="Chemin vers le fichier audio .wav")
    parser.add_argument("--onnx", default=str(SCRIPT_DIR / "models" / "resnet18_mel_finetuned_f.onnx"), help="Chemin vers le modèle .onnx")
    parser.add_argument("--scaler", default=str(SCRIPT_DIR / "models" / "tabular_scaler.json"), help="Chemin vers le scaler .json")
    args = parser.parse_args()
    
    test_onnx_local(args.wav, args.onnx, args.scaler)
