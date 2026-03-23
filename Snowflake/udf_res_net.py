from snowflake.snowpark.functions import udf

@udf(
    name="predict_respiratory",
    is_permanent=True,
    stage_location="@model_stage",
    packages=["onnxruntime", "scipy", "numpy", "scikit-learn"],
    imports=["@model_stage/resnet18_mel_finetuned.onnx"],
    replace=True,
    session=session
)
def predict_respiratory(audio_bytes: bytes) -> str:
    import onnxruntime as ort
    import numpy as np
    import json
    import io
    import os
    import sys
    from scipy.io import wavfile
    from scipy import signal as scipy_signal
    from scipy.ndimage import zoom

    TARGET_SR  = 22050
    TARGET_LEN = int(TARGET_SR * 6.0)
    CLASSES    = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

    # ── Chercher le modèle ONNX ───────────────────────────────────
    onnx_path = None
    for path in sys.path:
        candidate = os.path.join(path, "resnet18_mel_finetuned.onnx")
        if os.path.exists(candidate):
            onnx_path = candidate
            break

    if onnx_path is None:
        return json.dumps({"error": "Modèle ONNX introuvable"})

    sess       = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    # ── Lecture audio + infos fichier ─────────────────────────────
    file_size = len(audio_bytes)
    sr, y     = wavfile.read(io.BytesIO(audio_bytes))

    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.dtype != np.float32:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    original_sr       = int(sr)
    original_duration = round(len(y) / sr, 2)

    # ── Prétraitement audio ───────────────────────────────────────
    # Rééchantillonnage si nécessaire
    if sr != TARGET_SR:
        from scipy.signal import resample
        y = resample(y, int(len(y) * TARGET_SR / sr))

    # Suppression des silences
    energy    = np.convolve(y**2, np.ones(1024) / 1024, mode='same')
    threshold = energy.max() * 0.001
    nonsilent = np.where(energy > threshold)[0]
    if len(nonsilent) > 0:
        y = y[nonsilent[0]:nonsilent[-1]]

    # Durée fixe à 6 secondes
    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    else:
        y = np.pad(y, (0, TARGET_LEN - len(y)))

    # Filtre passe-bande (identique au Preprocessor)
    nyquist = TARGET_SR / 2
    b, a    = scipy_signal.butter(4, [100 / nyquist, 2000 / nyquist], btype='band')
    y_clean = scipy_signal.filtfilt(b, a, y)

    # ── Mel-spectrogram (identique à compute_mel_spectrogram) ─────
    n_fft      = 2048
    hop_length = 512
    n_mels     = 128

    # STFT via scipy pour reproduire librosa.feature.melspectrogram
    f, t, Sxx = scipy_signal.spectrogram(
        y_clean,
        fs=TARGET_SR,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window='hann',
        scaling='spectrum'
    )

    # Filtre mel triangulaire
    mel_freqs  = np.linspace(0, TARGET_SR / 2, n_mels)
    mel_matrix = np.zeros((n_mels, len(f)), dtype=np.float32)
    for i in range(n_mels):
        lower  = mel_freqs[i - 1] if i > 0          else 0.0
        upper  = mel_freqs[i + 1] if i < n_mels - 1 else mel_freqs[i]
        center = mel_freqs[i]
        for j, fq in enumerate(f):
            if lower <= fq <= center:
                mel_matrix[i, j] = (fq - lower) / (center - lower + 1e-10)
            elif center < fq <= upper:
                mel_matrix[i, j] = (upper - fq) / (upper - center + 1e-10)

    mel_spec = mel_matrix @ Sxx                      # (n_mels, T)
    mel_db   = 10 * np.log10(mel_spec + 1e-10)
    mel_db   = mel_db - mel_db.max()                 # ref=np.max comme librosa

    # Normalisation min-max (identique à MelResNetDataset._normalize)
    mn, mx = mel_db.min(), mel_db.max()
    if mx > mn:
        mel_norm = (mel_db - mn) / (mx - mn)
    else:
        mel_norm = np.zeros_like(mel_db)
    mel_norm = mel_norm.astype(np.float32)           # (128, T)

    # ── Resize en 224×224 (identique à F.interpolate bilinear) ───
    zh = 224 / mel_norm.shape[0]
    zw = 224 / mel_norm.shape[1]
    mel_resized = zoom(mel_norm, (zh, zw), order=1).astype(np.float32)  # (224, 224)

    # ── 3 canaux par répétition (identique à x.repeat(3,1,1)) ────
    tensor = np.stack([mel_resized, mel_resized, mel_resized], axis=0)  # (3, 224, 224)
    tensor = tensor[np.newaxis]                                          # (1, 3, 224, 224)

    # ── Inférence ─────────────────────────────────────────────────
    scores = sess.run(None, {input_name: tensor})[0]
    e      = np.exp(scores - scores.max())
    probs  = (e / e.sum()).squeeze()

    # ── Résultat enrichi ──────────────────────────────────────────
    sorted_idx = probs.argsort()[::-1]

    result = {
        cls: round(float(p), 4)
        for cls, p in zip(CLASSES, probs)
    }

    result['predicted_class'] = CLASSES[int(sorted_idx[0])]
    result['confidence']      = round(float(probs[sorted_idx[0]]), 4)
    result['second_class']    = CLASSES[int(sorted_idx[1])]
    result['second_prob']     = round(float(probs[sorted_idx[1]]), 4)
    result['file_size_bytes'] = file_size
    result['duration_sec']    = original_duration
    result['sample_rate']     = original_sr

    return json.dumps(result)

print("UDF créée avec succès")