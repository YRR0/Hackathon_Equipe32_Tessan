@udf(
    name="predict_respiratory",
    is_permanent=True,
    stage_location="@model_stage",
    packages=["onnxruntime", "scipy", "numpy", "scikit-learn"],
    imports=[
        "@model_stage/resnet18_mel_finetuned.onnx",
        "@my_packages_stage/librosa_snowflake.zip"  # ← ajout de notre librairie importé librosa
    ],
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

    # ── Import librosa depuis le zip ──────────────────────────────
    import zipfile, tempfile
    import_dir = sys._xoptions["snowflake_import_directory"]
    zip_path = os.path.join(import_dir, "librosa_snowflake.zip")
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)
    sys.path.insert(0, temp_dir)
    import librosa
    # ─────────────────────────────────────────────────────────────

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

    # ── Lecture audio ─────────────────────────────────────────────
    file_size = len(audio_bytes)
    sr, y     = wavfile.read(io.BytesIO(audio_bytes))

    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.dtype != np.float32:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    original_sr       = int(sr)
    original_duration = round(len(y) / sr, 2)

    # ── Rééchantillonnage avec librosa ────────────────────────────
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)  # ← LIBROSA

    # ── Suppression des silences ──────────────────────────────────
    energy    = np.convolve(y**2, np.ones(1024) / 1024, mode='same')
    threshold = energy.max() * 0.001
    nonsilent = np.where(energy > threshold)[0]
    if len(nonsilent) > 0:
        y = y[nonsilent[0]:nonsilent[-1]]

    # ── Durée fixe à 6 secondes ───────────────────────────────────
    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    else:
        y = np.pad(y, (0, TARGET_LEN - len(y)))

    # ── Filtre passe-bande ────────────────────────────────────────
    nyquist = TARGET_SR / 2
    b, a    = scipy_signal.butter(4, [100 / nyquist, 2000 / nyquist], btype='band')
    y_clean = scipy_signal.filtfilt(b, a, y)

    # ── Mel-spectrogram avec librosa ──────────────────────────────
    mel_spec = librosa.feature.melspectrogram(      # ← LIBROSA
        y=y_clean,
        sr=TARGET_SR,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)  # ← LIBROSA

    # ── Normalisation min-max ─────────────────────────────────────
    mn, mx = mel_db.min(), mel_db.max()
    if mx > mn:
        mel_norm = (mel_db - mn) / (mx - mn)
    else:
        mel_norm = np.zeros_like(mel_db)
    mel_norm = mel_norm.astype(np.float32)

    # ── Resize en 224×224 ─────────────────────────────────────────
    zh = 224 / mel_norm.shape[0]
    zw = 224 / mel_norm.shape[1]
    mel_resized = zoom(mel_norm, (zh, zw), order=1).astype(np.float32)

    # ── 3 canaux ──────────────────────────────────────────────────
    tensor = np.stack([mel_resized, mel_resized, mel_resized], axis=0)
    tensor = tensor[np.newaxis]

    # ── Inférence ─────────────────────────────────────────────────
    scores = sess.run(None, {input_name: tensor})[0]
    e      = np.exp(scores - scores.max())
    probs  = (e / e.sum()).squeeze()

    # ── Résultat ──────────────────────────────────────────────────
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