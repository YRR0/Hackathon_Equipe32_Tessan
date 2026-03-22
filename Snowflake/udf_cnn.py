from snowflake.snowpark.functions import udf

@udf(
    name="predict_respiratory",
    is_permanent=True,
    stage_location="@model_stage",
    packages=["onnxruntime", "scipy", "numpy", "scikit-learn"],
    imports=["@model_stage/cnn_respiratory.onnx"],
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
    CLASSES    = ['asthma', 'bronchial', 'copd', 'healthy', 'pneumonia']

    # ── Chercher le modèle ONNX ───────────────────────────────────
    onnx_path = None
    for path in sys.path:
        candidate = os.path.join(path, "cnn_respiratory.onnx")
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

    # Stocker les infos originales avant tout traitement
    original_sr       = int(sr)
    original_duration = round(len(y) / sr, 2)

    # ── Prétraitement ─────────────────────────────────────────────
    if sr != TARGET_SR:
        from scipy.signal import resample
        y = resample(y, int(len(y) * TARGET_SR / sr))

    energy    = np.convolve(y**2, np.ones(1024)/1024, mode='same')
    threshold = energy.max() * 0.001
    nonsilent = np.where(energy > threshold)[0]
    if len(nonsilent) > 0:
        y = y[nonsilent[0]:nonsilent[-1]]

    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    else:
        y = np.pad(y, (0, TARGET_LEN - len(y)))

    nyquist = TARGET_SR / 2
    b, a    = scipy_signal.butter(4, [100/nyquist, 2000/nyquist], btype='band')
    y_clean = scipy_signal.filtfilt(b, a, y)

    # ── Construction des 6 canaux ─────────────────────────────────
    def make_channel(arr, target_h=128, target_w=259):
        zh = target_h / arr.shape[0]
        zw = target_w / arr.shape[1]
        return zoom(arr, (zh, zw)).astype(np.float32)

    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    f_spec, t_spec, Sxx = scipy_signal.spectrogram(
        y_clean, TARGET_SR, nperseg=2048, noverlap=2048-512
    )
    mel_db = 10 * np.log10(Sxx + 1e-10)
    ch1    = make_channel(normalize(mel_db))
    delta  = np.diff(mel_db, axis=1, prepend=mel_db[:, :1])
    ch2    = make_channel(normalize(delta))
    delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
    ch3    = make_channel(normalize(delta2))

    frame_len  = 512
    zcr_frames = []
    for i in range(0, len(y_clean) - frame_len, frame_len // 2):
        frame = y_clean[i:i + frame_len]
        zcr_frames.append(np.mean(np.abs(np.diff(np.sign(frame)))) / 2)
    ch4 = make_channel(normalize(
        np.repeat(np.array(zcr_frames).reshape(1, -1), 128, axis=0)
    ))

    rms_frames = []
    for i in range(0, len(y_clean) - frame_len, frame_len // 2):
        frame = y_clean[i:i + frame_len]
        rms_frames.append(np.sqrt(np.mean(frame**2)))
    ch5 = make_channel(normalize(
        np.repeat(np.array(rms_frames).reshape(1, -1), 128, axis=0)
    ))

    centroid_frames = []
    for col_idx in range(Sxx.shape[1]):
        spec_col = Sxx[:, col_idx]
        total    = spec_col.sum()
        centroid_frames.append(
            float(np.sum(f_spec * spec_col) / total) if total > 0 else 0.0
        )
    ch6 = make_channel(normalize(
        np.repeat(np.array(centroid_frames).reshape(1, -1), 128, axis=0)
    ))

    # ── Inférence ─────────────────────────────────────────────────
    tensor = np.stack([ch1, ch2, ch3, ch4, ch5, ch6], axis=0)[np.newaxis]
    scores = sess.run(None, {input_name: tensor})[0]
    e      = np.exp(scores - scores.max())
    probs  = (e / e.sum()).squeeze()

    # ── Résultat enrichi ──────────────────────────────────────────
    # Trier les classes par probabilité décroissante
    sorted_idx = probs.argsort()[::-1]

    result = {
        # Probabilités par classe
        cls: round(float(p), 4)
        for cls, p in zip(CLASSES, probs)
    }

    # Diagnostic principal
    result['predicted_class'] = CLASSES[int(sorted_idx[0])]
    result['confidence']      = round(float(probs[sorted_idx[0]]), 4)

    # Deuxième classe la plus probable
    result['second_class'] = CLASSES[int(sorted_idx[1])]
    result['second_prob']  = round(float(probs[sorted_idx[1]]), 4)

    # Infos fichier
    result['file_size_bytes']  = file_size
    result['duration_sec']     = original_duration
    result['sample_rate']      = original_sr

    return json.dumps(result)

print("UDF créée avec succès")