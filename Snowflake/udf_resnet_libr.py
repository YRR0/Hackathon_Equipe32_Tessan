@udf(
    name="extract_mel_librosa",
    is_permanent=True,
    stage_location="@model_stage",
    packages=["scipy", "numpy", "numba", "packaging"],
    imports=[
        "@my_packages_stage/librosa_snowflake_312.zip"
    ],
    replace=True,
    session=session
)
def extract_mel_librosa(audio_bytes: bytes) -> str:
    import os, sys
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa_cache"

    import numba
    import numba.np.ufunc.decorators
    _orig_guv = numba.guvectorize
    def _safe_guvectorize(ftylist_or_function=None, signature=None, **kws):
        def decorator(func):
            try:
                return _orig_guv(ftylist_or_function, signature, **kws)(func)
            except Exception:
                return func
        if callable(ftylist_or_function):
            return ftylist_or_function
        return decorator
    numba.guvectorize = _safe_guvectorize
    numba.np.ufunc.decorators.guvectorize = _safe_guvectorize

    import numpy as np
    import io, zipfile, tempfile, json
    from scipy.io import wavfile
    from scipy import signal as scipy_signal

    sys.modules['soundfile'] = None
    sys.modules['sf'] = None

    import_dir = sys._xoptions["snowflake_import_directory"]
    zip_path = os.path.join(import_dir, "librosa_snowflake_312.zip")
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)
    sys.path = [p for p in sys.path if "librosa_snowflake_312.zip" not in p]
    if temp_dir not in sys.path:
        sys.path.insert(0, temp_dir)
    import librosa

    TARGET_SR  = 22050
    TARGET_LEN = int(TARGET_SR * 6.0)
    N_MELS     = 128
    HOP_LENGTH = 512

    file_size = len(audio_bytes)
    sr, y = wavfile.read(io.BytesIO(audio_bytes))

    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.dtype != np.float32:
        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.float32) / np.iinfo(y.dtype).max
        else:
            y = y.astype(np.float32)

    original_sr       = int(sr)
    original_duration = round(len(y) / sr, 2) if sr > 0 else 0.0

    if len(y) == 0:
        n_frames = 1 + TARGET_LEN // HOP_LENGTH
        mel_norm = np.zeros((N_MELS, n_frames), dtype=np.float32)
        return json.dumps({
            "mel":             mel_norm.tolist(),
            "file_size_bytes": file_size,
            "duration_sec":    0.0,
            "sample_rate":     original_sr
        })

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type='polyphase')

    energy    = np.convolve(y**2, np.ones(1024) / 1024, mode='same')
    threshold = energy.max() * 0.001
    nonsilent = np.where(energy > threshold)[0]
    if len(nonsilent) > 0:
        y = y[nonsilent[0]:nonsilent[-1]]

    if len(y) == 0:
        y = np.zeros(TARGET_LEN, dtype=np.float32)

    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    else:
        y = np.pad(y, (0, TARGET_LEN - len(y)))

    nyquist = TARGET_SR / 2
    b, a = scipy_signal.butter(4, [100 / nyquist, 2000 / nyquist], btype='band')
    y_clean = scipy_signal.filtfilt(b, a, y)

    mel_spec = librosa.feature.melspectrogram(
        y=y_clean, sr=TARGET_SR,
        n_fft=2048, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    mn, mx = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - mn) / (mx - mn) if mx > mn else np.zeros_like(mel_db)
    mel_norm = mel_norm.astype(np.float32)

    return json.dumps({
        "mel":             mel_norm.tolist(),
        "file_size_bytes": file_size,
        "duration_sec":    original_duration,
        "sample_rate":     original_sr
    })

print("Extraction librosa ok")


@udf(
    name="predict_from_mel",
    is_permanent=True,
    stage_location="@model_stage",
    packages=["onnxruntime", "numpy", "scikit-learn"],
    imports=["@model_stage/resnet18_mel_finetuned_c.onnx"],
    replace=True,
    session=session
)
def predict_from_mel(mel_json: str) -> str:
    import onnxruntime as ort
    import numpy as np
    import json, os, sys
    from scipy.ndimage import zoom

    CLASSES = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

    # ── Parser le JSON entrant ────────────────────────────────────
    data         = json.loads(mel_json)
    mel_norm     = np.array(data["mel"], dtype=np.float32)  # (128, T)
    file_size    = data["file_size_bytes"]
    duration     = data["duration_sec"]
    sample_rate  = data["sample_rate"]

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

    # ── Resize 224x224 ────────────────────────────────────────────
    zh = 224 / mel_norm.shape[0]
    zw = 224 / mel_norm.shape[1]
    mel_resized = zoom(mel_norm, (zh, zw), order=1).astype(np.float32)

    # ── 3 canaux + batch ──────────────────────────────────────────
    tensor = np.stack([mel_resized, mel_resized, mel_resized], axis=0)[np.newaxis]

    # ── Inférence ─────────────────────────────────────────────────
    scores = sess.run(None, {input_name: tensor})[0]
    e      = np.exp(scores - scores.max())
    probs  = (e / e.sum()).squeeze()

    # ── Résultat complet ──────────────────────────────────────────
    sorted_idx = probs.argsort()[::-1]

    result = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}
    result['predicted_class']  = CLASSES[int(sorted_idx[0])]
    result['confidence']       = round(float(probs[sorted_idx[0]]), 4)
    result['second_class']     = CLASSES[int(sorted_idx[1])]
    result['second_prob']      = round(float(probs[sorted_idx[1]]), 4)
    result['file_size_bytes']  = file_size
    result['duration_sec']     = duration
    result['sample_rate']      = sample_rate

    return json.dumps(result)

print("Inférence onnx librosa ok")