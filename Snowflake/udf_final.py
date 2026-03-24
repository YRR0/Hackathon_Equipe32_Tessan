from snowflake.snowpark.functions import udf

@udf(
    name="extract_mel_librosa",
    is_permanent=True,
    stage_location="@model_stage",
    packages=["scipy", "numpy", "numba", "packaging", "scikit-learn"],
    imports=[
        "@my_packages_stage/librosa_snowflake_312.zip"
    ],
    replace=True,
    session=session
)
def extract_mel_librosa(audio_bytes: bytes) -> str:
    import os, sys
    import json, io, zipfile, tempfile
    import numpy as np
    from scipy.io import wavfile
    from scipy import signal as scipy_signal
    import warnings
    warnings.filterwarnings('ignore')

    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa_cache"
    import numba
    import numba.np.ufunc.decorators
    _orig_guv = numba.guvectorize
    def _safe_guvectorize(ftylist_or_function=None, signature=None, **kws):
        def decorator(func):
            try: return _orig_guv(ftylist_or_function, signature, **kws)(func)
            except Exception: return func
        if callable(ftylist_or_function): return ftylist_or_function
        return decorator
    numba.guvectorize = _safe_guvectorize
    numba.np.ufunc.decorators.guvectorize = _safe_guvectorize

    sys.modules['soundfile'] = None
    sys.modules['sf'] = None

    import_dir = sys._xoptions.get("snowflake_import_directory", "")
    zip_path = os.path.join(import_dir, "librosa_snowflake_312.zip")
    if os.path.exists(zip_path):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)
        sys.path = [p for p in sys.path if "librosa_snowflake_312.zip" not in p]
        if temp_dir not in sys.path:
            sys.path.insert(0, temp_dir)

    import librosa

    TARGET_SR  = 22050
    DURATION   = 6.0
    TARGET_LEN = int(TARGET_SR * DURATION)
    N_MELS     = 128
    HOP_LENGTH = 512
    N_FFT      = 2048
    FRAME_LEN  = 2048

    file_size = len(audio_bytes)

    try:
        sr, y = wavfile.read(io.BytesIO(audio_bytes))
    except Exception as e:
        return json.dumps({"error": f"audio read error: {str(e)}"})

    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.dtype != np.float32:
        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.float32) / np.iinfo(y.dtype).max
        else:
            y = y.astype(np.float32)

    original_sr = int(sr)
    original_duration = round(len(y) / sr, 2) if sr > 0 else 0.0

    if sr != TARGET_SR and len(y) > 0:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type='polyphase')
        sr = TARGET_SR

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
    S_amp = np.sqrt(S)

    mel_spec = librosa.feature.melspectrogram(
        S=S, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    vmin, vmax = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - vmin) / (vmax - vmin + 1e-8)
    mel_norm = mel_norm.astype(np.float32)

    features = {}

    mfcc_13 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), sr=TARGET_SR, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = float(mfcc_13[i].mean())
        features[f'mfcc_{i}_std']  = float(mfcc_13[i].std())

    centroid = librosa.feature.spectral_centroid(S=S_amp, sr=TARGET_SR)
    features['centroid_mean'] = float(centroid.mean())
    features['centroid_std']  = float(centroid.std())

    bandwidth = librosa.feature.spectral_bandwidth(S=S_amp, sr=TARGET_SR)
    features['bandwidth_mean'] = float(bandwidth.mean())

    n_zcr_frames = 1 + (len(y_filtered) - FRAME_LEN) // HOP_LENGTH
    zcr_vals = np.zeros(max(n_zcr_frames, 1), dtype=np.float32)
    for i in range(max(n_zcr_frames, 1)):
        frame = y_filtered[i * HOP_LENGTH : i * HOP_LENGTH + FRAME_LEN]
        if len(frame) > 1:
            zcr_vals[i] = np.mean(np.abs(np.diff(np.sign(frame))) > 0)
    features['zcr_mean'] = float(zcr_vals.mean())
    features['zcr_std']  = float(zcr_vals.std())

    chroma = librosa.feature.chroma_stft(S=S, sr=TARGET_SR, tuning=0)
    features['chroma_mean'] = float(chroma.mean())

    rms_feat = librosa.feature.rms(S=S)
    features['rms_mean'] = float(rms_feat.mean())
    features['rms_std']  = float(rms_feat.std())

    threshold = 1e-3
    rolloff = librosa.feature.spectral_rolloff(S=S_amp, sr=TARGET_SR)
    features["rolloff_mean"] = float(rolloff.mean())
    features["rolloff_std"] = float(rolloff.std())

    contrast = librosa.feature.spectral_contrast(S=S, sr=TARGET_SR)
    for i in range(contrast.shape[0]):
        features[f"contrast_{i}_mean"] = float(contrast[i].mean())
        features[f"contrast_{i}_std"] = float(contrast[i].std())

    tonnetz_feat = librosa.feature.tonnetz(chroma=chroma)
    for i in range(tonnetz_feat.shape[0]):
        features[f"tonnetz_{i}_mean"] = float(tonnetz_feat[i].mean())
        features[f"tonnetz_{i}_std"] = float(tonnetz_feat[i].std())

    mfcc_full = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), sr=TARGET_SR)
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

    silence_ratio = np.mean(np.abs(y_filtered) < threshold)
    features["silence_ratio"] = float(silence_ratio)

    fft_vals = np.abs(np.fft.rfft(y_filtered))
    peak_freq = int(np.argmax(fft_vals))
    peak_freq_hz = float(peak_freq * TARGET_SR / max(1, len(y_filtered)))
    features["peak_freq_bin"] = float(peak_freq)
    features["peak_freq_hz"] = peak_freq_hz

    sorted_keys = sorted(list(features.keys()))
    tabular_vector = [features[k] for k in sorted_keys]

    return json.dumps({
        "mel": mel_norm.tolist(),
        "tabular": tabular_vector,
        "file_size_bytes": file_size,
        "duration_sec": original_duration,
        "sample_rate": original_sr
    })
print("Extraction librosa ok")


from snowflake.snowpark.functions import udf
@udf(
    name="predict_from_mel",
    is_permanent=True,
    stage_location="@model_stage",
    packages=["onnxruntime", "numpy", "scikit-learn"],
    imports=[
        "@model_stage/resnet18_mel_finetuned_f.onnx",
        "@model_stage/tabular_scaler.json"
    ],
    replace=True,
    session=session
)
def predict_from_mel(mel_json: str) -> str:
    import onnxruntime as ort
    import numpy as np
    import json, os, sys
    from scipy.ndimage import zoom

    CLASSES = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

    data = json.loads(mel_json)
    if "error" in data:
        return json.dumps({"error": data["error"]})

    mel_norm = np.array(data["mel"], dtype=np.float32)
    raw_tabular = np.array(data.get("tabular", []), dtype=np.float32)

    onnx_path = None
    scaler_path = None
    for path in sys.path:
        cand_onnx = os.path.join(path, "resnet18_mel_finetuned_f.onnx")
        if onnx_path is None and os.path.exists(cand_onnx):
            onnx_path = cand_onnx
        cand_scaler = os.path.join(path, "tabular_scaler.json")
        if scaler_path is None and os.path.exists(cand_scaler):
            scaler_path = cand_scaler

    if onnx_path is None:
        return json.dumps({"error": "ONNX model not found"})

    sess = ort.InferenceSession(onnx_path)

    tab_expected_dim = 186
    for inp in sess.get_inputs():
        if "tabular" in inp.name or "tab" in inp.name:
            tab_expected_dim = inp.shape[1] if len(inp.shape) > 1 else inp.shape[0]
            break

    has_scaler = False
    t_mean = None
    t_std = None
    if scaler_path is not None:
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        t_mean = np.array(scaler_data.get("mean", []), dtype=np.float32)
        t_std = np.array(scaler_data.get("std", []), dtype=np.float32)
        if len(t_mean) == tab_expected_dim:
            has_scaler = True

    if len(raw_tabular) >= tab_expected_dim:
        tab_vec = raw_tabular[:tab_expected_dim]
    elif len(raw_tabular) > 0:
        tab_vec = np.zeros(tab_expected_dim, dtype=np.float32)
        tab_vec[:len(raw_tabular)] = raw_tabular
    else:
        tab_vec = np.zeros(tab_expected_dim, dtype=np.float32)

    if has_scaler:
        tab_scaled = (tab_vec - t_mean) / (t_std + 1e-8)
    else:
        tab_scaled = tab_vec

    zh = 224 / mel_norm.shape[0]
    zw = 224 / mel_norm.shape[1]
    mel_resized = zoom(mel_norm, (zh, zw), order=1).astype(np.float32)
    tensor_img = np.stack([mel_resized, mel_resized, mel_resized], axis=0)[np.newaxis]
    tensor_tab = tab_scaled.reshape(1, -1)

    input_feed = {}
    for inp in sess.get_inputs():
        if "tabular" in inp.name or "tab" in inp.name:
            input_feed[inp.name] = tensor_tab
        else:
            input_feed[inp.name] = tensor_img

    scores = sess.run(None, input_feed)[0]
    e = np.exp(scores - np.max(scores))
    probs = (e / np.sum(e)).squeeze()

    sorted_idx = probs.argsort()[::-1]

    result = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}
    result['predicted_class']  = CLASSES[int(sorted_idx[0])]
    result['confidence']       = round(float(probs[sorted_idx[0]]), 4)
    result['second_class']     = CLASSES[int(sorted_idx[1])]
    result['file_size_bytes']  = data.get("file_size_bytes", 0)
    result['duration_sec']     = data.get("duration_sec", 0.0)
    result['sample_rate']      = data.get("sample_rate", 22050)

    return json.dumps(result)

print("Inférence onnx librosa ok")