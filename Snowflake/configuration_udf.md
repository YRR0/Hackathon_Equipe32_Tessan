# Configuration Snowflake — UDF d'inférence respiratory

## Prérequis

1. **Modèle ONNX prêt** : `cnn_respiratory.onnx` généré depuis `conversion.ipynb`
2. **Label encoder** : `label_encoder.pkl` sauvegardé lors de l'entraînement
3. **Stage Snowflake** : `@model_stage` pour stocker les artefacts
4. **Session active** : Connexion Python Snowflake via `snowpark`

---

## Étape 1 — Vérifier les artefacts dans le stage

Exécuter dans un notebook Snowflake (SQL) :

```sql
LIST @model_stage;
```

Attendu :
- `cnn_respiratory.onnx`
- `label_encoder.pkl`

Si absent, uploader les fichiers :

```python
session.file.put("cnn_respiratory.onnx", "@model_stage", auto_compress=False, overwrite=True)
session.file.put("label_encoder.pkl", "@model_stage", auto_compress=False, overwrite=True)
```

---

## Étape 2 — Créer la UDF `predict_respiratory`

Exécuter ce code Python dans une cellule Snowflake ou notebook :

### Code UDF

```python
@udf(
    name="predict_respiratory",
    is_permanent=True,
    stage_location="@model_stage",
    packages=["onnxruntime", "scipy", "numpy", "scikit-learn"],
    imports=[
        "@model_stage/cnn_respiratory.onnx",
        "@model_stage/label_encoder.pkl"
    ],
    replace=True,
    session=session
)
def predict_respiratory(audio_bytes: bytes) -> str:
    import onnxruntime as ort
    import numpy as np
    import pickle
    import json
    import io
    import os
    import sys
    from scipy.io import wavfile
    from scipy import signal as scipy_signal
    from scipy.ndimage import zoom

    TARGET_SR  = 22050
    TARGET_LEN = int(TARGET_SR * 6.0)

    # ── Chercher le modèle ONNX ──────────────────────────────────────
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

    # ── Chercher le LabelEncoder ──────────────────────────────────────
    le_path = None
    for path in sys.path:
        candidate = os.path.join(path, "label_encoder.pkl")
        if os.path.exists(candidate):
            le_path = candidate
            break

    if le_path is None:
        return json.dumps({"error": "LabelEncoder introuvable"})

    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    # ── Prétraitement audio ──────────────────────────────────────────
    sr, y = wavfile.read(io.BytesIO(audio_bytes))

    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.dtype != np.float32:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

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

    # ── Construction des 6 canaux ────────────────────────────────────
    def make_channel(arr, target_h=128, target_w=259):
        """Redimensionne n'importe quelle feature map vers (128, 259)"""
        zh = target_h / arr.shape[0]
        zw = target_w / arr.shape[1]
        return zoom(arr, (zh, zw)).astype(np.float32)

    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # Canal 1 — Spectrogramme Mel (feature principale)
    f_spec, t_spec, Sxx = scipy_signal.spectrogram(
        y_clean, TARGET_SR, nperseg=2048, noverlap=2048-512
    )
    mel_db = 10 * np.log10(Sxx + 1e-10)
    ch1    = make_channel(normalize(mel_db))

    # Canal 2 — Delta Mel (dérivée temporelle)
    delta = np.diff(mel_db, axis=1, prepend=mel_db[:, :1])
    ch2   = make_channel(normalize(delta))

    # Canal 3 — Delta-Delta Mel (dérivée seconde)
    delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
    ch3    = make_channel(normalize(delta2))

    # Canal 4 — ZCR (Zero Crossing Rate) par frame
    frame_len = 512
    zcr_frames = []
    for i in range(0, len(y_clean) - frame_len, frame_len // 2):
        frame = y_clean[i:i + frame_len]
        zcr_frames.append(np.mean(np.abs(np.diff(np.sign(frame)))) / 2)
    zcr_arr = np.array(zcr_frames).reshape(1, -1)
    zcr_arr = np.repeat(zcr_arr, 128, axis=0)
    ch4     = make_channel(normalize(zcr_arr))

    # Canal 5 — RMS Energy par frame
    rms_frames = []
    for i in range(0, len(y_clean) - frame_len, frame_len // 2):
        frame = y_clean[i:i + frame_len]
        rms_frames.append(np.sqrt(np.mean(frame**2)))
    rms_arr = np.array(rms_frames).reshape(1, -1)
    rms_arr = np.repeat(rms_arr, 128, axis=0)
    ch5     = make_channel(normalize(rms_arr))

    # Canal 6 — Spectral Centroid par frame
    centroid_frames = []
    freqs = f_spec
    for col_idx in range(Sxx.shape[1]):
        spec_col = Sxx[:, col_idx]
        total    = spec_col.sum()
        if total > 0:
            centroid_frames.append(np.sum(freqs * spec_col) / total)
        else:
            centroid_frames.append(0.0)
    centroid_arr = np.array(centroid_frames).reshape(1, -1)
    centroid_arr = np.repeat(centroid_arr, 128, axis=0)
    ch6          = make_channel(normalize(centroid_arr))

    # ── Assemblage des 6 canaux → (1, 6, 128, 259) ──────────────────
    tensor = np.stack([ch1, ch2, ch3, ch4, ch5, ch6], axis=0)  # (6, 128, 259)
    tensor = tensor[np.newaxis, :, :, :]                        # (1, 6, 128, 259)

    # ── Inférence ONNX ───────────────────────────────────────────────
    scores = sess.run(None, {input_name: tensor})[0]

    e     = np.exp(scores - scores.max())
    probs = (e / e.sum()).squeeze()

    result = {
        cls: round(float(p), 4)
        for cls, p in zip(le.classes_, probs)
    }
    result['predicted_class'] = le.classes_[probs.argmax()]
    result['confidence']      = round(float(probs.max()), 4)

    return json.dumps(result)


print("UDF 'predict_respiratory' créée avec succès")
```

Message attendu :
```
 UDF 'predict_respiratory' créée avec succès
```

---

## Étape 3 — Créer la table de stockage `PREDICTIONS`

Exécuter ce code SQL dans un notebook Snowflake (changer le type cellule en SQL) :

```sql
CREATE OR REPLACE TABLE predictions (
    prediction_id   VARCHAR DEFAULT UUID_STRING(),
    timestamp       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    pharmacie_id    VARCHAR,
    region          VARCHAR,
    filename        VARCHAR,
    predicted_class VARCHAR,
    prob_asthma     FLOAT,
    prob_copd       FLOAT,
    prob_bronchial  FLOAT,
    prob_pneumonia  FLOAT,
    prob_healthy    FLOAT,
    confidence      FLOAT
);
```

Message attendu :
```
Table PREDICTIONS successfully created.
```

---

## Étape 4 — Tester la UDF

Tester manuellement en SQL :

```sql
-- Charger un fichier WAV du stage pour test
SELECT predict_respiratory(GET(@audio_stage/test_sample.wav)) AS diagnostic;
```

Devrait retourner un JSON complet :

```json
{
    "asthma": 0.0321,
    "bronchial": 0.1144,
    "copd": 0.0723,
    "healthy": 0.0210,
    "pneumonia": 0.7602,
    "predicted_class": "pneumonia",
    "confidence": 0.7602
}
```

---

## Étape 5 — Charger l'app Streamlit

Dans Snowsight ou Snowpark, exécuter :

```python
import streamlit.io as streamlit_io
streamlit_io.run("streamlit_app.py")
```

Ou directement via l'interface Streamlit dans Snowflake si le fichier a été uploadé.

---

## Checklist de déploiement

- [ ] `@model_stage` contient `cnn_respiratory.onnx` et `label_encoder.pkl`
- [ ] UDF `predict_respiratory` créée et testée
- [ ] Table `predictions` créée
- [ ] Premiers appels à la UDF fonctionnels
- [ ] App Streamlit `streamlit_app.py` lancée
- [ ] Quelques prédictions enregistrées dans `predictions`

---

## Dépannage

### Erreur : `Modèle ONNX introuvable`
- Vérifier que `@model_stage/cnn_respiratory.onnx` existe
- Relancer la création de la UDF avec `replace=True`

### Erreur : `LabelEncoder introuvable`
- Vérifier que `@model_stage/label_encoder.pkl` existe
- Relancer la création de la UDF avec `replace=True`

### Erreur package manquant
- Ajouter le package en relançant la UDF avec le paramètre packages mis à jour