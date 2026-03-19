import streamlit as st
import json
import base64
import numpy as np
import altair as alt
import pandas as pd
from snowflake.snowpark.context import get_active_session
import io
import struct

st.set_page_config(
    page_title="Tessan — Diagnostic Respiratoire",
    layout="wide"
)

session = get_active_session()

# ── Lecture WAV sans scipy ────────────────────────────────────────
def read_wav(audio_bytes):
    """Lecture WAV robuste sans scipy"""
    buf = io.BytesIO(audio_bytes)
    
    # Vérifier signature RIFF
    riff = buf.read(4)
    if riff != b'RIFF':
        raise ValueError("Pas un fichier WAV valide")
    
    buf.read(4)  # taille totale — ignorée
    wave = buf.read(4)
    if wave != b'WAVE':
        raise ValueError("Pas un fichier WAV valide")

    # Valeurs par défaut
    sr         = 44100
    n_channels = 1
    sampwidth  = 2
    y          = np.array([], dtype=np.float32)

    # Parcourir tous les chunks jusqu'à trouver fmt et data
    while True:
        chunk_hdr = buf.read(8)
        if len(chunk_hdr) < 8:
            break

        chunk_id, chunk_size = struct.unpack('<4sI', chunk_hdr)

        if chunk_id == b'fmt ':
            fmt_data = buf.read(chunk_size)
            audio_fmt, n_channels, sr, byte_rate, block_align, bits = \
                struct.unpack('<HHIIHH', fmt_data[:16])
            sampwidth = bits // 8
            # Si chunk fmt > 16 octets (format étendu), ignorer le reste
            if chunk_size > 16:
                pass  # déjà lu les 16 premiers octets utiles

        elif chunk_id == b'data':
            raw = buf.read(chunk_size)
            if sampwidth == 1:
                y = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                y = (y - 128.0) / 128.0
            elif sampwidth == 2:
                y = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                y = y / 32768.0
            elif sampwidth == 3:
                # 24-bit — lecture manuelle octet par octet
                n_samples = len(raw) // 3
                y = np.zeros(n_samples, dtype=np.float32)
                for i in range(n_samples):
                    b0, b1, b2 = raw[3*i], raw[3*i+1], raw[3*i+2]
                    val = (b2 << 16) | (b1 << 8) | b0
                    if val >= 0x800000:
                        val -= 0x1000000
                    y[i] = val / 8388608.0
            elif sampwidth == 4:
                y = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
                y = y / 2147483648.0
            break  # data trouvé — on arrête

        else:
            # Chunk inconnu (LIST, INFO, etc.) — on saute
            # chunk_size impair → padding d'un octet
            skip = chunk_size + (chunk_size % 2)
            buf.seek(skip, 1)

    # Stéréo → mono
    if n_channels == 2 and len(y) > 0:
        y = y.reshape(-1, 2).mean(axis=1)

    if len(y) == 0:
        raise ValueError("Aucune donnée audio trouvée dans le fichier")

    return sr, y
# ── Spectrogramme simple sans scipy ──────────────────────────────
def simple_spectrogram(y, sr, nperseg=512):
    """STFT basique avec fenêtre de Hann"""
    hop    = nperseg // 2
    window = np.hanning(nperseg)
    frames = []
    for i in range(0, len(y) - nperseg, hop):
        frame  = y[i:i+nperseg] * window
        spec   = np.abs(np.fft.rfft(frame))
        frames.append(spec)
    Sxx   = np.array(frames).T          # (freq, time)
    freqs = np.fft.rfftfreq(nperseg, 1/sr)
    times = np.arange(len(frames)) * hop / sr
    return freqs, times, Sxx

# ── Header ────────────────────────────────────────────────────────
st.title("Détection des Maladies Respiratoires")
st.caption("Cabines Tessan · Analyse IA des sons respiratoires")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Informations cabine")
    pharmacie_id = st.text_input("ID Pharmacie", value="PH-75001")
    region = st.selectbox("Région", [
        "Île-de-France", "Auvergne-Rhône-Alpes",
        "Provence-Alpes-Côte d'Azur", "Occitanie",
        "Nouvelle-Aquitaine", "Autre"
    ])
    st.divider()
    st.caption("Tessan x Snowflake Hackathon")

# ── Upload ────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Déposez un fichier audio respiratoire (.wav)",
    type=["wav"]
)

if uploaded:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Signal audio")

        audio_bytes = uploaded.read()
        sr, y = read_wav(audio_bytes)

        display_len = int(min(6.0, len(y)/sr) * sr)
        y_display   = y[:display_len]

        # Sous-échantillonner pour affichage
        step  = max(1, len(y_display) // 1000)
        times = np.linspace(0, len(y_display)/sr, len(y_display[::step]))

        # Forme d'onde via altair
        df_wave = pd.DataFrame({
            'time': times,
            'amplitude': y_display[::step]
        })
        wave_chart = alt.Chart(df_wave).mark_line(
            color='steelblue', strokeWidth=0.8
        ).encode(
            x=alt.X('time:Q', title='Temps (s)'),
            y=alt.Y('amplitude:Q', title='Amplitude',
                    scale=alt.Scale(domain=[-1, 1]))
        ).properties(
            title="Forme d'onde", height=200
        )
        st.altair_chart(wave_chart, use_container_width=True)

        # Spectrogramme via altair heatmap
        freqs, times_s, Sxx = simple_spectrogram(y_display, sr, nperseg=512)
        freq_mask = freqs <= 4000
        Sxx_db    = 10 * np.log10(Sxx[freq_mask] + 1e-10)

        # Sous-échantillonner pour altair
        t_step = max(1, Sxx_db.shape[1] // 100)
        f_step = max(1, Sxx_db.shape[0] // 50)
        Sxx_sub  = Sxx_db[::f_step, ::t_step]
        freqs_sub = freqs[freq_mask][::f_step]
        times_sub = times_s[::t_step]

        rows = []
        for fi, f in enumerate(freqs_sub):
            for ti, t in enumerate(times_sub):
                rows.append({
                    'time': round(float(t), 3),
                    'freq': round(float(f), 1),
                    'db':   round(float(Sxx_sub[fi, ti]), 2)
                })
        df_spec = pd.DataFrame(rows)

        spec_chart = alt.Chart(df_spec).mark_rect().encode(
            x=alt.X('time:O', title='Temps (s)',
                    axis=alt.Axis(labelOverlap=True)),
            y=alt.Y('freq:O', title='Fréquence (Hz)',
                    sort='descending',
                    axis=alt.Axis(labelOverlap=True)),
            color=alt.Color('db:Q', scale=alt.Scale(scheme='magma'),
                            title='dB')
        ).properties(
            title="Spectrogramme", height=200
        )
        st.altair_chart(spec_chart, use_container_width=True)

    with col2:
        st.subheader("Diagnostic IA")

        if st.button("Analyser", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):

                import tempfile, os
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                session.file.put(
                    tmp_path, "@audio_stage",
                    auto_compress=False, overwrite=True
                )
                os.unlink(tmp_path)

                from snowflake.snowpark.files import SnowflakeFile
                stage_path = f"@audio_stage/{uploaded.name}"

                with SnowflakeFile.open(stage_path, 'rb') as f:
                    file_bytes = f.read()

                session.sql(
                    "CREATE OR REPLACE TEMP TABLE tmp_audio "
                    "(audio_data BINARY)"
                ).collect()

                audio_b64 = base64.b64encode(file_bytes).decode()
                session.sql(f"""
                    INSERT INTO tmp_audio
                    SELECT TO_BINARY('{audio_b64}', 'BASE64')
                """).collect()

                result_raw = session.sql("""
                    SELECT predict_respiratory(audio_data) AS diagnostic
                    FROM tmp_audio
                """).collect()[0]['DIAGNOSTIC']

                result = json.loads(result_raw)

            # -- Affichage résultat --------------------------------
            predicted  = result['predicted_class']
            confidence = result['confidence']

            emoji_map = {
                'pneumonia': '🔴',
                'copd':      '🟠',
                'asthma':    '🟡',
                'bronchial': '🟡',
                'healthy':   '🟢'
            }
            reco_map = {
                'pneumonia': 'Consultation urgente recommandée',
                'copd':      'Suivi pneumologue recommandé',
                'asthma':    'Consultation dans les 48h',
                'bronchial': 'Consultation dans les 48h',
                'healthy':   'Aucune anomalie détectée'
            }

            st.metric(
                label="Diagnostic principal",
                value=f"{emoji_map[predicted]} {predicted.upper()}",
                delta=f"Confiance : {confidence*100:.1f}%"
            )
            st.info(reco_map[predicted])
            st.divider()

            # Graphique probabilités altair
            classes = ['asthma', 'bronchial', 'copd', 'healthy', 'pneumonia']
            probs   = [result.get(c, 0) for c in classes]

            df_probs = pd.DataFrame({
                'classe':      classes,
                'probabilite': probs,
                'selected':    [c == predicted for c in classes]
            })

            prob_chart = alt.Chart(df_probs).mark_bar().encode(
                x=alt.X('probabilite:Q',
                        scale=alt.Scale(domain=[0, 1]),
                        title='Probabilité'),
                y=alt.Y('classe:N', sort='-x', title=''),
                color=alt.condition(
                    alt.datum.selected,
                    alt.value('#EF9F27'),
                    alt.value('#B5D4F4')
                ),
                text=alt.Text('probabilite:Q', format='.3f')
            ).mark_bar().properties(height=200, title="Probabilités par classe")

            text_layer = prob_chart.mark_text(
                align='left', dx=3
            ).encode(text=alt.Text('probabilite:Q', format='.3f'))

            st.altair_chart(
                alt.layer(prob_chart, text_layer),
                use_container_width=True
            )

            # Enregistrement PREDICTIONS
            session.sql(f"""
                INSERT INTO predictions (
                    pharmacie_id, region, filename,
                    predicted_class, confidence,
                    prob_asthma, prob_copd, prob_bronchial,
                    prob_pneumonia, prob_healthy
                ) VALUES (
                    '{pharmacie_id}', '{region}', '{uploaded.name}',
                    '{predicted}', {confidence},
                    {result.get('asthma',    0)},
                    {result.get('copd',      0)},
                    {result.get('bronchial', 0)},
                    {result.get('pneumonia', 0)},
                    {result.get('healthy',   0)}
                )
            """).collect()

            st.success("Résultat enregistré dans Snowflake")

# ── Dashboard épidémiologique ─────────────────────────────────────
st.divider()
st.subheader("Surveillance épidémiologique — réseau Tessan")

total = session.sql(
    "SELECT COUNT(*) as nb FROM predictions"
).collect()[0]['NB']

if total == 0:
    st.info("Aucune prédiction enregistrée pour l'instant.")
else:
    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Total diagnostics", total)

    with col4:
        top = session.sql("""
            SELECT predicted_class, COUNT(*) as nb
            FROM predictions
            GROUP BY predicted_class
            ORDER BY nb DESC LIMIT 1
        """).collect()[0]
        st.metric("Pathologie dominante",
                  top['PREDICTED_CLASS'].upper())

    with col5:
        conf_moy = session.sql("""
            SELECT ROUND(AVG(confidence)*100, 1) as moy
            FROM predictions
        """).collect()[0]['MOY']
        st.metric("Confiance moyenne", f"{conf_moy}%")

    st.divider()
    col6, col7 = st.columns([1, 1])

    with col6:
        st.markdown("**Distribution des diagnostics**")
        df_stats = session.sql("""
            SELECT
                predicted_class               AS Pathologie,
                COUNT(*)                      AS Nb,
                ROUND(AVG(confidence)*100, 1) AS Confiance_moy
            FROM predictions
            GROUP BY predicted_class
            ORDER BY Nb DESC
        """).to_pandas()

        bar_chart = alt.Chart(df_stats).mark_bar(
            color='#378ADD'
        ).encode(
            x=alt.X('PATHOLOGIE:N', title=''),
            y=alt.Y('NB:Q', title='Nombre de diagnostics'),
            tooltip=['PATHOLOGIE', 'NB', 'CONFIANCE_MOY']
        ).properties(height=250)
        st.altair_chart(bar_chart, use_container_width=True)

    with col7:
        st.markdown("**10 dernières prédictions**")
        df_recent = session.sql("""
            SELECT
                TO_CHAR(timestamp, 'DD/MM HH24:MI') AS Heure,
                pharmacie_id                         AS Pharmacie,
                region                               AS Region,
                predicted_class                      AS Diagnostic,
                ROUND(confidence*100, 1)             AS Confiance
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 10
        """).to_pandas()
        st.dataframe(df_recent, use_container_width=True)