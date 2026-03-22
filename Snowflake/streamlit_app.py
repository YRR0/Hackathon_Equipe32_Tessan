import streamlit as st
import json
import base64
import numpy as np
import altair as alt
import pandas as pd
from snowflake.snowpark.context import get_active_session
import io
import struct

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Tessan — Diagnostic Respiratoire",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

session = get_active_session()

# ── CSS Global ────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&family=Sora:wght@600;700;800&display=swap');

  /* ── Reset & base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #080C14 !important;
    font-family: 'DM Sans', sans-serif;
    color: #C8D4E8;
  }
  [data-testid="stAppViewContainer"] > .main {
    background-color: #080C14 !important;
  }
  [data-testid="block-container"] {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0D1220 !important;
    border-right: 1px solid #1A2540 !important;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stTextInput label {
    color: #5A7099 !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
  }
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] select {
    background: #131B2E !important;
    border: 1px solid #1E2D4A !important;
    border-radius: 6px !important;
    color: #C8D4E8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
  }

  /* ── Typography ── */
  h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
    color: #EEF3FB !important;
  }
  h1 { font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
  h2 { font-size: 1.1rem !important; font-weight: 600 !important; }
  h3 { font-size: 0.95rem !important; font-weight: 600 !important; }
  p, li { color: #8A9BBF; line-height: 1.65; font-size: 0.9rem; }

  /* ── Cards ── */
  .card {
    background: #0D1220;
    border: 1px solid #1A2540;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .card-glass {
    background: rgba(13, 18, 32, 0.7);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.5rem;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: #0D1220 !important;
    border: 1px solid #1A2540 !important;
    border-radius: 10px !important;
    padding: 1.1rem 1.3rem !important;
  }
  [data-testid="stMetricLabel"] {
    color: #5A7099 !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
  }
  [data-testid="stMetricValue"] {
    color: #EEF3FB !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
  }
  [data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #0D1220 !important;
    border: 1.5px dashed #1E2D4A !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    transition: border-color 0.2s ease;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #3D6AC4 !important;
  }
  [data-testid="stFileUploaderDropzoneInstructions"] {
    color: #5A7099 !important;
    font-size: 0.85rem !important;
  }

  /* ── Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 0 0 0 rgba(37,99,235,0) !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    box-shadow: 0 0 24px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #3B82F6 !important; }

  /* ── Alerts ── */
  .stInfo {
    background: rgba(37, 99, 235, 0.08) !important;
    border: 1px solid rgba(37, 99, 235, 0.25) !important;
    border-radius: 8px !important;
    color: #93B4F5 !important;
  }
  .stWarning {
    background: rgba(234, 179, 8, 0.07) !important;
    border: 1px solid rgba(234, 179, 8, 0.2) !important;
    border-radius: 8px !important;
    color: #FDE047 !important;
  }
  .stSuccess {
    background: rgba(16, 185, 129, 0.07) !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    border-radius: 8px !important;
    color: #6EE7B7 !important;
  }

  /* ── Divider ── */
  hr { border-color: #1A2540 !important; margin: 2rem 0 !important; }

  /* ── DataFrame ── */
  .dataframe {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #0D1220 !important;
    border: 1px solid #1A2540 !important;
    border-radius: 8px !important;
    color: #C8D4E8 !important;
  }
  [data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
  }

  /* ── Captions & labels ── */
  .stCaption, .caption-label {
    color: #5A7099 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
  }

  /* ── Section labels ── */
  .section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3D6AC4;
    margin-bottom: 0.75rem;
    display: block;
  }

  /* ── Diagnostic badge ── */
  .diag-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .badge-red    { background: rgba(239,68,68,0.12); color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3); }
  .badge-orange { background: rgba(249,115,22,0.12); color: #FDBA74; border: 1px solid rgba(249,115,22,0.3); }
  .badge-yellow { background: rgba(234,179,8,0.12); color: #FDE047; border: 1px solid rgba(234,179,8,0.3); }
  .badge-green  { background: rgba(16,185,129,0.12); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.3); }

  /* ── Stat row ── */
  .stat-row {
    display: flex;
    gap: 1px;
    background: #1A2540;
    border-radius: 8px;
    overflow: hidden;
    margin: 1rem 0;
  }
  .stat-cell {
    flex: 1;
    background: #0D1220;
    padding: 0.75rem 1rem;
    text-align: center;
  }
  .stat-cell .val {
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    color: #EEF3FB;
  }
  .stat-cell .lbl {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5A7099;
    margin-top: 2px;
  }

  /* ── Nav header ── */
  .nav-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1A2540;
  }
  .logo-area { display: flex; align-items: center; gap: 0.75rem; }
  .logo-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
  }
  .logo-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #EEF3FB;
    letter-spacing: -0.01em;
  }
  .logo-sub {
    font-size: 0.7rem;
    color: #5A7099;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px rgba(16,185,129,0.6);
    display: inline-block;
    margin-right: 0.4rem;
  }
  .status-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #10B981;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 100px;
    padding: 0.25rem 0.7rem;
  }

  /* ── Epidemio cards ── */
  .epi-metric {
    background: #0D1220;
    border: 1px solid #1A2540;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .epi-metric .epi-val {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #EEF3FB;
  }
  .epi-metric .epi-lbl {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5A7099;
    margin-top: 0.25rem;
  }
  
  /* ── Scroll fix ── */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: #0D1220; }
  ::-webkit-scrollbar-thumb { background: #1E2D4A; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers WAV & Spectro ────────────────────────────────────────
def read_wav(audio_bytes):
    buf = io.BytesIO(audio_bytes)
    riff = buf.read(4)
    if riff != b'RIFF':
        raise ValueError("Pas un fichier WAV valide")
    buf.read(4)
    wave = buf.read(4)
    if wave != b'WAVE':
        raise ValueError("Pas un fichier WAV valide")
    sr, n_channels, sampwidth = 44100, 1, 2
    y = np.array([], dtype=np.float32)
    while True:
        chunk_hdr = buf.read(8)
        if len(chunk_hdr) < 8:
            break
        chunk_id, chunk_size = struct.unpack('<4sI', chunk_hdr)
        if chunk_id == b'fmt ':
            fmt_data = buf.read(chunk_size)
            _, n_channels, sr, _, _, bits = struct.unpack('<HHIIHH', fmt_data[:16])
            sampwidth = bits // 8
        elif chunk_id == b'data':
            raw = buf.read(chunk_size)
            if sampwidth == 1:
                y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
            elif sampwidth == 2:
                y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 3:
                n_s = len(raw) // 3
                y = np.zeros(n_s, dtype=np.float32)
                for i in range(n_s):
                    b0, b1, b2 = raw[3*i], raw[3*i+1], raw[3*i+2]
                    v = (b2 << 16) | (b1 << 8) | b0
                    if v >= 0x800000:
                        v -= 0x1000000
                    y[i] = v / 8388608.0
            elif sampwidth == 4:
                y = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            break
        else:
            buf.seek(chunk_size + (chunk_size % 2), 1)
    if n_channels == 2 and len(y) > 0:
        y = y.reshape(-1, 2).mean(axis=1)
    if len(y) == 0:
        raise ValueError("Aucune donnée audio trouvée")
    return sr, y


def simple_spectrogram(y, sr, nperseg=512):
    hop, window = nperseg // 2, np.hanning(nperseg)
    frames = [np.abs(np.fft.rfft(y[i:i+nperseg] * window))
              for i in range(0, len(y) - nperseg, hop)]
    Sxx   = np.array(frames).T
    freqs = np.fft.rfftfreq(nperseg, 1/sr)
    times = np.arange(len(frames)) * hop / sr
    return freqs, times, Sxx


# ── Altair chart theme override ──────────────────────────────────
def altair_dark_theme():
    return {
        "config": {
            "background": "#0D1220",
            "view": {"stroke": "transparent"},
            "axis": {
                "domainColor": "#1A2540",
                "gridColor": "#131B2E",
                "tickColor": "#1A2540",
                "labelColor": "#5A7099",
                "titleColor": "#5A7099",
                "labelFont": "DM Mono",
                "titleFont": "DM Sans",
                "labelFontSize": 10,
                "titleFontSize": 11,
            },
            "legend": {
                "labelColor": "#8A9BBF",
                "titleColor": "#5A7099",
                "labelFont": "DM Mono",
                "labelFontSize": 10,
            },
            "title": {
                "color": "#C8D4E8",
                "font": "Sora",
                "fontSize": 12,
                "fontWeight": 600,
                "anchor": "start",
                "offset": 8,
            },
        }
    }

alt.themes.register("dark_medical", altair_dark_theme)
alt.themes.enable("dark_medical")


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem">
      <div style="font-family:'Sora',sans-serif;font-size:1rem;font-weight:700;color:#EEF3FB;letter-spacing:-0.01em;">Tessan</div>
      <div style="font-size:0.65rem;color:#3D6AC4;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px;">Diagnostic Respiratoire</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<span class="section-label">Configuration cabine</span>', unsafe_allow_html=True)
    pharmacie_id = st.text_input("ID Pharmacie", value="PH-75001", label_visibility="visible")
    region = st.selectbox("Région", [
        "Île-de-France", "Auvergne-Rhône-Alpes",
        "Provence-Alpes-Côte d'Azur", "Occitanie",
        "Nouvelle-Aquitaine", "Autre"
    ])

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#3A5080;line-height:1.7;">
      <div style="margin-bottom:6px;">🔒 Données chiffrées end-to-end</div>
      <div style="margin-bottom:6px;">☁️ Stockage Snowflake sécurisé</div>
      <div>🏥 Conforme RGPD & HDS</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#2A3F60;">
      Tessan × Snowflake Hackathon<br>v2.0.0 — 2025
    </div>
    """, unsafe_allow_html=True)


# ── Top nav header ─────────────────────────────────────────────────
st.markdown("""
<div class="nav-header">
  <div class="logo-area">
    <div class="logo-icon">🫁</div>
    <div>
      <div class="logo-title">Analyse Respiratoire</div>
      <div class="logo-sub">IA · Signal Audio · Épidémiologie</div>
    </div>
  </div>
  <div>
    <span class="status-tag"><span class="status-dot"></span>Système opérationnel</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Upload zone ───────────────────────────────────────────────────
st.markdown('<span class="section-label">01 — Chargement du signal</span>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Déposez un fichier audio respiratoire",
    type=["wav"],
    help="Format supporté : WAV mono ou stéréo, 16–44 100 Hz"
)


# ── Analyse principale ────────────────────────────────────────────
if uploaded:
    audio_bytes = uploaded.read()
    sr, y = read_wav(audio_bytes)

    st.markdown('<span class="section-label">02 — Visualisation du signal</span>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    # ── Colonne gauche : visualisations ─────────────────────────
    with col1:
        display_len = int(min(6.0, len(y)/sr) * sr)
        y_display   = y[:display_len]
        step        = max(1, len(y_display) // 1200)
        times_arr   = np.linspace(0, len(y_display)/sr, len(y_display[::step]))

        df_wave = pd.DataFrame({'time': times_arr, 'amplitude': y_display[::step]})

        wave_chart = alt.Chart(df_wave).mark_area(
            line={'color': '#3B82F6', 'strokeWidth': 1},
            color=alt.Gradient(
                gradient='linear',
                stops=[
                    alt.GradientStop(color='rgba(37,99,235,0.25)', offset=0),
                    alt.GradientStop(color='rgba(37,99,235,0.0)',  offset=1)
                ],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('time:Q', title='Temps (s)', axis=alt.Axis(tickCount=6)),
            y=alt.Y('amplitude:Q', title='Amplitude',
                    scale=alt.Scale(domain=[-1, 1]))
        ).properties(title="Forme d'onde", height=180)

        st.altair_chart(wave_chart, use_container_width=True)

        # Spectrogramme
        freqs, times_s, Sxx = simple_spectrogram(y_display, sr, nperseg=512)
        freq_mask = freqs <= 4000
        Sxx_db    = 10 * np.log10(Sxx[freq_mask] + 1e-10)
        t_step    = max(1, Sxx_db.shape[1] // 120)
        f_step    = max(1, Sxx_db.shape[0] // 60)
        Sxx_sub   = Sxx_db[::f_step, ::t_step]
        freqs_sub = freqs[freq_mask][::f_step]
        times_sub = times_s[::t_step]

        rows = [
            {'time': round(float(t), 3), 'freq': round(float(f), 1), 'db': round(float(Sxx_sub[fi, ti]), 2)}
            for fi, f in enumerate(freqs_sub)
            for ti, t in enumerate(times_sub)
        ]
        df_spec = pd.DataFrame(rows)

        spec_chart = alt.Chart(df_spec).mark_rect().encode(
            x=alt.X('time:O', title='Temps (s)', axis=alt.Axis(labelOverlap=True, tickCount=6)),
            y=alt.Y('freq:O', title='Fréquence (Hz)', sort='descending',
                    axis=alt.Axis(labelOverlap=True, tickCount=6)),
            color=alt.Color('db:Q', scale=alt.Scale(scheme='blues'), title='dB')
        ).properties(title="Spectrogramme temps-fréquence", height=200)

        st.altair_chart(spec_chart, use_container_width=True)

        # Infos techniques compactes
        duration = len(y) / sr
        file_kb   = len(audio_bytes) / 1024
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-cell">
            <div class="val">{sr:,} Hz</div>
            <div class="lbl">Fréq. d'échantillonnage</div>
          </div>
          <div class="stat-cell">
            <div class="val">{duration:.1f}s</div>
            <div class="lbl">Durée</div>
          </div>
          <div class="stat-cell">
            <div class="val">{file_kb:.0f} KB</div>
            <div class="lbl">Taille fichier</div>
          </div>
          <div class="stat-cell">
            <div class="val">{len(y):,}</div>
            <div class="lbl">Échantillons</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Colonne droite : diagnostic ──────────────────────────────
    with col2:
        st.markdown('<span class="section-label">03 — Analyse IA</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card" style="margin-bottom:1.25rem;">
          <div style="font-size:0.72rem;color:#5A7099;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;">Fichier sélectionné</div>
          <div style="font-family:'DM Mono',monospace;color:#C8D4E8;font-size:0.85rem;">📄 {uploaded.name}</div>
        </div>
        """, unsafe_allow_html=True)

        analyze_btn = st.button("⚡ Lancer le diagnostic", type="primary", use_container_width=True)

        if analyze_btn:
            with st.spinner("Analyse du signal en cours…"):
                # Encode audio directement depuis les bytes deja lus
                session.sql("CREATE OR REPLACE TEMP TABLE tmp_audio (audio_data BINARY)").collect()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                session.sql(f"INSERT INTO tmp_audio SELECT TO_BINARY('{audio_b64}', 'BASE64')").collect()
                result_raw = session.sql(
                    "SELECT predict_respiratory(audio_data) AS diagnostic FROM tmp_audio"
                ).collect()[0]['DIAGNOSTIC']
                result = json.loads(result_raw)

            # -- Résultats --
            predicted  = result['predicted_class']
            confidence = result['confidence']

            badge_map = {
                'pneumonia': ('badge-red',    '🔴', 'Consultation urgente recommandée'),
                'copd':      ('badge-orange', '🟠', 'Suivi pneumologue recommandé'),
                'asthma':    ('badge-yellow', '🟡', 'Consultation dans les 48h'),
                'bronchial': ('badge-yellow', '🟡', 'Consultation dans les 48h'),
                'healthy':   ('badge-green',  '🟢', 'Aucune anomalie détectée'),
            }
            badge_cls, emoji, reco = badge_map.get(predicted, ('badge-green', '❓', '—'))

            st.markdown(f"""
            <div class="card" style="border-color:#1E2D4A;margin-top:1rem;">
              <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;color:#5A7099;margin-bottom:0.85rem;">
                Résultat diagnostic
              </div>
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.75rem;">
                <span class="diag-badge {badge_cls}">{emoji} {predicted.upper()}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#8A9BBF;">
                  Confiance &nbsp;<strong style="color:#EEF3FB;">{confidence*100:.1f}%</strong>
                </span>
              </div>
              <div style="margin-top:1rem;padding:0.7rem 0.9rem;background:rgba(255,255,255,0.03);border-radius:7px;
                          font-size:0.82rem;color:#8A9BBF;border-left:2px solid #2563EB;">
                {reco}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Secondaire
            second      = result.get('second_class', '')
            second_prob = result.get('second_prob', 0)
            if second and second_prob > 0.15:
                st.markdown(f"""
                <div style="padding:0.6rem 0.9rem;background:rgba(234,179,8,0.06);
                            border:1px solid rgba(234,179,8,0.18);border-radius:8px;
                            font-size:0.8rem;color:#FDE047;margin-top:0.5rem;">
                  ⚠️ Diagnostic secondaire possible : <strong>{second.upper()}</strong>
                  &nbsp;<span style="color:#8A9BBF;">({second_prob*100:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

            # Graphique probabilités
            classes = ['asthma', 'bronchial', 'copd', 'healthy', 'pneumonia']
            probs   = [result.get(c, 0) for c in classes]
            df_probs = pd.DataFrame({
                'classe':      classes,
                'probabilite': probs,
                'selected':    [c == predicted for c in classes]
            }).sort_values('probabilite', ascending=True)

            bar = alt.Chart(df_probs).mark_bar(
                cornerRadiusTopRight=4, cornerRadiusBottomRight=4
            ).encode(
                x=alt.X('probabilite:Q', scale=alt.Scale(domain=[0, 1]),
                        title='Probabilité', axis=alt.Axis(format='%')),
                y=alt.Y('classe:N', sort=alt.EncodingSortField(field='probabilite'), title=''),
                color=alt.condition(
                    alt.datum.selected,
                    alt.value('#2563EB'),
                    alt.value('#1A2540')
                )
            ).properties(height=165, title="Distribution des probabilités")

            text_layer = bar.mark_text(align='left', dx=5, color='#8A9BBF',
                                       font='DM Mono', fontSize=10
                                       ).encode(text=alt.Text('probabilite:Q', format='.1%'))

            st.altair_chart(alt.layer(bar, text_layer), use_container_width=True)

            # Stats fichier
            st.markdown(f"""
            <div class="stat-row" style="margin-top:0.75rem;">
              <div class="stat-cell">
                <div class="val">{result.get('duration_sec', 0)}s</div>
                <div class="lbl">Durée</div>
              </div>
              <div class="stat-cell">
                <div class="val">{result.get('sample_rate', 0)}</div>
                <div class="lbl">Sample Rate</div>
              </div>
              <div class="stat-cell">
                <div class="val">{result.get('file_size_bytes', 0)}</div>
                <div class="lbl">Octets</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Enregistrement
            session.sql(f"""
                INSERT INTO predictions (
                    pharmacie_id, region, filename,
                    file_size_bytes, duration_sec, sample_rate,
                    predicted_class, confidence,
                    second_class, second_prob,
                    prob_asthma, prob_copd, prob_bronchial,
                    prob_pneumonia, prob_healthy
                ) VALUES (
                    '{pharmacie_id}', '{region}', '{uploaded.name}',
                    {result.get('file_size_bytes', 0)},
                    {result.get('duration_sec', 0)},
                    {result.get('sample_rate', 0)},
                    '{predicted}', {confidence},
                    '{result.get('second_class', '')}',
                    {result.get('second_prob', 0)},
                    {result.get('asthma', 0)},    {result.get('copd', 0)},
                    {result.get('bronchial', 0)}, {result.get('pneumonia', 0)},
                    {result.get('healthy', 0)}
                )
            """).collect()

            st.markdown("""
            <div style="font-size:0.75rem;color:#10B981;font-family:'DM Mono',monospace;
                        margin-top:0.75rem;text-align:right;">
              ✓ Enregistré dans Snowflake
            </div>
            """, unsafe_allow_html=True)


# ── Dashboard épidémiologique ─────────────────────────────────────
st.divider()
st.markdown('<span class="section-label">04 — Surveillance épidémiologique · Réseau Tessan</span>', unsafe_allow_html=True)

total = session.sql("SELECT COUNT(*) as nb FROM predictions").collect()[0]['NB']

if total == 0:
    st.markdown("""
    <div class="card" style="text-align:center;padding:3rem;color:#3A5080;">
      <div style="font-size:2rem;margin-bottom:0.75rem;">📊</div>
      <div style="font-size:0.85rem;">Aucune donnée de prédiction disponible.<br>Effectuez votre premier diagnostic ci-dessus.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    top = session.sql("""
        SELECT predicted_class, COUNT(*) as nb
        FROM predictions GROUP BY predicted_class ORDER BY nb DESC LIMIT 1
    """).collect()[0]
    conf_moy = session.sql("""
        SELECT ROUND(AVG(confidence)*100, 1) as moy FROM predictions
    """).collect()[0]['MOY']

    # KPI row
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("Total diagnostics", f"{total:,}")
    with col_k2:
        st.metric("Pathologie dominante", top['PREDICTED_CLASS'].upper())
    with col_k3:
        st.metric("Confiance moyenne", f"{conf_moy}%")

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    col6, col7 = st.columns([1, 1], gap="large")

    with col6:
        st.markdown('<span class="section-label">Distribution des diagnostics</span>', unsafe_allow_html=True)
        df_stats = session.sql("""
            SELECT
                predicted_class               AS Pathologie,
                COUNT(*)                      AS Nb,
                ROUND(AVG(confidence)*100, 1) AS Confiance_pct
            FROM predictions
            GROUP BY predicted_class
            ORDER BY Nb DESC
        """).to_pandas()

        bar_epi = alt.Chart(df_stats).mark_bar(
            cornerRadiusTopRight=5, cornerRadiusBottomRight=5
        ).encode(
            x=alt.X('PATHOLOGIE:N', title='', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('NB:Q', title='Diagnostics'),
            color=alt.Color('PATHOLOGIE:N',
                            scale=alt.Scale(
                                domain=['asthma','bronchial','copd','healthy','pneumonia'],
                                range=['#F59E0B','#6366F1','#F97316','#10B981','#EF4444']
                            ),
                            legend=None),
            tooltip=[
                alt.Tooltip('PATHOLOGIE:N', title='Pathologie'),
                alt.Tooltip('NB:Q', title='Cas'),
                alt.Tooltip('CONFIANCE_PCT:Q', title='Confiance moy. (%)')
            ]
        ).properties(height=220)

        st.altair_chart(bar_epi, use_container_width=True)

    with col7:
        st.markdown('<span class="section-label">10 dernières prédictions</span>', unsafe_allow_html=True)
        df_recent = session.sql("""
            SELECT
                TO_CHAR(timestamp, 'DD/MM HH24:MI')  AS Heure,
                pharmacie_id                          AS Pharmacie,
                region                                AS Region,
                filename                              AS Fichier,
                predicted_class                       AS Diagnostic,
                second_class                          AS Diag_2,
                ROUND(second_prob * 100, 1)           AS Prob_2,
                ROUND(confidence * 100, 1)            AS Confiance,
                duration_sec                          AS Duree_sec,
                sample_rate                           AS SampleRate,
                file_size_bytes                       AS Taille_bytes
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 10
        """).to_pandas()
        st.dataframe(df_recent, use_container_width=True, height=255)
