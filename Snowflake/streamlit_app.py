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

  /* ── Mode selector ── */
  .mode-selector {
    display: flex;
    gap: 0.5rem;
    background: #0D1220;
    border: 1px solid #1A2540;
    border-radius: 10px;
    padding: 0.35rem;
    width: fit-content;
  }
  .mode-btn {
    padding: 0.4rem 1.1rem;
    border-radius: 7px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.18s ease;
    border: none;
  }
  .mode-btn-active {
    background: #2563EB;
    color: #FFFFFF;
    box-shadow: 0 2px 10px rgba(37,99,235,0.4);
  }
  .mode-btn-inactive {
    background: transparent;
    color: #5A7099;
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

  /* ── Diagnostic result block (PRIMARY) ── */
  .diag-result-block {
    background: linear-gradient(135deg, #0D1220 0%, #101729 100%);
    border: 1px solid #1E2D4A;
    border-radius: 16px;
    padding: 2rem;
    margin: 1.25rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .diag-result-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
  }
  .diag-result-block.severity-red::before   { background: linear-gradient(90deg, #EF4444, #DC2626); }
  .diag-result-block.severity-orange::before{ background: linear-gradient(90deg, #F97316, #EA580C); }
  .diag-result-block.severity-yellow::before{ background: linear-gradient(90deg, #EAB308, #CA8A04); }
  .diag-result-block.severity-green::before { background: linear-gradient(90deg, #10B981, #059669); }

  .diag-icon { font-size: 3rem; margin-bottom: 0.75rem; }
  .diag-label-main {
    font-family: 'Sora', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #EEF3FB;
    margin-bottom: 0.35rem;
  }
  .diag-confidence-human {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.8rem;
    border-radius: 100px;
    margin-bottom: 1.25rem;
  }
  .conf-high   { background: rgba(16,185,129,0.12); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.3); }
  .conf-medium { background: rgba(234,179,8,0.12);  color: #FDE047; border: 1px solid rgba(234,179,8,0.3); }
  .conf-low    { background: rgba(239,68,68,0.12);  color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3); }

  /* ── Recommendation block ── */
  .reco-block {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    background: rgba(37,99,235,0.07);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    text-align: left;
  }
  .reco-icon { font-size: 1.25rem; flex-shrink: 0; margin-top: 1px; }
  .reco-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #93B4F5;
    margin-bottom: 0.25rem;
  }
  .reco-text {
    font-size: 0.82rem;
    color: #8A9BBF;
    line-height: 1.55;
  }

  /* ── Disclaimer ── */
  .disclaimer {
    background: rgba(90,112,153,0.07);
    border: 1px solid rgba(90,112,153,0.2);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.75rem;
    color: #5A7099;
    text-align: center;
    margin-top: 1rem;
  }

  /* ── Instruction card (patient) ── */
  .instruction-card {
    background: #0D1220;
    border: 1px solid #1A2540;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.25rem;
  }
  .instruction-step {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    margin-bottom: 0.85rem;
  }
  .step-num {
    flex-shrink: 0;
    width: 26px; height: 26px;
    background: #2563EB;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    color: #FFFFFF;
  }
  .step-text {
    font-size: 0.88rem;
    color: #C8D4E8;
    line-height: 1.55;
    padding-top: 2px;
  }

  /* ── Diagnostic badge (kept for secondary) ── */
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

  /* ── Credibility strip ── */
  .credibility-strip {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    background: rgba(37,99,235,0.05);
    border: 1px solid rgba(37,99,235,0.12);
    border-radius: 8px;
    padding: 0.65rem 1.1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
  }
  .cred-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: #5A7099;
  }
  .cred-item strong { color: #8A9BBF; }

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

  /* ── Progress bar ── */
  .stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2563EB, #3B82F6) !important;
  }

  /* ── Scroll fix ── */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: #0D1220; }
  ::-webkit-scrollbar-thumb { background: #1E2D4A; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Label maps ─────────────────────────────────────────────────────
CLASS_LABELS_FR = {
    'asthma':    "Suspicion d'asthme",
    'copd':      "Suspicion de BPCO",
    'bronchial': "Bronchite possible",
    'pneumonia': "Suspicion de pneumonie",
    'healthy':   "Fonction respiratoire normale",
}

CLASS_LABELS_FR_SHORT = {
    'asthma':    "Asthme",
    'copd':      "BPCO",
    'bronchial': "Bronchite",
    'pneumonia': "Pneumonie",
    'healthy':   "Sain",
}

BADGE_MAP = {
    'pneumonia': ('badge-red',    '🔴', 'severity-red',    '🫁', 'Consultation urgente recommandée — contactez un médecin ou rendez-vous aux urgences.',    'Élevée'),
    'copd':      ('badge-orange', '🟠', 'severity-orange', '💨', 'Un suivi par un pneumologue est recommandé dans les prochains jours.',                  'Modérée'),
    'asthma':    ('badge-yellow', '🟡', 'severity-yellow', '🌬️', 'Consultez votre médecin ou un allergologue dans les 48h.',                               'Modérée'),
    'bronchial': ('badge-yellow', '🟡', 'severity-yellow', '🫀', 'Consultez votre médecin dans les 48h pour un examen complet.',                            'Modérée'),
    'healthy':   ('badge-green',  '🟢', 'severity-green',  '✅', 'Aucune anomalie détectée. Continuez à surveiller votre santé régulièrement.',             'Élevée'),
}


def confidence_label(conf):
    if conf >= 0.75:
        return "Confiance élevée", "conf-high"
    elif conf >= 0.50:
        return "Confiance modérée", "conf-medium"
    else:
        return "Confiance faible", "conf-low"


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


def check_signal_quality(y, sr):
    """Returns (ok, warning_msg)."""
    duration = len(y) / sr
    if duration < 2.0:
        return False, "Signal trop court (< 2s). Veuillez enregistrer au moins 3 secondes d'audio."
    rms = float(np.sqrt(np.mean(y**2)))
    if rms < 0.001:
        return False, "Signal audio trop faible. Vérifiez le microphone et recommencez."
    if rms > 0.95:
        return False, "Signal saturé. Éloignez le micro et recommencez."
    return True, None


# ── Altair chart theme ────────────────────────────────────────────
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


# ── Session state ─────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "Patient"


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem">
      <div style="font-family:'Sora',sans-serif;font-size:1rem;font-weight:700;color:#EEF3FB;letter-spacing:-0.01em;">Tessan</div>
      <div style="font-size:0.65rem;color:#3D6AC4;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px;">Diagnostic Respiratoire</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Mode selector
    st.markdown('<span class="section-label">Mode d\'utilisation</span>', unsafe_allow_html=True)
    mode_choice = st.radio(
        "Mode",
        ["Patient", "Professionnel"],
        index=0 if st.session_state.mode == "Patient" else 1,
        label_visibility="collapsed",
        horizontal=True
    )
    st.session_state.mode = mode_choice

    if mode_choice == "Patient":
        st.markdown("""
        <div style="font-size:0.76rem;color:#3D6AC4;background:rgba(37,99,235,0.07);
                    border:1px solid rgba(37,99,235,0.15);border-radius:7px;padding:0.6rem 0.8rem;margin-top:0.5rem;">
          Mode simplifié 
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.76rem;color:#5A7099;background:rgba(90,112,153,0.06);
                    border:1px solid rgba(90,112,153,0.15);border-radius:7px;padding:0.6rem 0.8rem;margin-top:0.5rem;">
          Mode professionnel — détails techniques, probabilités brutes et métriques.
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

is_patient = (st.session_state.mode == "Patient")

# ── Top nav header ─────────────────────────────────────────────────
mode_label = "Mode Patient" if is_patient else "Mode Professionnel"
mode_color = "#3B82F6" if is_patient else "#8B5CF6"
st.markdown(f"""
<div class="nav-header">
  <div class="logo-area">
    <div class="logo-icon">🫁</div>
    <div>
      <div class="logo-title">Analyse Respiratoire</div>
      <div class="logo-sub">IA · Signal Audio · Épidémiologie</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:0.75rem;">
    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{mode_color};
                 background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);
                 border-radius:100px;padding:0.25rem 0.7rem;">
      {mode_label}
    </span>
    <span class="status-tag"><span class="status-dot"></span>Système opérationnel</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Credibility strip ─────────────────────────────────────────────
st.markdown("""
<div class="credibility-strip">
  <div class="cred-item"> <strong>Modèle clinique</strong> · Entraîné sur données validées</div>
  <div class="cred-item"> <strong>5 pathologies</strong> · Asthme, BPCO, Bronchite, Pneumonie, Sain</div>
  <div class="cred-item"> Ce diagnostic <strong>ne remplace pas</strong> un avis médical</div>
</div>
""", unsafe_allow_html=True)


# ── ÉTAPE 1 — Instructions patient ───────────────────────────────
st.markdown('<span class="section-label">Étape 1 — Préparation</span>', unsafe_allow_html=True)

if is_patient:
    st.markdown("""
    <div class="instruction-card">
      <div style="font-family:'Sora',sans-serif;font-size:0.92rem;font-weight:600;color:#EEF3FB;margin-bottom:1rem;">
        Comment préparer votre enregistrement ?
      </div>
      <div class="instruction-step">
        <div class="step-num">1</div>
        <div class="step-text">Placez-vous dans un endroit calme, sans bruit de fond.</div>
      </div>
      <div class="instruction-step">
        <div class="step-num">2</div>
        <div class="step-text">Tenez votre appareil à environ 10 cm de la bouche.</div>
      </div>
      <div class="instruction-step">
        <div class="step-num">3</div>
        <div class="step-text">Respirez normalement pendant au moins <strong>5 secondes</strong>, puis toussez si demandé.</div>
      </div>
      <div class="instruction-step" style="margin-bottom:0;">
        <div class="step-num">4</div>
        <div class="step-text">Chargez le fichier audio au format WAV ci-dessous.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="card" style="padding:1rem 1.4rem;">
      <div style="font-size:0.82rem;color:#8A9BBF;">
        Format supporté : <strong style="color:#C8D4E8;">WAV</strong> (mono ou stéréo, 16–44 100 Hz) · 
        Durée minimale recommandée : <strong style="color:#C8D4E8;">5 secondes</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── ÉTAPE 2 — Upload ──────────────────────────────────────────────
st.markdown('<span class="section-label">Étape 2 — Chargement du signal audio</span>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Déposez votre fichier audio respiratoire" if is_patient else "Charger un fichier audio (WAV)",
    type=["wav"],
    help="Format supporté : WAV mono ou stéréo, 16–44 100 Hz"
)


# ── Analyse principale ────────────────────────────────────────────
if uploaded:
    try:
        audio_bytes = uploaded.read()
        sr, y = read_wav(audio_bytes)
    except Exception as e:
        st.error(f"❌ Fichier audio invalide — {e}. Veuillez utiliser un fichier WAV valide.")
        st.stop()

    # Signal quality check
    quality_ok, quality_msg = check_signal_quality(y, sr)
    if not quality_ok:
        st.warning(f"⚠️ {quality_msg}")
        st.stop()

    # ── Visualisations techniques (toujours visible en Pro, collapsées en Patient) ──
    expander_label = "Analyse technique avancée — Signal & Spectrogramme"
    with st.expander(expander_label, expanded=not is_patient):
        col1, col2 = st.columns([1, 1], gap="large")

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

        with col2:
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
                x=alt.X('time:O', title='Temps (s)', axis=alt.Axis(labelOverlap=True, labelAngle=0, tickMinStep=5)),
                y=alt.Y('freq:O', title='Fréquence (Hz)', sort='descending',
                        axis=alt.Axis(labelOverlap=True)),
                color=alt.Color('db:Q', scale=alt.Scale(scheme='magma'), title='Intensité (dB)'),
                tooltip=[
                    alt.Tooltip('time:Q', title='Temps (s)'),
                    alt.Tooltip('freq:Q', title='Fréq (Hz)'),
                    alt.Tooltip('db:Q', title='dB')
                ]
            ).properties(title="Spectrogramme Audio", height=220)

            st.altair_chart(spec_chart, use_container_width=True)

        if not is_patient:
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

    # ── ÉTAPE 3 — Diagnostic ─────────────────────────────────────
    st.markdown('<span class="section-label">Étape 3 — Analyse par intelligence artificielle</span>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="margin-bottom:1rem;padding:1rem 1.4rem;">
      <div style="font-size:0.72rem;color:#5A7099;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">Fichier sélectionné</div>
      <div style="font-family:'DM Mono',monospace;color:#C8D4E8;font-size:0.85rem;">📄 {uploaded.name}</div>
    </div>
    """, unsafe_allow_html=True)

    analyze_btn = st.button(" Lancer l'analyse respiratoire", type="primary", use_container_width=True)

    if analyze_btn:

        # ── Progress feedback ──
        progress_bar  = st.progress(0)
        status_text   = st.empty()

        status_text.markdown('<div style="font-size:0.82rem;color:#5A7099;text-align:center;margin-top:0.5rem;">Analyse du signal audio…</div>', unsafe_allow_html=True)
        progress_bar.progress(15)

        try:
            session.sql("CREATE OR REPLACE TEMP TABLE tmp_audio (audio_data BINARY)").collect()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            session.sql(f"INSERT INTO tmp_audio SELECT TO_BINARY('{audio_b64}', 'BASE64')").collect()

            status_text.markdown('<div style="font-size:0.82rem;color:#5A7099;text-align:center;">Extraction des caractéristiques…</div>', unsafe_allow_html=True)
            progress_bar.progress(45)

            status_text.markdown('<div style="font-size:0.82rem;color:#5A7099;text-align:center;">Prédiction en cours…</div>', unsafe_allow_html=True)
            progress_bar.progress(75)

            result_raw = session.sql(
                "SELECT predict_from_mel(extract_mel_librosa(audio_data)) AS diagnostic FROM tmp_audio"
            ).collect()[0]['DIAGNOSTIC']

            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            result = json.loads(result_raw)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Une erreur est survenue lors de l'analyse. Veuillez réessayer. ({e})")
            st.stop()

        # ── Résultats ──
        predicted  = result['predicted_class']
        confidence = result['confidence']

        badge_cls, emoji_badge, severity_cls, diag_icon, reco_text, _ = BADGE_MAP.get(
            predicted, ('badge-green', '❓', 'severity-green', '❓', '—', 'Modérée')
        )
        conf_label, conf_cls = confidence_label(confidence)
        label_fr = CLASS_LABELS_FR.get(predicted, predicted.upper())

        # ─── PRIMARY: Diagnostic result block ────────────────────
        st.markdown(f"""
        <div class="diag-result-block {severity_cls}">
          <div class="diag-icon">{diag_icon}</div>
          <div class="diag-label-main">{label_fr}</div>
          <div>
            <span class="diag-confidence-human {conf_cls}">{conf_label} — {confidence*100:.0f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ─── SECONDARY: Recommendation ────────────────────────────
        if is_patient:
            st.markdown(f"""
            <div class="reco-block">
              <div>
                <div class="reco-title">Recommandation</div>
                <div class="reco-text">{reco_text}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # CTA buttons
            col_cta1, col_cta2 = st.columns(2)
            with col_cta1:
                st.button(" Prendre rendez-vous", use_container_width=True)
            with col_cta2:
                st.button(" Générer un compte rendu", use_container_width=True)

            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
              ⚠️ Ce résultat est fourni à titre indicatif. Il ne remplace pas un diagnostic médical établi par un professionnel de santé.
            </div>
            """, unsafe_allow_html=True)

        # ── Diagnostic secondaire ─────────────────────────────────
        second      = result.get('second_class', '')
        second_prob = result.get('second_prob', 0)
        if second and second_prob > 0.15:
            second_fr = CLASS_LABELS_FR_SHORT.get(second, second.upper())
            st.markdown(f"""
            <div style="padding:0.6rem 0.9rem;background:rgba(234,179,8,0.06);
                        border:1px solid rgba(234,179,8,0.18);border-radius:8px;
                        font-size:0.8rem;color:#FDE047;margin-top:0.75rem;">
              ⚠️ Diagnostic secondaire possible : <strong>{second_fr}</strong>
              &nbsp;<span style="color:#8A9BBF;">({second_prob*100:.0f}%)</span>
            </div>
            """, unsafe_allow_html=True)

        # ─── TERTIARY: Technical charts (always visible in Pro, collapsible in Patient) ──
        tech_expander_label = "Voir les détails techniques" if is_patient else "Détails de l'analyse"
        with st.expander(tech_expander_label, expanded=not is_patient):

            classes    = ['asthma', 'bronchial', 'copd', 'healthy', 'pneumonia']
            classes_fr = [CLASS_LABELS_FR_SHORT[c] for c in classes]
            probs      = [result.get(c, 0) for c in classes]

            df_probs = pd.DataFrame({
                'classe':      classes_fr,
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
            ).properties(height=170, title="Distribution des probabilités par pathologie")

            text_layer = bar.mark_text(
                align='left', dx=5, color='#8A9BBF', font='DM Mono', fontSize=10
            ).encode(text=alt.Text('probabilite:Q', format='.1%'))

            st.altair_chart(alt.layer(bar, text_layer), use_container_width=True)

            if not is_patient:
                st.markdown(f"""
                <div class="stat-row" style="margin-top:0.75rem;">
                  <div class="stat-cell">
                    <div class="val">{result.get('duration_sec', 0)}s</div>
                    <div class="lbl">Durée analysée</div>
                  </div>
                  <div class="stat-cell">
                    <div class="val">{result.get('sample_rate', 0)}</div>
                    <div class="lbl">Sample Rate</div>
                  </div>
                  <div class="stat-cell">
                    <div class="val">{result.get('file_size_bytes', 0)}</div>
                    <div class="lbl">Octets</div>
                  </div>
                  <div class="stat-cell">
                    <div class="val">{confidence*100:.1f}%</div>
                    <div class="lbl">Score confiance</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Snowflake logging ─────────────────────────────────────
        try:
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
              ✓ Résultat enregistré dans Snowflake
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass  # Non-blocking


if not is_patient:
    # ── Dashboard épidémiologique ─────────────────────────────────────
    st.divider()
    st.markdown('<span class="section-label">Surveillance épidémiologique · Réseau Tessan</span>', unsafe_allow_html=True)

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
        top_fr = CLASS_LABELS_FR_SHORT.get(top['PREDICTED_CLASS'], top['PREDICTED_CLASS'].upper())

        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.metric("Total diagnostics", f"{total:,}")
        with col_k2:
            st.metric("Pathologie dominante", top_fr)
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
            df_stats['PATHOLOGIE_FR'] = df_stats['PATHOLOGIE'].map(CLASS_LABELS_FR_SHORT)

            bar_epi = alt.Chart(df_stats).mark_bar(
                cornerRadiusTopRight=5, cornerRadiusBottomRight=5
            ).encode(
                x=alt.X('PATHOLOGIE_FR:N', title='', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('NB:Q', title='Diagnostics'),
                color=alt.Color('PATHOLOGIE_FR:N',
                                scale=alt.Scale(
                                    domain=['Asthme','Bronchite','BPCO','Sain','Pneumonie'],
                                    range=['#F59E0B','#6366F1','#F97316','#10B981','#EF4444']
                                ),
                                legend=None),
                tooltip=[
                    alt.Tooltip('PATHOLOGIE_FR:N', title='Pathologie'),
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
            df_recent['DIAGNOSTIC'] = df_recent['DIAGNOSTIC'].map(lambda x: CLASS_LABELS_FR_SHORT.get(x, x))
            df_recent['DIAG_2']     = df_recent['DIAG_2'].map(lambda x: CLASS_LABELS_FR_SHORT.get(x, x))
            st.dataframe(df_recent, use_container_width=True, height=255)