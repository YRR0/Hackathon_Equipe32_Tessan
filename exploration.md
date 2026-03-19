-- Créer un stage interne pour les fichiers audio
CREATE OR REPLACE STAGE audio_stage;

-- Uploader les fichiers WAV depuis ton terminal
PUT file:///chemin/data/*.wav @audio_stage AUTO_COMPRESS=FALSE;

-- Créer la table de métadonnées
CREATE OR REPLACE TABLE audio_metadata (
    file_name     VARCHAR,
    label         VARCHAR,        -- 'Asthma', 'COPD', 'Bronchial', 'Pneumonia', 'Healthy'
    duration_sec  FLOAT,
    sample_rate   INT,
    file_path     VARCHAR         -- chemin relatif dans le stage
);