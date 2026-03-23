#  Tessan Respiratory Diagnostic AI | Équipe 32

> **Transformer les 1 600+ cabines médicales Tessan en centres de diagnostic respiratoire primaires et intelligents grâce à l'IA.**

---

##  Project Overview

**Le stéthoscope de demain est connecté et intelligent.** Face à l'engorgement des cabinets médicaux, notre projet donne un "cerveau" analytique aux stéthoscopes des cabines Tessan. En captant simplement un son respiratoire (WAV), notre intelligence artificielle est capable de diagnostiquer en temps réel 5 états cliniques distincts. 
**Le problème :** L'accès limité aux médecins et l'incertitude lors de la téléconsultation. 
**La solution :** Une classification AI ultra-rapide hébergée sur Snowflake utilisant un modèle ResNet18 basé sur des spectrogrammes Mel.
**L'impact :** Une aide au diagnostic instantanée pour les médecins à distance, rassurant le patient et optimisant la prise en charge dans les déserts médicaux.

---

##  The Problem

En France, la désertification médicale est un enjeu critique. Les cabines médicales Tessan rapprochent la médecine des patients, mais lors d'une téléconsultation, le médecin ne peut s'appuyer que sur son écoute à distance via le stéthoscope connecté. 
- **L'incertitude :** Les bruits respiratoires peuvent être complexes à isoler et interpréter à distance, menant parfois à des diagnostics hésitants.
- **La perte de temps :** L'analyse manuelle des sons prend du temps sur des créneaux de consultation très courts.
- **Le besoin :** Un outil d'aide à la décision (Second Opinion) fiable et instantané pour classifier les anomalies respiratoires.

---

##  Our Solution

Nous avons conçu un pipeline complet transformant un fichier audio brut en un diagnostic clair, accessible via une interface web fluide. 
Au lieu de traiter l'audio comme une série temporelle complexe, nous l'abordons comme un problème de vision par ordinateur : nous convertissons les sons en représentations visuelles (**Spectrogrammes Mel**) et utilisons un réseau de neurones convolutif puissant (**ResNet18**) pour isoler les motifs distinctifs propres à chaque pathologie. Le tout est intégré via les capacités cloud de **Snowflake** pour une scalabilité immédiate sur tout le parc Tessan.

---

##  Tech Stack

* **Data & Machine Learning :** Python, PyTorch (ResNet18), Librosa (Audio Preprocessing), ONNX
* **Data Cloud & Backend :** Snowflake (User-Defined Functions - UDFs, Snowpark)
* **Frontend / UI :** Streamlit (Déployé sur Snowflake et en local)
* **Environnement Exploratoire :** Jupyter Notebooks, Pandas, NumPy

---

##  Features

*  **Prétraitement Audio Avancé** : Nettoyage, découpage et transformation des fichiers WAV bruts en spectrogrammes Mel haute fidélité (`spectres.npy`).
*  **Classification IA Multi-Classes** : Prédiction précise parmi 5 états de santé : *Sain, Asthme, BPCO (COPD), Bronchite, Pneumonie*.
*  **Intégration Cloud-Native** : Modèle exporté en ONNX et propulsé par les UDF Snowflake (`configuration_udf.md`) pour une inférence sécurisée et scalable.
*  **Dashboard Interactif** : Interface Streamlit permettant de charger un audio, de visualiser son signal spectral et d'obtenir la prédiction IA instantanément.

---

##  Data

**Sources :**
Le jeu de données comprend des enregistrements respiratoires (format `.wav`) classifiés médicalement. 
**Organisation :**
* Les données sont stratifiées en 5 classes cliniques.
* `data_original/` : Données brutes d'enregistrement.
* `data_updated/` : Données augmentées/nettoyées.
**Processing (Pipeline issu de `src/preprocessing.py`) :** 
Normalisation du volume, filtrage du bruit de fond, et conversion en images 2D (Spectrogrammes de Mel) pour alimenter le modèle de vision. 

---

##  Architecture

1. **Acquisition** : Le patient utilise le stéthoscope de la cabine Tessan.
2. **Ingestion** : Le fichier WAV est transmis à l'application.
3. **Prétraitement** : Conversion du son en Mel-Spectrogramme (via `src/utils/multispectredataset.py`).
4. **Inférence (Cerveau)** : Le modèle `resnet18_mel_finetuned.onnx` évalue la signature visuelle du son (en local ou via Snowflake UDF).
5. **Restitution** : Le médecin visualise instantanément les probabilités des pathologies sur l'interface Streamlit.

---

##  Team

**Équipe 32**

* [Paul](https://github.com/PaMilot) -
* [Pierre](https://github.com/PierreFournierInfo) - 
* [Tsiory](https://github.com/YRR0) - 

*(Pour consulter les détails de la répartition du travail technique, [document de suivi](https://docs.google.com/document/d/1EjFf9Z3SS0-IjFbZyLYTLG_0arUZyKefo2pR4YQaTz8/edit?usp=sharing)).*