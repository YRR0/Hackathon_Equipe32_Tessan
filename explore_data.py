import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def explore_dataset(data_path):
    print("=== Phase 1: Exploration du dataset ===")
    
    classes = ['asthma', 'Bronchial', 'copd', 'healthy', 'pneumonia']
    file_info = []
    
    print("Parcours des répertoires...")
    for class_name in classes:
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        print(f"Classe {class_name} : {len(files)} fichiers trouvés.")
        
        # On va analyser les premiers fichiers pour récupérer des stats rapides
        # (on ne fait pas tous pour que le script soit rapide au début)
        for f in files:
            file_path = os.path.join(class_dir, f)
            file_info.append({
                'class': class_name,
                'filename': f,
                'path': file_path
            })

    df = pd.DataFrame(file_info)
    print("\n--- Distribution des classes ---")
    print(df['class'].value_counts())
    
    return df

def analyze_audio_properties(df, sample_size_per_class=5):
    print("\n=== Phase 2: Analyse des propriétés audio (sur un échantillon) ===")
    
    stats = []
    
    grouped = df.groupby('class')
    for name, group in grouped:
        sample = group.head(sample_size_per_class)
        for _, row in sample.iterrows():
            try:
                # sr=None pour garder le sample rate d'origine
                y, sr = librosa.load(row['path'], sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                stats.append({
                    'class': name,
                    'sr': sr,
                    'duration_sec': duration
                })
            except Exception as e:
                pass
                
    stats_df = pd.DataFrame(stats)
    print("\nStatistiques d'échantillonnage (SR) :")
    print(stats_df['sr'].value_counts())
    
    print("\nStatistiques de durée (sec) :")
    print(stats_df.groupby('class')['duration_sec'].describe().apply(lambda x: round(x, 2)))

def plot_spectrograms(df):
    print("\n=== Phase 3: Visualisation d'un WAV par classe ===")
    classes = df['class'].unique()
    
    plt.figure(figsize=(20, 10))
    for i, cls in enumerate(classes):
        # Prendre le premier fichier de chaque classe
        sample_file = df[df['class'] == cls].iloc[0]['path']
        y, sr = librosa.load(sample_file, sr=22050) # Standardisation temporaire pour l'affichage
        
        # Onde
        plt.subplot(len(classes), 2, i*2 + 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f'Onde ({cls})')
        plt.xlabel('Temps (s)')
        
        # Spectrogramme Mel
        plt.subplot(len(classes), 2, i*2 + 2)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogramme ({cls})')

    plt.tight_layout()
    plt.savefig('comparaison_spectrogrammes.png')
    print("Graphique sauvegardé sous 'comparaison_spectrogrammes.png'")

if __name__ == "__main__":
    DATA_PATH = "data_set/"
    df = explore_dataset(DATA_PATH)
    analyze_audio_properties(df)
    plot_spectrograms(df)
    print("\nExploration terminée avec succès !")
    
