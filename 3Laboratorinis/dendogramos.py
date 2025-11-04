import os
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def load_numeric_csv(path):
    df = pd.read_csv(path, sep=';')
    features = [col for col in df.columns if col != 'label']
    X = df[features].apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    return X.values

X_visi_pozymiai = load_numeric_csv('../pilna_EKG_pupsniu_analize_uzpildyta_medianomis_visi_normuota_pagal_minmax.csv')
X_atrinkta = load_numeric_csv('duomenys/atrinkta_aibe.csv')
X_2D = load_numeric_csv('duomenys/tsne_2d_data.csv')

METHOD = 'ward'
Z_visi_pozymiai = linkage(X_visi_pozymiai, method=METHOD)
Z_atrinkta = linkage(X_atrinkta, method=METHOD)
Z_2D = linkage(X_2D, method=METHOD)

base_dir = 'grafikai/dendrogramos'
os.makedirs(base_dir, exist_ok=True)

def print_dendrograma(Z, aibes_pavadinimas, failo_pavadinimas, color_threshold_ratio=0.7):
    plt.figure(figsize=(12, 7))
    max_d = max(Z[:, 2])
    color_threshold = max_d * color_threshold_ratio

    dendrogram(
        Z,
        color_threshold=color_threshold,  # spalvų atskyrimo riba
        above_threshold_color='gray',     # virš ribos – pilka
    )

    plt.title(f"Dendrograma (aibė = {aibes_pavadinimas}, metodas = {METHOD})")
    plt.xlabel("Duomenų taškai")
    plt.ylabel("Euklidinis atstumas")

    plt.axhline(y=color_threshold, c='red', lw=1.5, linestyle='--', label='Spalvų riba')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"{failo_pavadinimas}.png"), dpi=300)
    plt.close()

print_dendrograma(Z_visi_pozymiai, 'visi požymiai', 'visi_pozymiai')
print_dendrograma(Z_atrinkta, 'atrinkti požymiai', 'atrinkti_pozymiai')
print_dendrograma(Z_2D, 'sumažinta iki 2D (t-SNE)', '2D')

print(f"✓ Visos dendrogramos išsaugotos į {base_dir} aplankalą")
