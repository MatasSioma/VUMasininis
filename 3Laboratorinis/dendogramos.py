import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA


def load_numeric_csv(path):
    df = pd.read_csv(path, sep=';')
    features = [col for col in df.columns if col != 'label']
    X = df[features].apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    return X.values

def load_full_csv(path):
    """Įkelia CSV su label stulpeliu"""
    df = pd.read_csv(path, sep=';')
    return df

def print_dendrograma(Z, aibes_pavadinimas, failo_pavadinimas, color_threshold_ratio=0.7):
    plt.figure(figsize=(12, 7))
    max_d = max(Z[:, 2])
    color_threshold = max_d * color_threshold_ratio

    dendrogram(
        Z,
        color_threshold=color_threshold,
        above_threshold_color='gray',
    )

    plt.title(f"Dendrograma (aibė = {aibes_pavadinimas}, metodas = {METHOD})")
    plt.xlabel("Duomenų taškai")
    plt.ylabel("Euklidinis atstumas")
    plt.axhline(y=color_threshold, c='red', lw=1.5, linestyle='--', label='Spalvų riba')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_dendograma, f"{failo_pavadinimas}.png"), dpi=300)
    plt.close()

def atlikti_klasterizavima(Z, max_ratio=0.7):
    max_d = max(Z[:, 2])
    color_threshold = max_d * max_ratio
    clusters = fcluster(Z, t=color_threshold, criterion='distance')
    return clusters

def atlikti_klasterizavima_su_n(Z, n_clusters):
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
    return clusters

def vizualizuoti_klasterius(X, clusters, pavadinimas, failo_pavadinimas):
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='tab10', s=35)
    plt.title(f"Hierarchinis klasterizavimas ({pavadinimas})")
    plt.xlabel("1 dimensija")
    plt.ylabel("2 dimensija")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_klasteriai, f"{failo_pavadinimas}.png"), dpi=300)
    plt.close()

def vizualizuoti_klasterius_sujungta(X_list, clusters_list, pavadinimai, failo_pavadinimas):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    for i, (X, clusters, pavadinimas) in enumerate(zip(X_list, clusters_list, pavadinimai)):
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X
        
        axes[i].scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='tab10', s=35)
        axes[i].set_title(f"Hierarchinis klasterizavimas\n({pavadinimas})")
        axes[i].set_xlabel("1 dimensija")
        axes[i].set_ylabel("2 dimensija")
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_klasteriai, f"{failo_pavadinimas}.png"), dpi=300)
    plt.close()

def vizualizuoti_palyginima(X_2d, tiksliosios_klases, hierarchiniai_klasteriai, failo_pavadinimas):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # 1. t-SNE su tiksliosiomis klasėmis
    unique_classes = sorted(np.unique(tiksliosios_klases))
    n_classes = len(unique_classes)
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=tiksliosios_klases, cmap='viridis', s=35, alpha=0.7)
    axes[0].set_title('t-SNE su tiksliosiomis klasėmis')
    axes[0].set_xlabel('Dimensija 1')
    axes[0].set_ylabel('Dimensija 2')

    # 2. Hierarchinio klasterizavimo rezultatas
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=hierarchiniai_klasteriai, cmap='tab10', s=35, alpha=0.7)
    axes[1].set_title('t-SNE su klasterių klasėmis')
    axes[1].set_xlabel('Dimensija 1')
    axes[1].set_ylabel('Dimensija 2')

    # 3. Persidengimo analizė
    # Skaičiuojame, kiek kiekvienos tiksliosios klasės taškų patenka į kiekvieną klasterį
    confusion_map = np.zeros((n_classes, len(np.unique(hierarchiniai_klasteriai))))
    for true_class in unique_classes:
        mask = tiksliosios_klases == true_class
        clusters_in_class = hierarchiniai_klasteriai[mask]
        for cluster in np.unique(hierarchiniai_klasteriai):
            confusion_map[int(true_class), cluster-1] = np.sum(clusters_in_class == cluster)
    
    # Sukuriame spalvų kodą pagal dominuojantį klasterį
    color_codes = np.zeros(len(tiksliosios_klases))
    for i, (true_class, hier_cluster) in enumerate(zip(tiksliosios_klases, hierarchiniai_klasteriai)):
        # Spalva = (tikroji_klasė * 10 + hierarchinis_klasteris)
        color_codes[i] = true_class * 10 + hier_cluster

    # Apskaičiuojame tikslumą
    correct = 0
    for true_class in unique_classes:
        mask = tiksliosios_klases == true_class
        clusters_in_class = hierarchiniai_klasteriai[mask]
        dominant_cluster = np.argmax(confusion_map[int(true_class), :]) + 1
        correct += np.sum(clusters_in_class == dominant_cluster)
    
    accuracy = (correct / len(tiksliosios_klases)) * 100
    print(f"\nPersidengimo tiksumas: {accuracy:.2f}%")
    print(f"Nesutampa: {100-accuracy:.2f}%")
    
    # Pažymime taškus pagal sutapimą/nesutapimą
    match_colors = []
    for true_class, hier_cluster in zip(tiksliosios_klases, hierarchiniai_klasteriai):
        # Randame, kuris klasteris dominuoja šiai klasei
        dominant_cluster = np.argmax(confusion_map[int(true_class), :]) + 1
        if hier_cluster == dominant_cluster:
            match_colors.append('gray')  # Sutampa
        else:
            match_colors.append('red')  # Nesutampa
    
    axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=match_colors, s=35, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[2].set_title(f"t-SNE: atitikimas pagal klasės\n{accuracy:.2f}% neatitinka")
    axes[2].set_xlabel('Dimensija 1')
    axes[2].set_ylabel('Dimensija 2')
    
    # Pridedame legendą
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', label='Atitinka'),
        Patch(facecolor='red', edgecolor='black', label='Neatitinka')
    ]
    axes[2].legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_klasteriai, f"{failo_pavadinimas}.png"), dpi=300)
    plt.close()

X_visi_pozymiai = load_numeric_csv('../pilna_EKG_pupsniu_analize_uzpildyta_medianomis_visi_normuota_pagal_minmax.csv')
X_atrinkta = load_numeric_csv('duomenys/atrinkta_aibe.csv')
X_2D = load_numeric_csv('duomenys/tsne_2d_data.csv')

# Įkeliame tsne_2d_data su label stulpeliu palyginimui
df_tsne = load_full_csv('duomenys/tsne_2d_data.csv')
tiksliosios_klases = df_tsne['label'].values

METHOD = 'ward'
Z_visi_pozymiai = linkage(X_visi_pozymiai, method=METHOD)
Z_atrinkta = linkage(X_atrinkta, method=METHOD)
Z_2D = linkage(X_2D, method=METHOD)


base_dir_dendograma = 'grafikai/dendogramos'
base_dir_klasteriai = 'grafikai/hierarchinis'
os.makedirs(base_dir_dendograma, exist_ok=True)
os.makedirs(base_dir_klasteriai, exist_ok=True)


print_dendrograma(Z_visi_pozymiai, 'visi požymiai', 'visi_pozymiai')
print_dendrograma(Z_atrinkta, 'atrinkti požymiai', 'atrinkti_pozymiai')
print_dendrograma(Z_2D, 'sumažinta iki 2D (t-SNE)', '2D')


# Visi požymiai - 3 variantai
klasteriai_visi_rekom = atlikti_klasterizavima(Z_visi_pozymiai)
n_rekom_visi = len(set(klasteriai_visi_rekom))
klasteriai_visi_plus1 = atlikti_klasterizavima_su_n(Z_visi_pozymiai, n_rekom_visi + 1)
klasteriai_visi_8 = atlikti_klasterizavima_su_n(Z_visi_pozymiai, 8)

print(f"\nVisi požymiai:")
print(f"  - Rekomenduojama: {n_rekom_visi} klasteriai")
print(f"  - +1 variantas: {n_rekom_visi + 1} klasteriai")
print(f"  - Fiksuota: 8 klasteriai")

# Sujungti grafikai - visi požymiai
vizualizuoti_klasterius_sujungta(
    [X_visi_pozymiai, X_visi_pozymiai, X_visi_pozymiai],
    [klasteriai_visi_rekom, klasteriai_visi_plus1, klasteriai_visi_8],
    [f'visi požymiai ({n_rekom_visi} klasteriai)', 
     f'visi požymiai ({n_rekom_visi + 1} klasteriai)',
     'visi požymiai (8 klasteriai)'],
    'visi_pozymiai_sujungta'
)


# Atrinkti požymiai - 3 variantai
klasteriai_atrinkta_rekom = atlikti_klasterizavima(Z_atrinkta)
n_rekom_atrinkta = len(set(klasteriai_atrinkta_rekom))
klasteriai_atrinkta_plus1 = atlikti_klasterizavima_su_n(Z_atrinkta, n_rekom_atrinkta + 1)
klasteriai_atrinkta_8 = atlikti_klasterizavima_su_n(Z_atrinkta, 8)

print(f"\nAtrinkti požymiai:")
print(f"  - Rekomenduojama: {n_rekom_atrinkta} klasteriai")
print(f"  - +1 variantas: {n_rekom_atrinkta + 1} klasteriai")
print(f"  - Fiksuota: 8 klasteriai")

# Sujungti grafikai - atrinkti požymiai
vizualizuoti_klasterius_sujungta(
    [X_atrinkta, X_atrinkta, X_atrinkta],
    [klasteriai_atrinkta_rekom, klasteriai_atrinkta_plus1, klasteriai_atrinkta_8],
    [f'atrinkti požymiai ({n_rekom_atrinkta} klasteriai)', 
     f'atrinkti požymiai ({n_rekom_atrinkta + 1} klasteriai)',
     'atrinkti požymiai (8 klasteriai)'],
    'atrinkti_pozymiai_sujungta'
)


# 2D (t-SNE) - 3 variantai
klasteriai_2D_rekom = atlikti_klasterizavima(Z_2D)
n_rekom_2D = len(set(klasteriai_2D_rekom))
klasteriai_2D_plus1 = atlikti_klasterizavima_su_n(Z_2D, n_rekom_2D + 1)
klasteriai_2D_8 = atlikti_klasterizavima_su_n(Z_2D, 8)

print(f"\n2D (t-SNE):")
print(f"  - Rekomenduojama: {n_rekom_2D} klasteriai")
print(f"  - +1 variantas: {n_rekom_2D + 1} klasteriai")
print(f"  - Fiksuota: 8 klasteriai")

# Sujungti grafikai - 2D
vizualizuoti_klasterius_sujungta(
    [X_2D, X_2D, X_2D],
    [klasteriai_2D_rekom, klasteriai_2D_plus1, klasteriai_2D_8],
    [f't-SNE duomenys ({n_rekom_2D} klasteriai)', 
     f't-SNE duomenys ({n_rekom_2D + 1} klasteriai)',
     't-SNE duomenys (8 klasteriai)'],
    '2D_sujungta'
)

# NAUJAS PALYGINIMAS: t-SNE tiksliosios klasės vs hierarchiniai klasteriai
print("\n" + "="*60)
print("Kuriam palyginimo grafikus...")
print("="*60)

vizualizuoti_palyginima(X_2D, tiksliosios_klases, klasteriai_2D_rekom, 
                        f'palyginimas_tsne_vs_hierarchinis_{n_rekom_2D}k')
vizualizuoti_palyginima(X_2D, tiksliosios_klases, klasteriai_2D_plus1, 
                        f'palyginimas_tsne_vs_hierarchinis_{n_rekom_2D+1}k')
vizualizuoti_palyginima(X_2D, tiksliosios_klases, klasteriai_2D_8, 
                        'palyginimas_tsne_vs_hierarchinis_8k')


print("\n" + "="*60)
print("✓ Dendrogramos išsaugotos į", base_dir_dendograma)
print("✓ Klasterių vizualizacijos išsaugotos į", base_dir_klasteriai)
print(f"✓ Sukurti 3 sujungti horizontalūs grafikai (po 3 subgrafikius kiekviename)")
print(f"✓ Sukurti 3 palyginimo grafikai (t-SNE vs hierarchinis klasterizavimas)")
print("="*60)