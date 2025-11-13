import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.manifold import TSNE
from matplotlib.patches import Patch

# Pagalbinės funkcijos
def load_numeric_csv(path):
    df = pd.read_csv(path, sep=';')
    features = [col for col in df.columns if col != 'label']
    X = df[features].apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    return X.values

def load_full_csv(path):
    df = pd.read_csv(path, sep=';')
    return df

def extreme_outlier_mask_iqr(X_std: np.ndarray, iqr_mult: float = 3.0):
    q1 = np.percentile(X_std, 25, axis=0)
    q3 = np.percentile(X_std, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr
    upper = q3 + iqr_mult * iqr
    mask = np.any((X_std < lower) | (X_std > upper), axis=1)
    return mask

def print_dendograma(Z, aibes_pavadinimas, failo_pavadinimas, color_threshold_ratio=0.7):
    plt.figure(figsize=(12, 7))
    max_d = max(Z[:, 2])
    color_threshold = max_d * color_threshold_ratio

    dendrogram(
        Z,
        color_threshold=color_threshold,
        above_threshold_color='gray',
    )

    plt.title(f"Dendograma (aibė = {aibes_pavadinimas}, metodas = {METHOD})")
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

def vizualizuoti_klasterius_sujungta(X_list, clusters_list, pavadinimai, failo_pavadinimas):
    _, axes = plt.subplots(1, len(X_list), figsize=(5 * len(X_list), 5))

    if len(X_list) == 1:
        axes = [axes]

    for i, (X, clusters, pavadinimas) in enumerate(zip(X_list, clusters_list, pavadinimai)):
        if X.shape[1] > 2:
            tsne = TSNE(n_components=2, perplexity=50, metric='canberra', random_state=42)
            X_2d = tsne.fit_transform(X)
        else:
            X_2d = X

        scatter = axes[i].scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='tab10', s=35)
        axes[i].set_title(f"Hierarchinis klasterizavimas\n({pavadinimas})")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Add legend with cluster labels
        unique_clusters = np.unique(clusters)
        legend_elements = [Patch(facecolor=scatter.cmap(scatter.norm(cluster)), 
                                 label=f'Klasteris {cluster}')
                          for cluster in unique_clusters]
        axes[i].legend(handles=legend_elements, loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_klasteriai, f"{failo_pavadinimas}.png"), dpi=300)
    plt.close()

def vizualizuoti_palyginima(X_2d, tiksliosios_klases, hierarchiniai_klasteriai, failo_pavadinimas):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    unique_classes = sorted(np.unique(tiksliosios_klases))
    n_classes = len(unique_classes)
    
    # First plot: True classes with legend
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=tiksliosios_klases, cmap='viridis', s=35, alpha=0.7)
    axes[0].set_title('t-SNE su tiksliosiomis klasėmis')
    legend_elements_classes = [Patch(facecolor=scatter1.cmap(scatter1.norm(cls)), 
                                     label=f'Klasė {cls}')
                               for cls in unique_classes]
    axes[0].legend(handles=legend_elements_classes, loc='best', fontsize=8)

    # Second plot: Hierarchical clusters with legend
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=hierarchiniai_klasteriai, cmap='tab10', s=35, alpha=0.7)
    axes[1].set_title('t-SNE su klasterių klasėmis')
    unique_clusters = np.unique(hierarchiniai_klasteriai)
    legend_elements_clusters = [Patch(facecolor=scatter2.cmap(scatter2.norm(cluster)), 
                                      label=f'Klasteris {cluster}')
                                for cluster in unique_clusters]
    axes[1].legend(handles=legend_elements_clusters, loc='best', fontsize=8)

    # Confusion map and accuracy calculation
    confusion_map = np.zeros((n_classes, len(np.unique(hierarchiniai_klasteriai))))
    for true_class in unique_classes:
        mask = tiksliosios_klases == true_class
        clusters_in_class = hierarchiniai_klasteriai[mask]
        for cluster in np.unique(hierarchiniai_klasteriai):
            confusion_map[int(true_class), cluster-1] = np.sum(clusters_in_class == cluster)

    correct = 0
    for true_class in unique_classes:
        mask = tiksliosios_klases == true_class
        clusters_in_class = hierarchiniai_klasteriai[mask]
        dominant_cluster = np.argmax(confusion_map[int(true_class), :]) + 1
        correct += np.sum(clusters_in_class == dominant_cluster)

    accuracy = (correct / len(tiksliosios_klases)) * 100
    print(f"\nPersidengimo tikslumas: {accuracy:.2f}%")

    match_colors = []
    for true_class, hier_cluster in zip(tiksliosios_klases, hierarchiniai_klasteriai):
        dominant_cluster = np.argmax(confusion_map[int(true_class), :]) + 1
        if hier_cluster == dominant_cluster:
            match_colors.append('gray')
        else:
            match_colors.append('red')

    axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=match_colors, s=35, alpha=0.8, edgecolors='black', linewidth=0.5)
    axes[2].set_title(f"Atitikimas pagal klasę\n{100-accuracy:.2f}% neatitinka")

    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', label='Atitinka'),
        Patch(facecolor='red', edgecolor='black', label='Neatitinka')
    ]
    axes[2].legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_klasteriai, f"{failo_pavadinimas}.png"), dpi=300)
    plt.close()

def spausdinti_neatitikimus(tiksliosios_klases, klasteriai):
    unique_classes = sorted(np.unique(tiksliosios_klases))
    neatitikimai_pagal_klase = {}

    for true_class in unique_classes:
        mask = tiksliosios_klases == true_class
        klasteriai_klaseje = klasteriai[mask]
        unique_clusters, counts = np.unique(klasteriai_klaseje, return_counts=True)
        dominant_cluster = unique_clusters[np.argmax(counts)]
        neatitinkantys = np.sum(klasteriai_klaseje != dominant_cluster)
        neatitikimai_pagal_klase[int(true_class)] = neatitinkantys

    print("\nNeatitinkančių objektų kiekis pagal klases:")
    for klase, kiekis in neatitikimai_pagal_klase.items():
        print(f"Neatitinkančių objektų kiekis {klase} klasei: {kiekis}")

    return neatitikimai_pagal_klase

# Duomenų įkėlimas
X_visi_pozymiai = load_numeric_csv('../pilna_EKG_pupsniu_analize_uzpildyta_medianomis_visi_normuota_pagal_minmax.csv')
X_atrinkta = load_numeric_csv('duomenys/atrinkta_aibe.csv')
X_2D = load_numeric_csv('duomenys/tsne_2d_data.csv')

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

# Dendogramos
print_dendograma(Z_visi_pozymiai, 'visi požymiai', 'visi_pozymiai')
print_dendograma(Z_atrinkta, 'atrinkti požymiai', 'atrinkti_pozymiai')
print_dendograma(Z_2D, 'sumažinta iki 2D (t-SNE)', '2D')

# Klasterizavimas
klasteriai_visi_rekom = atlikti_klasterizavima(Z_visi_pozymiai)
n_rekom_visi = len(set(klasteriai_visi_rekom))
klasteriai_visi_plus1 = atlikti_klasterizavima_su_n(Z_visi_pozymiai, n_rekom_visi + 1)
klasteriai_visi_8 = atlikti_klasterizavima_su_n(Z_visi_pozymiai, 8)

vizualizuoti_klasterius_sujungta(
    [X_visi_pozymiai, X_visi_pozymiai, X_visi_pozymiai],
    [klasteriai_visi_rekom, klasteriai_visi_plus1, klasteriai_visi_8],
    [f'visi požymiai ({n_rekom_visi} klasteriai)', 
     f'visi požymiai ({n_rekom_visi + 1} klasteriai)',
     'visi požymiai (8 klasteriai)'],
    'visi_pozymiai_sujungta'
)

klasteriai_atrinkta_rekom = atlikti_klasterizavima(Z_atrinkta)
n_rekom_atrinkta = len(set(klasteriai_atrinkta_rekom))
klasteriai_atrinkta_plus1 = atlikti_klasterizavima_su_n(Z_atrinkta, n_rekom_atrinkta + 1)
klasteriai_atrinkta_8 = atlikti_klasterizavima_su_n(Z_atrinkta, 8)

vizualizuoti_klasterius_sujungta(
    [X_atrinkta, X_atrinkta, X_atrinkta],
    [klasteriai_atrinkta_rekom, klasteriai_atrinkta_plus1, klasteriai_atrinkta_8],
    [f'atrinkti požymiai ({n_rekom_atrinkta} klasteriai)', 
     f'atrinkti požymiai ({n_rekom_atrinkta + 1} klasteriai)',
     'atrinkti požymiai (8 klasteriai)'],
    'atrinkti_pozymiai_sujungta'
)

klasteriai_2D_rekom = atlikti_klasterizavima(Z_2D)
n_rekom_2D = len(set(klasteriai_2D_rekom))
klasteriai_2D_plus1 = atlikti_klasterizavima_su_n(Z_2D, n_rekom_2D + 1)
klasteriai_2D_8 = atlikti_klasterizavima_su_n(Z_2D, 8)

vizualizuoti_klasterius_sujungta(
    [X_2D, X_2D, X_2D],
    [klasteriai_2D_rekom, klasteriai_2D_plus1, klasteriai_2D_8],
    [f't-SNE duomenys ({n_rekom_2D} klasteriai)', 
     f't-SNE duomenys ({n_rekom_2D + 1} klasteriai)',
     't-SNE duomenys (8 klasteriai)'],
    '2D_sujungta'
)

# Palyginimas
vizualizuoti_palyginima(X_2D, tiksliosios_klases, klasteriai_2D_rekom, f'palyginimas_tsne_vs_hierarchinis_{n_rekom_2D}k')
vizualizuoti_palyginima(X_2D, tiksliosios_klases, klasteriai_2D_plus1, f'palyginimas_tsne_vs_hierarchinis_{n_rekom_2D+1}k')
vizualizuoti_palyginima(X_2D, tiksliosios_klases, klasteriai_2D_8, 'palyginimas_tsne_vs_hierarchinis_8k')

spausdinti_neatitikimus(tiksliosios_klases, klasteriai_2D_rekom)
spausdinti_neatitikimus(tiksliosios_klases, klasteriai_2D_plus1)
spausdinti_neatitikimus(tiksliosios_klases, klasteriai_2D_8)

# 5. Išskirčių įtakos analizė naudojant IQR metodą
def analizuoti_isksirtis_iqr_ir_vizualizuoti(X, klasteriu_kiekis, pavadinimas, failo_pavadinimas, iqr_mult=3.0):
    print(f"\nAnalizuojama aibė: {pavadinimas}")
    print(f"Naudojamas IQR daugiklis: {iqr_mult}")
    
    # Detektuojame išskirčius naudojant IQR metodą
    outlier_mask = extreme_outlier_mask_iqr(X, iqr_mult=iqr_mult)
    n_outliers = np.sum(outlier_mask)
    print(f"Rasta {n_outliers} išskirčių iš {len(X)} ({(n_outliers/len(X))*100:.2f}%).")

    # Pašaliname išskirčius
    X_be_isksirciu = X[~outlier_mask]
    
    if len(X_be_isksirciu) < 2:
        print(f"ĮSPĖJIMAS: Po išskirčių pašalinimo liko per mažai duomenų ({len(X_be_isksirciu)})")
        return
    
    # Klasterizavimas be išskirčių
    Z_be_isksirciu = linkage(X_be_isksirciu, method=METHOD)
    klasteriai_po = atlikti_klasterizavima(Z_be_isksirciu)
    n_po = len(set(klasteriai_po))

    print(f"Be išskirčių – rekomenduojama {n_po} klasterių.")
    
    # Klasterizavimas su tuo pačiu klasterių skaičiumi kaip originale
    klasteriai_su = atlikti_klasterizavima_su_n(linkage(X, method=METHOD), len(set(klasteriu_kiekis)))
    klasteriai_be = atlikti_klasterizavima_su_n(Z_be_isksirciu, len(set(klasteriu_kiekis)))
    
    vizualizuoti_klasterius_sujungta(
        [X, X_be_isksirciu],
        [klasteriai_su, klasteriai_be],
        [f"Su išskirtimis ({len(set(klasteriai_su))} klasteriai, n={len(X)})",
         f"Be išskirčių ({len(set(klasteriai_be))} klasteriai, n={len(X_be_isksirciu)})"],
        failo_pavadinimas
    )
    print(f"Sukurtas grafikas: {failo_pavadinimas}.png")

# Analizė su IQR metodu (ekstremalūs išskirtys, iqr_mult=3.0)
print("\n" + "="*60)
print("IŠSKIRČIŲ ANALIZĖ SU IQR METODU (ekstremalūs išskirtys)")
print("="*60)

analizuoti_isksirtis_iqr_ir_vizualizuoti(
    X_visi_pozymiai, klasteriai_visi_rekom, 
    "Visi požymiai", "palyginimas_isksirtys_iqr_visi", iqr_mult=3.0
)

analizuoti_isksirtis_iqr_ir_vizualizuoti(
    X_atrinkta, klasteriai_atrinkta_rekom, 
    "Atrinkti požymiai", "palyginimas_isksirtys_iqr_atrinkta", iqr_mult=3.0
)

# PALYGINIMAS: t-SNE su išskirtimis vs be išskirčių
print("\n" + "="*60)
print("PALYGINIMAS: t-SNE su išskirtimis vs be išskirčių")
print("="*60)

# Įkeliame abiejų t-SNE variantų duomenis
X_2D_su_isskirtim = load_numeric_csv('duomenys/tsne_2d_data.csv')
X_2D_be_isskirciu = load_numeric_csv('duomenys/tsne_2d_data_be_isskirciu.csv')

# Hierarchinis klasterizavimas abiem
Z_2D_su = linkage(X_2D_su_isskirtim, method=METHOD)
Z_2D_be = linkage(X_2D_be_isskirciu, method=METHOD)

klasteriai_2D_su = atlikti_klasterizavima(Z_2D_su)
klasteriai_2D_be = atlikti_klasterizavima(Z_2D_be)

# Gautų klasterių kiekiai
n_2D_su = len(set(klasteriai_2D_su))
n_2D_be = len(set(klasteriai_2D_be))
print(f"t-SNE su išskirtimis: {n_2D_su} klasteriai")
print(f"t-SNE be išskirčių: {n_2D_be} klasteriai")

# Vizualus palyginimas
vizualizuoti_klasterius_sujungta(
    [X_2D_su_isskirtim, X_2D_be_isskirciu],
    [klasteriai_2D_su, klasteriai_2D_be],
    [f"t-SNE su išskirtimis ({n_2D_su} klasteriai, n={len(X_2D_su_isskirtim)})",
     f"t-SNE be išskirčių ({n_2D_be} klasteriai, n={len(X_2D_be_isskirciu)})"],
    "palyginimas_tsne_su_vs_be_isskirciu"
)

print("Sukurtas grafikas: grafikai/hierarchinis/palyginimas_tsne_su_vs_be_isskirciu.png")


# Dimensijų mažinimo įtaka
vizualizuoti_klasterius_sujungta(
    [X_atrinkta, X_2D],
    [klasteriai_atrinkta_rekom, klasteriai_2D_rekom],
    [f"Originalūs duomenys ({n_rekom_atrinkta} klasteriai)",
     f"t-SNE duomenys ({n_rekom_2D} klasteriai)"],
    "palyginimas_dimensiju_mazinimas"
)

print("\nSukurtas grafikas: dimensijų mažinimo įtaka (originalūs vs t-SNE duomenys)")
print("="*60)