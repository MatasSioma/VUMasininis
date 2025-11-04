import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score

# === Parametrai ===
N_COMPONENTS = 2  # PCA komponentų skaičius
N_NEIGHBORS = 15

# === Duomenų įkėlimas ===
df_nenormuota_su_atrinktais = pd.read_csv("EKG_pupsniu_analize_uzpildyta_medianomis.csv", sep=";")
df_normuota_su_atrinktais = pd.read_csv("pilna_EKG_pupsniu_analize_normuota_pagal_minmax.csv", sep=";")

df_nenormuota_su_visais = pd.read_csv("EKG_pupsniu_analize_uzpildyta_medianomis_visi.csv", sep=";")
df_normuota_su_visais = pd.read_csv("pilna_EKG_pupsniu_analize_uzpildyta_medianomis_visi_normuota_pagal_minmax.csv", sep=";")

columns_su_atrinktais = [col for col in df_normuota_su_atrinktais.columns if col.lower() != 'label']
columns_su_visais = [col for col in df_normuota_su_visais if col.lower() != 'label']

X_normuota_su_atrinktais = df_normuota_su_atrinktais[columns_su_atrinktais].values
Y_normuota_su_atrinktais = df_normuota_su_atrinktais['label'].values

X_nenormuota_su_atrinktais = df_nenormuota_su_atrinktais[columns_su_atrinktais].values
Y_nenormuota_su_atrinktais = df_nenormuota_su_atrinktais['label'].values

X_normuota_su_visais = df_normuota_su_visais[columns_su_visais].values
Y_normuota_su_visais = df_normuota_su_visais['label'].values

X_nenormuota_su_visais = df_nenormuota_su_visais[columns_su_visais].values
Y_nenormuota_su_visais = df_nenormuota_su_visais['label'].values

color_map = ["#71B1FA", "#87E775", "#E05C58"]

# === Išskirčių detekcija ===
def detect_outliers(df, columns):
    mild_outliers_set = set()
    extreme_outliers_set = set()

    for label in df['label'].unique():
        df_label = df[df['label'] == label]

        for col in columns:
            Q1 = df_label[col].quantile(0.25)
            Q3 = df_label[col].quantile(0.75)
            H = Q3 - Q1

            inner_lower = Q1 - 1.5 * H
            inner_upper = Q3 + 1.5 * H
            outer_lower = Q1 - 3 * H
            outer_upper = Q3 + 3 * H

            extreme_outliers = df_label[(df_label[col] < outer_lower) | (df_label[col] > outer_upper)]
            mild_outliers = df_label[((df_label[col] >= outer_lower) & (df_label[col] < inner_lower)) |
                                     ((df_label[col] > inner_upper) & (df_label[col] <= outer_upper))]

            extreme_outliers_set.update(extreme_outliers.index)
            mild_outliers_set.update(mild_outliers.index)

    return mild_outliers_set, extreme_outliers_set

# === Vizualizacija su PCA ===
def plot_pca(X, y, df, dataset_name, mild_outliers_set, extreme_outliers_set):
    colors = [color_map[int(c)] for c in y]

    print(f"\n=== {dataset_name} ===")
    print(f"Rasta mild išskirčių: {len(mild_outliers_set)}")
    print(f"Rasta extreme išskirčių: {len(extreme_outliers_set)}")

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca_result = pca.fit_transform(X)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"Paaiškinta dispersija PC1: {explained_variance[0]:.2%}")
    print(f"Paaiškinta dispersija PC2: {explained_variance[1]:.2%}")
    print(f"Bendra paaiškinta dispersija: {sum(explained_variance):.2%}")

    try:
        tw = trustworthiness(X, pca_result, n_neighbors=N_NEIGHBORS)
        print(f"Patikimumas: {tw:.4f}")
    except Exception as e:
        print("Klaida patikimumo skaičiavime:", e)

    try:
        sil = silhouette_score(pca_result, y)
        print(f"Kontūro įvertis: {sil:.4f}")
    except Exception as e:
        print("Klaida kontūro įverčio skaičiavime:", e)

    base_dir = 'grafikai/PCA'
    os.makedirs(base_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.title(f'PCA Dimensijos Mažinimas ({dataset_name})\n'
              f'PC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%}, '
              f'Bendra: {sum(explained_variance):.1%}')

    # Normalūs taškai
    normal_indices = [i for i in range(len(df)) if i not in mild_outliers_set and i not in extreme_outliers_set]
    plt.scatter(
        pca_result[normal_indices, 0],
        pca_result[normal_indices, 1],
        c=[colors[i] for i in normal_indices],
        s=50, alpha=0.8, linewidths=0
    )

    # Sąlyginės išskirtys – juodas storesnis kontūras
    mild_indices = list(mild_outliers_set)
    if mild_indices:
        plt.scatter(
            pca_result[mild_indices, 0],
            pca_result[mild_indices, 1],
            c=[colors[i] for i in mild_indices],
            s=50, alpha=0.9,
            edgecolors='black', linewidths=1.2,
            label='Sąlyginės išskirtys'
        )

    # Kritinės išskirtys – mėlynas kontūras (mediumBlue)
    extreme_indices = list(extreme_outliers_set)
    if extreme_indices:
        plt.scatter(
            pca_result[extreme_indices, 0],
            pca_result[extreme_indices, 1],
            c=[colors[i] for i in extreme_indices],
            s=50, alpha=0.9,
            edgecolors='mediumBlue', linewidths=1.2,
            label='Kritinės išskirtys'
        )

    # Legenda
    unique_labels = sorted(set(y))
    handles = [mpatches.Patch(color=color_map[i], label=f"Klasė {unique_labels[i]}") for i in range(len(unique_labels))]
    handles.append(mpatches.Patch(edgecolor='black', facecolor='white', label='Sąlyginės išskirtys'))
    handles.append(mpatches.Patch(edgecolor='mediumBlue', facecolor='white', label='Kritinės išskirtys'))

    plt.legend(handles=handles, title="Klasės / Išskirtys")
    plt.xlabel(f'PC1 ({explained_variance[0]:.1%})')
    plt.ylabel(f'PC2 ({explained_variance[1]:.1%})')
    plt.tight_layout()

    output_path = f"{base_dir}/{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Vizualizacija išsaugota: {output_path}")

# === Paleidimas ===
print("Apdorojama normuota duomenų aibė (atrinkti požymiai)...")
mild_normuota_su_atrinktais, extreme_normuota_su_atrinktais = detect_outliers(df_normuota_su_atrinktais, columns_su_atrinktais)
plot_pca(X_normuota_su_atrinktais, Y_normuota_su_atrinktais, df_normuota_su_atrinktais, "Normuota su atrinktais požymiais", mild_normuota_su_atrinktais, extreme_normuota_su_atrinktais)

print("\nApdorojama nenormuota duomenų aibė (atrinkti požymiai)...")
mild_nenormuota_su_atrinktais, extreme_nenormuota_su_atrinktais = detect_outliers(df_nenormuota_su_atrinktais, columns_su_atrinktais)
plot_pca(X_nenormuota_su_atrinktais, Y_nenormuota_su_atrinktais, df_nenormuota_su_atrinktais, "Nenormuota su atrinktais požymiais", mild_nenormuota_su_atrinktais, extreme_nenormuota_su_atrinktais)

print("\nApdorojama normuota duomenų aibė (visi požymiai)...")
mild_normuota_su_visais, extreme_normuota_su_visais = detect_outliers(df_normuota_su_visais, columns_su_visais)
plot_pca(X_normuota_su_visais, Y_normuota_su_visais, df_normuota_su_visais, "Normuota su visais požymiais", mild_normuota_su_visais, extreme_normuota_su_visais)

print("\nApdorojama nenormuota duomenų aibė (visi požymiai)...")
mild_nenormuota_su_visais, extreme_nenormuota_su_visais = detect_outliers(df_nenormuota_su_visais, columns_su_visais)
plot_pca(X_nenormuota_su_visais, Y_nenormuota_su_visais, df_nenormuota_su_visais, "Nenormuota su visais požymiais", mild_nenormuota_su_visais, extreme_nenormuota_su_visais)

print("\n✓ Visos keturios PCA vizualizacijos sėkmingai sukurtos ir išsaugotos")