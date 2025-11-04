import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score

# === Parametrai ===
PERPLEXITY = 50
LEARNING_RATE = 200
MAX_ITER = 1000
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

# === Vizualizacija su t-SNE ===
def plot_tsne(X, y, df, dataset_name, mild_outliers_set, extreme_outliers_set):
    colors = [color_map[int(c)] for c in y]

    print(f"\n=== {dataset_name} ===")
    print(f"Rasta mild išskirčių: {len(mild_outliers_set)}")
    print(f"Rasta extreme išskirčių: {len(extreme_outliers_set)}")

    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        init='pca',
        verbose=1,
        random_state=42
    ).fit_transform(X)

    # === Patikimumas ir kontūro įvertis ===
    tw, sil = None, None
    try:
        tw = trustworthiness(X, tsne, n_neighbors=N_NEIGHBORS)
        print(f"Patikimumas: {tw:.4f}")
    except Exception as e:
        print("Klaida patikimumo skaičiavime:", e)

    try:
        sil = silhouette_score(tsne, y)
        print(f"Kontūro įvertis: {sil:.4f}")
    except Exception as e:
        print("Klaida kontūro įverčio skaičiavime:", e)

    # === Vizualizacija ===
    base_dir = 'grafikai/tSNE'
    os.makedirs(base_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))

    # Antraštė su įverčiais
    subtitle = f"perplexity={PERPLEXITY}, lr={LEARNING_RATE}"
    if tw is not None and sil is not None:
        subtitle += f"\nPatikimumas = {tw:.4f} | Kontūro įvertis = {sil:.4f}"
    elif tw is not None:
        subtitle += f"\nPatikimumas = {tw:.4f}"
    elif sil is not None:
        subtitle += f"\nKontūro įvertis = {sil:.4f}"

    plt.title(f't-SNE Dimensijos Mažinimas ({dataset_name})\n{subtitle}')

    # Normalūs taškai
    normal_indices = [i for i in range(len(df)) if i not in mild_outliers_set and i not in extreme_outliers_set]
    plt.scatter(
        tsne[normal_indices, 0],
        tsne[normal_indices, 1],
        c=[colors[i] for i in normal_indices],
        s=50, alpha=0.8, linewidths=0
    )

    # Sąlyginės išskirtys – juodas kontūras
    mild_indices = list(mild_outliers_set)
    if mild_indices:
        plt.scatter(
            tsne[mild_indices, 0],
            tsne[mild_indices, 1],
            c=[colors[i] for i in mild_indices],
            s=50, alpha=0.9,
            edgecolors='black', linewidths=1.2,
            label='Sąlyginės išskirtys'
        )

    # Kritinės išskirtys – mėlynas kontūras
    extreme_indices = list(extreme_outliers_set)
    if extreme_indices:
        plt.scatter(
            tsne[extreme_indices, 0],
            tsne[extreme_indices, 1],
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
    plt.tight_layout()

    output_path = f"{base_dir}/{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Vizualizacija išsaugota: {output_path}")


# === Paleidimas ===
print("Apdorojama normuota duomenų aibė (atrinkti požymiai)...")
mild_normuota_su_atrinktais, extreme_normuota_su_atrinktais = detect_outliers(df_normuota_su_atrinktais, columns_su_atrinktais)
plot_tsne(X_normuota_su_atrinktais, Y_normuota_su_atrinktais, df_normuota_su_atrinktais, "Normuota su atrinktais požymiais", mild_normuota_su_atrinktais, extreme_normuota_su_atrinktais)

print("\nApdorojama nenormuota duomenų aibė (atrinkti požymiai)...")
mild_nenormuota_su_atrinktais, extreme_nenormuota_su_atrinktais = detect_outliers(df_nenormuota_su_atrinktais, columns_su_atrinktais)
plot_tsne(X_nenormuota_su_atrinktais, Y_nenormuota_su_atrinktais, df_nenormuota_su_atrinktais, "Nenormuota su atrinktais požymiais", mild_nenormuota_su_atrinktais, extreme_nenormuota_su_atrinktais)

print("\nApdorojama normuota duomenų aibė (visi požymiai)...")
mild_normuota_su_visais, extreme_normuota_su_visais = detect_outliers(df_normuota_su_visais, columns_su_visais)
plot_tsne(X_normuota_su_visais, Y_normuota_su_visais, df_normuota_su_visais, "Normuota su visais požymiais", mild_normuota_su_visais, extreme_normuota_su_visais)

print("\nApdorojama nenormuota duomenų aibė (visi požymiai)...")
mild_nenormuota_su_visais, extreme_nenormuota_su_visais = detect_outliers(df_nenormuota_su_visais, columns_su_visais)
plot_tsne(X_nenormuota_su_visais, Y_nenormuota_su_visais, df_nenormuota_su_visais, "Nenormuota su visais požymiais", mild_nenormuota_su_visais, extreme_nenormuota_su_visais)

print("\n✓ Visos keturios t-SNE vizualizacijos sėkmingai sukurtos ir išsaugotos")
