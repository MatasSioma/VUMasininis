import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

PERPLEXITY = 30
LEARNING_RATE = 200
MAX_ITER = 1000

df_nenormuota = pd.read_csv("EKG_pupsniu_analize_uzpildyta_medianomis.csv", sep=";")
df_normuota = pd.read_csv("pilna_EKG_pupsniu_analize_normuota_pagal_minmax.csv", sep=";")

columns = [col for col in df_normuota.columns if col.lower() != 'label']

X_normuota = df_normuota[columns].values
y_normuota = df_normuota['label'].values

X_nenormuota = df_nenormuota[columns].values
y_nenormuota = df_nenormuota['label'].values

color_map = ["#71B1FA", "#87E775", "#E05C58"]

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

    base_dir = 'grafikai/tSNE'
    os.makedirs(base_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.title(f't-SNE Dimensijos Mažinimas ({dataset_name})\nperplexity={PERPLEXITY}, lr={LEARNING_RATE}')

    normal_indices = [i for i in range(len(df)) if i not in mild_outliers_set and i not in extreme_outliers_set]
    plt.scatter(
        tsne[normal_indices, 0],
        tsne[normal_indices, 1],
        c=[colors[i] for i in normal_indices],
        s=50, alpha=0.8, linewidths=0
    )

    mild_indices = list(mild_outliers_set)
    if mild_indices:
        plt.scatter(
            tsne[mild_indices, 0],
            tsne[mild_indices, 1],
            c=[colors[i] for i in mild_indices],
            s=50, alpha=0.9,
            edgecolors='black', linewidths=0.6,
            label='Sąlyginės išskirtys'
        )

    extreme_indices = list(extreme_outliers_set)
    if extreme_indices:
        plt.scatter(
            tsne[extreme_indices, 0],
            tsne[extreme_indices, 1],
            c=[colors[i] for i in extreme_indices],
            s=50, alpha=0.9,
            edgecolors='#00FFFF', linewidths=0.6,
            label='Nesąlyginės išskirtys'
        )

    unique_labels = sorted(set(y))
    handles = [mpatches.Patch(color=color_map[i], label=f"Klasė {unique_labels[i]}") for i in range(len(unique_labels))]
    handles.append(mpatches.Patch(edgecolor='black', facecolor='white', label='Sąlyginės išskirtys'))
    handles.append(mpatches.Patch(edgecolor='#00FFFF', facecolor='white', label='Nesąlyginės išskirtys'))

    plt.legend(handles=handles, title="Klasės / Išskirtys")
    plt.tight_layout()

    output_path = f"{base_dir}/{dataset_name.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Vizualizacija išsaugota: {output_path}")

print("Apdorojama normuota duomenų aibė...")
mild_normuota, extreme_normuota = detect_outliers(df_normuota, columns)
plot_tsne(X_normuota, y_normuota, df_normuota, "Normuota", mild_normuota, extreme_normuota)

print("\nApdorojama nenormuota duomenų aibė...")
mild_nenormuota, extreme_nenormuota = detect_outliers(df_nenormuota, columns)
plot_tsne(X_nenormuota, y_nenormuota, df_nenormuota, "Nenormuota", mild_nenormuota, extreme_nenormuota)

print("\n✓ Abi t-SNE vizualizacijos sėkmingai sukurtos ir išsaugotos")