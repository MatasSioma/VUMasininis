import umap
import matplotlib.patches as mpatches
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

N_NEIGHBORS = 15
MIN_DIST = 0.3

filenames = [
    "EKG_pupsniu_analize_uzpildyta_medianomis.csv", 
    "pilna_EKG_pupsniu_analize_normuota_pagal_minmax.csv",
    "EKG_pupsniu_analize_uzpildyta_medianomis_visi.csv",
    "pilna_EKG_pupsniu_analize_uzpildyta_medianomis_visi_normuota_pagal_minmax.csv"
]

for filename in filenames:
    # Duomenų įkėlimas
    df = pd.read_csv(filename, sep=";")

    # Požymių ir klasių atskyrimas
    X = df[["Q_val", "R_val", "S_val", "RR_l_0/RR_l_1", "signal_std", "seq_size"]]
    y = df["label"]

    # LDA pritaikymas
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)

    # Rezultatai sudedami į DataFrame
    lda_df = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
    lda_df['Class'] = y

    os.makedirs('grafikai/LDA', exist_ok=True)

    # Outlier'iu identifikavimas pagal IQR metodą kiekvienai klasei

    lda_df['Outlier'] = 'None'

    for c in lda_df['Class'].unique():
        class_data = lda_df[lda_df['Class'] == c]

        mild_outliers_set = set()
        extreme_outliers_set = set()

        for col in ['LD1', 'LD2']:
            Q1 = class_data[col].quantile(0.25)
            Q3 = class_data[col].quantile(0.75)
            H = Q3 - Q1

            inner_lower = Q1 - 1.5 * H
            inner_upper = Q3 + 1.5 * H

            outer_lower = Q1 - 3 * H
            outer_upper = Q3 + 3 * H

            extreme_outliers = class_data[(class_data[col] < outer_lower) | (class_data[col] > outer_upper)]
            mild_outliers = class_data[((class_data[col] >= outer_lower) & (class_data[col] < inner_lower)) |
                                       ((class_data[col] > inner_upper) & (class_data[col] <= outer_upper))]

            extreme_outliers_set.update(extreme_outliers.index)
            mild_outliers_set.update(mild_outliers.index)

        # Spausidnama kiek kokių išskirčių buvo kiekvienoj klasėj
        print(f"Class {c}: Mild outliers = {len(mild_outliers_set)}, Extreme outliers = {len(extreme_outliers_set)}")

        lda_df.loc[list(mild_outliers_set), 'Outlier'] = 'Mild'
        lda_df.loc[list(extreme_outliers_set), 'Outlier'] = 'Extreme'

    # Vizualizacija
    plt.figure(figsize=(10, 6))

    color_map = ["#71B1FA", "#87E775", "#E05C58"]
    colors = [color_map[int(c)] for c in y]

    palette = color_map
    class_order = sorted(lda_df['Class'].unique())
    palette_dict = dict(zip(class_order, palette))

    # Pagrindiniai taškai
    sns.scatterplot(
        data=lda_df[lda_df['Outlier']=='None'],
        x='LD1', 
        y='LD2', 
        hue='Class', 
        palette=palette_dict, 
        s=50, 
        edgecolor=None, 
        alpha = 0.9
    )

    # Mild outlieriu vizualizacija
    for cls in class_order:
        subset = lda_df[(lda_df['Class'] == cls) & (lda_df['Outlier'] == 'Mild')]
        plt.scatter(
            subset['LD1'], 
            subset['LD2'], 
            s=50, 
            facecolors=palette_dict[cls], 
            edgecolors='black', 
            alpha = 0.9, 
            linewidths=1.5, 
            label=None
        )

    # Extreme outlieriu vizualizacija
    for cls in class_order:
        subset = lda_df[(lda_df['Class'] == cls) & (lda_df['Outlier'] == 'Extreme')]
        plt.scatter(
            subset['LD1'],
            subset['LD2'],
            s=50, facecolors=palette_dict[cls],
            edgecolors='mediumBlue', 
            alpha = 0.9,
            linewidths=1.5, 
            label=None
        )

    plt.title('LDA vizualizacija')

    # Sukuriama legenda
    unique_labels = sorted(lda_df['Class'].unique())
    handles = [mpatches.Patch(color=palette_dict[label], label=f"Klasė {label}") for label in unique_labels]
    handles.append(mpatches.Patch(edgecolor='black', facecolor='white', label='Sąlyginės išskirtys'))
    handles.append(mpatches.Patch(edgecolor='mediumBlue', facecolor='white', label='Kritinės išskirtys'))
    plt.legend(handles=handles, title="Klasės / Išskirtys", loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.xlabel('')
    plt.ylabel('')
    plt.savefig("grafikai/LDA/LDA.png", dpi=300, bbox_inches='tight')
    plt.show()

    # "Testai"

    print(f"=={filename}==")
    try:
        tw = trustworthiness(X, X_lda, n_neighbors=N_NEIGHBORS)
        print(f"Patikimumas: {tw:.4f}")
    except Exception as e:
        print("Klaida patikimumo skaičiavime.", e)

    try:
        sil = silhouette_score(X_lda, y)
        print(f"Kontūro įvertis: {sil:.4f}")
    except Exception as e:
        print("Klaida kontūro įverčio skaičiavime.", e)

    base_dir = 'grafikai/UMAP'
    os.makedirs(base_dir, exist_ok=True)