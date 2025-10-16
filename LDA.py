import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Duomenų įkėlimas
df = pd.read_csv("pilna_EKG_pupsniu_analize_normuota_pagal_minmax.csv", sep=";")

# Požymių ir klasių atskyrimas
X = df[["Q_val", "R_val", "S_val", "RR_l_0/RR_l_1", "signal_std", "seq_size"]]
y = df["label"]

# Normalizacija
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LDA pritaikymas
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

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

palette = sns.color_palette("Set1", n_colors=lda_df['Class'].nunique())
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
plt.legend(title='Klasė')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('')
plt.ylabel('')
plt.savefig("grafikai/LDA/LDA.png", dpi=300, bbox_inches='tight')
plt.show()