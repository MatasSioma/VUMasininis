import umap
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

N_NEIGHBORS = 15
MIN_DIST = 0.2

df = pd.read_csv('EKG_pupsniu_analize_normuota_pagal_minmax.csv', sep=';')

columns = [col for col in df.columns if col.lower() != 'label']
X = df[columns].values
y = df['label'].values

color_map = ["#71B1FA", "#87E775", "#E05C58"]
colors = [color_map[int(c)] for c in y]

base_dir = 'grafikai/UMAP'
os.makedirs(base_dir, exist_ok=True)

umap_2d = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    n_components=2,
).fit_transform(X)

plt.figure(figsize=(10, 7))
plt.title('Dimensijos mažinimas UMAP metodu\nn_neighbors={}, min_dist={}'.format(N_NEIGHBORS, MIN_DIST))

plt.scatter(
    umap_2d[:, 0],
    umap_2d[:, 1],
    c=colors,
    s=15,
    alpha=0.8
)

# Custom legend for 3 classes
unique_labels = sorted(set(y))
handles = [
    mpatches.Patch(color=color_map[i], label=f"Klasė {unique_labels[i]}")
    for i in range(len(unique_labels))
]
plt.legend(handles=handles, title="Klasės")

plt.tight_layout()
plt.savefig(f"{base_dir}/UMAP-nN_{N_NEIGHBORS}-mD_{MIN_DIST}.png", dpi=150)
plt.close()

print("2D UMAP vizualizacija sėkmingai sukurta ir išsaugota")
