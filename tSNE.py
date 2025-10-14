import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

df = pd.read_csv('EKG_pupsniu_analize_normuota_pagal_minmax.csv', sep=';')

columns = [col for col in df.columns if col.lower() != 'label']
X = df[columns].values
y = df['label'].values

color_map = ["#71B1FA", "#87E775", "#E05C58"]
colors = [color_map[int(c)] for c in y]

base_dir = 'grafikai/tSNE'
os.makedirs(base_dir, exist_ok=True)

tsne_2d = TSNE(
    n_components=2,
    perplexity=30,      # analogous to n_neighbors
    learning_rate=200,  # affects convergence speed
).fit_transform(X)

# === 5. Plot results ===
plt.figure(figsize=(10, 7))
plt.title('Dimensijos mažinimas t-SNE metodu')

plt.scatter(
    tsne_2d[:, 0],
    tsne_2d[:, 1],
    c=colors,
    s=15,
    alpha=0.8
)

unique_labels = sorted(set(y))
handles = [
    mpatches.Patch(color=color_map[i], label=f"Klasė {unique_labels[i]}")
    for i in range(len(unique_labels))
]
plt.legend(handles=handles, title="Klasės")

plt.tight_layout()
plt.savefig(f"{base_dir}/tSNE.png", dpi=150)
plt.close()

print("2D t-SNE vizualizacija sėkmingai sukurta ir išsaugota")
