import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.manifold import TSNE

df = pd.read_csv('duomenys/atrinkta_aibe.csv', sep=';')
X = df[[col for col in df.columns if col != 'label']].values
Y = df['label'].values

PERPLEXITY = 50
MAX_ITER = 500
METRIC = 'canberra'
RANDOM_STATE = 42

tsne = TSNE(n_components=2,
    perplexity=PERPLEXITY,
    max_iter=MAX_ITER,
    metric=METRIC,
    random_state=RANDOM_STATE
    )
data_tsne = tsne.fit_transform(X)

# Min-Max normavimas t-SNE rezultatams
data_tsne_normalized = data_tsne.copy()
for i in range(data_tsne.shape[1]):
    xmin = data_tsne[:, i].min()
    xmax = data_tsne[:, i].max()
    data_tsne_normalized[:, i] = (data_tsne[:, i] - xmin) / (xmax - xmin)

# Išsaugoti sunormuotus 2D duomenis į CSV
tsne_df = pd.DataFrame(data_tsne_normalized, columns=['tsne_dim1', 'tsne_dim2'])
tsne_df['label'] = Y
os.makedirs("grafikai/tSNE", exist_ok=True)
os.makedirs("duomenys", exist_ok=True)
tsne_df.to_csv('duomenys/tsne_2d_data.csv', sep=';', index=False)
print("✓ t-SNE 2D data (normalized) saved to duomenys/tsne_2d_data.csv")

plt.figure(figsize=(10, 8))

class_values = sorted(df['label'].unique())
class_labels = [f'Klasė {int(c)}' for c in class_values]
colors = cm.viridis(np.linspace(0, 1, len(class_values)))

for val, label, color in zip(class_values, class_labels, colors):
    mask = df['label'] == val
    plt.scatter(
        data_tsne_normalized[mask, 0],
        data_tsne_normalized[mask, 1],
        color=color,
        label=label,
        alpha=0.7
    )

subtitle = f"perplexity={PERPLEXITY}, max_iter={MAX_ITER}, metric={METRIC}, random_state={RANDOM_STATE}"
plt.title(f't-SNE Dimensijos Mažinimas (atrinkti, sunormuoti požymiai)\n{subtitle}')
plt.xlabel('Dimensija 1')
plt.ylabel('Dimensija 2')
plt.legend(title='Klasės')
plt.tight_layout()
plt.savefig("grafikai/tSNE/atrinkta_aibe.png", dpi=300)
plt.close()
print("✓ t-SNE plot saved to grafikai/tSNE/atrinkta_aibe.png")