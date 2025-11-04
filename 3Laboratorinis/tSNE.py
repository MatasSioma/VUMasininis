import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.manifold import TSNE

df = pd.read_csv('atrinkta_aibe.csv', sep=';')
X = df[[col for col in df.columns if col != 'label']].values
Y = df['label'].values

PERPLEXITY = 50
MAX_ITER = 3000
METRIC = 'canberra'
RANDOM_STATE = 2

tsne = TSNE(n_components=2,
    perplexity=PERPLEXITY,
    max_iter=MAX_ITER,
    metric=METRIC,
    random_state=RANDOM_STATE
    )
data_tsne = tsne.fit_transform(X)

# Išsaugoti 2D duomenis į CSV
tsne_df = pd.DataFrame(data_tsne, columns=['tsne_dim1', 'tsne_dim2'])
tsne_df['label'] = Y
os.makedirs("grafikai/tSNE", exist_ok=True)
tsne_df.to_csv('grafikai/tSNE/tsne_2d_data.csv', sep=';', index=False)
print("✓ t-SNE 2D data saved to grafikai/tSNE/tsne_2d_data.csv")

plt.figure(figsize=(10, 8))

class_values = sorted(df['label'].unique())
class_labels = [f'Klasė {int(c)}' for c in class_values]
colors = cm.viridis(np.linspace(0, 1, len(class_values)))

for val, label, color in zip(class_values, class_labels, colors):
    mask = df['label'] == val
    plt.scatter(
        data_tsne[mask, 0],
        data_tsne[mask, 1],
        color=color,
        label=label,
        alpha=0.7
    )

subtitle = f"perplexity={PERPLEXITY}, max_iter={MAX_ITER}, metric={METRIC}, random_state={RANDOM_STATE}"
plt.title(f't-SNE Dimensijos Mažinimas (atrinkti, sunormuoti požymiai)\n{subtitle}')
plt.xlabel('Dimensija 1')
plt.ylabel('Dimensija 2')
plt.legend(title='Klasės / Išskirtys')
plt.tight_layout()
plt.savefig("grafikai/tSNE/atrinkta_aibe.png", dpi=300)
plt.close()
print("✓ t-SNE plot saved to grafikai/tSNE/atrinkta_aibe.png")