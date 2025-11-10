import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE

# === PARAMETRAI ===
PERPLEXITY = 50
MAX_ITER = 500
METRIC = 'canberra'
RANDOM_STATE = 42

# === FUNKCIJA t-SNE vykdymui ir saugojimui ===
def run_tsne_and_save(X, Y, name_prefix):
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        max_iter=MAX_ITER,
        metric=METRIC,
        random_state=RANDOM_STATE
    )
    data_tsne = tsne.fit_transform(X)

    # Min-Max normalizavimas
    data_tsne_normalized = np.zeros_like(data_tsne)
    for i in range(data_tsne.shape[1]):
        xmin, xmax = data_tsne[:, i].min(), data_tsne[:, i].max()
        data_tsne_normalized[:, i] = (data_tsne[:, i] - xmin) / (xmax - xmin)

    # Išsaugoti į CSV
    tsne_df = pd.DataFrame(data_tsne_normalized, columns=['tsne_dim1', 'tsne_dim2'])
    tsne_df['label'] = Y
    os.makedirs("duomenys", exist_ok=True)
    csv_path = f'duomenys/tsne_2d_data{name_prefix}.csv'
    tsne_df.to_csv(csv_path, sep=';', index=False)
    print(f"✓ t-SNE 2D data (normalized) saved to {csv_path}")

    # Sukurti grafiką
    plt.figure(figsize=(10, 8))
    class_values = sorted(np.unique(Y))
    class_labels = [f'Klasė {int(c)}' for c in class_values]
    colors = cm.viridis(np.linspace(0, 1, len(class_values)))

    for val, label, color in zip(class_values, class_labels, colors):
        mask = Y == val
        plt.scatter(
            data_tsne_normalized[mask, 0],
            data_tsne_normalized[mask, 1],
            color=color,
            label=label,
            alpha=0.7
        )

    subtitle = f"perplexity={PERPLEXITY}, max_iter={MAX_ITER}, metric={METRIC}, random_state={RANDOM_STATE}"
    plt.title(f't-SNE Dimensijos Mažinimas ({name_prefix})\n{subtitle}')
    plt.xlabel('Dimensija 1')
    plt.ylabel('Dimensija 2')
    plt.legend(title='Klasės')
    plt.tight_layout()

    os.makedirs("grafikai/tSNE", exist_ok=True)
    img_path = f"grafikai/tSNE/{name_prefix}.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"✓ t-SNE plot saved to {img_path}\n")


# === ĮKELTI DUOMENIS ===
df_su_isskirtim = pd.read_csv('duomenys/atrinkta_aibe.csv', sep=';')
X_su_isskirtim = df_su_isskirtim.drop(columns=['label']).values
Y_su_isskirtim = df_su_isskirtim['label'].values

df_be_isskirciu = pd.read_csv('../EKG_pupsniu_analize_be_isskirciu.csv', sep=';')
X_be_isskirciu = df_be_isskirciu.drop(columns=['label']).values
Y_be_isskirciu = df_be_isskirciu['label'].values

# === t-SNE abiems rinkinams ===
run_tsne_and_save(X_su_isskirtim, Y_su_isskirtim, '')
run_tsne_and_save(X_be_isskirciu, Y_be_isskirciu, '_be_isskirciu')
