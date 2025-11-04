import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('EKG_pupsniu_analize_be_isskirciu.csv', sep=';')

columns = [col for col in df.columns if col.lower() != 'label']

colors = ['#00B894', '#FDCB6E', '#E17055']
labels = sorted(df['label'].unique(), key=lambda x: str(x))

base_dir = 'grafikai/histogramos'
os.makedirs(base_dir, exist_ok=True)

for col in columns:
    plt.figure(figsize=(8, 6))

    min_val = df[col].min()
    max_val = df[col].max()
    bins = np.linspace(min_val, max_val, 31)

    for i, label in enumerate(labels):
        klases_data = df[df['label'] == label][col]
        plt.hist(klases_data, bins=bins, alpha=1,
                label=f'Klasė {label}',
                color=colors[i],
                edgecolor="black")

    plt.xlabel(col)
    plt.ylabel('Dažnis')
    plt.title(f'{col} reikšmių pasiskirstymas pagal klases')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{col.replace('/', '_')}.png"
    plt.savefig(os.path.join(base_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()