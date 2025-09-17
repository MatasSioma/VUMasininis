import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('EKG_pupsniu_analize_normalizuota_pagal_minmax.csv', sep=';')

# exclude label column
columns = [col for col in df.columns if col.lower() != 'label']

base_dir = 'grafikai/taskiniai'

def replace(pozymis: str) -> str:
    return pozymis.replace('/', '_')

for col_x in columns:
    col_x_folder = os.path.join(base_dir, replace(col_x))
    os.makedirs(col_x_folder, exist_ok=True)

    for col_y in columns:
        if col_x != col_y:
            plt.figure(figsize=(6, 4))
            plt.scatter(df[col_x], df[col_y], alpha=0.5)
            plt.xlabel(col_x)
            plt.ylabel(col_y)
            plt.title(f'{col_x} vs {col_y}')
            plt.tight_layout()
            plt.savefig(f"{col_x_folder}/{replace(col_x)}_vs_{replace(col_y)}.png", dpi=150)
            plt.close()
