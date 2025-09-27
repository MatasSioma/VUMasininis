import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

base_dir = 'grafikai/koreliacijos'
os.makedirs(base_dir, exist_ok=True)

df = pd.read_csv('EKG_pupsniu_analize_normalizuota_pagal_minmax.csv', sep=';')

columns = [col for col in df.columns if col.lower() != 'label']
labels = sorted(df['label'].unique(), key=lambda x: str(x))

def replace(pozymis: str) -> str:
    return pozymis.replace('/', '_')

# Koreliacijos kiekvienai klasei
for label in labels:
    df_class = df[df['label'] == label]
    print(f"\n=== {label} Klasė ===")

    class_folder = os.path.join(base_dir, f"klasė_{int(label)}")
    os.makedirs(class_folder, exist_ok=True)

    # Koreliacijos lentelė
    table_headers = [''] + columns
    table_rows = []

    for col_x in columns:
        row = [col_x]
        for col_y in columns:
            if col_x == col_y:
                row.append(1.0)
            else:
                corr_coef = df_class[col_x].corr(df_class[col_y], method='pearson')
                row.append(round(corr_coef, 4))
        table_rows.append(row)

    print(tabulate(table_rows, headers=table_headers, tablefmt='fancy_grid'))

    for col_x in columns:
        col_x_folder = os.path.join(class_folder, replace(col_x))
        os.makedirs(col_x_folder, exist_ok=True)

        for col_y in columns:
            if col_x != col_y:
                x = df_class[col_x].values
                y = df_class[col_y].values

                corr_coef = round(df_class[col_x].corr(df_class[col_y], method='pearson'), 4)

                plt.figure(figsize=(6, 4))
                plt.scatter(x, y, alpha=0.5)

                # Regresijos tiesė
                m, b = np.polyfit(x, y, 1)
                plt.plot(x, m*x + b, color='red', linestyle='-')

                plt.xlabel(col_x)
                plt.ylabel(col_y)
                plt.title(f'Klasė {label}: {col_x} vs {col_y}\nPearson r = {corr_coef}')
                plt.tight_layout()
                plt.savefig(f"{col_x_folder}/{replace(col_x)}_vs_{replace(col_y)}_class_{label}.png", dpi=150)
                plt.close()

# Visos df aibės koreliacijos
print(f"\n=== Bendrai ===")

overall_folder = os.path.join(base_dir, f"{'bendrai'}")
os.makedirs(overall_folder, exist_ok=True)

# Koreliacijos lentelė
table_headers = [''] + columns
table_rows = []

for col_x in columns:
    row = [col_x]
    for col_y in columns:
        if col_x == col_y:
            row.append(1.0)
        else:
            corr_coef = df[col_x].corr(df[col_y], method='pearson')
            row.append(round(corr_coef, 4))
    table_rows.append(row)

print(tabulate(table_rows, headers=table_headers, tablefmt='fancy_grid'))

for col_x in columns:
    col_x_folder = os.path.join(overall_folder, replace(col_x))
    os.makedirs(col_x_folder, exist_ok=True)

    for col_y in columns:
        if col_x != col_y:
            x = df[col_x].values
            y = df[col_y].values

            corr_coef = round(df[col_x].corr(df[col_y], method='pearson'), 4)

            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, alpha=0.5)

            # Regresijos tiesė
            m, b = np.polyfit(x, y, 1)
            plt.plot(x, m*x + b, color='red', linestyle='-')

            plt.xlabel(col_x)
            plt.ylabel(col_y)
            plt.title(f'Bendrai: {col_x} vs {col_y}\nPearson r = {corr_coef}')
            plt.tight_layout()
            plt.savefig(f"{col_x_folder}/{replace(col_x)}_vs_{replace(col_y)}_bendrai.png", dpi=150)
            plt.close()