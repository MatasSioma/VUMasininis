import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('EKG_pupsniu_analize_be_isskirciu.csv', sep=';')

columns = [col for col in df.columns if col.lower() != 'label']

base_dir = 'grafikai/histogramos'
os.makedirs(base_dir, exist_ok=True)

for col in columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=40, edgecolor='black', color='#542788', alpha=0.7)
    plt.xlabel(col)
    plt.ylabel('Dažnis')
    plt.title(f'{col} reikšmių dažnio diagrama')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"{col.replace('/', '_')}.png"), dpi=150)
    plt.close()