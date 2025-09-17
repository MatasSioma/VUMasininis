import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = 'grafikai/dazniai'
os.makedirs(base_dir, exist_ok=True)

df = pd.read_csv('EKG_pupsniu_analize_normalizuota_pagal_minmax.csv', sep=';')

label_kiekis = df['label'].value_counts()

label_kiekis_df = label_kiekis.rename_axis('label').reset_index(name='count')
ax = label_kiekis_df.plot.bar(x='label', y='count', legend=False, color='green', edgecolor='black', figsize=(6,4))
ax.set_xlabel('Klasė')
ax.set_ylabel('Dažnis')
ax.set_title('Kiekvienos klasės dažnis')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'Pasiskirstymas_pagal_klases.png'), dpi=150)