import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

PERPLEXITY = 50
MAX_ITER = 500
METRIC = 'canberra'
RANDOM_STATE = 42

DUOMENU_DIREKTORIJA = 'duomenys'
DUOMENU_FAILAS = 'sugeneruota_aibe.csv'
DUOMENU_FAILAS_2D = 'sugeneruota_aibe_2D.csv'

GRAFIKU_DIREKTORIJA = 'grafikai'
TSNE_DIREKTORIJA = 'TSNE',
TSNE_GRAFIKA = '2D'

df = pd.read_csv('../EKG_pupsniu_analize.csv', sep=';')
df_atfiltruotos_klases = df[df['label'].isin([0, 2])]

# 1 Žingsnis – pašalinti eilutes su praleistomis reikšmėmis prieš atrinkimą.
df_be_praleista = df_atfiltruotos_klases.dropna()

# 2 Žingsnis – atsirinkti duomenis pagal klases (0 ir 2) 2x1000.
df_sugeneruota = df_be_praleista.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=1000, random_state=RANDOM_STATE)
).reset_index(drop=True)

# 3 Žingsnis – konvertuojame iš object į float64.
for col in df_sugeneruota.columns:
    df_sugeneruota[col] = pd.to_numeric(df_sugeneruota[col], errors='coerce')

# 4 Žingsnis – sunormuoti duomenų aibę min-max metodu.
df_normuota = df_sugeneruota.copy()
for col in df_normuota.columns:
    if col != 'label':
        x_min = df_normuota[col].min()
        x_max = df_normuota[col].max()
        df_normuota[col] = ((df_normuota[col] - x_min) / (x_max - x_min))

# 5 Žingsnis – padaryti 2D duomenų aibę.
X = df_normuota.drop(columns='label').values
Y = df_normuota['label'].values

tsne = TSNE(
    n_components=2,
    perplexity=PERPLEXITY,
    max_iter=MAX_ITER,
    metric=METRIC,
    random_state=RANDOM_STATE
)
tsne_duomenys = tsne.fit_transform(X)

tsne_duomenys_normuota = np.zeros_like(tsne_duomenys)
for i in range(tsne_duomenys.shape[1]):
    xmin, xmax = tsne_duomenys[:, i].min(), tsne_duomenys[:, i].max()
    tsne_duomenys_normuota[:, i] = (tsne_duomenys[:, i] - xmin) / (xmax - xmin)

tsne_df = pd.DataFrame(tsne_duomenys_normuota, columns=['dimensija_1', 'dimensija_2'])
tsne_df['label'] = Y


# Paskutinis žingsnis – viską išsaugoti.
os.makedirs(DUOMENU_DIREKTORIJA, exist_ok=True)
df_normuota.to_csv(os.path.join(DUOMENU_DIREKTORIJA, DUOMENU_FAILAS), index=False, sep=';')
tsne_df.to_csv(os.path.join(DUOMENU_DIREKTORIJA, DUOMENU_FAILAS_2D), index=False, sep=';')