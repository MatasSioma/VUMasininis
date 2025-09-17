import pandas as pd
from tabulate import tabulate

df = pd.read_csv('EKG_pupsniu_analize_be_isskirciu.csv', sep=';')
columns = [col for col in df.columns if col.lower() != 'label']

# Min-max normalizavimas
df_minmax = df.copy()
print("\n##################### PRIEŠ MIN-MAX NORMALIZAVIMĄ #####################\n")
print(tabulate(df_minmax.head(), headers='keys', tablefmt='fancy_grid', showindex=True))
for col in columns:
    xmin = df_minmax[col].min()
    xmax = df_minmax[col].max()
    df_minmax[col] = ((df_minmax[col] - xmin) / (xmax - xmin)).round(4)

print("\n##################### PO MIN-MAX NORMALZAVIMO #####################\n")
print(tabulate(df_minmax.head(), headers='keys', tablefmt='fancy_grid', showindex=True))
for col in columns:
    print(f'\n{col} Min: {df_minmax[col].min()}')
    print(f'{col} Max: {df_minmax[col].max()}')

df_minmax.to_csv('EKG_pupsniu_analize_normalizuota_pagal_minmax.csv', index=False, sep=';')

# Normalizavimas pagal vidurkį
df_z_reiksme = df.copy()
print("\n##################### PRIEŠ NORMALIZAVIMĄ PAGAL VIDURKĮ IR DISPERSIJĄ #####################\n")
print(tabulate(df_z_reiksme.head(), headers='keys', tablefmt='fancy_grid', showindex=True))
for col in columns:
    vidurkis = df_z_reiksme[col].mean()
    std = df_z_reiksme[col].std()
    df_z_reiksme[col] = ((df_z_reiksme[col] - vidurkis) / std).round(4)

print("\n##################### PO NORMALIZAVIMO PAGAL VIDURKĮ IR DISPERSIJĄ #####################\n")
print(tabulate(df_z_reiksme.head(), headers='keys', tablefmt='fancy_grid', showindex=True))
for col in columns:
    print(f'\n{col} Min: {df_z_reiksme[col].min()}')
    print(f'{col} Max: {df_z_reiksme[col].max()}')

df_z_reiksme.to_csv('EKG_pupsniu_analize_normalizuota_pagal_vidurki_ir_dispersija.csv', index=False, sep=';')