import os

import pandas as pd
from tabulate import tabulate

df_original = pd.read_csv('../EKG_pupsniu_analize.csv', sep=';')
columns = df_original.columns.drop('label')

os.makedirs("duomenys", exist_ok=True)

print('=== Bazinė informacija apie visą duomenų rinkinį ===')
print(df_original.info())

print('\n=== Praleistų reikšmių kiekis pagal požymį ===')
print(df_original.isnull().sum())

print('\n=== Bendras praleistų reikšmių kiekis ===')
print(df_original.isnull().sum().sum())

df_original = df_original.dropna(subset=['label'])
print(f'\n=== Pašalintos eilutės be label (liko {len(df_original)} eilučių) ===')

# Konvertuojame iš object į float64
for col in columns:
    df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

print('\n=== Praleistų reikšmių užpildymas medianomis ===')
df_uzpildyta = df_original.copy()
for col in columns:
   df_uzpildyta[col] = df_uzpildyta.groupby('label')[col].transform(lambda x: x.fillna(x.median()))

print('=== Bazinė informacija apie visą užpildytą rinkinį ===')
print(df_uzpildyta.info())

print('\n=== Bendras praleistų reikšmių kiekis ===')
print(df_uzpildyta.isnull().sum().sum())

print('\n=== Generuojama aibė su 1500 objektų ===')
pozymiai = ['Q_val', 'R_val', 'S_val', 'RR_l_0/RR_l_1', 'signal_std', 'seq_size', 'label']
random_seed = 22

df_atrinkta = df_uzpildyta[pozymiai].groupby('label', group_keys=False).apply(lambda x: x.sample(n=500, random_state=random_seed)).reset_index(drop=True)

print('\n=== Sugeneruota duomenų aibė ===')
print(df_atrinkta.info())

print('\n=== Normavimas ===')
df_minmax = df_atrinkta.copy()
for col in pozymiai:
    if col != 'label':
        xmin = df_minmax[col].min()
        xmax = df_minmax[col].max()
        df_minmax[col] = ((df_minmax[col] - xmin) / (xmax - xmin))

print(tabulate(df_minmax.describe(), headers='keys', tablefmt='fancy_grid', showindex=True, numalign='left'))

df_minmax.to_csv('duomenys/atrinkta_aibe.csv', index=False, sep=';')