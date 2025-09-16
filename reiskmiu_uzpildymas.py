import pandas as pd

df = pd.read_csv('EKG_pupsniu_analize_subalansuota.csv', sep=';')

columns = list(df.columns)
print(columns)

# Patikriname praleistas reikšmes

print("\n##################### BENDRAS #####################\n")
print(df[columns][df[columns].isnull().any(axis=1)])

# Praleistų reikšmių užpildymas pagal vidurkį

print("\n##################### PAGAL VIDURKĮ #####################\n")
df_pagal_vidurki = df.copy()
for column in columns:
    df_pagal_vidurki[column] = df_pagal_vidurki[column].fillna(df_pagal_vidurki.groupby('label')[column].transform('mean'))

print(df_pagal_vidurki[columns][df[columns].isnull().any(axis=1)])

# Praleistų reikšmių užpildymas pagal medianą

print("\n##################### PAGAL MEDIANĄ #####################\n")
df_pagal_mediana = df.copy()
for column in columns:
    df_pagal_mediana[column] = df_pagal_mediana[column].fillna(df_pagal_mediana.groupby('label')[column].transform('median'))


print(df_pagal_mediana[columns][df[columns].isnull().any(axis=1)])

# Praleistų reikšmių užpildymas pagal modą

print("\n##################### PAGAL MODĄ #####################\n")
df_pagal_moda = df.copy()
for column in columns:
    df_pagal_moda[column] = df_pagal_moda[column].fillna(df_pagal_moda.groupby('label')[column].transform(lambda x: x.mode()[0]))

print(df_pagal_moda[columns][df[columns].isnull().any(axis=1)])


df_pagal_vidurki.to_csv('EKG_pupsniu_analize_uzpildyta_vidurkiais.csv', index=False, sep=';')