import pandas as pd
from tabulate import tabulate

# Pasirenkame su kuriuo failu dirbti
print("Pasirinkite su kuriuo CSV failu norite dirbti:")
print("1 - EKG_pupsniu_analize_subalansuota.csv")
print("2 - EKG_pupsniu_analize_subalansuota_visi_pozymiai.csv")
pasirinkimas = input("Įveskite 1 arba 2: ").strip()

if pasirinkimas == "2":
    csv_failas = "EKG_pupsniu_analize_subalansuota_visi_pozymiai.csv"
    output_failas = "EKG_pupsniu_analize_uzpildyta_medianomis_visi.csv"
else:
    csv_failas = "EKG_pupsniu_analize_subalansuota.csv"
    output_failas = "EKG_pupsniu_analize_uzpildyta_medianomis.csv"

df = pd.read_csv(csv_failas, sep=';')

columns = list(df.columns)
print(columns)

# Patikriname praleistas reikšmes
print("\n##################### BENDRAS #####################\n")
praleistos_reikmes = df[df[columns].isnull().any(axis=1)]
if not praleistos_reikmes.empty:
    print(tabulate(praleistos_reikmes, headers='keys', tablefmt='fancy_grid', showindex=True))
else:
    print("Praleistų reikšmių nėra!")

# Praleistų reikšmių užpildymas pagal vidurkį
print("\n##################### PAGAL VIDURKĮ #####################\n")
df_pagal_vidurki = df.copy()
for column in columns:
    df_pagal_vidurki[column] = df_pagal_vidurki.groupby('label')[column].transform(lambda x: x.fillna(x.mean())).round(4)

uzpildyta_vidurkiu = df_pagal_vidurki[columns][df[columns].isnull().any(axis=1)]
print(tabulate(uzpildyta_vidurkiu, headers='keys', tablefmt='fancy_grid', showindex=True))

# Praleistų reikšmių užpildymas pagal medianą
print("\n##################### PAGAL MEDIANĄ #####################\n")
df_pagal_mediana = df.copy()
for column in columns:
    df_pagal_mediana[column] = df_pagal_mediana.groupby('label')[column].transform(lambda x: x.fillna(x.median())).round(4)

uzpildyta_mediana = df_pagal_mediana[columns][df[columns].isnull().any(axis=1)]
print(tabulate(uzpildyta_mediana, headers='keys', tablefmt='fancy_grid', showindex=True))

# Praleistų reikšmių užpildymas pagal modą
print("\n##################### PAGAL MODĄ #####################\n")
df_pagal_moda = df.copy()
for column in columns:
    df_pagal_moda[column] = df_pagal_moda.groupby('label')[column].transform(
            lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
        ).round(4)

uzpildyta_moda = df_pagal_moda[columns][df[columns].isnull().any(axis=1)]
print(tabulate(uzpildyta_moda, headers='keys', tablefmt='fancy_grid', showindex=True))


df_pagal_mediana.to_csv(output_failas, index=False, sep=';')