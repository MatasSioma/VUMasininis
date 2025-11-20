import pandas as pd
from tabulate import tabulate

df_originali = pd.read_csv('duomenys/sugeneruota_aibe_nenormuota.csv', sep=';')

columns = [col for col in df_originali.columns if col.lower() != 'label']

df_klases = {
    0: df_originali[df_originali['label'] == 0.0].reset_index(drop=True),
    2: df_originali[df_originali['label'] == 2.0].reset_index(drop=True)
}

def print_statistika(df_pagal_klase, class_name):
    print(f"\n##################### {class_name} #####################")
    for col in columns:
        apr_statistika = df_pagal_klase[col].describe().round(4)
        mediana = df_pagal_klase[col].median().round(4)
        moda = df_pagal_klase[col].mode().round(4).tolist()

        print(f"\n#### {col.upper()} ####\n")
        print(tabulate(apr_statistika.to_frame(), headers=['Rodiklis', 'Reikšmė'], tablefmt='fancy_grid'))
        print(f"Mediana: {mediana}")
        if len(moda) > 3:
            print(f"Modos nėra! :(((")
        else:
            print(f'Moda: {moda}')

for klases_nr, df_pagal_klase in df_klases.items():
    print_statistika(df_pagal_klase, f"{klases_nr} KLASĖ")

print_statistika(df_originali, "BENDRAI")