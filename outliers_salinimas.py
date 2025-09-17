import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pd.read_csv('EKG_pupsniu_analize_uzpildyta_medianomis.csv', sep=';')

columns = [col for col in df.columns if col.lower() != 'label']

os.makedirs('grafikai/isskirtys', exist_ok=True)

# Išskirčių printinimas
for col in columns:
    plt.figure()
    df.boxplot(column=col, by='label', grid=False)
    plt.title(f'{col} boxplot pagal klasę')
    plt.suptitle("")
    plt.xlabel('Klasė')
    plt.ylabel(col)
    plt.savefig(f'grafikai/isskirtys/{col.replace("/", "_")}_pagal_klase.png')
    plt.close()



# Išskirčių šalinimas
index_set = set()
for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    H = Q3 - Q1

    inner_lower = round(Q1 - 1.5 * H, 4)
    inner_upper = round(Q3 + 1.5 * H, 4)

    outer_lower = round(Q1 - 3 * H, 4)
    outer_upper = round(Q3 + 3 * H, 4)

    extreme_outliers = df[(df[col] < outer_lower) | (df[col] > outer_upper)]
    mild_outliers = df[((df[col] >= outer_lower) & (df[col] < inner_lower)) |
                   ((df[col] > inner_upper) & (df[col] <= outer_upper))]

    index_set.update(extreme_outliers.index)

    table_data = [
        ["Q1", Q1],
        ["Q3", Q3],
        ["Vidiniai barjerai", f"[{inner_lower}, {inner_upper}]"],
        ["Išoriniai barjerai", f"[{outer_lower}, {outer_upper}]"],
        ["Mild outliers", len(mild_outliers)],
        ["Extreme outliers", len(extreme_outliers)]
    ]

    print(f"\nColumn: {col}")
    print(tabulate(table_data, headers=['Rodiklis', 'reikšmė/intervalas'], tablefmt='fancy_grid'))

print(f'\nŠalinamos eilutės: {sorted(index_set)}\n')
print(f"Viso 'extreme' išskirčių: {len(index_set)}")

df_cleaned = df.drop(index_set).reset_index(drop=True)

df_cleaned.to_csv('EKG_pupsniu_analize_be_isskirciu.csv', index=False, sep=';')