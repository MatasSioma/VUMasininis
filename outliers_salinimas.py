import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pd.read_csv('EKG_pupsniu_analize_uzpildyta_medianomis.csv', sep=';')

columns = [col for col in df.columns if col.lower() != 'label']

os.makedirs('grafikai/isskirtys', exist_ok=True)

# Išskirčių printinimas pagal klases
for col in columns:
    plt.figure()
    df.boxplot(column=col, by='label', grid=False)
    plt.savefig(f'grafikai/isskirtys/{col.replace("/", "_")}_pagal_klase.png')
    plt.close()

    print(f"\n{'='*80}")
    print(f"Požymis: {col}")
    print(f"{'='*80}\n")

    # Sukuriame lentelę su visomis klasėmis
    table_data = []
    headers = ['Klasė', 'Q1', 'Q3', 'Vidiniai barjerai', 
               'Išoriniai barjerai', 'Mild', 'Extreme']

    for label in sorted(df['label'].unique()):
        df_label = df[df['label'] == label]

        Q1 = df_label[col].quantile(0.25)
        Q3 = df_label[col].quantile(0.75)
        H = Q3 - Q1

        inner_lower = Q1 - 1.5 * H
        inner_upper = Q3 + 1.5 * H

        outer_lower = Q1 - 3 * H
        outer_upper = Q3 + 3 * H

        extreme_outliers = df_label[(df_label[col] < outer_lower) | (df_label[col] > outer_upper)].round(4)
        mild_outliers = df_label[((df_label[col] >= outer_lower) & (df_label[col] < inner_lower)) |
                                  ((df_label[col] > inner_upper) & (df_label[col] <= outer_upper))].round(4)

        row = [
            label,
            round(Q1, 4),
            round(Q3, 4),
            f"[{round(inner_lower, 4)}, {round(inner_upper, 4)}]",
            f"[{round(outer_lower, 4)}, {round(outer_upper, 4)}]",
            len(mild_outliers),
            len(extreme_outliers)
        ]
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

# Išskirčių šalinimas (bendrai visam duomenų rinkiniui)
mild_outliers_set = set()
index_set = set()

print(f"\n{'='*80}")
print("BENDRAS IŠSKIRČIŲ ŠALINIMAS (visam duomenų rinkiniui)")
print(f"{'='*80}\n")

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
    mild_outliers_set.update(mild_outliers.index)

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
print(f"Viso 'mild' išskirčių: {len(mild_outliers_set)}")

df_cleaned = df.drop(index_set).reset_index(drop=True)

print(f"Liko eilučių kiekis: {df_cleaned.shape[0]}")

df_cleaned.to_csv('EKG_pupsniu_analize_be_isskirciu.csv', index=False, sep=';')