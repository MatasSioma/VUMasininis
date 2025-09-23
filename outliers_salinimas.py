import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pd.read_csv('EKG_pupsniu_analize_uzpildyta_medianomis.csv', sep=';')

columns = [col for col in df.columns if col.lower() != 'label']

os.makedirs('grafikai/isskirtys', exist_ok=True)

# Boxplot'ai
for col in columns:
    plt.figure()
    df.boxplot(column=col, by='label', grid=False)
    plt.title(f'{col} boxplot pagal klasę')
    plt.suptitle("")
    plt.xlabel('Klasė')
    plt.ylabel(col)
    plt.savefig(f'grafikai/isskirtys/{col.replace("/", "_")}_pagal_klase.png')
    plt.close()

# Išskirčių lentelės ir šalinimas
mild_outliers_set = set()
index_set = set()

for col in columns:
    print(f"\n=== Stulpelis: {col} ===")

    rows = []

    for klasė, grupė in df.groupby('label'):
        # Skaičiavimai konkrečiai klasei
        Q1 = round(grupė[col].quantile(0.25), 4)
        Q3 = round(grupė[col].quantile(0.75), 4)
        H = Q3 - Q1

        inner_lower = round(Q1 - 1.5 * H, 4)
        inner_upper = round(Q3 + 1.5 * H, 4)

        outer_lower = round(Q1 - 3 * H, 4)
        outer_upper = round(Q3 + 3 * H, 4)

        extreme_outliers = grupė[(grupė[col] < outer_lower) | (grupė[col] > outer_upper)]
        mild_outliers = grupė[((grupė[col] >= outer_lower) & (grupė[col] < inner_lower)) |
                              ((grupė[col] > inner_upper) & (grupė[col] <= outer_upper))]

        index_set.update(extreme_outliers.index)
        mild_outliers_set.update(mild_outliers.index)

        rows.append([
            klasė,
            Q1,
            Q3,
            f"[{inner_lower}, {inner_upper}]",
            f"[{outer_lower}, {outer_upper}]",
            len(mild_outliers),
            len(extreme_outliers)
        ])

    # Gražesnė lentelė su klasėmis kaip eilutėmis
    table_df = pd.DataFrame(rows, columns=[
        'Klasė', 'Q1', 'Q3', 'Vidiniai barjerai', 'Išoriniai barjerai',
        'Mild outliers', 'Extreme outliers'
    ]).set_index('Klasė')

    print(tabulate(table_df, headers='keys', tablefmt='fancy_grid'))

print(f'\nŠalinamos eilutės: {sorted(index_set)}\n')
print(f"Viso 'extreme' išskirčių: {len(index_set)}")
print(f"Viso 'mild' išskirčių: {len(mild_outliers_set)}")

df_cleaned = df.drop(index_set).reset_index(drop=True)
print(f"Liko eilučių: {df_cleaned.shape[0]}")

df_cleaned.to_csv('EKG_pupsniu_analize_be_isskirciu.csv', index=False, sep=';')
