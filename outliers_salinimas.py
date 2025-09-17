import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('EKG_pupsniu_analize_uzpildyta_medianomis.csv', sep=';')

columns = [col for col in df.columns if col.lower() != 'label']

os.makedirs('grafikai/isskirtys', exist_ok=True)

# Išskirčių printinimas
for col in columns:
    plt.figure()
    df.boxplot(column=col, by='label', grid=False)
    plt.title(f'Boxplot of {col} by label')
    plt.suptitle("")
    plt.xlabel('Label')
    plt.ylabel(col)
    plt.savefig(f'grafikai/isskirtys/{col.replace("/", "_")}_by_label.png')
    plt.close()



# Išskirčių šalinimas
index_set = set()
for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    H = Q3 - Q1

    inner_lower = Q1 - 1.5 * H
    inner_upper = Q3 + 1.5 * H

    outer_lower = Q1 - 3 * H
    outer_upper = Q3 + 3 * H

    extreme_outliers = df[(df[col] < outer_lower) | (df[col] > outer_upper)]
    mild_outliers = df[((df[col] >= outer_lower) & (df[col] < inner_lower)) |
                   ((df[col] > inner_upper) & (df[col] <= outer_upper))]

    index_set.update(extreme_outliers.index)

    print(f"\nColumn: {col}")
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"Inner limits: [{inner_lower:.3f}, {inner_upper:.3f}]")
    print(f"Outer limits: [{outer_lower:.3f}, {outer_upper:.3f}]")
    print(f"Mild outliers: {len(mild_outliers)}")
    print(f"Extreme outliers: {len(extreme_outliers)}")

print(f'\n{sorted(index_set)}\n')
print(f"Viso 'extreme' išskirčių: {len(index_set)}")

df_cleaned = df.drop(index_set).reset_index(drop=True)

df_cleaned.to_csv('EKG_pupsniu_analize_be_isskirciu.csv', index=False, sep=';')