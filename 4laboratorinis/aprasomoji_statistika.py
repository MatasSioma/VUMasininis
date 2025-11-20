import pandas as pd
from tabulate import tabulate

df = pd.read_csv('rezultatai/sugeneruota_aibe.csv', sep=';')

print(df.info())
print('=== Atrinktos klasės ===')
print(' ir '.join(map(str, df['label'].unique().astype(int))))