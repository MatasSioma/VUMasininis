import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Konstantos
RANDOM_STATE = 42
DUOMENU_DIREKTORIJA = 'duomenys'

# Nuskaitome normuotą duomenų aibę
df_normuota = pd.read_csv(
    os.path.join(DUOMENU_DIREKTORIJA, 'sugeneruota_aibe_normuota.csv'),
    sep=';'
)

# Atskiriam požymius nuo klasių
X = df_normuota.drop(columns='label')
y = df_normuota['label']

# PIRMAS DALIJIMAS: 80/20 (80% mokymas+validavimas, 20% testavimas)
X_mokymas_validavimas, X_testavimas, y_mokymas_validavimas, y_testavimas = train_test_split(
    X, y,
    test_size=0.2,  # 20% testavimui
    random_state=RANDOM_STATE,
    stratify=y  # Išlaikome proporcijas tarp klasių
)

# ANTRAS DALIJIMAS: 80/20 iš likusių (80% mokymas, 20% validavimas)
X_mokymas, X_validavimas, y_mokymas, y_validavimas = train_test_split(
    X_mokymas_validavimas, y_mokymas_validavimas,
    test_size=0.2,  # 20% validavimui iš likusių 80%
    random_state=RANDOM_STATE,
    stratify=y_mokymas_validavimas
)

# Sujungiam požymius su klasėmis atgal
df_mokymas = pd.concat([X_mokymas, y_mokymas], axis=1)
df_validavimas = pd.concat([X_validavimas, y_validavimas], axis=1)
df_testavimas = pd.concat([X_testavimas, y_testavimas], axis=1)

# Išsaugome į failus
df_mokymas.to_csv(
    os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'),
    index=False,
    sep=';'
)
df_validavimas.to_csv(
    os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'),
    index=False,
    sep=';'
)
df_testavimas.to_csv(
    os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'),
    index=False,
    sep=';'
)

# Išspausdinam statistiką
print("=" * 60)
print("DUOMENŲ AIBĖS DALIJIMO STATISTIKA")
print("=" * 60)
print(f"\nBendras įrašų skaičius: {len(df_normuota)}")
print(f"\nMokymo aibė: {len(df_mokymas)} įrašų ({len(df_mokymas)/len(df_normuota)*100:.1f}%)")
print(f"  - Klasė 0: {(y_mokymas == 0).sum()} įrašų")
print(f"  - Klasė 2: {(y_mokymas == 2).sum()} įrašų")

print(f"\nValidavimo aibė: {len(df_validavimas)} įrašų ({len(df_validavimas)/len(df_normuota)*100:.1f}%)")
print(f"  - Klasė 0: {(y_validavimas == 0).sum()} įrašų")
print(f"  - Klasė 2: {(y_validavimas == 2).sum()} įrašų")

print(f"\nTestavimo aibė: {len(df_testavimas)} įrašų ({len(df_testavimas)/len(df_normuota)*100:.1f}%)")
print(f"  - Klasė 0: {(y_testavimas == 0).sum()} įrašų")
print(f"  - Klasė 2: {(y_testavimas == 2).sum()} įrašų")

print("\n" + "=" * 60)
print("Failai išsaugoti:")
print(f"  - {os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv')}")
print(f"  - {os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv')}")
print(f"  - {os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv')}")
print("=" * 60)