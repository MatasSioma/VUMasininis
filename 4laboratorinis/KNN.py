import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

"""
=============================================================================
K-ARTIMIAUSIŲ KAIMYNŲ (k-NN) ALGORITMO APRAŠYMAS
=============================================================================

VEIKIMO PRINCIPAS:
k-NN yra vienas paprasčiausių mašininio mokymosi algoritmų, paremtas intuityviu
principu: "panašūs objektai yra arti vieni kitų erdvėje".

KAIP VEIKIA:
1. Mokymo fazė:
   - Algoritmas tiesiog "įsimena" visus mokymo duomenis (lazy learning)
   - Nejokių skaičiavimų neatliekama

2. Prognozavimo fazė (naujiems duomenims):
   a) Apskaičiuojamas atstumas nuo naujo taško iki VISŲ mokymo taškų
   b) Išrenkami k artimiausi kaimynai
   c) Naujas taškas priskiriamas tai klasei, kuri dominuoja tarp k kaimynų
      (balsavimo principu - majority voting)

PARAMETRAI:
- n_neighbors (k): kiek kaimynų naudoti balsavimui
  * Mažas k (pvz., 1-3): jautrus triukšmui, gali būti overfitting
  * Didelis k (pvz., 20+): labiau išlygintas, bet gali prarasti detales
  * Dažniausiai renkasi nelyginį skaičių, kad išvengtų lygiųjų

- metric: kaip matuoti atstumą tarp taškų
  * 'euclidean': tiesus atstumas (√[(x₁-x₂)² + (y₁-y₂)²])
  * 'manhattan': "miesto kvartalų" atstumas (|x₁-x₂| + |y₁-y₂|)
  * 'minkowski': generalizuotas atstumas

- weights: kaip svertis kaimynų balsus
  * 'uniform': visi kaimynai turi vienodą svorį
  * 'distance': artimesni kaimynai turi didesnį svorį

PRIVALUMAI:
+ Paprastas suprasti ir implementuoti
+ Nereikia mokymo fazės (lazy learning)
+ Gerai veikia su neliniškais duomenimis
+ Gali būti naudojamas klasifikacijai ir regresijai

TRŪKUMAI:
- Lėtas prognozuojant (reikia skaičiuoti atstumą iki visų taškų)
- Jautrus duomenų skalei (būtina normalizacija)
- Neefektyvus didelėms duomenų aibėms
- Jautrus irrelevantiems požymiams

TAIKYMAS ŠIU DUOMENIMS:
- Turime 2D t-SNE duomenis (2 požymiai)
- Duomenys jau normuoti [0,1] intervale
- 2 klasės (0 ir 2) - binarinė klasifikacija
- k-NN puikiai tinka tokiai problemai
=============================================================================
"""

# Konstantos
RANDOM_STATE = 42
DUOMENU_DIREKTORIJA = 'duomenys'
GRAFIKU_DIREKTORIJA = 'grafikai'
KNN_DIREKTORIJA = 'KNN'

# k-NN hiperparametrai
K_NEIGHBORS = 5  # Kiek kaimynų naudoti
METRIC = 'euclidean'  # Atstumo metrika
WEIGHTS = 'uniform'  # Kaip svertis kaimynų balsus

# Sukuriame grafikai/KNN direktoriją
os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA), exist_ok=True)

# 1. Įkeliame duomenis
print("=" * 70)
print("1. DUOMENŲ ĮKĖLIMAS")
print("=" * 70)

df_mokymas = pd.read_csv(
    os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'),
    sep=';'
)
df_validavimas = pd.read_csv(
    os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'),
    sep=';'
)
df_testavimas = pd.read_csv(
    os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'),
    sep=';'
)

# Atskiriam požymius (X) ir klases (y)
X_mokymas = df_mokymas.drop(columns='label').values
y_mokymas = df_mokymas['label'].values

X_validavimas = df_validavimas.drop(columns='label').values
y_validavimas = df_validavimas['label'].values

X_testavimas = df_testavimas.drop(columns='label').values
y_testavimas = df_testavimas['label'].values

print(f"Mokymo aibė: {X_mokymas.shape[0]} įrašų, {X_mokymas.shape[1]} požymiai")
print(f"Validavimo aibė: {X_validavimas.shape[0]} įrašų")
print(f"Testavimo aibė: {X_testavimas.shape[0]} įrašų")

# 2. Sukuriame ir apmokom k-NN modelį
print("\n" + "=" * 70)
print("2. MODELIO MOKYMAS")
print("=" * 70)

knn = KNeighborsClassifier(
    n_neighbors=K_NEIGHBORS,
    metric=METRIC,
    weights=WEIGHTS
)

print(f"Algoritmas: k-Nearest Neighbors (k-NN)")
print(f"Parametrai:")
print(f"  - n_neighbors (k): {K_NEIGHBORS}")
print(f"  - metric: {METRIC}")
print(f"  - weights: {WEIGHTS}")
print(f"\nMokymas...")

knn.fit(X_mokymas, y_mokymas)
print("✓ Modelis apmokytas!")

# 3. Prognozuojame ir vertiname
print("\n" + "=" * 70)
print("3. MODELIO VERTINIMAS")
print("=" * 70)

# Prognozės
y_mokymas_pred = knn.predict(X_mokymas)
y_validavimas_pred = knn.predict(X_validavimas)
y_testavimas_pred = knn.predict(X_testavimas)

# Metrikos
def spausdinti_metrikos(y_tikros, y_prognozes, rinkinys_pavadinimas):
    print(f"\n{rinkinys_pavadinimas}:")
    print(f"  Tikslumas (Accuracy): {accuracy_score(y_tikros, y_prognozes):.4f}")
    print(f"  Precizija (Precision): {precision_score(y_tikros, y_prognozes, average='weighted'):.4f}")
    print(f"  Atšaukimas (Recall): {recall_score(y_tikros, y_prognozes, average='weighted'):.4f}")
    print(f"  F1 balas: {f1_score(y_tikros, y_prognozes, average='weighted'):.4f}")

spausdinti_metrikos(y_mokymas, y_mokymas_pred, "MOKYMO AIBĖ")
spausdinti_metrikos(y_validavimas, y_validavimas_pred, "VALIDAVIMO AIBĖ")
spausdinti_metrikos(y_testavimas, y_testavimas_pred, "TESTAVIMO AIBĖ")

# 4. Detalus klasifikacijos ataskaita testavimo aibei
print("\n" + "=" * 70)
print("4. DETALI KLASIFIKACIJOS ATASKAITA (TESTAVIMO AIBĖ)")
print("=" * 70)
print(classification_report(
    y_testavimas,
    y_testavimas_pred,
    target_names=['Klasė 0', 'Klasė 2']
))

# 5. Painiavos matrica (Confusion Matrix)
print("\n" + "=" * 70)
print("5. PAINIAVOS MATRICA")
print("=" * 70)

cm = confusion_matrix(y_testavimas, y_testavimas_pred)
print("\nTestavimo aibė:")
print(cm)
print(f"\nTeisingai klasifikuota: {cm[0,0] + cm[1,1]}")
print(f"Klaidingai klasifikuota: {cm[0,1] + cm[1,0]}")

# 6. Vizualizacija: Painiavos matrica
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Klasė 0', 'Klasė 2'],
    yticklabels=['Klasė 0', 'Klasė 2']
)
plt.title(f'Painiavos matrica (k-NN, k={K_NEIGHBORS})')
plt.ylabel('Tikra klasė')
plt.xlabel('Prognozuota klasė')
plt.tight_layout()
plt.savefig(
    os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'confusion_matrix.png'),
    dpi=3000
)
plt.close()

# 7. Vizualizacija: Klasifikacijos rezultatai
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def nubraizti_rezultatus(ax, X, y_tikros, y_prognozes, pavadinimas):
    # Teisingai klasifikuoti
    teisinga_maska = y_tikros == y_prognozes
    # Klaidingai klasifikuoti
    klaida_maska = ~teisinga_maska

    # Braižome teisingus
    for klase in [0.0, 2.0]:
        maska = (y_tikros == klase) & teisinga_maska
        ax.scatter(
            X[maska, 0], 
            X[maska, 1],
            label=f'Klasė {int(klase)} (teisingai)',
            alpha=0.6,
            s=30
        )

    # Braižome klaidingus raudonais kryžiais
    ax.scatter(
        X[klaida_maska, 0],
        X[klaida_maska, 1],
        c='red',
        marker='x',
        s=100,
        label='Klaidingai klasifikuota',
        linewidths=2
    )

    ax.set_title(f'{pavadinimas}\nTikslumas: {accuracy_score(y_tikros, y_prognozes):.4f}')
    ax.set_xlabel('Požymis 1')
    ax.set_ylabel('Požymis 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

nubraizti_rezultatus(axes[0], X_mokymas, y_mokymas, y_mokymas_pred, 'Mokymo aibė')
nubraizti_rezultatus(axes[1], X_validavimas, y_validavimas, y_validavimas_pred, 'Validavimo aibė')
nubraizti_rezultatus(axes[2], X_testavimas, y_testavimas, y_testavimas_pred, 'Testavimo aibė')

plt.suptitle(f'k-NN Klasifikacijos rezultatai (k={K_NEIGHBORS}, metric={METRIC})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(
    os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'classification_results.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.close()

print(f"\n✓ Grafikai išsaugoti '{os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA)}' direktorijoje")
print("=" * 70)