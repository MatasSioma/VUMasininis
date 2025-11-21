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

# Sukuriame grafikai/KNN direktoriją
os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA), exist_ok=True)

# 1. Įkeliame duomenis
print("=" * 70)
print("1. DUOMENŲ ĮKĖLIMAS")
print("=" * 70)

try:
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
except FileNotFoundError:
    # Fallback if running from parent directory
    DUOMENU_DIREKTORIJA = '4laboratorinis/duomenys'
    GRAFIKU_DIREKTORIJA = '4laboratorinis/grafikai'
    os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA), exist_ok=True)
    
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

# 2. Hiperparametrų parinkimas (Tuning)
print("\n" + "=" * 70)
print("2. HIPERPARAMETRŲ PARINKIMAS (VALIDAVIMAS)")
print("=" * 70)

k_values = range(1, 22, 2)  # 1, 3, 5, ..., 21
results = []

print(f"{'k':<5} | {'Accuracy':<10} | {'F1 Score':<10}")
print("-" * 30)

best_k = -1
best_f1 = -1

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
    knn.fit(X_mokymas, y_mokymas)
    
    y_val_pred = knn.predict(X_validavimas)
    
    acc = accuracy_score(y_validavimas, y_val_pred)
    f1 = f1_score(y_validavimas, y_val_pred, average='weighted')
    
    results.append({'k': k, 'accuracy': acc, 'f1': f1})
    
    print(f"{k:<5} | {acc:.4f}     | {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_k = k

print("-" * 30)
print(f"Geriausias k pagal F1 balą: {best_k} (F1={best_f1:.4f})")

# Vizualizuojame parametru paiešką
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.plot(results_df['k'], results_df['accuracy'], marker='o', label='Accuracy')
plt.plot(results_df['k'], results_df['f1'], marker='s', label='F1 Score')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
plt.xlabel('Kaimynų skaičius (k)')
plt.ylabel('Metrikos reikšmė')
plt.title('k-NN Hiperparametrų įtaka tikslumui (Validavimo aibė)')
plt.legend()
plt.grid(True)
plt.xticks(k_values)
plt.savefig(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'parameter_tuning.png'))
plt.close()

# 3. Galutinis modelio mokymas
print("\n" + "=" * 70)
print(f"3. GALUTINIO MODELIO MOKYMAS (k={best_k})")
print("=" * 70)

final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='uniform')
final_knn.fit(X_mokymas, y_mokymas)
print("✓ Modelis apmokytas su optimaliais parametrais!")

# 4. Prognozuojame ir vertiname
print("\n" + "=" * 70)
print("4. MODELIO VERTINIMAS (TESTAVIMO AIBĖ)")
print("=" * 70)

# Prognozės
y_mokymas_pred = final_knn.predict(X_mokymas)
y_testavimas_pred = final_knn.predict(X_testavimas)

# Metrikos
def spausdinti_metrikos(y_tikros, y_prognozes, rinkinys_pavadinimas):
    print(f"\n{rinkinys_pavadinimas}:")
    print(f"  Tikslumas (Accuracy): {accuracy_score(y_tikros, y_prognozes):.4f}")
    print(f"  Precizija (Precision): {precision_score(y_tikros, y_prognozes, average='weighted'):.4f}")
    print(f"  Atšaukimas (Recall): {recall_score(y_tikros, y_prognozes, average='weighted'):.4f}")
    print(f"  F1 balas: {f1_score(y_tikros, y_prognozes, average='weighted'):.4f}")

spausdinti_metrikos(y_mokymas, y_mokymas_pred, "MOKYMO AIBĖ")
spausdinti_metrikos(y_testavimas, y_testavimas_pred, "TESTAVIMO AIBĖ")

# 5. Detalus klasifikacijos ataskaita testavimo aibei
print("\n" + "=" * 70)
print("5. DETALI KLASIFIKACIJOS ATASKAITA")
print("=" * 70)
print(classification_report(
    y_testavimas,
    y_testavimas_pred,
    target_names=['Klasė 0', 'Klasė 2']
))

# 6. Painiavos matrica (Confusion Matrix)
print("\n" + "=" * 70)
print("6. PAINIAVOS MATRICA")
print("=" * 70)

cm = confusion_matrix(y_testavimas, y_testavimas_pred)
print("\nTestavimo aibė:")
print(cm)
print(f"\nTeisingai klasifikuota: {cm[0,0] + cm[1,1]}")
print(f"Klaidingai klasifikuota: {cm[0,1] + cm[1,0]}")

# Vizualizacija: Painiavos matrica
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Klasė 0', 'Klasė 2'],
    yticklabels=['Klasė 0', 'Klasė 2']
)
plt.title(f'Painiavos matrica (k-NN, k={best_k})')
plt.ylabel('Tikra klasė')
plt.xlabel('Prognozuota klasė')
plt.tight_layout()
plt.savefig(
    os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'confusion_matrix.png'),
    dpi=300
)
plt.close()

# 7. Klaidų analizė
print("\n" + "=" * 70)
print("7. KLAIDŲ ANALIZĖ")
print("=" * 70)

klaidos_maska = y_testavimas != y_testavimas_pred
klaidingi_indeksai = np.where(klaidos_maska)[0]
klaidingi_X = X_testavimas[klaidos_maska]
klaidingi_y_tikri = y_testavimas[klaidos_maska]
klaidingi_y_pred = y_testavimas_pred[klaidos_maska]

print(f"Viso klaidų: {len(klaidingi_indeksai)}")
if len(klaidingi_indeksai) > 0:
    print("\nPirmieji 10 klaidingai klasifikuotų taškų:")
    print(f"{'Indeksas':<10} | {'Tikra':<10} | {'Prognozė':<10} | {'Koordinatės'}")
    print("-" * 60)
    for i in range(min(10, len(klaidingi_indeksai))):
        print(f"{klaidingi_indeksai[i]:<10} | {klaidingi_y_tikri[i]:<10} | {klaidingi_y_pred[i]:<10} | {klaidingi_X[i]}")

# 8. Vizualizacija: Klasifikacijos rezultatai
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
nubraizti_rezultatus(axes[1], X_testavimas, y_testavimas, y_testavimas_pred, 'Testavimo aibė')

plt.suptitle(f'k-NN Klasifikacijos rezultatai (k={best_k})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(
    os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'classification_results.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.close()

print(f"\n✓ Grafikai išsaugoti '{os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA)}' direktorijoje")
print("=" * 70)