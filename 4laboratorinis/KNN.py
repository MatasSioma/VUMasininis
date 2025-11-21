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
    classification_report,
    roc_curve,
    auc
)

"""
=============================================================================
K-ARTIMIAUSIŲ KAIMYNŲ (k-NN) ALGORITMO DETALUS APRAŠYMAS
=============================================================================

1. ALGORITMO ESMĖ:
   k-NN (k-Nearest Neighbors) yra vienas paprasčiausių ir intuityviausių
   mašininio mokymosi algoritmų. Jis remiasi principu: "panašūs objektai
   yra arti vieni kitų požymių erdvėje".

2. ALGORITMO TIPAS:
   - Priskiriamas "tingaus mokymosi" (lazy learning) algoritmams
   - Instance-based learning algoritmas
   - Non-parametric algoritmas (nedaro prielaidų apie duomenų pasiskirstymą)

3. KAIP VEIKIA ALGORITMAS (ŽINGSNIS PO ŽINGSNIO):

   MOKYMO FAZĖ:
   - Algoritmas TIESIOG ĮSIMENA visus mokymo duomenis
   - Nėra jokių skaičiavimų ar modelio konstravimo
   - Visi duomenys saugomi atmintyje

   KLASIFIKAVIMO FAZĖ (naujiems duomenims):
   Žingsnis 1: ATSTUMO SKAIČIAVIMAS
      - Apskaičiuojamas atstumas nuo naujo taško iki VISŲ mokymo taškų
      - Dažniausiai naudojami atstumai:
        * Euklido: d = √[(x₁-x₂)² + (y₁-y₂)² + ... + (xₙ-yₙ)²]
        * Manheteno: d = |x₁-x₂| + |y₁-y₂| + ... + |xₙ-yₙ|

   Žingsnis 2: KAIMYNŲ ATRINKIMAS
      - Surūšiuojami visi taškai pagal atstumą (nuo mažiausio iki didžiausio)
      - Išrenkami k artimiausi kaimynai

   Žingsnis 3: BALSAVIMAS (VOTING)
      - Kiekvienas iš k kaimynų "balsuoja" už savo klasę
      - Naujas taškas priskiriamas daugumos klasei
      - Pvz.: jei k=5, ir 3 kaimynai yra klasės 0, o 2 klasės 2,
        naujas taškas bus priskirtas klasei 0

4. PAGRINDINIAI PARAMETRAI:

   n_neighbors (k):
   - Kiek kaimynų naudoti klasifikacijai
   - MAŽAS k (1-3):
     ✓ Privalumas: tikslus lokalus sprendimas
     ✗ Trūkumas: jautrus triukšmui, gali būti overfitting
   - DIDELIS k (>20):
     ✓ Privalumas: stabilesni rezultatai, mažiau triukšmo įtakos
     ✗ Trūkumas: gali prarasti detales, underfitting
   - REKOMENDUACIJA: dažniausiai renkamasi nelyginis skaičius
     (pvz., 3, 5, 7, 9), kad išvengtų lygiųjų balsavime

   metric (atstumo metrika):
   - 'euclidean': tiesus atstumas, labiausiai paplitęs
   - 'manhattan': "miesto kvartalų" atstumas
   - 'minkowski': generalizuotas, su parametru p

   weights (kaimynų svoriai):
   - 'uniform': visi k kaimynai turi vienodą balsą
   - 'distance': artimesni kaimynai turi didesnį svorį
     (svoris = 1/atstumas)

5. ALGORITMO MATEMATINĖ IŠRAIŠKA:

   Klasė(x_naujas) = argmax Σ(w_i * I(y_i = c))
                     c∈C  i∈Nk(x)

   Kur:
   - x_naujas: naujas klasifikuojamas taškas
   - Nk(x): k artimiausių kaimynų aibė
   - w_i: kaimyno svoris (1 jei uniform, 1/d_i jei distance)
   - I(y_i = c): indikatorinė funkcija (1 jei kaimynas klasės c, 0 kitaip)
   - C: visų klasių aibė

6. PRIVALUMAI:
   ✓ Labai paprastas suprasti ir implementuoti
   ✓ Nereikia mokymo fazės (greitas "mokymas")
   ✓ Efektyvus su neliniškais duomenimis
   ✓ Gali būti naudojamas klasifikacijai IR regresijai
   ✓ Lengvai prisitaiko prie naujų duomenų
   ✓ Neparametrinis - nedaro prielaidų apie duomenų pasiskirstymą

7. TRŪKUMAI:
   ✗ Lėtas klasifikuojant (O(n*d) kur n-objektų skaičius, d-dimensijos)
   ✗ Didelės atminties sąnaudos (reikia saugoti visus mokymo duomenis)
   ✗ Labai jautrus požymių skalei (BŪTINA normalizacija!)
   ✗ Neefektyvus didelėms duomenų aibėms
   ✗ Jautrus irrelevantiems požymiams (curse of dimensionality)
   ✗ Nepasakys, kurie požymiai svarbiausi (no feature importance)

8. KADA NAUDOTI k-NN:
   ✓ Mažos-vidutinės apimties duomenų aibės
   ✓ Kai turime nedaug požymių (iki ~20)
   ✓ Kai duomenų klasės persidengiantys arba sudėtingai išsibarstę
   ✓ Kai reikia paprasto baseline modelio
   ✓ Kai duomenys nėra labai triukšmingi

9. TAIKYMAS ŠIEMS DUOMENIMS:
   - Turime 2D t-SNE duomenis (tik 2 požymiai) ✓
   - Duomenys normuoti [0,1] intervale ✓
   - 2 klasės (0 ir 2) - binarine klasifikacija ✓
   - Klasės nesubalansuotos (96% vs 4%) - reikia atsargumo
   - k-NN puikiai tinka tokiai žemadimensei problemai

10. KLASIFIKAVIMO PROCESO SCHEMA:

    [Mokymo duomenys] → [Saugojimas atmintyje]
                              ↓
    [Naujas taškas] → [Atstumo skaičiavimas] → [k artimiausių]
                              ↓                      ↓
                        [Balsavimas] → [Klasės priskyrimas]

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
print("=" * 80)
print(" 1. DUOMENŲ ĮKĖLIMAS IR PARUOŠIMAS ".center(80, "="))
print("=" * 80)

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

print(f"✓ Mokymo aibė: {X_mokymas.shape[0]} įrašų, {X_mokymas.shape[1]} požymiai")
print(f"✓ Validavimo aibė: {X_validavimas.shape[0]} įrašų")
print(f"✓ Testavimo aibė: {X_testavimas.shape[0]} įrašų")

# 3. Hiperparametrų parinkimas (Tuning)
print("\n" + "=" * 80)
print(" 3. HIPERPARAMETRŲ PARINKIMAS (k REIKŠMĖS) ".center(80, "="))
print("=" * 80)

print("\nTiriamos k reikšmės: nuo 1 iki 21 (nelyginiai skaičiai)")
print("Tikslas: rasti optimalų k, kuris duotų geriausią F1 balą validavimo aibėje")
print()

k_values = range(1, 22, 2)  # 1, 3, 5, ..., 21
results = []

print(f"{'k':<5} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
print("-" * 60)

best_k = -1
best_f1 = -1
best_accuracy = -1

for k in k_values:
    # Sukuriame k-NN klasifikatorių su dabartine k reikšme
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='manhattan',  # Euklido atstumas
        weights='uniform'     # Visi kaimynai turi vienodą svorį
    )

    # Apmokiname modelį
    knn.fit(X_mokymas, y_mokymas)

    # Prognozuojame validavimo aibę
    y_val_pred = knn.predict(X_validavimas)

    # Skaičiuojame metrikos
    acc = accuracy_score(y_validavimas, y_val_pred)
    prec = precision_score(y_validavimas, y_val_pred, average='weighted')
    rec = recall_score(y_validavimas, y_val_pred, average='weighted')
    f1 = f1_score(y_validavimas, y_val_pred, average='weighted')

    results.append({
        'k': k,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })

    print(f"{k:<5} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

    # Saugome geriausią k
    if f1 > best_f1:
        best_f1 = f1
        best_k = k
        best_accuracy = acc

print("-" * 60)
print(f"\n🏆 GERIAUSIAS k = {best_k}")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"   F1 Score: {best_f1:.4f}")

# Vizualizuojame parametrų paiešką
results_df = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axes[0, 0].plot(results_df['k'], results_df['accuracy'], marker='o', linewidth=2)
axes[0, 0].axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
axes[0, 0].set_xlabel('Kaimynų skaičius (k)')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Tikslumo (Accuracy) priklausomybė nuo k')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(k_values)

# Precision
axes[0, 1].plot(results_df['k'], results_df['precision'], marker='s', color='green', linewidth=2)
axes[0, 1].axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
axes[0, 1].set_xlabel('Kaimynų skaičius (k)')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precizijos (Precision) priklausomybė nuo k')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(k_values)

# Recall
axes[1, 0].plot(results_df['k'], results_df['recall'], marker='^', color='orange', linewidth=2)
axes[1, 0].axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
axes[1, 0].set_xlabel('Kaimynų skaičius (k)')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].set_title('Atšaukimo (Recall) priklausomybė nuo k')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(k_values)

# F1 Score
axes[1, 1].plot(results_df['k'], results_df['f1'], marker='D', color='purple', linewidth=2)
axes[1, 1].axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
axes[1, 1].set_xlabel('Kaimynų skaičius (k)')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('F1 balo priklausomybė nuo k')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(k_values)

plt.suptitle('k-NN Hiperparametrų įtaka klasifikavimo kokybei (Validavimo aibė)',
             fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(
    os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'parameter_tuning_detailed.png'),
    dpi=300
)
plt.close()

# Visos metrikos viename grafike
plt.figure(figsize=(12, 6))
plt.plot(results_df['k'], results_df['accuracy'], marker='o', label='Accuracy', linewidth=2)
plt.plot(results_df['k'], results_df['precision'], marker='s', label='Precision', linewidth=2)
plt.plot(results_df['k'], results_df['recall'], marker='^', label='Recall', linewidth=2)
plt.plot(results_df['k'], results_df['f1'], marker='D', label='F1 Score', linewidth=2)
plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
plt.xlabel('Kaimynų skaičius (k)', fontsize=12)
plt.ylabel('Metrikos reikšmė', fontsize=12)
plt.title('k-NN: Visų metrikų palyginimas', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.savefig(
    os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'all_metrics_comparison.png'),
    dpi=300
)
plt.close()

print("✓ Parametrų parinkimo grafikai išsaugoti")

# 4. Galutinis modelio mokymas
print("\n" + "=" * 80)
print(f" 4. GALUTINIO MODELIO MOKYMAS (k={best_k}) ".center(80, "="))
print("=" * 80)

final_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric='euclidean',
    weights='uniform'
)
final_knn.fit(X_mokymas, y_mokymas)

print(f"✓ Modelis sėkmingai apmokytas!")
print(f"  Parametrai:")
print(f"    - n_neighbors (k): {best_k}")
print(f"    - metric: euclidean")
print(f"    - weights: uniform")
print(f"  Mokymo duomenų kiekis: {len(X_mokymas)}")

# 5. Prognozuojame ir vertiname
print("\n" + "=" * 80)
print(" 5. MODELIO VERTINIMAS ".center(80, "="))
print("=" * 80)

# Prognozės
y_mokymas_pred = final_knn.predict(X_mokymas)
y_validavimas_pred = final_knn.predict(X_validavimas)
y_testavimas_pred = final_knn.predict(X_testavimas)

# Funkcija metrikoms spausdinti
def spausdinti_metrikos(y_tikros, y_prognozes, rinkinys_pavadinimas):
    acc = accuracy_score(y_tikros, y_prognozes)
    prec = precision_score(y_tikros, y_prognozes, average='weighted')
    rec = recall_score(y_tikros, y_prognozes, average='weighted')
    f1 = f1_score(y_tikros, y_prognozes, average='weighted')

    print(f"\n{'─'*50}")
    print(f"{rinkinys_pavadinimas:^50}")
    print(f"{'─'*50}")
    print(f"  Tikslumas (Accuracy):   {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precizija (Precision):  {prec:.4f}")
    print(f"  Atšaukimas (Recall):    {rec:.4f}")
    print(f"  F1 balas:               {f1:.4f}")
    print(f"{'─'*50}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

metrikos_mokymas = spausdinti_metrikos(y_mokymas, y_mokymas_pred, "MOKYMO AIBĖ")
metrikos_validavimas = spausdinti_metrikos(y_validavimas, y_validavimas_pred, "VALIDAVIMO AIBĖ")
metrikos_testavimas = spausdinti_metrikos(y_testavimas, y_testavimas_pred, "TESTAVIMO AIBĖ")

# Palyginimo lentelė
print("\n" + "=" * 80)
print(" METRIKŲ PALYGINIMAS ".center(80, "="))
print("=" * 80)

comparison_df = pd.DataFrame({
    'Mokymo aibė': metrikos_mokymas,
    'Validavimo aibė': metrikos_validavimas,
    'Testavimo aibė': metrikos_testavimas
})

print(comparison_df.to_string())

# Vizualizuojame metrikas
fig, ax = plt.subplots(figsize=(10, 6))
comparison_df.T.plot(kind='bar', ax=ax)
ax.set_title('k-NN Klasifikavimo metrikų palyginimas', fontsize=14)
ax.set_xlabel('Duomenų rinkinys', fontsize=12)
ax.set_ylabel('Metrikos reikšmė', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Metrika', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])
plt.tight_layout()
plt.savefig(
    os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'metrics_comparison.png'),
    dpi=300
)
plt.close()

# 6. Detalus klasifikacijos ataskaita
print("\n" + "=" * 80)
print(" 6. DETALI KLASIFIKACIJOS ATASKAITA (TESTAVIMO AIBĖ) ".center(80, "="))
print("=" * 80)

print(classification_report(
    y_testavimas,
    y_testavimas_pred,
    target_names=['Klasė 0 (Normalūs)', 'Klasė 2 (Aritmija)'],
    digits=4
))

# 7. Painiavos matrica (Confusion Matrix)
print("\n" + "=" * 80)
print(" 7. PAINIAVOS MATRICA ".center(80, "="))
print("=" * 80)

# Skaičiuojame painiavos matricas
cm_mokymas = confusion_matrix(y_mokymas, y_mokymas_pred)
cm_validavimas = confusion_matrix(y_validavimas, y_validavimas_pred)
cm_testavimas = confusion_matrix(y_testavimas, y_testavimas_pred)

# Spausdiname testavimo matricos detales
print("\nTestavimo aibės painiavos matrica:")
print(f"\n{'':>15} | {'Prognozė 0':>12} | {'Prognozė 2':>12}")
print("-" * 45)
print(f"{'Tikroji 0':>15} | {cm_testavimas[0,0]:>12} | {cm_testavimas[0,1]:>12}")
print(f"{'Tikroji 2':>15} | {cm_testavimas[1,0]:>12} | {cm_testavimas[1,1]:>12}")

print(f"\nTeisingai klasifikuota: {cm_testavimas[0,0] + cm_testavimas[1,1]} "
      f"({((cm_testavimas[0,0] + cm_testavimas[1,1])/len(y_testavimas)*100):.2f}%)")
print(f"Klaidingai klasifikuota: {cm_testavimas[0,1] + cm_testavimas[1,0]} "
      f"({((cm_testavimas[0,1] + cm_testavimas[1,0])/len(y_testavimas)*100):.2f}%)")

# Vizualizuojame painiavos matricas
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (cm, title) in enumerate([
    (cm_mokymas, 'Mokymo aibė'),
    (cm_validavimas, 'Validavimo aibė'),
    (cm_testavimas, 'Testavimo aibė')
]):
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Klasė 0', 'Klasė 2'],
        yticklabels=['Klasė 0', 'Klasė 2'],
        ax=axes[idx],
        cbar_kws={'label': 'Kiekis'}
    )
    axes[idx].set_title(f'{title}\n(Accuracy: {accuracy_score(
        [y_mokymas, y_validavimas, y_testavimas][idx],
        [y_mokymas_pred, y_validavimas_pred, y_testavimas_pred][idx]
    ):.4f})')
    axes[idx].set_ylabel('Tikra klasė')
    axes[idx].set_xlabel('Prognozuota klasė')

plt.suptitle(f'Painiavos matricos (k-NN, k={best_k})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(
    os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'confusion_matrices.png'),
    dpi=300
)
plt.close()

print("✓ Painiavos matricos vizualizacija išsaugota")

# 8. Klaidų analizė
print("\n" + "=" * 80)
print(" 8. KLAIDŲ ANALIZĖ ".center(80, "="))
print("=" * 80)

klaidos_maska = y_testavimas != y_testavimas_pred
klaidingi_indeksai = np.where(klaidos_maska)[0]
klaidingi_X = X_testavimas[klaidos_maska]
klaidingi_y_tikri = y_testavimas[klaidos_maska]
klaidingi_y_pred = y_testavimas

print("\n" + "="*80)
print(" 9. KITŲ ALGORITMŲ PALYGINIMAS: NB, TREE, RF ".center(80, "="))
print("="*80)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ALG_DIREKTORIJA = os.path.join(GRAFIKU_DIREKTORIJA, "KITIALGORITMAI")
os.makedirs(ALG_DIREKTORIJA, exist_ok=True)

# 1. ALGORITMŲ SĄRAŠAS
algoritmai = {
    "KNN": final_knn,
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=RANDOM_STATE
    )
}

# 2. METRIKŲ IR ROC REIKŠMIŲ TALPYKLA
rezultatai = {}
roc_duomenys = {}

# 3. VISŲ ALGORITMŲ TRENIRAVIMAS IR VERTINIMAS
for pavadinimas, modelis in algoritmai.items():
    print("\n" + "-"*60)
    print(f" ALGORITMAS: {pavadinimas}".center(60))
    print("-"*60)

    # Apmokymas
    modelis.fit(X_mokymas, y_mokymas)

    # Prognozės
    y_pred = modelis.predict(X_testavimas)

    # Metrikos
    acc = accuracy_score(y_testavimas, y_pred)
    prec = precision_score(y_testavimas, y_pred, average="weighted")
    rec = recall_score(y_testavimas, y_pred, average="weighted")
    f1 = f1_score(y_testavimas, y_pred, average="weighted")

    rezultatai[pavadinimas] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    # Painiavos matrica
    cm = confusion_matrix(y_testavimas, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=["0", "2"], yticklabels=["0", "2"])
    plt.title(f"{pavadinimas} Painiavos matrica")
    plt.xlabel("Prognozuota")
    plt.ylabel("Tikra")
    plt.tight_layout()
    plt.savefig(os.path.join(ALG_DIREKTORIJA, f"{pavadinimas}_confusion.png"))
    plt.close()

    # ROC (tik tiems, kurie turi predict_proba)
    if hasattr(modelis, "predict_proba"):
        y_proba = modelis.predict_proba(X_testavimas)[:, 1]
        fpr, tpr, _ = roc_curve(y_testavimas, y_proba, pos_label=2)
        roc_duomenys[pavadinimas] = (fpr, tpr)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")


print("\n" + "="*80)
print(" 9–11 ŽINGSNIAI: ATSITIKTINĖ POŽYMIŲ PERMUTACIJA ".center(80, "="))
print("="*80)

from copy import deepcopy

# 9 žingsnis – atsitiktinai sumaišyti požymių reikšmes
def permutuoti_pozymius(X):
    X_permutuotas = deepcopy(X)
    for col in range(X.shape[1]):
        np.random.shuffle(X_permutuotas[:, col])
    return X_permutuotas

X_mokymas_perm = permutuoti_pozymius(X_mokymas)
X_validavimas_perm = permutuoti_pozymius(X_validavimas)
X_testavimas_perm = permutuoti_pozymius(X_testavimas)

# 10 žingsnis – treniruoti modelį naujoje permutuotoje erdvėje
knn_perm = KNeighborsClassifier(
    n_neighbors=best_k,
    metric='euclidean',
    weights='uniform'
)
knn_perm.fit(X_mokymas_perm, y_mokymas)

# 11 žingsnis – įvertinti permutuoto modelio veikimą
y_test_pred_perm = knn_perm.predict(X_testavimas_perm)

acc_perm = accuracy_score(y_testavimas, y_test_pred_perm)
prec_perm = precision_score(y_testavimas, y_test_pred_perm, average='weighted')
rec_perm = recall_score(y_testavimas, y_test_pred_perm, average='weighted')
f1_perm = f1_score(y_testavimas, y_test_pred_perm, average='weighted')

print("\nRezultatai po atsitiktinės požymių permutacijos:")
print(f"  Accuracy:  {acc_perm:.4f}")
print(f"  Precision: {prec_perm:.4f}")
print(f"  Recall:    {rec_perm:.4f}")
print(f"  F1-score:  {f1_perm:.4f}")

print("\n--- PALYGINIMAS (originalus vs permutuotas) ---")
print(f"Originalus F1:   {metrikos_testavimas['f1']:.4f}")
print(f"Permutuoto F1:   {f1_perm:.4f}")

if f1_perm < metrikos_testavimas['f1'] * 0.5:
    print("\n✓ Klasifikatorius teisingai naudoja požymius — permutacija smarkiai pablogino rezultatus.")
else:
    print("\n⚠️ DĖMESIO: permutacija smarkiai nesumažino veikimo — požymiai gali būti neinformatyvūs.")

print("\n" + "="*80)
print(" BENDRA ROC/AUC DIAGRAMA VISIEMS ALGORITMAMS ".center(80, "="))
print("="*80)

plt.figure(figsize=(8, 6))

auc_lentele = {}

for alg, (fpr, tpr) in roc_duomenys.items():
    auc_val = auc(fpr, tpr)
    auc_lentele[alg] = auc_val
    plt.plot(fpr, tpr, linewidth=2, label=f"{alg} (AUC={auc_val:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess")

plt.title("ROC kreivės palyginimas (KNN, NB, Tree, RF)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.savefig(os.path.join(ALG_DIREKTORIJA, "ROC_ALL_MODELS.png"), dpi=300)
plt.close()

print("✓ Bendra ROC kreivė sugeneruota")
print("\nAUC reikšmės:")
for alg, val in auc_lentele.items():
    print(f"  {alg}: {val:.4f}")

