import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# ---------- NUSTATYMAI ----------
RANDOM_STATE = 42
DUOMENU_DIREKTORIJA = '../duomenys'
GRAFIKU_DIREKTORIJA = '../grafikai'
DT_DIREKTORIJA = 'DecisionTree'
CV_DIREKTORIJA = os.path.join(GRAFIKU_DIREKTORIJA, DT_DIREKTORIJA, 'Cross_Validation')
JSON_DIREKTORIJA = '../JSON'
JSON_FAILAS = os.path.join(JSON_DIREKTORIJA, 'geriausias_rinkinys.json')

os.makedirs(CV_DIREKTORIJA, exist_ok=True)

DEPTH_RANGE = range(1, 21)

# ---------- 0. POŽYMIŲ NUSKAITYMAS ----------
print("=" * 100)
print(" KONFIGŪRACIJOS ĮKĖLIMAS (DT CV) ".center(100, "="))

try:
    with open(JSON_FAILAS, 'r', encoding='utf-8') as f:
        config = json.load(f)
        OPTIMALUS_POZYMIAI = config.get("GERIAUSIAS_MODELIS_6_POZYMIAI", [])
    if not OPTIMALUS_POZYMIAI:
        OPTIMALUS_POZYMIAI = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]
        print("[INFO] JSON raktas tuščias, naudojami numatytieji požymiai.")
    else:
        print(f"[OK] Įkelti požymiai iš JSON: {len(OPTIMALUS_POZYMIAI)} vnt.")
except FileNotFoundError:
    OPTIMALUS_POZYMIAI = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]
    print("[INFO] JSON nerastas, naudojami numatytieji požymiai.")

# ---------- 1. DUOMENŲ ĮKELIMAS ----------
print("-" * 100)
print(" VYKDOMAS KRYŽMINIS VALIDAVIMAS (10-Fold, Decision Tree) ".center(100, " "))

try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
except FileNotFoundError:
    print(f"[KLAIDA] Nerasti duomenų failai aplanke '{DUOMENU_DIREKTORIJA}'.")
    exit()

try:
    X_full = np.concatenate([df_mokymas[OPTIMALUS_POZYMIAI].values, df_validavimas[OPTIMALUS_POZYMIAI].values])
    y_full = np.concatenate([df_mokymas['label'].values, df_validavimas['label'].values])
except KeyError as e:
    print(f"[KLAIDA] Trūksta stulpelių: {e}")
    exit()

print(f"Bendra aibė: {X_full.shape[0]} eilučių.")

# ---------- 2. SKAIČIAVIMAI ----------
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

scoring_metrics = {
    'Accuracy': 'accuracy',
    'Precision': 'precision_weighted',
    'Recall': 'recall_weighted',
    'F1 Score': 'f1_weighted'
}

rezultatai_gyliai = []
geriausias_gylis = None
geriausias_f1 = -1
geriausio_gylio_raw = None

for depth in DEPTH_RANGE:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    results = cross_validate(dt, X_full, y_full, cv=cv_strategy, scoring=scoring_metrics)

    mean_acc = results['test_Accuracy'].mean()
    mean_prec = results['test_Precision'].mean()
    mean_rec = results['test_Recall'].mean()
    mean_f1 = results['test_F1 Score'].mean()

    rezultatai_gyliai.append({
        'Depth': depth,
        'Accuracy': mean_acc,
        'Precision': mean_prec,
        'Recall': mean_rec,
        'F1 Score': mean_f1
    })

    if mean_f1 > geriausias_f1:
        geriausias_f1 = mean_f1
        geriausias_gylis = depth
        geriausio_gylio_raw = results

# ---------- 3. SUVESTINĖ LENTELĖ ----------
df_rezultatai = pd.DataFrame(rezultatai_gyliai)
print("\n" + "=" * 100)
print(" VIDUTINĖS METRIKOS PAGAL GYLĮ ".center(100, "="))
print("=" * 100)
print(df_rezultatai.to_string(index=False, float_format="%.4f"))

print(f"\n[BEST] Geriausias gylis pagal vidutinį F1: {geriausias_gylis} (F1={geriausias_f1:.4f})")

# ---------- 4. DETALI STATISTIKA GERIAUSIAM GYLIUI ----------
print("\n" + "=" * 100)
print(f" DETALI STATISTIKA (gylys={geriausias_gylis}) ".center(100, "="))
print(f"{'METRIKA':<15} | {'VIDURKIS':<10} | {'STD (Nuokrypis)':<18} | {'VAR (Dispersija)':<18} | {'KLAIDA (Error)':<15}")
print("-" * 100)

df_best = pd.DataFrame({
    'Accuracy': geriausio_gylio_raw['test_Accuracy'],
    'Precision': geriausio_gylio_raw['test_Precision'],
    'Recall': geriausio_gylio_raw['test_Recall'],
    'F1 Score': geriausio_gylio_raw['test_F1 Score']
})

for col in df_best.columns:
    values = df_best[col]
    mean_val = values.mean()
    std_val = values.std()
    var_val = std_val ** 2
    error_val = 1.0 - mean_val
    print(f"{col:<15} | {mean_val:.4f}    | {std_val:.4f}            | {var_val:.4f}            | {error_val:.4f}")

# ---------- 5. VIZUALIZACIJA ----------
df_melted = df_best.melt(var_name='Metrika', value_name='Reikšmė')

plt.figure(figsize=(12, 7))

sns.boxplot(
    x='Metrika',
    y='Reikšmė',
    data=df_melted,
    color='white',
    width=0.5,
    linewidth=1.5,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    medianprops=dict(color='black', linewidth=2),
    flierprops=dict(marker='o', markerfacecolor='black', markersize=5, linestyle='none')
)

sns.stripplot(
    x='Metrika',
    y='Reikšmė',
    data=df_melted,
    color='black',
    size=5,
    jitter=True,
    alpha=0.6
)

plt.title(f'10-lankstų kryžminis validavimas\n(gylys={geriausias_gylis}, Požymiai: Optimalūs)', fontsize=14, fontweight='bold', color='black')
plt.ylabel('Reikšmė (0.0 - 1.0)', fontsize=12, color='black')
plt.xlabel('', color='black')
plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')
plt.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')

y_min = df_melted['Reikšmė'].min()
plt.ylim(max(0, y_min - 0.02), 1.005)

failo_pav = os.path.join(CV_DIREKTORIJA, 'DT_Kryzminis_Validavimas.png')
plt.savefig(failo_pav, dpi=300)
plt.close()

print(f"\n[OK] Kryžminio validavimo grafikas išsaugotas: {failo_pav}")
print("=" * 100)
