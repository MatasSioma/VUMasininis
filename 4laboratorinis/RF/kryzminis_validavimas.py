import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

# ---------- NUSTATYMAI ----------
DUOMENU_DIREKTORIJA = '../duomenys'
GRAFIKU_DIREKTORIJA = '../grafikai'
RF_DIREKTORIJA = 'RF'
CV_DIREKTORIJA = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'Cross_Validation')
JSON_DIREKTORIJA = '../JSON'
JSON_FAILAS = os.path.join(JSON_DIREKTORIJA, 'geriausias_rinkinys.json')
OPTIMAL_PARAMS_JSON = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'RF_optimal_params.json')

os.makedirs(CV_DIREKTORIJA, exist_ok=True)

# ---------- 0. OPTIMALIŲ PARAMETRŲ NUSKAITYMAS ----------
print("=" * 100)
print(" KONFIGŪRACIJOS ĮKĖLIMAS ".center(100, "="))

def load_optimal_params(exp_name):
    """Bandoma ikelti optimizuotus parametrus iš JSON. Jei nėra, naudoja defaults."""
    try:
        with open(OPTIMAL_PARAMS_JSON, 'r', encoding='utf-8') as f:
            params_dict = json.load(f)
            if exp_name in params_dict:
                params = params_dict[exp_name].copy()
                # Konvertuojame atgal tinkamus duomenų tipus
                if params.get('max_depth') == "None":
                    params['max_depth'] = None
                else:
                    params['max_depth'] = int(params.get('max_depth', 10))
                return params
    except FileNotFoundError:
        pass

    # Numatytieji parametrai
    return {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    }

optimal_params = load_optimal_params("Vsi požymiai")
print(f"[OK] Optimalūs parametrai įkelti:")
print(f"  n_estimators: {optimal_params['n_estimators']}")
print(f"  max_depth: {optimal_params['max_depth']}")
print(f"  min_samples_split: {optimal_params['min_samples_split']}")
print(f"  min_samples_leaf: {optimal_params['min_samples_leaf']}")
print(f"  max_features: {optimal_params['max_features']}")

# ---------- 1. DUOMENŲ ĮKELIMAS ----------
print("-" * 100)
print(" VYKDOMAS KRYŽMINIS VALIDAVIMAS (10-Fold) ".center(100, " "))

try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
except FileNotFoundError:
    print(f"[KLAIDA] Nerasti duomenų failai aplanke '{DUOMENU_DIREKTORIJA}'.")
    exit()

# Naudojame visus požymius (full dataset)
visi_pozymiai = [col for col in df_mokymas.columns if col != 'label']

try:
    X_full = np.concatenate([df_mokymas[visi_pozymiai].values, df_validavimas[visi_pozymiai].values])
    y_full = np.concatenate([df_mokymas['label'].values, df_validavimas['label'].values])
except KeyError as e:
    print(f"[KLAIDA] Trūksta stulpelių: {e}")
    exit()

print(f"Bendra aibė: {X_full.shape[0]} eilučių, {len(visi_pozymiai)} požymiai.")

# ---------- 2. SKAIČIAVIMAI ----------
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf = RandomForestClassifier(
    n_estimators=optimal_params['n_estimators'],
    max_depth=optimal_params['max_depth'],
    min_samples_split=optimal_params['min_samples_split'],
    min_samples_leaf=optimal_params['min_samples_leaf'],
    max_features=optimal_params['max_features'],
    random_state=42,
    n_jobs=1
)

scoring_metrics = {
    'Accuracy': 'accuracy',
    'Precision': 'precision_weighted',
    'Recall': 'recall_weighted',
    'F1 Score': 'f1_weighted'
}

# Vykdome validavimą
results = cross_validate(rf, X_full, y_full, cv=cv_strategy, scoring=scoring_metrics)

# Konvertuojame į DataFrame
df_results = pd.DataFrame({
    'Accuracy': results['test_Accuracy'],
    'Precision': results['test_Precision'],
    'Recall': results['test_Recall'],
    'F1 Score': results['test_F1 Score']
})

# ---------- 3. DETALI STATISTIKA (Mean, Std, Var, Error) ----------
print("\n" + "=" * 100)
print(f" DETALI STATISTIKA (Atsitiktinio medžio, visų požymiai) ".center(100, "="))
print(f"{'METRIKA':<15} | {'VIDURKIS':<10} | {'STD (Nuokrypis)':<18} | {'VAR (Dispersija)':<18} | {'KLAIDA (Error)':<15}")
print("-" * 100)

for col in df_results.columns:
    values = df_results[col]

    mean_val = values.mean()
    std_val = values.std()
    var_val = std_val ** 2  # Dispersija yra standatinio nuokrypio kvadratas
    error_val = 1.0 - mean_val # Klaida yra 1 minus vidurkis

    print(f"{col:<15} | {mean_val:.4f}    | {std_val:.4f}            | {var_val:.4f}            | {error_val:.4f}")

print("=" * 100)

# ---------- 4. VIZUALIZACIJA (JUODAI-BALTA) ----------
df_melted = df_results.melt(var_name='Metrika', value_name='Reikšmė')

plt.figure(figsize=(12, 7))

# Boxplot su juodais rėmeliais ir baltu vidumi
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

# Stripplot (taškai)
sns.stripplot(
    x='Metrika',
    y='Reikšmė',
    data=df_melted,
    color='black',
    size=5,
    jitter=True,
    alpha=0.6
)

plt.title(f'10-lanksčio kryžminio validavimo rezultatai\nAtsitiktinio miško modelis (Visi požymiai)', fontsize=14, fontweight='bold', color='black')
plt.ylabel('Reikšmė (0.0 - 1.0)', fontsize=12, color='black')
plt.xlabel('', color='black')

plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')
plt.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')

# Ribos
y_min = df_melted['Reikšmė'].min()
plt.ylim(max(0, y_min - 0.02), 1.005)

failo_pav = os.path.join(CV_DIREKTORIJA, 'Kryzminis_Validavimas.png')
plt.savefig(failo_pav, dpi=300)
plt.close()

print(f"\n[OK] Juodai-baltas grafikas išsaugotas: {failo_pav}")
print("=" * 100)
