import os
import pandas as pd
import numpy as np
import json
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ---------- KONSTANTOS IR NUSTATYMAI ----------
DUOMENU_DIREKTORIJA = '../duomenys'
GRAFIKU_DIREKTORIJA = '../grafikai'
RF_DIREKTORIJA = 'RF'
JSON_DIREKTORIJA = '../JSON'
JSON_FAILAS = os.path.join(JSON_DIREKTORIJA, 'geriausias_rinkinys.json')
OPTIMAL_PARAMS_JSON = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'RF_optimal_params.json')

# Sukuriame reikiamas direktorijas
os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA), exist_ok=True)

# --- AUTOMATINIS PARAMETRU IKELIMAS IŠ TUNING REZULTATU ---
def load_optimal_params(exp_name):
    """Bandoma ikelti optimizuotus parametrus iš JSON. Jei nėra, naudoja defaults."""
    try:
        with open(OPTIMAL_PARAMS_JSON, 'r', encoding='utf-8') as f:
            params_dict = json.load(f)
            if exp_name in params_dict:
                params = params_dict[exp_name].copy()
                # Konvertuojame atgal tinkamus duomenų tipus
                if params['max_depth'] == "None":
                    params['max_depth'] = None
                else:
                    params['max_depth'] = int(params['max_depth'])
                return params
    except FileNotFoundError:
        pass

    # Numatytieji parametrai, jei tuning nebuvo atliktas
    return {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    }

# ---------- 1. DUOMENU IKELIMAS ----------
print("=" * 100)
print(" 1. DUOMENU IKELIMAS IR PARUOSIMAS ".center(100, "="))

try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
    df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')
    print(f"[OK] Duomenys ikelti. Mokymo imtis: {len(df_mokymas)} eiluciu.")
except FileNotFoundError:
    print(f"[KLAIDA] Nerasti duomenys aplanke '{DUOMENU_DIREKTORIJA}'.")
    exit()

# Isskiriame Target kintamaji (klases)
y_mokymas = df_mokymas['label'].values
y_validavimas = df_validavimas['label'].values
y_testavimas = df_testavimas['label'].values

# ---------- 2. EKSPERIMENTU APIBREZIMAS ----------

pozymiai_full = [col for col in df_mokymas.columns if col != 'label']

try:
    with open(JSON_FAILAS, 'r', encoding='utf-8') as f:
        config = json.load(f)
        pozymiai_subset = config.get("GERIAUSIAS_MODELIS_6_POZYMIAI", [])
        print(f"[OK] Ikelti geriausi pozymiai is JSON ({len(pozymiai_subset)} vnt.)")
except FileNotFoundError:
    print("[INFO] JSON nerastas. Naudojami numatytieji QRS pozymiai.")
    pozymiai_subset = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]

experiments = {
    "Vsi požymiai": pozymiai_full,
    "Optimalūs požymiai": pozymiai_subset
}

# Sąrašai duomenų kaupimui
roc_data_storage = []
cm_data_storage = []
summary_results = []
visu_eksperimentu_duomenys = []

# ---------- 3. PAGRINDINIS CIKLAS ----------

for exp_name, features in experiments.items():
    print("\n" + "#" * 100)
    print(f" VYKDOMAS EKSPERIMENTAS: {exp_name} ".center(100, "#"))

    X_mok = df_mokymas[features].values
    X_val = df_validavimas[features].values
    X_test = df_testavimas[features].values

    # Ikelti optimalius parametrus
    optimal_params = load_optimal_params(exp_name)

    print(f"\n--- Naudojami parametrai ---")
    print(f"  n_estimators: {optimal_params['n_estimators']}")
    print(f"  max_depth: {optimal_params['max_depth']}")
    print(f"  min_samples_split: {optimal_params['min_samples_split']}")
    print(f"  min_samples_leaf: {optimal_params['min_samples_leaf']}")
    print(f"  max_features: {optimal_params['max_features']}")

    # Kintamieji geriausio paieškai
    auto_best_n = optimal_params['n_estimators']
    best_val_f1 = -1
    tuning_data_table = []

    # Ciklas su didesniu diapazonu apie optimal n_estimators
    base_n = optimal_params['n_estimators']
    n_estimators_list = [max(10, base_n - 100), base_n - 50, base_n, base_n + 50, base_n + 100]
    n_estimators_list = sorted(list(set(n_estimators_list)))  # Unique ir sorted

    print(f"\n--- Validavimo su skirtingais n_estimators ---")
    for n_est in n_estimators_list:
        rf_temp = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=optimal_params['max_depth'],
            min_samples_split=optimal_params['min_samples_split'],
            min_samples_leaf=optimal_params['min_samples_leaf'],
            max_features=optimal_params['max_features'],
            random_state=42,
            n_jobs=1
        )
        rf_temp.fit(X_mok, y_mokymas)
        y_val_pred = rf_temp.predict(X_val)

        acc_val = accuracy_score(y_validavimas, y_val_pred)
        prec_val = precision_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)
        rec_val = recall_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)
        f1_val = f1_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)

        tuning_data_table.append([n_est, acc_val, prec_val, rec_val, f1_val])

        # Kaupiame duomenis grafikams
        visu_eksperimentu_duomenys.append({
            'Dataset': exp_name,
            'n_estimators': n_est,
            'Accuracy': acc_val,
            'Precision': prec_val,
            'Recall': rec_val,
            'F1 Score': f1_val
        })

        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            auto_best_n = n_est

    print(f"\nValidavimo rezultatai ({exp_name}):")
    headers = ["n_estimators", "Accuracy", "Precision", "Recall", "F1 Score"]
    print(tabulate(tuning_data_table, headers=headers, tablefmt="psql", floatfmt=".4f"))

    best_n_est = auto_best_n
    print(f"\n[BEST] Optimalus n_estimators: {best_n_est} (Validavimo F1={best_val_f1:.4f})")

    # -----------------------------------------------------------
    # GALUTINIS TESTAVIMAS SU TESTAVIMO AIBE
    # -----------------------------------------------------------
    final_rf = RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=optimal_params['max_depth'],
        min_samples_split=optimal_params['min_samples_split'],
        min_samples_leaf=optimal_params['min_samples_leaf'],
        max_features=optimal_params['max_features'],
        random_state=42,
        n_jobs=1
    )
    final_rf.fit(X_mok, y_mokymas)
    y_test_pred = final_rf.predict(X_test)

    # Metrikos
    acc = accuracy_score(y_testavimas, y_test_pred)
    prec = precision_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    rec = recall_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    f1_final = f1_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)

    summary_results.append({
        'Dataset': exp_name,
        'n_estimators': best_n_est,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1_final
    })

    # --- SVARBU: Išvedame detalią ataskaitą (Classification Report) ---
    print(f"\n>>> DETALI KLASIFIKAVIMO ATASKAITA: {exp_name} (n_estimators={best_n_est}) <<<")
    print(classification_report(y_testavimas, y_test_pred, target_names=["Normalus (0)", "Aritmija (2)"], digits=4))

    # --- SVARBU: Sumaišymo matricos skaičiai tekstui ---
    cm = confusion_matrix(y_testavimas, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f">>> KLAIDŲ ANALIZĖ: TN={tn} (Tikrai sveiki), FP={fp} (Klaidingi aliarmai), FN={fn} (Praleista liga), TP={tp} (Rasta liga)")
    print("-" * 60)

    cm_data_storage.append({
        'cm': cm,
        'title': f'{exp_name}\nn_estimators={best_n_est}'
    })

    # ROC Data
    if hasattr(final_rf, "predict_proba"):
        y_proba = final_rf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_testavimas, y_proba, pos_label=2)
        roc_auc = auc(fpr, tpr)
        roc_data_storage.append({'name': exp_name, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})

    # --- FEATURE IMPORTANCE ---
    feature_importance = final_rf.feature_importances_
    feature_names = features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Random Forest Feature Importance: {exp_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    importance_filename = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, f'Feature_Importance_{exp_name.replace(" ", "_")}.png')
    plt.savefig(importance_filename, dpi=300)
    plt.close()
    print(f"[OK] Sukurtas feature importance grafikas: {importance_filename}")

# ---------- 4. BENDRŲ GRAFIKŲ GENERAVIMAS (ROC ir CM) ----------
print("\n" + "=" * 100)
print(" 4. GENERUOJAMI BENDRI GRAFIKAI (ROC IR CM) ".center(100, "="))

# --- 4.1 BENDRAS ROC GRAFIKAS ---
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e']
for i, data in enumerate(roc_data_storage):
    plt.plot(data['fpr'], data['tpr'], color=colors[i % len(colors)], lw=3, label=f"{data['name']} (AUC = {data['auc']:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Atsitiktinis')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Kreivių Palyginimas')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_filename = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'BENDRAS_ROC_Grafikas.png')
plt.savefig(roc_filename, dpi=300)
plt.close()
print(f"[OK] Sukurtas bendras ROC grafikas: {roc_filename}")

# --- 4.2 BENDRAS SUMAIŠYMO MATRICŲ GRAFIKAS ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for i, data in enumerate(cm_data_storage):
    ax = axes[i]
    cm = data['cm']
    title = data['title']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax, annot_kws={"size": 14})
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Tikroji klasė', fontsize=10)
    ax.set_xlabel('Prognozuota klasė', fontsize=10)
    ax.set_xticklabels(['Normalus (0)', 'Aritmija (2)'])
    ax.set_yticklabels(['Normalus (0)', 'Aritmija (2)'])

plt.suptitle("Random Forest Sumaišymo Matricos Palyginimas", fontsize=16, y=1.02)
plt.tight_layout()
cm_filename = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'BENDRAS_Confusion_Matrix_Grid.png')
plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Sukurtas bendras CM grafikas (grid): {cm_filename}")


# ---------- 5. N_ESTIMATORS PRIKLAUSOMYBES GRAFIKAI ----------
print("\n" + "=" * 100)
print(" 5. GENERUOJAMA METRIKU SUVESTINE (2x2 GRID) ".center(100, "="))

df_visos_metrikos = pd.DataFrame(visu_eksperimentu_duomenys)
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
custom_palette = {"Vsi požymiai": "#1f77b4", "Optimalūs požymiai": "#ff7f0e"}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]

    sns.lineplot(
        data=df_visos_metrikos,
        x='n_estimators',
        y=metric,
        hue='Dataset',
        style='Dataset',
        markers=True,
        dashes=False,
        palette=custom_palette,
        linewidth=2.5,
        markersize=8,
        ax=ax,
        legend=(i == 0)
    )

    ax.set_title(f'{metric} priklausomybė nuo n_estimators', fontsize=12, fontweight='bold')
    ax.set_xlabel('n_estimators (Medžių skaičius)', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

plt.suptitle("Random Forest Metrikų Priklausomybė nuo n_estimators Reikšmės (Mokymo procesas)", fontsize=16, y=1.02)
plt.tight_layout()

combined_filename = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'RF_Metriku_Suvestine_Grid.png')
plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Sukurtas bendras grafikas: {combined_filename}")
print(f"\n[INFO] Visi failai issaugoti aplanke: {os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA)}")
print("=" * 100)
