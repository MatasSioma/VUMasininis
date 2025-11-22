import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ---------- KONSTANTOS ----------
RANDOM_STATE = 42
DUOMENU_DIREKTORIJA = 'duomenys'
GRAFIKU_DIREKTORIJA = 'grafikai'
RF_PAGRINDINE_DIR = 'RF_eksperimentai'  # Pakeistas aplanko pavadinimas
JSON_DIREKTORIJA = 'JSON'
JSON_FAILAS = 'pozymiu_rinkiniai.json'

JSON_FAILAS_PATH = os.path.join(JSON_DIREKTORIJA, JSON_FAILAS)

# Sukuriame pagrindinę direktoriją rezultatams
BASE_OUTPUT_DIR = os.path.join(GRAFIKU_DIREKTORIJA, RF_PAGRINDINE_DIR)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ---------- 1. ĮKELIAME DUOMENIS ----------
print("=" * 80)
print(" 1. DUOMENŲ ĮKĖLIMAS (RANDOM FOREST) ".center(80, "="))
print("=" * 80)

try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
    df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')
except FileNotFoundError:
    print(f"KLAIDA: Nerasti duomenų failai kataloge '{DUOMENU_DIREKTORIJA}'.")
    exit()

# ---------- 2. ĮKELIAME JSON KONFIGŪRACIJĄ ----------
try:
    with open(JSON_FAILAS_PATH, 'r', encoding='utf-8') as f:
        eksperimentai = json.load(f)
    print(f"✓ Rastas JSON failas. Įkelti {len(eksperimentai)} eksperimentų rinkiniai.")
except FileNotFoundError:
    print(f"KLAIDA: Nerastas '{JSON_FAILAS_PATH}'. Pirmiausia sugeneruokite jį.")
    exit()

visu_eksperimentu_rezultatai = []
roc_duomenys_bendrai = []

# ---------- 3. CIKLAS PER VISUS EKSPERIMENTUS ----------
print("\nPRADEDAMAS CIKLAS PER EKSPERIMENTUS...")

for eksp_pavadinimas, pozymiai in eksperimentai.items():
    SAVE_DIR = os.path.join(BASE_OUTPUT_DIR, eksp_pavadinimas)
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        X_mok = df_mokymas[pozymiai].values
        y_mok = df_mokymas['label'].values
        X_val = df_validavimas[pozymiai].values
        y_val = df_validavimas['label'].values
        X_test = df_testavimas[pozymiai].values
        y_test = df_testavimas['label'].values
    except KeyError as e:
        print(f"  [!] KLAIDA: Požymis {e} nerastas. Praleidžiamas '{eksp_pavadinimas}'.")
        continue

    # --- 3.3. Hiperparametrų parinkimas (RF: n_estimators) ---
    best_n = -1
    best_val_f1 = -1
    tuning_data = []

    # Tikriname skirtingą medžių skaičių
    n_estimators_list = [10, 25, 50, 100, 200, 300]

    for n in n_estimators_list:
        rf = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_mok, y_mok)
        y_val_pred = rf.predict(X_val)

        f1_val = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        acc_val = accuracy_score(y_val, y_val_pred)

        tuning_data.append({'n_estimators': n, 'f1': f1_val, 'accuracy': acc_val})

        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            best_n = n

    # Tuning grafikas
    td_df = pd.DataFrame(tuning_data)
    plt.figure(figsize=(8, 5))
    plt.plot(td_df['n_estimators'], td_df['f1'], marker='o', label='F1 Score')
    plt.plot(td_df['n_estimators'], td_df['accuracy'], marker='s', linestyle='--', label='Accuracy')
    plt.axvline(x=best_n, color='r', linestyle=':', label=f'Best N={best_n}')
    plt.title(f'Hiperparametrų parinkimas: {eksp_pavadinimas}')
    plt.xlabel('Medžių skaičius (n_estimators)')
    plt.ylabel('Metrika')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'tuning_results.png'), dpi=300)
    plt.close()

    # --- 3.4. Galutinis modelis ---
    final_model = RandomForestClassifier(n_estimators=best_n, random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(X_mok, y_mok)
    y_test_pred = final_model.predict(X_test)

    # --- 3.5. Metrikos ---
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    visu_eksperimentu_rezultatai.append([
        eksp_pavadinimas, len(pozymiai), best_n, acc, prec, rec, f1
    ])

    print(f"  ✓ Atlikta: {eksp_pavadinimas} (N={best_n}, F1={f1:.4f})")

    # --- 3.6. Painiavos matrica ---
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False) # Pakeičiau spalvą į Greens
    plt.title(f'CM: {eksp_pavadinimas}\n(Acc: {acc:.4f})')
    plt.xlabel('Prognozuota')
    plt.ylabel('Tikra')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # --- 3.7. ROC ir AUC ---
    if hasattr(final_model, "predict_proba"):
        y_proba = final_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=2)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='green', lw=2, label=f'AUC={roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(f'ROC: {eksp_pavadinimas}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, 'roc_curve.png'), dpi=300)
        plt.close()

        roc_duomenys_bendrai.append({
            'label': f"{eksp_pavadinimas} (AUC={roc_auc:.3f})",
            'fpr': fpr, 'tpr': tpr, 'auc': roc_auc
        })

# ---------- 4. REZULTATŲ SUVESTINĖ ----------
df_rezultatai = pd.DataFrame(
    visu_eksperimentu_rezultatai,
    columns=['Eksperimentas', 'Požymių sk.', 'Best N Trees', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
)
df_rezultatai = df_rezultatai.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)

print("\n" + "=" * 100)
print(" RF GALUTINIAI REZULTATAI ".center(100, "="))
print("=" * 100)
print(tabulate(df_rezultatai, headers='keys', tablefmt='psql', floatfmt=".4f", showindex=False))

# F1 grafikas
plt.figure(figsize=(12, 8))
sns.barplot(x='F1 Score', y='Eksperimentas', data=df_rezultatai, palette='viridis')
plt.title('Random Forest: F1 balų palyginimas')
plt.xlabel('F1 Score')
plt.xlim(0, 1.05)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'f1_score_summary.png'), dpi=300)
plt.close()

# ROC Combined
if roc_duomenys_bendrai:
    plt.figure(figsize=(10, 8))
    roc_duomenys_bendrai.sort(key=lambda x: x['auc'], reverse=True)
    for data in roc_duomenys_bendrai:
        plt.plot(data['fpr'], data['tpr'], lw=2, label=data['label'])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Visų eksperimentų ROC (Random Forest)')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'roc_curves_combined.png'), dpi=300)
    plt.close()

print(f"\n✓ Visi RF rezultatai išsaugoti: {BASE_OUTPUT_DIR}")