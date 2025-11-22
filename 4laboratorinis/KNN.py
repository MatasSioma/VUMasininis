import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ---------- KONSTANTOS ----------
RANDOM_STATE = 42
DUOMENU_DIREKTORIJA = 'duomenys'
GRAFIKU_DIREKTORIJA = 'grafikai'
KNN_DIREKTORIJA = 'KNN_Comparison' # Pakeičiau aplanką, kad nesimaišytų su senais

# Sukuriame grafikai/KNN_Comparison direktoriją
os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA), exist_ok=True)

# ---------- 1. ĮKELIAME DUOMENIS ----------
print("=" * 80)
print(" 1. DUOMENŲ ĮKĖLIMAS ".center(80, "="))
print("=" * 80)

try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
    df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')
except FileNotFoundError:
    # Fallback
    DUOMENU_DIREKTORIJA = '4laboratorinis/duomenys'
    GRAFIKU_DIREKTORIJA = '4laboratorinis/grafikai'
    os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA), exist_ok=True)
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
    df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')

# Išskiriame y (klases) - jos vienodos visiems eksperimentams
y_mokymas = df_mokymas['label'].values
y_validavimas = df_validavimas['label'].values
y_testavimas = df_testavimas['label'].values

# Apibrėžiame du eksperimentus: Pilną ir Sumažintą
pozymiai_subset = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]
pozymiai_full = [col for col in df_mokymas.columns if col != 'label']

experiments = {
    "FULL_DATASET": pozymiai_full,
    "SUBSET_BEST": pozymiai_subset
}

# Čia kaupsime duomenis bendram ROC grafikui
roc_data_storage = []

# ---------- 2. PAGRINDINIS CIKLAS PER EKSPERIMENTUS ----------

for exp_name, features in experiments.items():
    print("\n" + "#" * 80)
    print(f" VYKDOMAS EKSPERIMENTAS: {exp_name} ".center(80, "#"))
    print("#" * 80)
    print(f"Naudojami požymiai ({len(features)}): {features}")

    # Paruošiame X matricas šiam eksperimentui
    X_mok = df_mokymas[features].values
    X_val = df_validavimas[features].values
    X_test = df_testavimas[features].values

    # --- A. HIPERPARAMETRŲ PARINKIMAS ---
    print(f"\n--- {exp_name}: Hiperparametrų parinkimas ---")

    best_k = -1
    best_val_f1 = -1
    tuning_results = []
    k_values = range(1, 22, 2)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
        knn.fit(X_mok, y_mokymas)
        y_val_pred = knn.predict(X_val)

        f1 = f1_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)
        acc = accuracy_score(y_validavimas, y_val_pred)

        tuning_results.append({'k': k, 'f1': f1, 'acc': acc})

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_k = k

    print(f"🏆 Geriausias k eksperimentui '{exp_name}': {best_k} (F1={best_val_f1:.4f})")

    # Išsaugome tuning grafiką
    tr_df = pd.DataFrame(tuning_results)
    plt.figure(figsize=(8, 5))
    plt.plot(tr_df['k'], tr_df['f1'], marker='o', label='F1 Score')
    plt.plot(tr_df['k'], tr_df['acc'], marker='s', linestyle='--', label='Accuracy')
    plt.axvline(x=best_k, color='r', linestyle=':', label=f'Best k={best_k}')
    plt.title(f'Tuning: {exp_name}')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, f'{exp_name}_tuning.png'), dpi=300)
    plt.close()

    # --- B. GALUTINIS MODELIS ---
    final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='uniform')
    final_knn.fit(X_mok, y_mokymas)

    # --- C. PROGNOZAVIMAS IR METRIKOS ---
    y_test_pred = final_knn.predict(X_test)

    acc = accuracy_score(y_testavimas, y_test_pred)
    prec = precision_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    rec = recall_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)

    print(f"\nREZULTATAI ({exp_name}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # --- D. PAINIAVOS MATRICA (CM) ---
    cm = confusion_matrix(y_testavimas, y_test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'CM: {exp_name}\n(Acc: {acc:.4f})')
    plt.ylabel('Tikra')
    plt.xlabel('Prognozuota')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, f'{exp_name}_confusion_matrix.png'), dpi=300)
    plt.close()

    # --- E. DUOMENŲ PARUOŠIMAS BENDRAM ROC ---
    if hasattr(final_knn, "predict_proba"):
        y_proba = final_knn.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_testavimas, y_proba, pos_label=2)
        roc_auc = auc(fpr, tpr)

        roc_data_storage.append({
            'name': exp_name,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        })
        print(f"  -> ROC duomenys išsaugoti (AUC={roc_auc:.4f})")

# ---------- 3. BENDRAS ROC GRAFIKAS ----------
print("\n" + "=" * 80)
print(" 3. GENERUOJAMAS BENDRAS ROC GRAFIKAS ".center(80, "="))
print("=" * 80)

plt.figure(figsize=(10, 8))

colors = ['#1f77b4', '#ff7f0e'] # Mėlyna ir Oranžinė
for i, data in enumerate(roc_data_storage):
    plt.plot(
        data['fpr'],
        data['tpr'],
        color=colors[i % len(colors)],
        lw=2.5,
        label=f"{data['name']} (AUC = {data['auc']:.4f})"
    )

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Atsitiktinė')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Palyginimas: Full Dataset vs. Subset')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file = os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'COMBINED_ROC_curve.png')
plt.savefig(output_file, dpi=300)
plt.close()

print(f"✓ Bendras ROC grafikas išsaugotas: {output_file}")
print(f"✓ Visi rezultatai kataloge: {os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA)}")