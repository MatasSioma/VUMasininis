import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ---------- KONSTANTOS IR NUSTATYMAI ----------
DUOMENU_DIREKTORIJA = 'duomenys'
GRAFIKU_DIREKTORIJA = 'grafikai'
KNN_DIREKTORIJA = 'KNN'
JSON_FAILAS = os.path.join('JSON', 'geriausias_rinkinys.json')

# Sukuriame reikiamas direktorijas
os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA), exist_ok=True)

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
    "Visi požymiai": pozymiai_full,
    "Optimalūs požymiai": pozymiai_subset
}

roc_data_storage = []
summary_results = []
visu_eksperimentu_duomenys = []

# ---------- 3. PAGRINDINIS CIKLAS ----------

for exp_name, features in experiments.items():
    print("\n" + "#" * 100)
    print(f" VYKDOMAS EKSPERIMENTAS: {exp_name} ".center(100, "#"))

    X_mok = df_mokymas[features].values
    X_val = df_validavimas[features].values
    X_test = df_testavimas[features].values

    print(f"--- Ieskomas geriausias k (Validavimo aibe) ---")
    best_k = 1
    best_val_f1 = -1
    tuning_data_table = []

    for k in range(1, 22, 2):
        knn_temp = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
        knn_temp.fit(X_mok, y_mokymas)
        y_val_pred = knn_temp.predict(X_val)

        acc_val = accuracy_score(y_validavimas, y_val_pred)
        prec_val = precision_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)
        rec_val = recall_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)
        f1_val = f1_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)

        tuning_data_table.append([k, acc_val, prec_val, rec_val, f1_val])

        # Kaupiame duomenis grafikams
        visu_eksperimentu_duomenys.append({
            'Dataset': exp_name,
            'k': k,
            'Accuracy': acc_val,
            'Precision': prec_val,
            'Recall': rec_val,
            'F1 Score': f1_val
        })

        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            best_k = k

    print(f"\nParametru paieskos rezultatai ({exp_name}):")
    headers = ["k", "Accuracy", "Precision", "Recall", "F1 Score"]
    print(tabulate(tuning_data_table, headers=headers, tablefmt="psql", floatfmt=".4f"))
    print(f"\n[BEST] Pasirinktas k: {best_k} (Maksimalus F1={best_val_f1:.4f})")

    # Final training
    final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='uniform')
    final_knn.fit(X_mok, y_mokymas)
    y_test_pred = final_knn.predict(X_test)

    # Metrics
    acc = accuracy_score(y_testavimas, y_test_pred)
    prec = precision_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    rec = recall_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    f1_final = f1_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)

    summary_results.append({
        'Dataset': exp_name,
        'k': best_k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1_final
    })

    # Confusion Matrix
    cm = confusion_matrix(y_testavimas, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Sumaišymo matrica: {exp_name}\n(k={best_k}, F1={f1_final:.4f})')
    plt.ylabel('Tikroji klase')
    plt.xlabel('Prognozuota klase')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, f'{exp_name}_Confusion_Matrix.png'), dpi=300)
    plt.close()

    # ROC Data
    if hasattr(final_knn, "predict_proba"):
        y_proba = final_knn.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_testavimas, y_proba, pos_label=2)
        roc_auc = auc(fpr, tpr)
        roc_data_storage.append({'name': exp_name, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})

# ---------- 4. ROC GRAFIKAS ----------
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e']
for i, data in enumerate(roc_data_storage):
    plt.plot(data['fpr'], data['tpr'], color=colors[i % len(colors)], lw=3, label=f"{data['name']} (AUC = {data['auc']:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Atsitiktinis')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Kreivių Palyginimas')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'BENDRAS_ROC_Grafikas.png'), dpi=300)
plt.close()

# ---------- 5. K PRIKLAUSOMYBES GRAFIKAI (VIENAME FAILE 2x2) ----------
print("\n" + "=" * 100)
print(" 5. GENERUOJAMA METRIKU SUVESTINE (2x2 GRID) ".center(100, "="))

df_visos_metrikos = pd.DataFrame(visu_eksperimentu_duomenys)
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
custom_palette = {"Visi požymiai": "#1f77b4", "Optimalūs požymiai": "#ff7f0e"}

# Sukuriame 2x2 paveiksla
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten() # Išlyginame masyvą, kad galėtume iteruoti cikle (0, 1, 2, 3)

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]

    sns.lineplot(
        data=df_visos_metrikos,
        x='k',
        y=metric,
        hue='Dataset',
        style='Dataset',
        markers=True,
        dashes=False,
        palette=custom_palette,
        linewidth=2.5,
        markersize=8,
        ax=ax,          # Nurodome konkretu sub-grafika
        legend=(i == 0) # Legendą rodome tik pirmame grafike, kad neapkrautume vaizdo
    )

    ax.set_title(f'{metric} priklausomybė nuo k', fontsize=12, fontweight='bold')
    ax.set_xlabel('k (Kaimynų skaičius)', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xticks(range(1, 22, 2))
    ax.grid(True, linestyle='--', alpha=0.6)

# Sutvarkome išdėstymą
plt.suptitle("KNN Metrikų Priklausomybė nuo k Reikšmės (Mokymo procesas)", fontsize=16, y=1.02)
plt.tight_layout()

# Išsaugome vieną bendrą failą
combined_filename = os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA, 'KNN_Metriku_Suvestine_Grid.png')
plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Sukurtas bendras grafikas: {combined_filename}")
print(f"\n[INFO] Visi failai issaugoti aplanke: {os.path.join(GRAFIKU_DIREKTORIJA, KNN_DIREKTORIJA)}")
print("=" * 100)