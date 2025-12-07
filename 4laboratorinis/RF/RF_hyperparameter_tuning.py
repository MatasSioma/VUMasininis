import os
import pandas as pd
import numpy as np
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

warnings.filterwarnings('ignore', category=UserWarning)

DUOMENU_DIREKTORIJA = '../duomenys'
GRAFIKU_DIREKTORIJA = '../grafikai'
RF_DIREKTORIJA = 'RF'
JSON_DIREKTORIJA = '../JSON'
JSON_FAILAS = os.path.join(JSON_DIREKTORIJA, 'geriausias_rinkinys.json')

os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'hyperparameter_tuning'), exist_ok=True)

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

y_mokymas = df_mokymas['label'].values
y_validavimas = df_validavimas['label'].values
y_testavimas = df_testavimas['label'].values

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

best_params_storage = {}
summary_results = []


for exp_name, features in experiments.items():
    print("\n" + "#" * 100)
    print(f" HYPERPARAMETRU PAIESKA: {exp_name} (su GridSearchCV) ".center(100, "#"))

    X_mok = df_mokymas[features].values
    X_val = df_validavimas[features].values
    X_test = df_testavimas[features].values

    print(f"\nVykdoma grieta parametru paieska...")

    param_grid = {
        'n_estimators': [100, 125, 150, 225, 250],
        'max_depth': [6, 7, 8],
        'min_samples_split': [6,7,8,9,10],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2'],
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=1),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=8,
        verbose=1
    )

    print(f"Sudeda iš: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} kombinacijų")

    rf_grid.fit(X_mok, y_mokymas)

    print(f"\n[BEST] Geriausi parametrai:")
    print(f"  - n_estimators: {rf_grid.best_params_['n_estimators']}")
    print(f"  - max_depth: {rf_grid.best_params_['max_depth']}")
    print(f"  - min_samples_split: {rf_grid.best_params_['min_samples_split']}")
    print(f"  - min_samples_leaf: {rf_grid.best_params_['min_samples_leaf']}")
    print(f"  - max_features: {rf_grid.best_params_['max_features']}")
    print(f"  - CV F1 Score: {rf_grid.best_score_:.4f}")

    best_params_storage[exp_name] = rf_grid.best_params_

    cv_results = pd.DataFrame(rf_grid.cv_results_)

    results_csv = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'hyperparameter_tuning', f'gridsearch_results_{exp_name.replace(" ", "_")}.csv')
    cv_results.to_csv(results_csv, index=False)
    print(f"[OK] GridSearch rezultatai išsaugoti: {results_csv}")

    final_rf = RandomForestClassifier(**rf_grid.best_params_, random_state=42, n_jobs=-1)
    final_rf.fit(X_mok, y_mokymas)
    y_test_pred = final_rf.predict(X_test)

    # Metrikos
    acc = accuracy_score(y_testavimas, y_test_pred)
    prec = precision_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    rec = recall_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)
    f1_final = f1_score(y_testavimas, y_test_pred, average='weighted', zero_division=0)

    summary_results.append({
        'Dataset': exp_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1_final,
        'CV F1': rf_grid.best_score_
    })

    print(f"\n>>> DETALI KLASIFIKAVIMO ATASKAITA: {exp_name} <<<")
    print(classification_report(y_testavimas, y_test_pred, target_names=["Normalus (0)", "Aritmija (2)"], digits=4))

    cm = confusion_matrix(y_testavimas, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f">>> KLAIDŲ ANALIZĖ: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("-" * 60)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 14})
    plt.title(f'Confusion Matrix: {exp_name}\n(Accuracy: {acc:.4f})')
    plt.ylabel('Tikroji klasė')
    plt.xlabel('Prognozuota klasė')
    plt.tight_layout()
    cm_filename = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'hyperparameter_tuning', f'confusion_matrix_{exp_name.replace(" ", "_")}.png')
    plt.savefig(cm_filename, dpi=300)
    plt.close()
    print(f"[OK] CM grafikas: {cm_filename}")

    feature_importance = final_rf.feature_importances_
    feature_names = features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Feature Importance: {exp_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    importance_filename = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'hyperparameter_tuning', f'feature_importance_{exp_name.replace(" ", "_")}.png')
    plt.savefig(importance_filename, dpi=300)
    plt.close()
    print(f"[OK] Feature Importance: {importance_filename}")

    if hasattr(final_rf, "predict_proba"):
        y_proba = final_rf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_testavimas, y_proba, pos_label=2)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC={roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {exp_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_filename = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'hyperparameter_tuning', f'roc_curve_{exp_name.replace(" ", "_")}.png')
        plt.savefig(roc_filename, dpi=300)
        plt.close()
        print(f"[OK] ROC Curve: {roc_filename}")

print("\n" + "=" * 100)
print(" GALUTINIAI REZULTATAI ".center(100, "="))

df_summary = pd.DataFrame(summary_results)
print(tabulate(df_summary, headers='keys', tablefmt='psql', floatfmt=".4f", showindex=False))

output_params = {}
for exp_name, params in best_params_storage.items():
    output_params[exp_name] = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': params['max_depth'] if params['max_depth'] is not None else "None",
        'min_samples_split': int(params['min_samples_split']),
        'min_samples_leaf': int(params['min_samples_leaf']),
        'max_features': params['max_features']
    }

params_json_file = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'RF_optimal_params.json')
with open(params_json_file, 'w', encoding='utf-8') as f:
    json.dump(output_params, f, indent=4, ensure_ascii=False)

print(f"\n[OK] Optimalūs parametrai išsaugoti: {params_json_file}")
print("=" * 100)
