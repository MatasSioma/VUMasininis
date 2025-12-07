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
OPTIMAL_PARAMS_JSON = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'RF_optimal_params.json')

os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA), exist_ok=True)

def load_optimal_params(exp_name):
    try:
        with open(OPTIMAL_PARAMS_JSON, 'r', encoding='utf-8') as f:
            params_dict = json.load(f)
            if exp_name in params_dict:
                params = params_dict[exp_name].copy()

                if params.get('max_depth') == "None":
                    params['max_depth'] = None
                else:
                    params['max_depth'] = int(params.get('max_depth', 10))
                return params
    except FileNotFoundError:
        pass

    return {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    }

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

all_sweep_data = []

param_sweeps = {
    'n_estimators': list(range(50, 301, 50)),
    'max_depth': [None] + list(range(1, 21, 1)),
    'min_samples_split': list(range(2, 21, 1)),
    'min_samples_leaf': list(range(1, 9, 1)),
    'max_features': ['sqrt', 'log2']
}

for param_name, param_values in param_sweeps.items():
    print("\n" + "#" * 100)
    print(f" HYPERPARAMETRŲ PAIESKA (grupuojama pagal parametą): {param_name} ".center(100, "#"))


    for exp_name, features in experiments.items():
        print("\n" + "-" * 80)
        print(f" Vykdoma validacija: {exp_name}  (parametras: {param_name})")

        X_mok = df_mokymas[features].values
        X_val = df_validavimas[features].values
        X_test = df_testavimas[features].values

        param_tuning_table = []
        best_val_f1 = -1
        best_param_value = None

        for param_value in param_values:

            optimal_params = load_optimal_params(exp_name)
            current_params = optimal_params.copy()
            current_params[param_name] = param_value

            rf_temp = RandomForestClassifier(
                n_estimators=current_params['n_estimators'],
                max_depth=current_params['max_depth'],
                min_samples_split=current_params['min_samples_split'],
                min_samples_leaf=current_params['min_samples_leaf'],
                max_features=current_params['max_features'],
                random_state=42,
                n_jobs=1
            )
            rf_temp.fit(X_mok, y_mokymas)
            y_val_pred = rf_temp.predict(X_val)

            acc_val = accuracy_score(y_validavimas, y_val_pred)
            prec_val = precision_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)
            rec_val = recall_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)
            f1_val = f1_score(y_validavimas, y_val_pred, average='weighted', zero_division=0)

            param_tuning_table.append([str(param_value), acc_val, prec_val, rec_val, f1_val])


            all_sweep_data.append({
                'Dataset': exp_name,
                'Parameter': param_name,
                'Value': str(param_value),
                'Accuracy': acc_val,
                'Precision': prec_val,
                'Recall': rec_val,
                'F1 Score': f1_val
            })

            if f1_val > best_val_f1:
                best_val_f1 = f1_val
                best_param_value = param_value

        print(f"\nValidavimo rezultatai - {param_name} - {exp_name}:")
        headers = [param_name, "Accuracy", "Precision", "Recall", "F1 Score"]
        print(tabulate(param_tuning_table, headers=headers, tablefmt="psql", floatfmt=".4f"))
        print(f"[BEST] Optimalus {param_name} for {exp_name}: {best_param_value} (Validavimo F1={best_val_f1:.4f})")


print("\n" + "=" * 100)
print(" 4. GENERUOJAMI HYPERPARAMETRU SWEEP GRAFIKAI ".center(100, "="))

df_all_sweeps = pd.DataFrame(all_sweep_data)

# Gauta parametrų rinkinys
param_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
custom_palette = {"Vsi požymiai": "#1f77b4", "Optimalūs požymiai": "#ff7f0e"}


for param_name in param_names:
    df_param = df_all_sweeps[df_all_sweeps['Parameter'] == param_name].copy()

    if len(df_param) == 0:
        print(f"[INFO] Nėra duomenų parametrui {param_name}")
        continue

    print(f"\n--- Generuojami grafikai parametrui {param_name} ---")

    if param_name != 'max_features':
        try:
            # Konvertuojame į float reikšmę, jei įmanoma
            df_param['Value_sort'] = pd.to_numeric(df_param['Value'], errors='coerce')
            df_param = df_param.sort_values('Value_sort')
        except:
            pass

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        sns.lineplot(
            data=df_param,
            x='Value',
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

        ax.set_title(f'{metric} priklausomybė nuo {param_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'{param_name}', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)


        if len(df_param['Value'].unique()) > 5:
            ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f"Random Forest rodklių priklausomybė nuo {param_name} reikšmės",
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    grid_filename = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA,
                                 f'RF_Metriku_Suvestine_{param_name}_Grid.png')
    plt.savefig(grid_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Sukurtas grafikas: {grid_filename}")


print("\n" + "=" * 100)
print(" 5. IŠSAUGOMI SWEEP REZULTATAI ".center(100, "="))


csv_filename = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'RF_hyperparameter_sweep_results.csv')
df_all_sweeps.to_csv(csv_filename, index=False)
print(f"[OK] Sweep rezultatai išsaugoti: {csv_filename}")

print(f"\n[INFO] Visi failai issaugoti aplanke: {os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA)}")
print("=" * 100)
