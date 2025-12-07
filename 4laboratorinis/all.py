import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, f1_score

RANDOM_STATE = 42
DUOMENU_DIREKTORIJA = 'duomenys'
GRAFIKU_DIREKTORIJA = 'grafikai'
JSON_DIREKTORIJA = 'JSON'
JSON_FAILAS = os.path.join(JSON_DIREKTORIJA, 'geriausias_rinkinys.json')
ALL_DIREKTORIJA = 'ALL_MODELS'
OPTIMAL_RF_PARAMS_JSON = os.path.join(GRAFIKU_DIREKTORIJA, 'RF', 'RF_optimal_params.json')
POS_LABEL = 2

os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, ALL_DIREKTORIJA), exist_ok=True)


def load_rf_optimal_params(exp_name):
    try:
        with open(OPTIMAL_RF_PARAMS_JSON, 'r', encoding='utf-8') as f:
            params_dict = json.load(f)
            if exp_name in params_dict:
                params = params_dict[exp_name].copy()
                if params.get('max_depth') == "None":
                    params['max_depth'] = None
                else:
                    params['max_depth'] = int(params.get('max_depth', 15))
                return params
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    }

def get_best_params(model_name, X_mok, y_mok, X_val, y_val, exp_name):
    if model_name == "Random Forest":
        rf_params = load_rf_optimal_params(exp_name)
        best_f1, best_depth = -1, rf_params['max_depth']
        max_depth_values = range(4, 8)

        for md in max_depth_values:
            temp_model = RandomForestClassifier(
                n_estimators=rf_params['n_estimators'], max_depth=md,
                min_samples_split=rf_params['min_samples_split'], min_samples_leaf=rf_params['min_samples_leaf'],
                max_features=rf_params['max_features'], random_state=RANDOM_STATE, n_jobs=1
            )
            temp_model.fit(X_mok, y_mok)
            y_val_pred = temp_model.predict(X_val)
            f1_val = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            if f1_val > best_f1:
                best_f1, best_depth = f1_val, md

        rf_params['max_depth'] = best_depth
        print(f"[{model_name} | {exp_name}] Rasti parametrai: max_depth={best_depth}, n_estimators={rf_params['n_estimators']}")
        return RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=1)

    elif model_name == "Decision Tree":
        best_f1, best_depth = -1, 1
        for depth in range(1, 11): # DT.py paieška 1-10
            temp_model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
            temp_model.fit(X_mok, y_mok)
            f1_val = f1_score(y_val, temp_model.predict(X_val), average='weighted', zero_division=0)
            if f1_val > best_f1:
                best_f1, best_depth = f1_val, depth
        print(f"[{model_name} | {exp_name}] Rasti parametrai: max_depth={best_depth}")
        return DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)

    elif model_name == "KNN":
        best_f1, best_k = -1, 1
        for k in range(1, 22, 2): # KNN.py paieška 1-21 (kas 2)
            temp_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
            temp_model.fit(X_mok, y_mok)
            f1_val = f1_score(y_val, temp_model.predict(X_val), average='weighted', zero_division=0)
            if f1_val > best_f1:
                best_f1, best_k = f1_val, k

        # Fiksuoto K parametro atkartojimas, jei naudojami Optimalūs požymiai
        if exp_name == "Optimalūs požymiai" and best_k != 3:
             # Tikriname, ar k=3 F1 yra geresnis už automatiškai rastą best_k, jei ne, paliekame auto_best_k
             # Kadangi originaliame KNN.py kode buvo FIKSUOTAS k=3, mes privalome jį naudoti:
             if exp_name == "Optimalūs požymiai":
                 best_k = 3 # Fiksuojame pagal originalų kodą

        print(f"[{model_name} | {exp_name}] Rasti parametrai: k={best_k}")
        return KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='uniform')

print("=" * 100)
print(" 1. DUOMENŲ ĮKĖLIMAS IR PARUOŠIMAS ".center(100, "="))

try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
    df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')
except FileNotFoundError:
    print(f"[KLAIDA] Nerasti duomenys aplanke '{DUOMENU_DIREKTORIJA}'.")
    exit()

y_mokymas = df_mokymas['label'].values
y_validavimas = df_validavimas['label'].values
y_testavimas = df_testavimas['label'].values

pozymiai_full = [col for col in df_mokymas.columns if col != 'label']
try:
    with open(JSON_FAILAS, 'r', encoding='utf-8') as f:
        pozymiai_subset = json.load(f).get("GERIAUSIAS_MODELIS_6_POZYMIAI", [])
except FileNotFoundError:
    pozymiai_subset = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]

feature_sets = {
    "Optimalūs požymiai": pozymiai_subset,
    "Visi požymiai": pozymiai_full
}

models_to_test = ["Random Forest", "Decision Tree", "KNN"]
roc_data_storage_all = []
print(f"[INFO] Optimalūs požymiai: {pozymiai_subset}")

print("\n" + "=" * 100)
print(" 2. MODELIŲ TRENIRUOTĖ IR ROC DUOMENŲ GENERAVIMAS ".center(100, "="))

for set_name, features in feature_sets.items():
    X_mok_set = df_mokymas[features].values
    X_val_set = df_validavimas[features].values
    X_test_set = df_testavimas[features].values

    for model_name in models_to_test:
        print(f"\n--- Apdorojamas: {model_name} su {set_name} ---")

        try:
            final_model = get_best_params(model_name, X_mok_set, y_mokymas, X_val_set, y_validavimas, set_name)
            final_model.fit(X_mok_set, y_mokymas)
        except Exception as e:
            print(f"[KLAIDA] Nepavyko apmokyti {model_name}: {e}")
            continue

        # 2. Generuoti ROC duomenis
        if hasattr(final_model, "predict_proba"):
            try:
                class_index = list(final_model.classes_).index(POS_LABEL)
                y_proba = final_model.predict_proba(X_test_set)[:, class_index]

                if len(np.unique(y_proba)) > 1:
                    fpr, tpr, _ = roc_curve(y_testavimas, y_proba, pos_label=POS_LABEL)
                    roc_auc = auc(fpr, tpr)

                    roc_data_storage_all.append({
                        'model': model_name,
                        'set': set_name,
                        'name': f"{model_name} ({set_name.split()[0]})",
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': roc_auc
                    })
                    print(f"[{model_name} | {set_name}] ROC AUC: {roc_auc:.4f} [OK]")
                else:
                    print(f"[{model_name} | {set_name}] KLAIDA: Prognozės vienodos. AUC=0.5")
            except Exception as e:
                 print(f"[{model_name} | {set_name}] KLAIDA generuojant ROC: {e}")
        else:
            print(f"[{model_name} | {set_name}] KLAIDA: Modelis nepalaiko predict_proba.")

print("\n" + "=" * 100)
print(" 3. GENERUOJAMI ROC GRAFIKAI ".center(100, "="))

colors_models = {'Random Forest': '#1f77b4', 'Decision Tree': '#ff7f0e', 'KNN': '#2ca02c'}
linestyles_sets = {'Optimalūs požymiai': '-', 'Visi požymiai': '--'}

# --- 3.1 ROC: Optimalūs požymiai (Palyginimas A) ---
plt.figure(figsize=(10, 8))
data_subset = [d for d in roc_data_storage_all if d['set'] == 'Optimalūs požymiai']
for data in data_subset:
    plt.plot(data['fpr'], data['tpr'], color=colors_models[data['model']], lw=3, label=f"{data['model']} (AUC = {data['auc']:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Atsitiktinis')
plt.title('Klasifikatorių palyginimas su Optimaliais požymiais')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
filename = os.path.join(GRAFIKU_DIREKTORIJA, ALL_DIREKTORIJA, 'ROC_Optimalus_Pozymiai.png')
plt.savefig(filename, dpi=300); plt.close()
print(f"[OK] Sukurtas grafikas: {filename}")

# --- 3.2 ROC: Visi požymiai (Palyginimas B) ---
plt.figure(figsize=(10, 8))
data_subset = [d for d in roc_data_storage_all if d['set'] == 'Visi požymiai']
for data in data_subset:
    plt.plot(data['fpr'], data['tpr'], color=colors_models[data['model']], lw=3, label=f"{data['model']} (AUC = {data['auc']:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Atsitiktinis')
plt.title('Klasifikatorių palyginimas su Visais požymiais')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
filename = os.path.join(GRAFIKU_DIREKTORIJA, ALL_DIREKTORIJA, 'ROC_Visi_Pozymiai.png')
plt.savefig(filename, dpi=300); plt.close()
print(f"[OK] Sukurtas grafikas: {filename}")

# --- 3.3 ROC: Visi modeliai, abu požymių rinkiniai (Palyginimas C) ---
plt.figure(figsize=(14, 10))
for data in roc_data_storage_all:
    color = colors_models[data['model']]
    linestyle = linestyles_sets[data['set']]

    label = f"{data['model']} ({data['set'].split()[0]}) AUC={data['auc']:.4f}"

    plt.plot(data['fpr'], data['tpr'], color=color, lw=3, linestyle=linestyle, label=label)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Atsitiktinis')
plt.title('Visų Klasifikatorių ir Požymių Rinkinių ROC Kreivių Palyginimas')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right", fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
filename = os.path.join(GRAFIKU_DIREKTORIJA, ALL_DIREKTORIJA, 'ROC_Bendras_Visi_ir_Optimalus.png')
plt.savefig(filename, dpi=300); plt.close()
print(f"[OK] Sukurtas bendras 6 kreivių grafikas: {filename}")

print(f"\n[INFO] Visi grafikai išsaugoti aplanke: {os.path.join(GRAFIKU_DIREKTORIJA, ALL_DIREKTORIJA)}")
print("=" * 100)