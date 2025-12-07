import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

DUOMENU_DIREKTORIJA = '../duomenys'
GRAFIKU_DIREKTORIJA = '../grafikai'
DT_DIREKTORIJA = 'DecisionTree'
JSON_FAILAS = os.path.join('../JSON', 'geriausias_rinkinys.json')
KLAIDU_DIREKTORIJA = os.path.join(GRAFIKU_DIREKTORIJA, DT_DIREKTORIJA, 'Klaidos')
RANDOM_STATE = 42
DEPTH_RANGE = range(1, 21)

os.makedirs(KLAIDU_DIREKTORIJA, exist_ok=True)

print("=" * 50)
try:
    with open(JSON_FAILAS, 'r', encoding='utf-8') as f:
        config = json.load(f)
        OPTIMALUS_POZYMIAI = config.get("GERIAUSIAS_MODELIS_6_POZYMIAI", [])

        if not OPTIMALUS_POZYMIAI:
            raise ValueError("Raktas nerastas arba sąrašas tuščias")
        print(f"[OK] Požymiai įkelti iš JSON ({len(OPTIMALUS_POZYMIAI)} vnt.):")
        print(OPTIMALUS_POZYMIAI)

except (FileNotFoundError, ValueError) as e:
    print(f"[INFO] Nepavyko nuskaityti JSON ({e}). Naudojami numatytieji.")
    OPTIMALUS_POZYMIAI = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]

print("=" * 50)

print("Įkeliami duomenys...")
try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
    df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')
except FileNotFoundError:
    print(f"[KLAIDA] Nerasti duomenų failai aplanke: {DUOMENU_DIREKTORIJA}")
    exit()

X_train = df_mokymas[OPTIMALUS_POZYMIAI].values
y_train = df_mokymas['label'].values
X_val = df_validavimas[OPTIMALUS_POZYMIAI].values
y_val = df_validavimas['label'].values
X_test = df_testavimas[OPTIMALUS_POZYMIAI].values
y_test = df_testavimas['label'].values

best_depth = None
best_val_f1 = -1

for depth in DEPTH_RANGE:
    dt_tmp = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    dt_tmp.fit(X_train, y_train)
    y_val_pred = dt_tmp.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    if f1_val > best_val_f1:
        best_val_f1 = f1_val
        best_depth = depth

print(f"Apmokomas Decision Tree su optimaliu gyliu (depth={best_depth}) ...")
dt = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

klaidu_indeksai = np.where(y_test != y_pred)[0]
print(f"\nRasta klaidų: {len(klaidu_indeksai)}")

if len(klaidu_indeksai) == 0:
    print("Sveikiname! Klaidų nerasta. Nėra ką vizualizuoti.")
    exit()

vidurkiai = df_testavimas.groupby('label')[OPTIMALUS_POZYMIAI].mean()

klasiu_pavadinimai = {0: "Normalus (0)", 2: "Aritmija (2)"}

for i, idx in enumerate(klaidu_indeksai):
    objektas = df_testavimas.iloc[idx]
    tikra_klase = int(objektas['label'])
    prognozuota_klase = int(y_pred[idx])

    print(f"\n--- Klaida #{i+1} (Indeksas testavimo aibėje: {idx}) ---")
    print(f"Tikroji klasė: {klasiu_pavadinimai.get(tikra_klase, tikra_klase)}")
    print(f"Prognozuota:   {klasiu_pavadinimai.get(prognozuota_klase, prognozuota_klase)}")

    reiksmes_objekto = objektas[OPTIMALUS_POZYMIAI].values
    reiksmes_tikros = vidurkiai.loc[tikra_klase].values
    reiksmes_prognozuotos = vidurkiai.loc[prognozuota_klase].values

    x = np.arange(len(OPTIMALUS_POZYMIAI))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, reiksmes_objekto, width, label='KLAIDINGAS OBJEKTAS', color='#d62728', alpha=0.9)
    ax.bar(x, reiksmes_tikros, width, label=f'Vidurkis: {klasiu_pavadinimai.get(tikra_klase, tikra_klase)} (Tikras)', color='#2ca02c', alpha=0.7)
    ax.bar(x + width, reiksmes_prognozuotos, width, label=f'Vidurkis: {klasiu_pavadinimai.get(prognozuota_klase, prognozuota_klase)} (Spėtas)', color='#7f7f7f', alpha=0.5)

    ax.set_ylabel('Požymio reikšmė (sunormuota)')
    ax.set_title(f'Klaidos analizė: Objektas #{idx}\nTikra: {klasiu_pavadinimai.get(tikra_klase, tikra_klase)} -> Spėta: {klasiu_pavadinimai.get(prognozuota_klase, prognozuota_klase)}')
    ax.set_xticks(x)
    ax.set_xticklabels(OPTIMALUS_POZYMIAI)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    failo_pavadinimas = os.path.join(KLAIDU_DIREKTORIJA, f'Klaida_{idx}_Tikra_{tikra_klase}_Speta_{prognozuota_klase}.png')
    plt.savefig(failo_pavadinimas, dpi=300)
    plt.close()
    print(f"Grafikas išsaugotas: {failo_pavadinimas}")

print("\n" + "=" * 50)
print(f"Visos klaidų vizualizacijos išsaugotos aplanke: {KLAIDU_DIREKTORIJA}")
