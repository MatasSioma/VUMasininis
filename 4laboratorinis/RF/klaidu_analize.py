import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ---------- NUSTATYMAI ----------
DUOMENU_DIREKTORIJA = '../duomenys'
GRAFIKU_DIREKTORIJA = '../grafikai'
RF_DIREKTORIJA = 'RF'
JSON_DIREKTORIJA = '../JSON'
JSON_FAILAS = os.path.join(JSON_DIREKTORIJA, 'geriausias_rinkinys.json')
OPTIMAL_PARAMS_JSON = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'RF_optimal_params.json')
KLAIDU_DIREKTORIJA = os.path.join(GRAFIKU_DIREKTORIJA, RF_DIREKTORIJA, 'Klaidos')

os.makedirs(KLAIDU_DIREKTORIJA, exist_ok=True)

# ---------- 0. OPTIMALIŲ PARAMETRŲ NUSKAITYMAS IŠ JSON ----------
print("=" * 50)

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

print("=" * 50)

# ---------- 1. DUOMENŲ ĮKELIMAS ----------
print("Įkeliami duomenys...")
try:
    df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
    df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')
except FileNotFoundError:
    print(f"[KLAIDA] Nerasti duomenų failai aplanke: {DUOMENU_DIREKTORIJA}")
    exit()

# Naudojame visus požymius (full dataset)
visi_pozymiai = [col for col in df_mokymas.columns if col != 'label']

# Paruošiame X ir y
X_train = df_mokymas[visi_pozymiai].values
y_train = df_mokymas['label'].values

X_test = df_testavimas[visi_pozymiai].values
y_test = df_testavimas['label'].values

print(f"Mokymo aibė: {X_train.shape[0]} eilučių, {len(visi_pozymiai)} požymiai")
print(f"Testavimo aibė: {X_test.shape[0]} eilučių")

# ---------- 2. MODELIO APMOKYMAS ----------
print(f"\nApmokomas Atsitiktinis medis su visais parametrais...")
rf = RandomForestClassifier(
    n_estimators=optimal_params['n_estimators'],
    max_depth=optimal_params['max_depth'],
    min_samples_split=optimal_params['min_samples_split'],
    min_samples_leaf=optimal_params['min_samples_leaf'],
    max_features=optimal_params['max_features'],
    random_state=42,
    n_jobs=1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ---------- 3. KLAIDŲ PAIEŠKA ----------
klaidu_indeksai = np.where(y_test != y_pred)[0]
print(f"Rasta klaidų: {len(klaidu_indeksai)} iš {len(y_test)}")

if len(klaidu_indeksai) == 0:
    print("Klaidų nerasta. Nėra ką vizualizuoti.")
    exit()

# Paskaičiuojame klasių vidurkius (kad turėtume su kuo lyginti)
vidurkiai = df_testavimas.groupby('label')[visi_pozymiai].mean()

# ---------- 4. VIZUALIZACIJA KIEKVIENAI KLAIDAI ----------
klasiu_pavadinimai = {0: "Normalus (0)", 2: "Aritmija (2)"}

for i, idx in enumerate(klaidu_indeksai):
    objektas = df_testavimas.iloc[idx]
    tikra_klase = int(objektas['label'])
    prognozuota_klase = int(y_pred[idx])

    print(f"\n--- Klaida #{i+1} (Indeksas testavimo aibėje: {idx}) ---")
    print(f"Tikroji klasė: {klasiu_pavadinimai[tikra_klase]}")
    print(f"Prognozuota:   {klasiu_pavadinimai[prognozuota_klase]}")
    
    # Pasiruošiame duomenis grafikui
    reiksmes_objekto = objektas[visi_pozymiai].values
    reiksmes_tikros = vidurkiai.loc[tikra_klase].values
    reiksmes_prognozuotos = vidurkiai.loc[prognozuota_klase].values
    
    # Spausdiname parametrų reikšmes
    print(f"\nParametrų reikšmės (požymiai):")
    print(f"{'Požymis':<20} | {'Objekto':<10} | {klasiu_pavadinimai[tikra_klase]:<20} | {klasiu_pavadinimai[prognozuota_klase]:<20}")
    print("-" * 75)
    for j, pozymis in enumerate(visi_pozymiai):
        print(f"{pozymis:<20} | {reiksmes_objekto[j]:<10.4f} | {reiksmes_tikros[j]:<20.4f} | {reiksmes_prognozuotos[j]:<20.4f}")
    
    # Spausdiname klaidų analizę
    print(f"\nKlaidų analizė:")
    print(f"Skirtumas nuo tikrosios klasės vidurkio: {np.mean(np.abs(reiksmes_objekto - reiksmes_tikros)):.4f}")
    print(f"Skirtumas nuo prognozuotos klasės vidurkio: {np.mean(np.abs(reiksmes_objekto - reiksmes_prognozuotos)):.4f}")

    # Pasiruošiame duomenis grafikui
    reiksmes_objekto = objektas[visi_pozymiai].values
    reiksmes_tikros = vidurkiai.loc[tikra_klase].values
    reiksmes_prognozuotos = vidurkiai.loc[prognozuota_klase].values

    # Braižome grafiką
    x = np.arange(len(visi_pozymiai))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    # Stulpeliai
    rects1 = ax.bar(x - width, reiksmes_objekto, width, label='KLAIDINGAS OBJEKTAS', color='#d62728', alpha=0.9) # Raudona
    rects2 = ax.bar(x, reiksmes_tikros, width, label=f'Vidurkis: {klasiu_pavadinimai[tikra_klase]} (Tikras)', color='#2ca02c', alpha=0.7) # Žalia
    rects3 = ax.bar(x + width, reiksmes_prognozuotos, width, label=f'Vidurkis: {klasiu_pavadinimai[prognozuota_klase]} (Spėtas)', color='#7f7f7f', alpha=0.5) # Pilka

    ax.set_ylabel('Požymio reikšmė (normuota)')
    ax.set_title(f'Klaidos analizė (Atsitiktinis medis): Objektas #{idx}\nTikra: {klasiu_pavadinimai[tikra_klase]} -> Spėta: {klasiu_pavadinimai[prognozuota_klase]}')
    ax.set_xticks(x)
    ax.set_xticklabels(visi_pozymiai, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    failo_pavadinimas = os.path.join(KLAIDU_DIREKTORIJA, f'Klaida_{idx}_Tikra_{tikra_klase}_Speta_{prognozuota_klase}.png')
    plt.savefig(failo_pavadinimas, dpi=300)
    plt.close()
    print(f"Grafikas išsaugotas: {failo_pavadinimas}")

print("\n" + "="*50)
print(f"Visos klaidų vizualizacijos išsaugotos aplanke: {KLAIDU_DIREKTORIJA}")
