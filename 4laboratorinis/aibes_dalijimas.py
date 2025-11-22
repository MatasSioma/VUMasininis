import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------- KONSTANTOS ----------
RANDOM_STATE = 42
DUOMENU_DIREKTORIJA = 'duomenys'
GRAFIKU_DIREKTORIJA = 'grafikai'
AIBES_DIREKTORIJA = os.path.join(GRAFIKU_DIREKTORIJA, 'aibes')
JSON_DIREKTORIJA = 'JSON'
JSON_FAILAS = 'geriausias_rinkinys.json' # Tik failo vardas

# Pilnas kelias iki JSON failo
JSON_FAILAS_PATH = os.path.join(JSON_DIREKTORIJA, JSON_FAILAS)

# Mūsų nugalėtojai (6 geriausi požymiai)
BEST_FEATURES = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]

# Sukuriame reikalingas direktorijas
os.makedirs(AIBES_DIREKTORIJA, exist_ok=True)
os.makedirs(DUOMENU_DIREKTORIJA, exist_ok=True)
os.makedirs(JSON_DIREKTORIJA, exist_ok=True) # Svarbu: sukuriame JSON aplanką

# ---------- 1. ĮKELIAME 31D DUOMENIS ----------
print("=" * 60)
print(" 1. DUOMENŲ ĮKĖLIMAS ".center(60, "="))
df_normuota = pd.read_csv(
    os.path.join(DUOMENU_DIREKTORIJA, 'sugeneruota_aibe_normuota.csv'),
    sep=';'
)
X = df_normuota.drop(columns='label')
y = df_normuota['label']

# ---------- 2. ĮKELIAME 2D t-SNE DUOMENIS ----------
df_tsne = pd.read_csv(
    os.path.join(DUOMENU_DIREKTORIJA, 'sugeneruota_aibe_2D_normuota.csv'),
    sep=';'
)
p1, p2 = 'dimensija_1', 'dimensija_2'

# ---------- 3. PRADINĖ VISOS AIBĖS VIZUALIZACIJA ----------
plt.figure(figsize=(7,6))
for klase in sorted(df_tsne['label'].unique()):
    mask = df_tsne['label'] == klase
    plt.scatter(df_tsne.loc[mask, p1],
                df_tsne.loc[mask, p2],
                s=20, alpha=0.6,
                label=f'Klasė {int(klase)}')

plt.title("Pilna 2D duomenų aibė (t-SNE projekcija)")
plt.xlabel("Dimensija 1")
plt.ylabel("Dimensija 2")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(AIBES_DIREKTORIJA, "pradine_2D_aibe.png"), dpi=300)
plt.close()

print("✓ Išsaugota pradine_2D_aibe.png")

# ---------- 4. DALIJAME AIBĘ (MOKYMAS / VALIDAVIMAS / TESTAVIMAS) ----------
# Pirmas padalijimas: atskiriame 20% testavimui
X_mok_val, X_test, y_mok_val, y_test, tsne_mok_val, tsne_test = train_test_split(
    X, y, df_tsne, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Antras padalijimas: likusius 80% skeliame į mokymą ir validavimą (dar 20% nuo likučio)
X_mok, X_val, y_mok, y_val, tsne_mok, tsne_val = train_test_split(
    X_mok_val, y_mok_val, tsne_mok_val,
    test_size=0.2, random_state=RANDOM_STATE, stratify=y_mok_val
)

# ---------- 5. IŠSAUGOME DUOMENIS ----------
print("\n" + "=" * 60)
print(" 5. FAILŲ SAUGOJIMAS ".center(60, "="))

# --- A. Pilni 31D duomenys (Backup ir bendram naudojimui) ---
df_mok = pd.concat([X_mok, y_mok], axis=1)
df_val = pd.concat([X_val, y_val], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

df_mok.to_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';', index=False)
df_val.to_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';', index=False)
df_test.to_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';', index=False)

print("✓ Išsaugotos PILNOS (31 požymio) aibės (mokymo_aibe.csv ir t.t.)")

# --- B. SPECIFINIAI DUOMENYS (Tik 6 geriausi požymiai) ---
# Sukuriame versijas tik su pasirinktais stulpeliais + label
cols_to_keep = BEST_FEATURES + ['label']

df_mok_final = df_mok[cols_to_keep]
df_val_final = df_val[cols_to_keep]
df_test_final = df_test[cols_to_keep]

df_mok_final.to_csv(os.path.join(DUOMENU_DIREKTORIJA, 'eksperimento_mokymo_aibe.csv'), sep=';', index=False)
df_val_final.to_csv(os.path.join(DUOMENU_DIREKTORIJA, 'eksperimento_validavimo_aibe.csv'), sep=';', index=False)
df_test_final.to_csv(os.path.join(DUOMENU_DIREKTORIJA, 'eksperimento_testavimo_aibe.csv'), sep=';', index=False)

print(f"Išsaugotos OPTIMIZUOTOS (6 požymių) aibės failuose: eksperimento_*_aibe.csv")

# --- C. Atnaujiname JSON failą eksperimentams ---
config_data = {
    "GERIAUSIAS_MODELIS_6_POZYMIAI": BEST_FEATURES
}

# Saugome į JSON aplanką
with open(JSON_FAILAS_PATH, 'w', encoding='utf-8') as f:
    json.dump(config_data, f, indent=4)
print(f"Sukurtas konfigūracijos failas: '{JSON_FAILAS_PATH}'")


# ---------- 6. VIZUALIZUOJAME 2D AIBES ----------
def nupiesti_viena(tsne_df, pavadinimas, failas):
    plt.figure(figsize=(7,6))
    n = len(tsne_df)
    for klase in sorted(tsne_df['label'].unique()):
        mask = tsne_df['label'] == klase
        plt.scatter(
            tsne_df.loc[mask, p1],
            tsne_df.loc[mask, p2],
            s=20, alpha=0.6,
            label=f"Klasė {int(klase)}"
        )
    plt.title(f"{pavadinimas} (n={n})")
    plt.xlabel("Dimensija 1")
    plt.ylabel("Dimensija 2")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(AIBES_DIREKTORIJA, failas), dpi=300)
    plt.close()
    print(f"✓ Išsaugota {failas}")

def nupiesti_bendra(tsne_mok, tsne_val, tsne_test):
    _, axes = plt.subplots(1, 3, figsize=(20, 6))
    duomenys = [
        (tsne_mok,  "Mokymo aibė",     axes[0]),
        (tsne_val,  "Validavimo aibė", axes[1]),
        (tsne_test, "Testavimo aibė",  axes[2])
    ]
    for tsne_df, pavadinimas, ax in duomenys:
        n = len(tsne_df)
        for klase in sorted(tsne_df['label'].unique()):
            mask = tsne_df['label'] == klase
            ax.scatter(
                tsne_df.loc[mask, p1],
                tsne_df.loc[mask, p2],
                s=20, alpha=0.6,
                label=f"Klasė {int(klase)}"
            )
        ax.set_title(f"{pavadinimas} (n={n})")
        ax.set_xlabel("Dimensija 1")
        ax.set_ylabel("Dimensija 2")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    plt.tight_layout()
    failas = os.path.join(AIBES_DIREKTORIJA, "aibes_visos.png")
    plt.savefig(failas, dpi=300)
    plt.close()
    print(f"Išsaugota bendra vizualizacija: {failas}")

nupiesti_viena(tsne_mok, "Mokymo aibė", "mokymo_aibe.png")
nupiesti_viena(tsne_val, "Validavimo aibė", "validavimo_aibe.png")
nupiesti_viena(tsne_test, "Testavimo aibė", "testavimo_aibe.png")
nupiesti_bendra(tsne_mok, tsne_val, tsne_test)

# ---------- 7. STATISTIKA ----------
print("\n=== DUOMENŲ DALIJIMO STATISTIKA ===")
print(f"Mokymo aibė:      {len(df_mok)} (0: {(y_mok==0).sum()}, 2: {(y_mok==2).sum()})")
print(f"Validavimo aibė:  {len(df_val)} (0: {(y_val==0).sum()}, 2: {(y_val==2).sum()})")
print(f"Testavimo aibė:   {len(df_test)} (0: {(y_test==0).sum()}, 2: {(y_test==2).sum()})")
print("-" * 60)