import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- KONSTANTOS ----------
RANDOM_STATE = 42
DUOMENU_DIREKTORIJA = 'duomenys'
GRAFIKU_DIREKTORIJA = 'grafikai'
AIBES_DIREKTORIJA = os.path.join(GRAFIKU_DIREKTORIJA, 'aibes')

os.makedirs(AIBES_DIREKTORIJA, exist_ok=True)

# ---------- 1. ĮKELIAME 31D DUOMENIS ----------
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

# ---------- 4. DALIJAME 31D AIBĘ ----------
from sklearn.model_selection import train_test_split

X_mok_val, X_test, y_mok_val, y_test, tsne_mok_val, tsne_test = train_test_split(
    X, y, df_tsne, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

X_mok, X_val, y_mok, y_val, tsne_mok, tsne_val = train_test_split(
    X_mok_val, y_mok_val, tsne_mok_val,
    test_size=0.2, random_state=RANDOM_STATE, stratify=y_mok_val
)

# ---------- 5. IŠSAUGOME 31D AIBES ----------
df_mok = pd.concat([X_mok, y_mok], axis=1)
df_val = pd.concat([X_val, y_val], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

df_mok.to_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';', index=False)
df_val.to_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';', index=False)
df_test.to_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';', index=False)

print("✓ Išsaugotos 31D mok/val/test aibės")

# ---------- 6. VIZUALIZUOJAME 2D MOK/VAL/TEST AIBES ----------
def nupiesti_viena(tsne_df, pavadinimas, failas):
    """Išsaugo individualų scatter grafiką su legenda."""
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
    """Sukuria vieną bendrą PNG su legendomis visuose grafikuose."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

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

    print(f"✓ Išsaugota bendra vizualizacija: {failas}")


# Individualūs grafikai (jei reikia)
nupiesti_viena(tsne_mok, "Mokymo aibė (2D projekcija)", "mokymo_aibe.png")
nupiesti_viena(tsne_val, "Validavimo aibė (2D projekcija)", "validavimo_aibe.png")
nupiesti_viena(tsne_test, "Testavimo aibė (2D projekcija)", "testavimo_aibe.png")

# Bendra 3 grafų vizualizacija
nupiesti_bendra(tsne_mok, tsne_val, tsne_test)


# ---------- 7. STATISTIKA ----------
print("\n=== DUOMENŲ DALIJIMO STATISTIKA ===")
print(f"Mokymo aibė:      {len(df_mok)} (0: {(y_mok==0).sum()}, 2: {(y_mok==2).sum()})")
print(f"Validavimo aibė:  {len(df_val)} (0: {(y_val==0).sum()}, 2: {(y_val==2).sum()})")
print(f"Testavimo aibė:   {len(df_test)} (0: {(y_test==0).sum()}, 2: {(y_test==2).sum()})")
print("\nPNG failai išsaugoti kataloge grafikai/aibes/")
