import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

# ---- t-SNE nustatymai (vizualizacijoms) ----
PERPLEXITY = 50
MAX_ITER = 500
TSNE_METRIC = 'canberra'
RANDOM_STATE = 42

# ---- Išvesties katalogai ----
base_dir_gmm = 'grafikai/gmm'
base_dir_results = 'rezultatai'
os.makedirs(base_dir_gmm, exist_ok=True)
os.makedirs(base_dir_results, exist_ok=True)


# ============================ ĮKĖLIMAS ============================

def load_numeric_csv(path):
    df = pd.read_csv(path, sep=';')
    features = [c for c in df.columns if c != 'label']
    Xdf = df[features].apply(pd.to_numeric, errors='coerce').dropna()
    return Xdf.values, Xdf  # numpy ir DataFrame (statistikoms)

def load_full_csv(path):
    return pd.read_csv(path, sep=';')


# ============================ PAGALBINĖS ============================

def standartizuoti(X: np.ndarray):
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    return scaler.fit_transform(X_imp)

def tsne_2d(X: np.ndarray):
    # Naudojama tik vizualizacijoms. Jei X jau 2D, nieko nekeičiame.
    if X.shape[1] <= 2:
        return X
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        metric=TSNE_METRIC,
        random_state=RANDOM_STATE,
    )
    return tsne.fit_transform(X)

def best_k_by_silhouette_gmm(X_std: np.ndarray, k_min=2, k_max=12, covariance_type='full'):
    ks, scores = [], []
    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                  random_state=RANDOM_STATE, n_init=1)
            gmm.fit(X_std)
            labels = gmm.predict(X_std)   # kieti priskyrimai
            score = silhouette_score(X_std, labels, metric='euclidean')
        except Exception:
            score = -1.0
        ks.append(k)
        scores.append(score)
    best_idx = int(np.argmax(scores))
    return ks[best_idx], ks, scores

def spausdinti_klasterio_statistika(df_numeric: pd.DataFrame, labels: np.ndarray, aibes_pavadinimas: str):
    df_stats = df_numeric.reset_index(drop=True).copy()
    labels = labels[:len(df_stats)]
    df_stats['cluster'] = labels
    grupes = df_stats.groupby('cluster')

    print(f"\n--- Klasterių statistika: {aibes_pavadinimas} ---")
    for cl, g in grupes:
        print(f"\n[Klasteris {cl}] n={len(g)}")
        for col in g.drop(columns=['cluster']).columns:
            col_vals = pd.to_numeric(g[col], errors='coerce').to_numpy()
            mean = float(np.nanmean(col_vals))
            median = float(np.nanmedian(col_vals))
            std = float(np.nanstd(col_vals, ddof=1)) if len(col_vals) > 1 else float('nan')
            vmin = float(np.nanmin(col_vals))
            vmax = float(np.nanmax(col_vals))
            print(f"  {col}: mean={mean:.6f}, median={median:.6f}, std={std if not np.isnan(std) else np.nan}, "
                  f"min={vmin:.6f}, max={vmax:.6f}")

def vizualizuoti_klasterius_sujungta(X_list, clusters_list, pavadinimai, failo_pavadinimas):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, X, labels, ttl in zip(axes, X_list, clusters_list, pavadinimai):
        X2 = X if X.shape[1] <= 2 else tsne_2d(X)
        ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='tab10', s=35)
        ax.set_title(f"GMM\n({ttl})")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_gmm, f"{failo_pavadinimas}.png"), dpi=300)
    plt.close()

def vizualizuoti_palyginima(X, tiksliosios_klases, gmm_klasteriai, failo_pavadinimas):
    # X jau 2D (t-SNE failas) arba bus nupieštas kaip 2D jei >2
    X2 = X if X.shape[1] <= 2 else tsne_2d(X)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(X2[:, 0], X2[:, 1], c=tiksliosios_klases, cmap='viridis', s=35, alpha=0.7)
    axes[0].set_title('t-SNE su tiksliosiomis klasėmis')
    axes[0].set_xlabel('Dimensija 1'); axes[0].set_ylabel('Dimensija 2')

    axes[1].scatter(X2[:, 0], X2[:, 1], c=gmm_klasteriai, cmap='tab10', s=35, alpha=0.7)
    axes[1].set_title('t-SNE su GMM klasteriais')
    axes[1].set_xlabel('Dimensija 1'); axes[1].set_ylabel('Dimensija 2')

    # Atitikimai: dominuojantis klasteris kiekvienai klasei
    unique_classes = sorted(np.unique(tiksliosios_klases))
    dominant = {}
    for c in unique_classes:
        mask = (tiksliosios_klases == c)
        if np.sum(mask) > 0:
            u, cnt = np.unique(gmm_klasteriai[mask], return_counts=True)
            dominant[c] = u[np.argmax(cnt)]
        else:
            dominant[c] = None
    match_colors = [('gray' if dominant.get(int(t), None) == k else 'red')
                    for t, k in zip(tiksliosios_klases, gmm_klasteriai)]

    axes[2].scatter(X2[:, 0], X2[:, 1], c=match_colors, s=35, alpha=0.85,
                    edgecolors='black', linewidth=0.4)
    axes[2].set_title("t-SNE: atitikimai (pilka) / neatitikimai (raudona)")
    axes[2].set_xlabel('Dimensija 1'); axes[2].set_ylabel('Dimensija 2')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_gmm, f"{failo_pavadinimas}.png"), dpi=300)
    plt.close()


# ============================ VIENOS AIBĖS EIGA ============================

def gmm_su_silhouette(X_raw: np.ndarray, df_numeric: pd.DataFrame,
                      aibes_pavadinimas: str, failo_prefix: str,
                      k_min=2, k_max=18, covariance_type='full'):
    # 1) Standartizacija
    X_std = standartizuoti(X_raw)

    # 2) Optimalus k pagal vidutinį siluetą
    best_k, ks, sil_scores = best_k_by_silhouette_gmm(X_std, k_min=k_min, k_max=k_max,
                                                      covariance_type=covariance_type)

    # Silueto kreivė -> failas
    plt.figure(figsize=(8, 5))
    plt.plot(ks, sil_scores, marker='o')
    plt.axvline(best_k, ls='--')
    plt.title(f"Vidutinis siluetas pagal k (GMM) – {aibes_pavadinimas}, best k={best_k}")
    plt.xlabel("k (klasterių skaičius)")
    plt.ylabel("Silueto vidurkis (euclidean)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_gmm, f"silhouette_curve_{failo_prefix}.png"), dpi=300)
    plt.close()

    # 3) Galutinis GMM su best_k
    gmm = GaussianMixture(n_components=best_k, covariance_type=covariance_type,
                          random_state=RANDOM_STATE, n_init=1)
    gmm.fit(X_std)
    labels = gmm.predict(X_std)
    sil_best = silhouette_score(X_std, labels, metric='euclidean')

    # 4) Spausdinama suvestinė (kaip pavyzdyje – konsolėje)
    print(f"\n{aibes_pavadinimas}:")
    print(f"  - k kandidatų diapazonas: {k_min}..{k_max}")
    print("  - Silueto reikšmės: " + ", ".join([f"k={k}:{s:.4f}" for k, s in zip(ks, sil_scores)]))
    # Klasterių dydžiai
    uniq, cnts = np.unique(labels, return_counts=True)
    print("  - Klasterių dydžiai: " + ", ".join([f"kl{u}={c}" for u, c in zip(uniq, cnts)]))
    print(f"  - Pasirinktas k = {best_k} (max siluetas = {sil_best:.4f})")

    # 5) t-SNE vizualizacija (arba originali 2D, jei jau 2D)
    X2 = X_raw if X_raw.shape[1] <= 2 else tsne_2d(X_raw)
    plt.figure(figsize=(6, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='tab10', s=35)
    plt.title(f"{aibes_pavadinimas}: GMM (k={best_k})")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_gmm, f"gmm_{failo_prefix}.png"), dpi=300)
    plt.close()

    # 6) Išsaugome žymas
    labels_csv = os.path.join(base_dir_results, f"labels_{failo_prefix}.csv")
    pd.DataFrame({'cluster': labels}).to_csv(labels_csv, index=False)

    # 7) Per-klasterio statistika (spausdinama)
    spausdinti_klasterio_statistika(df_numeric, labels, aibes_pavadinimas)

    return labels, best_k, sil_best


# ============================ PAGRINDINĖ EIGA ============================

# Įkeliame 3 aibes (TAIP, GMM taikomas ir t-SNE 2D duomenims)
X_visi_pozymiai, df_visi = load_numeric_csv('../pilna_EKG_pupsniu_analize_uzpildyta_medianomis_visi_normuota_pagal_minmax.csv')
X_atrinkta, df_atrinkta   = load_numeric_csv('duomenys/atrinkta_aibe.csv')
X_2D, df_2D               = load_numeric_csv('duomenys/tsne_2d_data.csv')

# Palyginimui: ar turime 'label' t-SNE rinkinyje?
df_tsne_full = load_full_csv('duomenys/tsne_2d_data.csv')
tiksliosios_klases = df_tsne_full['label'].values if 'label' in df_tsne_full.columns else None

# Vykdome GMM + silhouette pasirinkimą visoms trims aibėms
labels_visi, k_visi, sil_visi = gmm_su_silhouette(X_visi_pozymiai, df_visi, 'Visi požymiai', 'visi_pozymiai')
labels_atr,  k_atr,  sil_atr  = gmm_su_silhouette(X_atrinkta,     df_atrinkta, 'Atrinkti požymiai', 'atrinkti_pozymiai')
labels_2D,   k_2d,   sil_2d   = gmm_su_silhouette(X_2D,            df_2D,      '2D (t-SNE)', '2D')

# Sujungtas 3-aibių grafikas (kaip pavyzdyje)
vizualizuoti_klasterius_sujungta(
    [X_visi_pozymiai, X_atrinkta, X_2D],
    [labels_visi, labels_atr, labels_2D],
    [f'visi požymiai (k={k_visi})', f'atrinkti požymiai (k={k_atr})', f't-SNE 2D (k={k_2d})'],
    'gmm_trys_aibes'
)

# Palyginimas su tiksliosiomis klasėmis (jei yra)
if tiksliosios_klases is not None and len(tiksliosios_klases) == X_2D.shape[0]:
    vizualizuoti_palyginima(X_2D, tiksliosios_klases, labels_2D, 'palyginimas_tsne_vs_gmm')

print("\n" + "="*60)
print("✓ Silueto kreivės ir t-SNE vizualizacijos išsaugotos į", base_dir_gmm)
print("✓ Žymos išsaugotos į", base_dir_results)
print("="*60)
