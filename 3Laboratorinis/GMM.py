# ===========================================
# GMM klasterizavimas: su ir be išskirčių
# (vienas failas, be atskirų aplankų)
# Reikalavimai:
# - *_trys_aibes* paveiksluose rodyti k kiekvienai aibei.
# - „be išskirčių“ eigoje NENAUDOTI naujo k parinkimo – naudoti "su išskirtimis" gautą k.
# - „be išskirčių“ klasterių paveiksluose NEPIEŠTI pašalintų išskirčių.
# - Sugeneruoti naują bendrą palyginimo paveikslą (su vs be išskirčių).
# - „palyginimas“ grafikuose skaičiuoti neatitikimo procentą ir perkelti paaiškinimą į legendą:
#   legendos įrašai „Atitinka“, „Neatitinka“, legendos pavadinimas „XX,XX % neatitinka“.
# - Spausdinti DUOMENIS sakiniui parašyti: neatitinkančių objektų skaičių kiekvienai klasei ir bendrą procentą.
# - Pridėta vėliava PRINT_CLUSTER_STATS klasterių statistikos spausdinimui.
# ===========================================

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

# ---- Vėliavos / nustatymai ----
PRINT_CLUSTER_STATS = False  # <- pakeiskite į True, jei reikia spausdinti klasterių statistiką
PERPLEXITY = 50
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
    return Xdf.values, Xdf  # numpy ir DataFrame statistikoms

def load_full_csv(path):
    return pd.read_csv(path, sep=';')

# ============================ PAGALBINĖS ============================

def standartizuoti(X: np.ndarray):
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    return scaler.fit_transform(X_imp)

def tsne_2d(X: np.ndarray):
    # Vizualizacijoms; jei jau 2D – grąžiname kaip yra.
    if X.shape[1] <= 2:
        return X
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, metric=TSNE_METRIC, random_state=RANDOM_STATE)
    return tsne.fit_transform(X)

def best_k_by_silhouette_gmm(X_std: np.ndarray, k_min=2, k_max=24, covariance_type='full'):
    ks, scores = [], []
    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                  random_state=RANDOM_STATE, n_init=1)
            gmm.fit(X_std)
            labels = gmm.predict(X_std)
            score = silhouette_score(X_std, labels, metric='euclidean')
        except Exception:
            score = -1.0
        ks.append(k)
        scores.append(score)
    best_idx = int(np.argmax(scores))
    return ks[best_idx], ks, scores

def spausdinti_klasterio_statistika(df_numeric: pd.DataFrame, labels: np.ndarray, aibes_pavadinimas: str):
    if not PRINT_CLUSTER_STATS:
        return
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
            print(
                f"  {col}: mean={mean:.6f}, median={median:.6f}, "
                f"std={std if not np.isnan(std) else np.nan}, min={vmin:.6f}, max={vmax:.6f}"
            )

def extreme_outlier_mask_iqr(X_std: np.ndarray, iqr_mult: float = 3.0):
    q1 = np.percentile(X_std, 25, axis=0)
    q3 = np.percentile(X_std, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr
    upper = q3 + iqr_mult * iqr
    mask = np.any((X_std < lower) | (X_std > upper), axis=1)
    return mask  # True = išskirtis

def format_percent(p: float) -> str:
    # 0.2033 -> "20,33 %"
    return f"{p*100:.2f}".replace('.', ',') + " %"

def mismatch_percentage(y_true: np.ndarray, y_pred_clusters: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    (Palikta suderinamumui) Grąžina neatitikimo procentą.
    """
    if mask is None:
        idx = np.ones_like(y_true, dtype=bool)
    else:
        idx = mask.astype(bool)

    y_t = y_true[idx]
    y_p = y_pred_clusters[idx]

    classes = np.unique(y_t)
    dominant = {}
    for c in classes:
        sel = (y_t == c)
        if np.any(sel):
            u, cnt = np.unique(y_p[sel], return_counts=True)
            dominant[c] = u[np.argmax(cnt)]
        else:
            dominant[c] = None
    matches = np.array([dominant.get(t, None) == k for t, k in zip(y_t, y_p)], dtype=bool)
    mismatch_rate = 1.0 - matches.mean() if len(matches) else 0.0
    return mismatch_rate

def mismatch_stats_by_class(y_true: np.ndarray, y_pred_clusters: np.ndarray, mask: np.ndarray | None = None):
    """
    Grąžina:
      - per_klase: {klase: neatitinkančių skaičius}
      - neatitinkantys_viso: int
      - viso: int
      - neatitikimo_dalis: float (0..1)
    """
    if mask is None:
        idx = np.ones_like(y_true, dtype=bool)
    else:
        idx = mask.astype(bool)

    y_t = y_true[idx]
    y_p = y_pred_clusters[idx]

    classes = np.unique(y_t)
    dominant = {}
    for c in classes:
        sel = (y_t == c)
        if np.any(sel):
            u, cnt = np.unique(y_p[sel], return_counts=True)
            dominant[c] = u[np.argmax(cnt)]
        else:
            dominant[c] = None

    matches = np.array([dominant.get(t, None) == k for t, k in zip(y_t, y_p)], dtype=bool)
    mism = ~matches
    per_klase = {}
    for c in classes:
        per_klase[c] = int(np.sum((y_t == c) & mism))
    neatitinkantys_viso = int(mism.sum())
    viso = int(len(y_t))
    neatitikimo_dalis = (neatitinkantys_viso / viso) if viso else 0.0
    return per_klase, neatitinkantys_viso, viso, neatitikimo_dalis

# ============================ BRAIŽYMAS ============================

def plot_outliers_overview(X_raw: np.ndarray, out_mask: np.ndarray, title: str, save_path: str):
    # Tik informacinis žemėlapis prieš šalinimą
    X2 = X_raw if X_raw.shape[1] <= 2 else tsne_2d(X_raw)
    plt.figure(figsize=(6, 5))
    plt.scatter(X2[~out_mask, 0], X2[~out_mask, 1], s=20, alpha=0.6, c='lightgray', label='duomenys')
    plt.scatter(X2[out_mask, 0], X2[out_mask, 1], s=35, c='black', marker='*', label='kritinės išskirtys')
    plt.title(title)
    leg = plt.legend(loc='best', frameon=True)
    leg.get_frame().set_edgecolor('black'); leg.get_frame().set_linewidth(0.8)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def scatter_clusters(X2: np.ndarray, labels: np.ndarray, title: str, save_path: str, exclude_mask: np.ndarray | None = None):
    """
    Jei exclude_mask pateikta, į grafiką NEįtraukiami tie taškai (pvz., pašalintos išskirtys).
    """
    plt.figure(figsize=(6, 5))
    if exclude_mask is None:
        plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='tab10', s=35)
    else:
        keep = ~exclude_mask
        plt.scatter(X2[keep, 0], X2[keep, 1], c=labels[keep], cmap='tab10', s=35)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def vizualizuoti_klasterius_sujungta(X2_list, labels_list, titles_list, save_name: str, exclude_masks=None):
    """
    Trijų aibių sujungtas grafikas (1x3).
    exclude_masks (sąrašas) – jei pateikta, atitinkami taškai nebraižomi.
    """
    if exclude_masks is None:
        exclude_masks = [None] * len(X2_list)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, X2, lbls, ttl, ex in zip(axes, X2_list, labels_list, titles_list, exclude_masks):
        if ex is None:
            ax.scatter(X2[:, 0], X2[:, 1], c=lbls, cmap='tab10', s=35)
        else:
            keep = ~ex
            ax.scatter(X2[keep, 0], X2[keep, 1], c=lbls[keep], cmap='tab10', s=35)
        ax.set_title(f"GMM\n({ttl})")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_gmm, save_name), dpi=300)
    plt.close()

def vizualizuoti_palyginima(X2: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                            dataset_tag: str, save_name: str, exclude_mask: np.ndarray | None = None):
    """
    3 pan., kairė – tikslios klasės, vidurys – GMM klasteriai, dešinė – atitikimai.
    Neatitikimo procentas paskaičiuojamas ir perkeliamas į legendą (legendos title).
    exclude_mask – jei pateikta, braižoma tik ten, kur exclude_mask==False (pvz., be išskirčių).
    Taip pat atspausdinami skaičiai „kiek neatitinka kiekvienoje klasėje“ ir bendras procentas.
    """
    if exclude_mask is None:
        keep = np.ones(len(y_true), dtype=bool)
    else:
        keep = ~exclude_mask

    X2k = X2[keep]; yt = y_true[keep]; yp = y_pred[keep]

    # Statistika neatitikimams
    per_klase, mism_cnt, total_cnt, mism_rate = mismatch_stats_by_class(yt, yp, mask=None)
    mismatch_txt = f"{format_percent(mism_rate)} neatitinka"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(X2k[:, 0], X2k[:, 1], c=yt, cmap='viridis', s=35, alpha=0.8)
    axes[0].set_title(f"{dataset_tag}: tiksliosios klasės")

    axes[1].scatter(X2k[:, 0], X2k[:, 1], c=yp, cmap='tab10', s=35, alpha=0.8)
    axes[1].set_title(f"{dataset_tag}: GMM klasteriai")

    # Atitikimų/Neatitikimų žemėlapis
    classes = np.unique(yt)
    dominant = {}
    for c in classes:
        mask_c = (yt == c)
        u, cnt = np.unique(yp[mask_c], return_counts=True)
        dominant[c] = u[np.argmax(cnt)]
    match = np.array([dominant.get(t, None) == k for t, k in zip(yt, yp)], dtype=bool)
    colors = np.where(match, 'gray', 'red')
    axes[2].scatter(X2k[:, 0], X2k[:, 1], c=colors, s=35, alpha=0.9, edgecolors='black', linewidth=0.4)
    axes[2].set_title(f"{dataset_tag}: atitikimas pagal klases")

    # Legenda su paaiškinimu (legendos pavadinimas – neatitikimo procentas)
    handles = [
        Line2D([0], [0], marker='o', linestyle='', color='gray', label='Atitinka', markersize=7),
        Line2D([0], [0], marker='o', linestyle='', color='red',  label='Neatitinka', markersize=7),
    ]
    leg = axes[2].legend(handles=handles, title=mismatch_txt, frameon=True, loc='best')
    leg.get_frame().set_edgecolor('black'); leg.get_frame().set_linewidth(0.8)

    for ax in axes:
        ax.set_xlabel('Dimensija 1'); ax.set_ylabel('Dimensija 2')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_gmm, save_name), dpi=300)
    plt.close()

    # Spausdiname DUOMENIS (be sakinio), kaip paprašyta:
    print(f"Bendrai neatitinkančių objektų: {mism_cnt} / {total_cnt} ({format_percent(mism_rate)})")
    for klas, kiekis in sorted(per_klase.items(), key=lambda x: x[0]):
        print(f"Neatitinkančių objektų kiekis {klas} klasei: {kiekis}")

# ============================ VIENOS AIBĖS EIGA ============================

def gmm_with_outliers(X_raw: np.ndarray, df_numeric: pd.DataFrame,
                      aibes_pavadinimas: str, failo_prefix: str,
                      k_min=2, k_max=24, covariance_type='full'):
    """
    Pilna eiga SU išskirtimis:
    - stand., siluetas -> best_k
    - GMM (best_k) ant visų taškų
    - 2D vizualizacija (visiems)
    Grąžina: labels, best_k, X_std, X2 (vizualizacijai)
    """
    X_std = standartizuoti(X_raw)
    best_k, ks, sil_scores = best_k_by_silhouette_gmm(X_std, k_min=k_min, k_max=k_max,
                                                      covariance_type=covariance_type)

    # Silueto kreivė
    plt.figure(figsize=(8, 5))
    plt.plot(ks, sil_scores, marker='o'); plt.axvline(best_k, ls='--')
    plt.title(f"Vidutinis siluetas pagal k (GMM) – {aibes_pavadinimas}, best k={best_k}")
    plt.xlabel("k (klasterių skaičius)"); plt.ylabel("Silueto vidurkis (euclidean)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_gmm, f"silhouette_curve_{failo_prefix}.png"), dpi=300)
    plt.close()

    # GMM
    gmm = GaussianMixture(n_components=best_k, covariance_type=covariance_type,
                          random_state=RANDOM_STATE, n_init=1)
    gmm.fit(X_std)
    labels = gmm.predict(X_std)
    sil = silhouette_score(X_std, labels, metric='euclidean')

    # 2D vaizdas
    X2 = X_raw if X_raw.shape[1] <= 2 else tsne_2d(X_raw)
    scatter_clusters(
        X2, labels,
        f"{aibes_pavadinimas}: GMM (k={best_k})",
        os.path.join(base_dir_gmm, f"gmm_{failo_prefix}_with_outliers.png"),
        exclude_mask=None
    )

    # Išsaugojimai / logai
    pd.DataFrame({'cluster': labels}).to_csv(os.path.join(base_dir_results, f"labels_{failo_prefix}_with_outliers.csv"), index=False)

    # (neprivaloma) statistika
    spausdinti_klasterio_statistika(df_numeric, labels, aibes_pavadinimas)

    # Konsolė
    u, c = np.unique(labels, return_counts=True)
    print(f"\n{aibes_pavadinimas} [su išskirtimis]: k={best_k}, siluetas={sil:.4f}, klasterių dydžiai: " +
          ", ".join([f"kl{uu}={cc}" for uu, cc in zip(u, c)]))

    return labels, best_k, X_std, X2

def gmm_no_outliers_with_fixed_k(X_raw: np.ndarray, df_numeric: pd.DataFrame,
                                 aibes_pavadinimas: str, failo_prefix: str,
                                 k_fixed: int, covariance_type='full'):
    """
    Eiga BE išskirčių:
    - stand., detektuojamos išskirtys (vieną kartą)
    - GMM treniruojamas su INLIERIAIS, su NUSTATYTU k_fixed (NENAUJAS parinkimas)
    - vizualizacija tik INLIERIAMS (išskirtys nebraižomos)
    Grąžina: labels_all (prognozės visiems), out_mask, X2_inliers
    """
    X_std = standartizuoti(X_raw)
    out_mask = extreme_outlier_mask_iqr(X_std, iqr_mult=3.0)
    inliers = ~out_mask

    # Informacinis žemėlapis (prieš šalinimą)
    plot_outliers_overview(
        X_raw, out_mask,
        title=f"{aibes_pavadinimas}: kritinių išskirčių žymėjimas",
        save_path=os.path.join(base_dir_gmm, f"outliers_marked_{failo_prefix}.png")
    )

    # GMM tik su inlieriais (naudojant k_fixed)
    X_in = X_std[inliers]
    gmm = GaussianMixture(n_components=k_fixed, covariance_type=covariance_type,
                          random_state=RANDOM_STATE, n_init=1).fit(X_in)
    # Prognozės skaičiavimams (pilnam rinkiniui), bet braižymui – tik inlieriai
    labels_all = gmm.predict(X_std)

    # 2D tik inlieriams (išskirtys nebraižomos)
    X2_in = X_raw[inliers] if X_raw.shape[1] <= 2 else tsne_2d(X_raw[inliers])

    # Braižome tik inlierius
    scatter_clusters(
        X2_in, labels_all[inliers],
        f"{aibes_pavadinimas}: GMM (k={k_fixed}) [be išskirčių]",
        os.path.join(base_dir_gmm, f"gmm_{failo_prefix}_no_outliers.png"),
        exclude_mask=None
    )

    # Išsaugojimai / logai
    df_out = pd.DataFrame({'cluster': labels_all, 'is_outlier': out_mask.astype(int)})
    df_out.to_csv(os.path.join(base_dir_results, f"labels_{failo_prefix}_no_outliers.csv"), index=False)

    # (neprivaloma) statistika – pagal prognozes (visiems)
    spausdinti_klasterio_statistika(df_numeric, labels_all, f"{aibes_pavadinimas} [be išskirčių]")

    # Konsolė
    n_out = int(out_mask.sum())
    u, c = np.unique(labels_all[inliers], return_counts=True)
    print(f"\n{aibes_pavadinimas} [be išskirčių]: k={k_fixed}, pašalinta iš mokymo {n_out}/{len(out_mask)} "
          f"({100*n_out/len(out_mask):.2f}%), klasterių dydžiai (inlieriams): " +
          ", ".join([f"kl{uu}={cc}" for uu, cc in zip(u, c)]))

    return labels_all, out_mask, X2_in

# ============================ PAGRINDINĖ EIGA ============================

if __name__ == "__main__":
    # Keliai
    kelias_visi = '../pilna_EKG_pupsniu_analize_uzpildyta_medianomis_visi_normuota_pagal_minmax.csv'
    kelias_atr  = 'duomenys/atrinkta_aibe.csv'
    kelias_2d   = 'duomenys/tsne_2d_data.csv'

    # Įkėlimas
    X_visi_pozymiai, df_visi = load_numeric_csv(kelias_visi)
    X_atrinkta, df_atrinkta   = load_numeric_csv(kelias_atr)
    X_2D, df_2D               = load_numeric_csv(kelias_2d)

    # Dataset žymos
    tag_visi = 'visi požymiai'
    tag_atr  = 'atrinkti požymiai'
    tag_2d   = '2D duomenys'

    # ----- SU išskirtimis -----
    labels_visi_w, k_visi, Xstd_visi_w, X2_visi_w = gmm_with_outliers(X_visi_pozymiai, df_visi, 'Visi požymiai', 'visi_pozymiai')
    labels_atr_w,  k_atr,  Xstd_atr_w,  X2_atr_w  = gmm_with_outliers(X_atrinkta,     df_atrinkta, 'Atrinkti požymiai', 'atrinkti_pozymiai')
    labels_2d_w,   k_2d,   Xstd_2d_w,   X2_2d_w   = gmm_with_outliers(X_2D,            df_2D,      '2D duomenys', '2D')

    # Sujungtas (su išskirtimis) – pavadinimuose rodom k
    vizualizuoti_klasterius_sujungta(
        [X2_visi_w, X2_atr_w, X2_2d_w],
        [labels_visi_w, labels_atr_w, labels_2d_w],
        [f'{tag_visi} (k={k_visi})', f'{tag_atr} (k={k_atr})', f'{tag_2d} (k={k_2d})'],
        save_name='gmm_trys_aibes_with_outliers.png',
        exclude_masks=None
    )

    # ----- BE išskirčių (naudojant k iš „su išskirtimis“) -----
    labels_visi_n, out_visi, X2_visi_n = gmm_no_outliers_with_fixed_k(
        X_visi_pozymiai, df_visi, 'Visi požymiai', 'visi_pozymiai', k_fixed=k_visi
    )
    labels_atr_n,  out_atr,  X2_atr_n  = gmm_no_outliers_with_fixed_k(
        X_atrinkta,     df_atrinkta, 'Atrinkti požymiai', 'atrinkti_pozymiai', k_fixed=k_atr
    )
    labels_2d_n,   out_2d,   X2_2d_n   = gmm_no_outliers_with_fixed_k(
        X_2D,            df_2D,      '2D duomenys', '2D', k_fixed=k_2d
    )

    # Sujungtas (be išskirčių) – piešiame TIK inlierius; pavadinimuose rodom originalų k
    vizualizuoti_klasterius_sujungta(
        [X2_visi_n, X2_atr_n, X2_2d_n],
        [labels_visi_n[~out_visi], labels_atr_n[~out_atr], labels_2d_n[~out_2d]],
        [f'{tag_visi} (k={k_visi})', f'{tag_atr} (k={k_atr})', f'{tag_2d} (k={k_2d})'],
        save_name='gmm_trys_aibes_no_outliers.png',
        exclude_masks=None
    )

    # ----- Bendras palyginimo paveikslas: su vs be išskirčių -----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Eilutė 1: su išskirtimis
    axes[0,0].scatter(X2_visi_w[:,0], X2_visi_w[:,1], c=labels_visi_w, cmap='tab10', s=30)
    axes[0,0].set_title(f"{tag_visi} – su išskirtimis (k={k_visi})")
    axes[0,1].scatter(X2_atr_w[:,0], X2_atr_w[:,1], c=labels_atr_w, cmap='tab10', s=30)
    axes[0,1].set_title(f"{tag_atr} – su išskirtimis (k={k_atr})")
    axes[0,2].scatter(X2_2d_w[:,0], X2_2d_w[:,1], c=labels_2d_w, cmap='tab10', s=30)
    axes[0,2].set_title(f"{tag_2d} – su išskirtimis (k={k_2d})")
    # Eilutė 2: be išskirčių (piešiami tik inlieriai)
    axes[1,0].scatter(X2_visi_n[:,0], X2_visi_n[:,1], c=labels_visi_n[~out_visi], cmap='tab10', s=30)
    axes[1,0].set_title(f"{tag_visi} – be išskirčių (k={k_visi})")
    axes[1,1].scatter(X2_atr_n[:,0], X2_atr_n[:,1], c=labels_atr_n[~out_atr], cmap='tab10', s=30)
    axes[1,1].set_title(f"{tag_atr} – be išskirčių (k={k_atr})")
    axes[1,2].scatter(X2_2d_n[:,0], X2_2d_n[:,1], c=labels_2d_n[~out_2d], cmap='tab10', s=30)
    axes[1,2].set_title(f"{tag_2d} – be išskirčių (k={k_2d})")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_gmm, "gmm_trys_aibes_compare_with_vs_no_outliers.png"), dpi=300)
    plt.close()

    # ----- PALYGINIMAI su tiksliosiomis klasėmis (legendos su paaiškinimu) -----
    df_visi_full = load_full_csv(kelias_visi)
    df_atr_full  = load_full_csv(kelias_atr)
    df_tsne_full = load_full_csv(kelias_2d)

    tikslios_visi = df_visi_full['label'].values if 'label' in df_visi_full.columns else None
    tikslios_atr  = df_atr_full['label'].values  if 'label' in df_atr_full.columns  else None
    tikslios_2d   = df_tsne_full['label'].values if 'label' in df_tsne_full.columns else None

    # SU išskirtimis (vertinama ant VISŲ taškų)
    if tikslios_visi is not None and len(tikslios_visi) == X_visi_pozymiai.shape[0]:
        vizualizuoti_palyginima(
            X2_visi_w, tikslios_visi, labels_visi_w,
            dataset_tag=tag_visi,
            save_name='palyginimas_visi_vs_gmm_with_outliers.png',
            exclude_mask=None
        )
    if tikslios_atr is not None and len(tikslios_atr) == X_atrinkta.shape[0]:
        vizualizuoti_palyginima(
            X2_atr_w, tikslios_atr, labels_atr_w,
            dataset_tag=tag_atr,
            save_name='palyginimas_atrinkti_vs_gmm_with_outliers.png',
            exclude_mask=None
        )
    if tikslios_2d is not None and len(tikslios_2d) == X_2D.shape[0]:
        vizualizuoti_palyginima(
            X2_2d_w, tikslios_2d, labels_2d_w,
            dataset_tag=tag_2d,
            save_name='palyginimas_tsne_vs_gmm_with_outliers.png',
            exclude_mask=None
        )

    # BE išskirčių (vertinama tik INLIERIAMS – atitinka grafikus)
    if tikslios_visi is not None and len(tikslios_visi) == X_visi_pozymiai.shape[0]:
        vizualizuoti_palyginima(
            X2_visi_w,  # koordinatės tinka, atranka per exclude_mask
            tikslios_visi, labels_visi_n,
            dataset_tag=f"{tag_visi} (be išskirčių)",
            save_name='palyginimas_visi_vs_gmm_no_outliers.png',
            exclude_mask=out_visi
        )
    if tikslios_atr is not None and len(tikslios_atr) == X_atrinkta.shape[0]:
        vizualizuoti_palyginima(
            X2_atr_w,
            tikslios_atr, labels_atr_n,
            dataset_tag=f"{tag_atr} (be išskirčių)",
            save_name='palyginimas_atrinkti_vs_gmm_no_outliers.png',
            exclude_mask=out_atr
        )
    if tikslios_2d is not None and len(tikslios_2d) == X_2D.shape[0]:
        vizualizuoti_palyginima(
            X2_2d_w,
            tikslios_2d, labels_2d_n,
            dataset_tag=f"{tag_2d} (be išskirčių)",
            save_name='palyginimas_tsne_vs_gmm_no_outliers.png',
            exclude_mask=out_2d
        )

    print("\n" + "="*60)
    print("✓ Paveikslai išsaugoti į:", base_dir_gmm)
    print("   - gmm_trys_aibes_with_outliers.png")
    print("   - gmm_trys_aibes_no_outliers.png")
    print("   - gmm_trys_aibes_compare_with_vs_no_outliers.png")
    print("   - palyginimas_*_with_outliers.png ir palyginimas_*_no_outliers.png")
    print("✓ Žymos (CSV) išsaugotos į:", base_dir_results)
    print("="*60)
