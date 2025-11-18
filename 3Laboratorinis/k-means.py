import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

PRINT_CLUSTER_STATS = True
PERPLEXITY = 50
TSNE_METRIC = 'canberra'
RANDOM_STATE = 42

base_dir_kmeans = 'grafikai/kmeans'
base_dir_results = 'rezultatai'
os.makedirs(base_dir_kmeans, exist_ok=True)
os.makedirs(base_dir_results, exist_ok=True)

def load_numeric_csv(path):
    df = pd.read_csv(path, sep=';')
    features = [c for c in df.columns if c != 'label']
    Xdf = df[features].apply(pd.to_numeric, errors='coerce').dropna()
    return Xdf.values, Xdf

def load_full_csv(path):
    return pd.read_csv(path, sep=';')

def standartizuoti(X: np.ndarray):
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    return scaler.fit_transform(X_imp)

def tsne_2d(X: np.ndarray):
    if X.shape[1] <= 2:
        return X
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, metric=TSNE_METRIC, random_state=RANDOM_STATE)
    return tsne.fit_transform(X)

def best_k_by_elbow_kmeans(X_std: np.ndarray, k_min=2, k_max=24):
    ks, inertias = [], []
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            km.fit(X_std)
            inertias.append(float(km.inertia_))
        except Exception:
            inertias.append(np.inf)
        ks.append(k)

    best_k = None
    kl = KneeLocator(
        ks, inertias,
        curve='convex', direction='decreasing',
        interp_method='polynomial'
    )
    if kl.elbow is not None:
        best_k = int(kl.elbow)

    return best_k, ks, inertias

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
    return mask

def format_percent(p: float) -> str:
    return f"{p*100:.2f}".replace('.', ',') + " %"

def mismatch_percentage(y_true: np.ndarray, y_pred_clusters: np.ndarray, mask: np.ndarray | None = None) -> float:
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


def plot_outliers_overview(X_raw: np.ndarray, out_mask: np.ndarray, title: str, save_path: str):
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
    plt.figure(figsize=(6, 5))
    if exclude_mask is None:
        scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='tab10', s=35)
        plot_labels = labels
    else:
        keep = ~exclude_mask
        scatter = plt.scatter(X2[keep, 0], X2[keep, 1], c=labels[keep], cmap='tab10', s=35)
        plot_labels = labels[keep]
    
    plt.title(title)
    
    # Add legend with cluster labels
    unique_clusters = np.unique(plot_labels)
    legend_elements = [Patch(facecolor=scatter.cmap(scatter.norm(cluster)), 
                             label=f'Klasteris {cluster}')
                      for cluster in unique_clusters]
    leg = plt.legend(handles=legend_elements, loc='best', fontsize=8, frameon=True)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.8)
    
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def vizualizuoti_klasterius_sujungta(X2_list, labels_list, titles_list, save_name: str, exclude_masks=None):
    if exclude_masks is None:
        exclude_masks = [None] * len(X2_list)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, X2, lbls, ttl, ex in zip(axes, X2_list, labels_list, titles_list, exclude_masks):
        if ex is None:
            scatter = ax.scatter(X2[:, 0], X2[:, 1], c=lbls, cmap='tab10', s=35)
            plot_labels = lbls
        else:
            keep = ~ex
            scatter = ax.scatter(X2[keep, 0], X2[keep, 1], c=lbls[keep], cmap='tab10', s=35)
            plot_labels = lbls[keep]
        
        ax.set_title(f"KMeans\n({ttl})")
        ax.set_xticks([]); ax.set_yticks([])
        
        # Add legend with cluster labels
        unique_clusters = np.unique(plot_labels)
        legend_elements = [Patch(facecolor=scatter.cmap(scatter.norm(cluster)), 
                                 label=f'Klasteris {cluster}')
                          for cluster in unique_clusters]
        leg = ax.legend(handles=legend_elements, loc='best', fontsize=7, frameon=True)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_kmeans, save_name), dpi=300)
    plt.close()

def vizualizuoti_palyginima(X2: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                            dataset_tag: str, save_name: str, exclude_mask: np.ndarray | None = None):
    if exclude_mask is None:
        keep = np.ones(len(y_true), dtype=bool)
    else:
        keep = ~exclude_mask

    X2k = X2[keep]; yt = y_true[keep]; yp = y_pred[keep]

    per_klase, mism_cnt, total_cnt, mism_rate = mismatch_stats_by_class(yt, yp, mask=None)
    mismatch_txt = f"{format_percent(mism_rate)} neatitinka"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # First plot: True classes with legend
    scatter1 = axes[0].scatter(X2k[:, 0], X2k[:, 1], c=yt, cmap='viridis', s=35, alpha=0.8)
    axes[0].set_title(f"{dataset_tag}: tiksliosios klasės")
    unique_classes = np.unique(yt)
    legend_elements_classes = [Patch(facecolor=scatter1.cmap(scatter1.norm(cls)), 
                                     label=f'Klasė {int(cls)}')
                               for cls in unique_classes]
    leg1 = axes[0].legend(handles=legend_elements_classes, loc='best', fontsize=8, frameon=True)
    leg1.get_frame().set_edgecolor('black')
    leg1.get_frame().set_linewidth(0.8)

    # Second plot: K-means clusters with legend
    scatter2 = axes[1].scatter(X2k[:, 0], X2k[:, 1], c=yp, cmap='tab10', s=35, alpha=0.8)
    axes[1].set_title(f"{dataset_tag}: KMeans klasteriai")
    unique_clusters = np.unique(yp)
    legend_elements_clusters = [Patch(facecolor=scatter2.cmap(scatter2.norm(cluster)), 
                                      label=f'Klasteris {cluster}')
                                for cluster in unique_clusters]
    leg2 = axes[1].legend(handles=legend_elements_clusters, loc='best', fontsize=8, frameon=True)
    leg2.get_frame().set_edgecolor('black')
    leg2.get_frame().set_linewidth(0.8)

    # Third plot: Match/mismatch
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

    handles = [
        Line2D([0], [0], marker='o', linestyle='', color='gray', label='Atitinka', markersize=7),
        Line2D([0], [0], marker='o', linestyle='', color='red',  label='Neatitinka', markersize=7),
    ]
    leg3 = axes[2].legend(handles=handles, title=mismatch_txt, frameon=True, loc='best')
    leg3.get_frame().set_edgecolor('black'); leg3.get_frame().set_linewidth(0.8)

    for ax in axes:
        ax.set_xlabel('Dimensija 1'); ax.set_ylabel('Dimensija 2')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_kmeans, save_name), dpi=300)
    plt.close()

    print(f"Bendrai neatitinkančių objektų: {mism_cnt} / {total_cnt} ({format_percent(mism_rate)})")
    for klas, kiekis in sorted(per_klase.items(), key=lambda x: x[0]):
        print(f"Neatitinkančių objektų kiekis {int(klas)} klasei: {kiekis}")


def kmeans_with_outliers(X_raw: np.ndarray, df_numeric: pd.DataFrame,
                         aibes_pavadinimas: str, failo_prefix: str,
                         k_min=2, k_max=24):
    X_std = standartizuoti(X_raw)
    best_k, ks, inertias = best_k_by_elbow_kmeans(X_std, k_min=k_min, k_max=k_max)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o')
    plt.axvline(best_k, ls='--')
    plt.title(f"Optimalus k pagal Elbow – {aibes_pavadinimas}, best k={best_k}")
    plt.xlabel("k (klasterių skaičius)")
    plt.ylabel("Kvadratinė paklaida")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_kmeans, f"elbow_curve_{failo_prefix}.png"), dpi=300)
    plt.close()

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_std)

    X2 = X_raw if X_raw.shape[1] <= 2 else tsne_2d(X_raw)
    scatter_clusters(
        X2, labels,
        f"{aibes_pavadinimas}: KMeans (k={best_k})",
        os.path.join(base_dir_kmeans, f"kmeans_{failo_prefix}_with_outliers.png"),
        exclude_mask=None
    )

    pd.DataFrame({'cluster': labels}).to_csv(os.path.join(base_dir_results, f"labels_{failo_prefix}_with_outliers.csv"), index=False)

    spausdinti_klasterio_statistika(df_numeric, labels, aibes_pavadinimas)

    u, c = np.unique(labels, return_counts=True)
    print(f"\n{aibes_pavadinimas} [su išskirtimis]: k={best_k}, klasterių dydžiai: " +
          ", ".join([f"kl{uu}={cc}" for uu, cc in zip(u, c)]))

    return labels, best_k, X_std, X2

def kmeans_no_outliers_with_fixed_k(X_raw: np.ndarray, df_numeric: pd.DataFrame,
                                    aibes_pavadinimas: str, failo_prefix: str,
                                    k_fixed: int):
    X_std = standartizuoti(X_raw)
    out_mask = extreme_outlier_mask_iqr(X_std, iqr_mult=3.0)
    inliers = ~out_mask

    plot_outliers_overview(
        X_raw, out_mask,
        title=f"{aibes_pavadinimas}: kritinių išskirčių žymėjimas",
        save_path=os.path.join(base_dir_kmeans, f"outliers_marked_{failo_prefix}.png")
    )

    X_in = X_std[inliers]
    km = KMeans(n_clusters=k_fixed, random_state=RANDOM_STATE, n_init=10).fit(X_in)
    labels_all = km.predict(X_std)

    X2_in = X_raw[inliers] if X_raw.shape[1] <= 2 else tsne_2d(X_raw[inliers])

    scatter_clusters(
        X2_in, labels_all[inliers],
        f"{aibes_pavadinimas}: KMeans (k={k_fixed}) [be išskirčių]",
        os.path.join(base_dir_kmeans, f"kmeans_{failo_prefix}_no_outliers.png"),
        exclude_mask=None
    )

    df_out = pd.DataFrame({'cluster': labels_all, 'is_outlier': out_mask.astype(int)})
    df_out.to_csv(os.path.join(base_dir_results, f"labels_{failo_prefix}_no_outliers.csv"), index=False)

    spausdinti_klasterio_statistika(df_numeric, labels_all, f"{aibes_pavadinimas} [be išskirčių]")

    n_out = int(out_mask.sum())
    u, c = np.unique(labels_all[inliers], return_counts=True)
    print(f"\n{aibes_pavadinimas} [be išskirčių]: k={k_fixed}, pašalinta iš mokymo {n_out}/{len(out_mask)} "
          f"({100*n_out/len(out_mask):.2f}%), klasterių dydžiai (inlieriams): " +
          ", ".join([f"kl{uu}={cc}" for uu, cc in zip(u, c)]))

    return labels_all, out_mask, X2_in

def _save_k_vs_kp1(X2, labels_k, labels_kp1, tag, k, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # k clusters
    scatter1 = axes[0].scatter(X2[:, 0], X2[:, 1], c=labels_k, cmap='tab10', s=30)
    axes[0].set_title(f"{tag} – k={k}")
    unique_k = np.unique(labels_k)
    legend_elements_k = [Patch(facecolor=scatter1.cmap(scatter1.norm(cluster)), 
                               label=f'Klasteris {cluster}')
                        for cluster in unique_k]
    leg1 = axes[0].legend(handles=legend_elements_k, loc='best', fontsize=8, frameon=True)
    leg1.get_frame().set_edgecolor('black')
    leg1.get_frame().set_linewidth(0.8)
    
    # k+1 clusters
    scatter2 = axes[1].scatter(X2[:, 0], X2[:, 1], c=labels_kp1, cmap='tab10', s=30)
    axes[1].set_title(f"{tag} – k={k+1}")
    unique_kp1 = np.unique(labels_kp1)
    legend_elements_kp1 = [Patch(facecolor=scatter2.cmap(scatter2.norm(cluster)), 
                                 label=f'Klasteris {cluster}')
                          for cluster in unique_kp1]
    leg2 = axes[1].legend(handles=legend_elements_kp1, loc='best', fontsize=8, frameon=True)
    leg2.get_frame().set_edgecolor('black')
    leg2.get_frame().set_linewidth(0.8)
    
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_kmeans, filename), dpi=300)
    plt.close()

if __name__ == "__main__":
    kelias_visi = '../pilna_EKG_pupsniu_analize_uzpildyta_medianomis_visi_normuota_pagal_minmax.csv'
    kelias_atr  = 'duomenys/atrinkta_aibe.csv'
    # kelias_2d   = 'duomenys/tsne_2d_data_be_isskirciu.csv'
    kelias_2d   = 'duomenys/tsne_2d_data.csv'

    X_visi_pozymiai, df_visi = load_numeric_csv(kelias_visi)
    X_atrinkta, df_atrinkta   = load_numeric_csv(kelias_atr)
    X_2D, df_2D               = load_numeric_csv(kelias_2d)

    tag_visi = 'visi požymiai'
    tag_atr  = 'atrinkti požymiai'
    tag_2d   = '2D duomenys'

    labels_visi_w, k_visi, Xstd_visi_w, X2_visi_w = kmeans_with_outliers(X_visi_pozymiai, df_visi, 'Visi požymiai', 'visi_pozymiai')
    labels_atr_w,  k_atr,  Xstd_atr_w,  X2_atr_w  = kmeans_with_outliers(X_atrinkta,     df_atrinkta, 'Atrinkti požymiai', 'atrinkti_pozymiai')
    labels_2d_w, k_2d, Xstd_2d_w, X2_2d_w   = kmeans_no_outliers_with_fixed_k(X_2D,            df_2D,      '2D duomenys', '2D',k_fixed=k_2d)
    labels_2d_w,   k_2d,   Xstd_2d_w,   X2_2d_w   = kmeans_with_outliers(X_2D,            df_2D,      '2D duomenys', '2D')
    # k_2d = 6

    vizualizuoti_klasterius_sujungta(
        [X2_visi_w, X2_atr_w, X2_2d_w],
        [labels_visi_w, labels_atr_w, labels_2d_w],
        [f'{tag_visi} (k={k_visi})', f'{tag_atr} (k={k_atr})', f'{tag_2d} (k={k_2d})'],
        save_name='kmeans_trys_aibes_with_outliers.png',
        exclude_masks=None
    )

    # #Palyginimas k, k+1
    # labels_visi_w_kp1 = KMeans(n_clusters=k_visi + 1, random_state=RANDOM_STATE, n_init=10).fit_predict(Xstd_visi_w)
    # labels_atr_w_kp1  = KMeans(n_clusters=k_atr  + 1, random_state=RANDOM_STATE, n_init=10).fit_predict(Xstd_atr_w)
    # labels_2d_w_kp1   = KMeans(n_clusters=k_2d   + 1, random_state=RANDOM_STATE, n_init=10).fit_predict(Xstd_2d_w)

    # # 1) Visi požymiai
    # _save_k_vs_kp1(
    #     X2_visi_w, labels_visi_w, labels_visi_w_kp1,
    #     tag_visi, k_visi,
    #     "kmeans_visi_k_vs_kplus1_with_outliers.png"
    # )

    # # 2) Atrinkti požymiai
    # _save_k_vs_kp1(
    #     X2_atr_w, labels_atr_w, labels_atr_w_kp1,
    #     tag_atr, k_atr,
    #     "kmeans_atr_k_vs_kplus1_with_outliers.png"
    # )

    # # 3) 2D duomenys
    # _save_k_vs_kp1(
    #     X2_2d_w, labels_2d_w, labels_2d_w_kp1,
    #     tag_2d, k_2d,
    #     "kmeans_2d_k_vs_kplus1_with_outliers.png"
    # )

    labels_visi_n, out_visi, X2_visi_n = kmeans_no_outliers_with_fixed_k(
        X_visi_pozymiai, df_visi, 'Visi požymiai', 'visi_pozymiai', k_fixed=k_visi
    )
    labels_atr_n,  out_atr,  X2_atr_n  = kmeans_no_outliers_with_fixed_k(
        X_atrinkta,     df_atrinkta, 'Atrinkti požymiai', 'atrinkti_pozymiai', k_fixed=k_atr
    )
    labels_2d_n,   out_2d,   X2_2d_n   = kmeans_no_outliers_with_fixed_k(
        X_2D,            df_2D,      '2D duomenys', '2D', k_fixed=k_2d
    )

    vizualizuoti_klasterius_sujungta(
        [X2_visi_n, X2_atr_n, X2_2d_n],
        [labels_visi_n[~out_visi], labels_atr_n[~out_atr], labels_2d_n[~out_2d]],
        [f'{tag_visi} (k={k_visi})', f'{tag_atr} (k={k_atr})', f'{tag_2d} (k={k_2d})'],
        save_name='kmeans_trys_aibes_no_outliers.png',
        exclude_masks=None
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # su išskirtimis
    scatter00 = axes[0,0].scatter(X2_visi_w[:,0], X2_visi_w[:,1], c=labels_visi_w, cmap='tab10', s=30)
    axes[0,0].set_title(f"{tag_visi} – su išskirtimis (k={k_visi})")
    unique_00 = np.unique(labels_visi_w)
    legend_00 = [Patch(facecolor=scatter00.cmap(scatter00.norm(c)), label=f'Klasteris {c}') for c in unique_00]
    axes[0,0].legend(handles=legend_00, loc='best', fontsize=6, frameon=True)
    
    scatter01 = axes[0,1].scatter(X2_atr_w[:,0], X2_atr_w[:,1], c=labels_atr_w, cmap='tab10', s=30)
    axes[0,1].set_title(f"{tag_atr} – su išskirtimis (k={k_atr})")
    unique_01 = np.unique(labels_atr_w)
    legend_01 = [Patch(facecolor=scatter01.cmap(scatter01.norm(c)), label=f'Klasteris {c}') for c in unique_01]
    axes[0,1].legend(handles=legend_01, loc='best', fontsize=6, frameon=True)
    
    scatter02 = axes[0,2].scatter(X2_2d_w[:,0], X2_2d_w[:,1], c=labels_2d_w, cmap='tab10', s=30)
    axes[0,2].set_title(f"{tag_2d} – su išskirtimis (k={k_2d})")
    unique_02 = np.unique(labels_2d_w)
    legend_02 = [Patch(facecolor=scatter02.cmap(scatter02.norm(c)), label=f'Klasteris {c}') for c in unique_02]
    axes[0,2].legend(handles=legend_02, loc='best', fontsize=6, frameon=True)
    
    # be išskirčių
    scatter10 = axes[1,0].scatter(X2_visi_n[:,0], X2_visi_n[:,1], c=labels_visi_n[~out_visi], cmap='tab10', s=30)
    axes[1,0].set_title(f"{tag_visi} – be išskirčių (k={k_visi})")
    unique_10 = np.unique(labels_visi_n[~out_visi])
    legend_10 = [Patch(facecolor=scatter10.cmap(scatter10.norm(c)), label=f'Klasteris {c}') for c in unique_10]
    axes[1,0].legend(handles=legend_10, loc='best', fontsize=6, frameon=True)
    
    scatter11 = axes[1,1].scatter(X2_atr_n[:,0], X2_atr_n[:,1], c=labels_atr_n[~out_atr], cmap='tab10', s=30)
    axes[1,1].set_title(f"{tag_atr} – be išskirčių (k={k_atr})")
    unique_11 = np.unique(labels_atr_n[~out_atr])
    legend_11 = [Patch(facecolor=scatter11.cmap(scatter11.norm(c)), label=f'Klasteris {c}') for c in unique_11]
    axes[1,1].legend(handles=legend_11, loc='best', fontsize=6, frameon=True)
    
    scatter12 = axes[1,2].scatter(X2_2d_n[:,0], X2_2d_n[:,1], c=labels_2d_n[~out_2d], cmap='tab10', s=30)
    axes[1,2].set_title(f"{tag_2d} – be išskirčių (k={k_2d})")
    unique_12 = np.unique(labels_2d_n[~out_2d])
    legend_12 = [Patch(facecolor=scatter12.cmap(scatter12.norm(c)), label=f'Klasteris {c}') for c in unique_12]
    axes[1,2].legend(handles=legend_12, loc='best', fontsize=6, frameon=True)
    
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir_kmeans, "kmeans_trys_aibes_compare_with_vs_no_outliers.png"), dpi=300)
    plt.close()

    df_visi_full = load_full_csv(kelias_visi)
    df_atr_full  = load_full_csv(kelias_atr)
    df_tsne_full = load_full_csv(kelias_2d)

    tikslios_visi = df_visi_full['label'].values if 'label' in df_visi_full.columns else None
    tikslios_atr  = df_atr_full['label'].values  if 'label' in df_atr_full.columns  else None
    tikslios_2d   = df_tsne_full['label'].values if 'label' in df_tsne_full.columns else None

    if tikslios_visi is not None and len(tikslios_visi) == X_visi_pozymiai.shape[0]:
        vizualizuoti_palyginima(
            X2_visi_w, tikslios_visi, labels_visi_w,
            dataset_tag=tag_visi,
            save_name='palyginimas_visi_vs_kmeans_with_outliers.png',
            exclude_mask=None
        )
    if tikslios_atr is not None and len(tikslios_atr) == X_atrinkta.shape[0]:
        vizualizuoti_palyginima(
            X2_atr_w, tikslios_atr, labels_atr_w,
            dataset_tag=tag_atr,
            save_name='palyginimas_atrinkti_vs_kmeans_with_outliers.png',
            exclude_mask=None
        )
    if tikslios_2d is not None and len(tikslios_2d) == X_2D.shape[0]:
        vizualizuoti_palyginima(
            X2_2d_w, tikslios_2d, labels_2d_w,
            dataset_tag=tag_2d,
            save_name='palyginimas_tsne_vs_kmeans_with_outliers.png',
            exclude_mask=None
        )

    if tikslios_visi is not None and len(tikslios_visi) == X_visi_pozymiai.shape[0]:
        vizualizuoti_palyginima(
            X2_visi_w,
            tikslios_visi, labels_visi_n,
            dataset_tag=f"{tag_visi} (be išskirčių)",
            save_name='palyginimas_visi_vs_kmeans_no_outliers.png',
            exclude_mask=out_visi
        )
    if tikslios_atr is not None and len(tikslios_atr) == X_atrinkta.shape[0]:
        vizualizuoti_palyginima(
            X2_atr_w,
            tikslios_atr, labels_atr_n,
            dataset_tag=f"{tag_atr} (be išskirčių)",
            save_name='palyginimas_atrinkti_vs_kmeans_no_outliers.png',
            exclude_mask=out_atr
        )
    if tikslios_2d is not None and len(tikslios_2d) == X_2D.shape[0]:
        vizualizuoti_palyginima(
            X2_2d_w,
            tikslios_2d, labels_2d_n,
            dataset_tag=f"{tag_2d} (be išskirčių)",
            save_name='palyginimas_tsne_vs_kmeans_no_outliers.png',
            exclude_mask=out_2d
        )

    print("\n" + "="*60)
    print("Pabaiga")
    print("Grafikai išsaugoti į:", base_dir_kmeans)
    print("Tarpiniai skaičiavimai išsaugoti į:", base_dir_results)
    print("="*60)