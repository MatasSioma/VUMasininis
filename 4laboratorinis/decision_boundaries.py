import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Directories
DUOMENU_DIREKTORIJA = 'duomenys'
GRAFIKU_DIREKTORIJA = 'grafikai'
BOUNDARY_DIREKTORIJA = 'decision_boundaries'
JSON_DIREKTORIJA = 'JSON'
JSON_FAILAS = os.path.join(JSON_DIREKTORIJA, 'geriausias_rinkinys.json')
RF_JSON = os.path.join(GRAFIKU_DIREKTORIJA, 'RF', 'RF_optimal_params.json')

os.makedirs(os.path.join(GRAFIKU_DIREKTORIJA, BOUNDARY_DIREKTORIJA), exist_ok=True)

# t-SNE parameters (based on 3Laboratorinis/tSNE.py)
PERPLEXITY = 50
MAX_ITER = 500
METRIC = 'canberra'
RANDOM_STATE = 42

print("=" * 100)
print(" DECISION BOUNDARY VISUALIZATION WITH t-SNE ".center(100, "="))
print("=" * 100)

# Load datasets
print("\n[1] Loading datasets...")
df_mokymas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'mokymo_aibe.csv'), sep=';')
df_testavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'testavimo_aibe.csv'), sep=';')
print(f"[OK] Mokymo: {len(df_mokymas)} samples, Testavimo: {len(df_testavimas)} samples")

# Extract features
pozymiai_full = [col for col in df_mokymas.columns if col != 'label']

# Load optimal features from JSON
try:
    with open(JSON_FAILAS, 'r', encoding='utf-8') as f:
        config = json.load(f)
        pozymiai_subset = config.get("GERIAUSIAS_MODELIS_6_POZYMIAI", [])
        print(f"[OK] Loaded optimal features from JSON ({len(pozymiai_subset)} features)")
except FileNotFoundError:
    print("[INFO] JSON not found. Using default QRS features.")
    pozymiai_subset = ["Q_val", "R_val", "S_val", "Q_pos", "R_pos", "S_pos"]

# Load RF optimal params
def load_rf_optimal_params(exp_name):
    try:
        with open(RF_JSON, 'r', encoding='utf-8') as f:
            params_dict = json.load(f)
            # Handle typo in JSON key
            exp_key = exp_name
            if exp_name == "Visi požymiai" and "Vsi požymiai" in params_dict:
                exp_key = "Vsi požymiai"

            if exp_key in params_dict:
                params = params_dict[exp_key].copy()
                if params['max_depth'] == "None":
                    params['max_depth'] = None
                else:
                    params['max_depth'] = int(params['max_depth'])
                return params
    except FileNotFoundError:
        pass
    return {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    }

# Function to find best k for KNN through validation
def find_best_knn_k(X_train, y_train, X_val, y_val):
    """Find best k value through validation (mimics KNN.py logic)"""
    from sklearn.metrics import f1_score
    best_k = 1
    best_f1 = -1

    for k in range(1, 22, 2):
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
        knn.fit(X_train, y_train)
        y_val_pred = knn.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    return best_k

# Function to find best depth for DT through validation
def find_best_dt_depth(X_train, y_train, X_val, y_val):
    """Find best depth value through validation (mimics DT.py logic)"""
    from sklearn.metrics import f1_score
    best_depth = 1
    best_f1 = -1

    for depth in range(1, 11):
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        y_val_pred = dt.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_depth = depth

    return best_depth

# Define experiments
experiments = {
    "Visi požymiai": pozymiai_full,
    "Optimalūs požymiai": pozymiai_subset
}

# Model configurations - will be populated dynamically
model_configs = {
    'RF': {
        'name': 'Atsitiktinis miškas',
        'params': {}  # Will be filled during processing
    },
    'KNN': {
        'name': 'K-NN',
        'params': {
            'Visi požymiai': {'k': 3},
            'Optimalūs požymiai': {'k': 3}
        }
    },
    'DT': {
        'name': 'Sprendimų medis',
        'params': {
            'Visi požymiai': {'depth': 5},
            'Optimalūs požymiai': {'depth': 4}
        }
    }
}

def apply_tsne_transform(X_train, X_test):
    """Apply t-SNE transformation to reduce data to 2D"""
    print(f"  [t-SNE] Transforming {X_train.shape[0]} + {X_test.shape[0]} samples to 2D...")

    # Combine train and test for consistent t-SNE transformation
    X_combined = np.vstack([X_train, X_test])

    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        max_iter=MAX_ITER,
        metric=METRIC,
        random_state=RANDOM_STATE
    )
    X_tsne = tsne.fit_transform(X_combined)

    # Split back
    X_train_tsne = X_tsne[:len(X_train)]
    X_test_tsne = X_tsne[len(X_train):]

    return X_train_tsne, X_test_tsne

def plot_decision_boundary(model, X, y, ax, title, feature_names=None):
    """Plot decision boundary for 2D data"""
    # Create mesh
    h = 0.01  # step size in mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary (filled regions)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm', levels=[-0.5, 0.5, 1.5, 2.5])

    # Plot data points for each class separately
    class_0_mask = y == 0
    class_2_mask = y == 2

    ax.scatter(X[class_0_mask, 0], X[class_0_mask, 1],
              c='blue', label='Normalus (0)', edgecolors='k', s=50, alpha=0.7)
    ax.scatter(X[class_2_mask, 0], X[class_2_mask, 1],
              c='red', label='Aritmija (2)', edgecolors='k', s=50, alpha=0.7)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dimensija 1', fontsize=10)
    ax.set_ylabel('Dimensija 2', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Simple legend
    ax.legend(loc='best', fontsize=9)

# Load validation data for parameter tuning
df_validavimas = pd.read_csv(os.path.join(DUOMENU_DIREKTORIJA, 'validavimo_aibe.csv'), sep=';')
print(f"[OK] Validavimo: {len(df_validavimas)} samples (for parameter selection)")

# Process each model
for model_key, model_info in model_configs.items():
    print(f"\n{'=' * 100}")
    print(f" PROCESSING: {model_info['name']} ".center(100, "="))
    print(f"{'=' * 100}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (exp_name, features) in enumerate(experiments.items()):
        print(f"\n[{exp_name}]")

        # Extract data
        X_train = df_mokymas[features].values
        y_train = df_mokymas['label'].values
        X_val = df_validavimas[features].values
        y_val = df_validavimas['label'].values
        X_test = df_testavimas[features].values
        y_test = df_testavimas['label'].values

        print(f"  Features: {len(features)}")
        print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Apply t-SNE
        X_train_2d, X_test_2d = apply_tsne_transform(X_train, X_test)
        X_val_2d, _ = apply_tsne_transform(X_val, X_test)  # Use same t-SNE space

        # Determine model parameters and create model
        if model_key == 'RF':
            # Get optimal RF parameters from JSON
            rf_params = load_rf_optimal_params(exp_name)
            print(f"  RF params: n_estimators={rf_params['n_estimators']}, max_depth={rf_params['max_depth']}")
            model = RandomForestClassifier(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth'],
                min_samples_split=rf_params['min_samples_split'],
                min_samples_leaf=rf_params['min_samples_leaf'],
                max_features=rf_params['max_features'],
                random_state=42,
                n_jobs=1
            )
            param_str = f"n_est={rf_params['n_estimators']}, depth={rf_params['max_depth']}"

        elif model_key == 'KNN':
            # Use fixed k value
            best_k = model_info['params'][exp_name]['k']
            print(f"  KNN k: {best_k} (fixed)")

            model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='uniform')
            param_str = f"k={best_k}"

        elif model_key == 'DT':
            # Use fixed depth value
            best_depth = model_info['params'][exp_name]['depth']
            print(f"  DT depth: {best_depth} (fixed)")

            model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
            param_str = f"depth={best_depth}"

        # Train model on 2D t-SNE data
        model.fit(X_train_2d, y_train)

        # Calculate accuracy on test set
        accuracy = model.score(X_test_2d, y_test)
        print(f"  Test Accuracy (2D t-SNE): {accuracy:.4f}")

        # Plot decision boundary
        ax = axes[idx]
        title = f"{exp_name}\n{param_str}, Acc: {accuracy:.4f}"
        # title = f"{exp_name}\n{param_str}"
        plot_decision_boundary(model, X_test_2d, y_test, ax, title)

    # Add main title
    plt.suptitle(f"{model_info['name']} - Sprendimo riba (t-SNE 2D)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    filename = os.path.join(GRAFIKU_DIREKTORIJA, BOUNDARY_DIREKTORIJA,
                           f'{model_key}_Decision_Boundaries.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Saved: {filename}")

print("\n" + "=" * 100)
print(" COMPLETED ".center(100, "="))
print("=" * 100)
print(f"All decision boundary plots saved to: {os.path.join(GRAFIKU_DIREKTORIJA, BOUNDARY_DIREKTORIJA)}")
