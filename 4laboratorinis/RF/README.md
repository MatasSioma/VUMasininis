# Random Forest Classifier - Hyperparameter Tuning Guide

## Quick Start

This folder contains two main scripts for Random Forest classification:

### 1. **RF_hyperparameter_tuning.py** - Find Optimal Parameters
Use this script to discover the best hyperparameters using GridSearchCV.

```bash
cd 4laboratorinis/RF
python RF_hyperparameter_tuning.py
```

**What it does:**
- Performs exhaustive grid search over hyperparameters
- Uses 5-fold cross-validation to find optimal settings
- Generates detailed results CSV files with all tested combinations
- Saves optimal parameters to `RF_optimal_params.json`
- Tests parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

**Output:**
- `grafikai/RF/hyperparameter_tuning/gridsearch_results_*.csv` - Detailed results
- `grafikai/RF/RF_optimal_params.json` - Best parameters found
- `grafikai/RF/hyperparameter_tuning/confusion_matrix_*.png`
- `grafikai/RF/hyperparameter_tuning/feature_importance_*.png`
- `grafikai/RF/hyperparameter_tuning/roc_curve_*.png`

---

### 2. **RF.py** - Train Final Model with Optimal Parameters
Use this script after finding the optimal parameters.

```bash
cd 4laboratorinis/RF
python RF.py
```

**What it does:**
- Automatically loads optimal parameters from `RF_optimal_params.json`
- Fine-tunes n_estimators around the optimal value
- Trains final Random Forest model
- Generates comprehensive visualizations and reports

**Output:**
- Detailed classification reports
- Confusion matrices
- ROC curves with AUC scores
- Feature importance visualizations
- Performance metrics (Accuracy, Precision, Recall, F1)

---

## Workflow

### Step 1: Find Optimal Parameters
```bash
# First time setup - this takes some time (30 minutes - 2 hours)
python RF_hyperparameter_tuning.py
```

This will generate `RF_optimal_params.json` with the best hyperparameters for your dataset.

### Step 2: Train Final Model
```bash
# Use the optimal parameters to train the final model
python RF.py
```

This will use the parameters found in step 1 and generate visualizations.

### Step 3: Iterate (Optional)
If you want to modify the grid search ranges, edit `RF_hyperparameter_tuning.py`:

```python
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300],      # Change these ranges
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}
```

---

## Virtual Environment Setup

Before running, activate the virtual environment:

```bash
# On Windows
env\Scripts\activate

# On Linux/Mac
source env/bin/activate

# Install/verify dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tabulate
```

---

## Parameter Explanation

| Parameter | Purpose | Default (if tuning skipped) |
|-----------|---------|-----|
| **n_estimators** | Number of trees in the forest | 200 |
| **max_depth** | Maximum depth of trees | 15 |
| **min_samples_split** | Min samples to split a node | 5 |
| **min_samples_leaf** | Min samples in leaf nodes | 2 |
| **max_features** | Features to consider per split | 'sqrt' |

---

## Output Files

All results are saved to `grafikai/RF/`:

```
grafikai/RF/
├── RF_optimal_params.json              # Optimal parameters (used by RF.py)
├── BENDRAS_ROC_Grafikas.png           # ROC curves comparison
├── BENDRAS_Confusion_Matrix_Grid.png  # Confusion matrices
├── RF_Metriku_Suvestine_Grid.png      # Metrics vs n_estimators
├── Feature_Importance_*.png            # Feature importance charts
└── hyperparameter_tuning/
    ├── gridsearch_results_*.csv        # Detailed GridSearch results
    ├── confusion_matrix_*.png
    ├── feature_importance_*.png
    └── roc_curve_*.png
```

---

## Tips for Finding Better Parameters

1. **First Run**: Use the default grid to get a baseline
2. **Refine**: Based on results, narrow the search space around best values
3. **Time vs Accuracy**: Reduce grid size if it takes too long:
   ```python
   param_grid = {
       'n_estimators': [100, 200, 300],          # Fewer values
       'max_depth': [15, 20, None],
       'min_samples_split': [5],                 # Fixed value
       'min_samples_leaf': [2],                  # Fixed value
       'max_features': ['sqrt'],                 # Single option
   }
   ```

4. **Monitor Progress**: The script prints progress during GridSearch

---

## Troubleshooting

**Q: GridSearch is taking too long**
- Reduce the number of CV folds: change `cv=5` to `cv=3`
- Reduce the number of parameters to test
- Use fewer workers: remove `n_jobs=-1` and specify a number like `n_jobs=4`

**Q: Getting out of memory errors**
- Reduce dataset size temporarily for testing
- Use fewer trees: `n_estimators: [50, 100, 150]`
- Reduce CV folds: `cv=3`

**Q: Want to use different features**
- Modify `pozymiai_full` or `pozymiai_subset` in the script
- Or update the JSON files that define feature sets

---

## Data Files Expected

The scripts expect these files to exist:
- `../duomenys/mokymo_aibe.csv` - Training data
- `../duomenys/validavimo_aibe.csv` - Validation data
- `../duomenys/testavimo_aibe.csv` - Test data
- `../JSON/geriausias_rinkinys.json` - Optimal feature sets

---

## Next Steps

After getting optimal parameters:
1. Review `RF_optimal_params.json` to understand the best configuration
2. Run `RF.py` to generate final results
3. Compare with other algorithms (KNN, Decision Tree) in this project
4. Use feature importance to understand which attributes matter most
