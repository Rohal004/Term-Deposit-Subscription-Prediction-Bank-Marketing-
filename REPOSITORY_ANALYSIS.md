# Repository Analysis: Term Deposit Subscription Prediction

## 📊 Repository Overview

| Attribute | Details |
|-----------|---------|
| **Repository Name** | Term-Deposit-Subscription-Prediction-Bank-Marketing- |
| **Repository ID** | 1263854079 |
| **Owner** | Rohal004 |
| **URL** | https://github.com/Rohal004/Term-Deposit-Subscription-Prediction-Bank-Marketing- |
| **Visibility** | Public |
| **Status** | Active |
| **Created** | June 9, 2026 (38 minutes ago) |
| **Last Updated** | June 9, 2026 (26 minutes ago) |
| **Last Pushed** | June 9, 2026 at 10:34:34 UTC |
| **Default Branch** | main |

---

## 📁 Repository Structure

```
Term-Deposit-Subscription-Prediction-Bank-Marketing-/
├── README.md
├── bank_marketing_prediction.py
├── REPOSITORY_ANALYSIS.md
└── outputs/
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    ├── roc_curves.png
    └── shap_top_contributions.csv or lime_explanations.txt
```

---

## 🔤 Language Composition

### Primary Language: **Python** (100%)

| Language | Bytes | Percentage |
|----------|-------|-----------|
| Python   | 9,670 | 100%      |

```
Python: ████████████████████████████████████ 100%
```

---

## 📝 Project Description

**Predict whether a bank customer will subscribe to a term deposit as a result of a marketing campaign.**

This project uses machine learning classification models to predict customer subscription behavior in response to bank marketing campaigns. The implementation includes:

- **Data Loading**: UCI Bank Marketing dataset (semicolon-delimited CSV)
- **Exploratory Data Analysis**: Dataset shape, columns, target distribution
- **Feature Engineering**: One-Hot Encoding for categorical variables
- **Model Training**: 
  - Logistic Regression
  - Random Forest Classifier (300 estimators)
- **Model Evaluation**:
  - Confusion Matrix (both models)
  - F1-Score
  - ROC-AUC Score
  - ROC Curves
- **Model Explainability**:
  - SHAP (Tree Explainer) for feature importance
  - Fallback to LIME (Local Interpretable Model-agnostic Explanations)

---

## 🔧 Technical Stack

### Core Libraries
- **pandas**: Data manipulation and exploration
- **scikit-learn**: Machine learning models and pipelines
  - LogisticRegression
  - RandomForestClassifier
  - ColumnTransformer
  - OneHotEncoder
  - Confusion Matrix & ROC Curve displays
- **matplotlib**: Visualization
- **numpy**: Numerical operations

### Optional Libraries
- **SHAP**: Model explainability (preferred)
- **LIME**: Fallback explainability method

---

## ⚙️ How to Run

### Basic Usage
```bash
python bank_marketing_prediction.py --output-dir outputs
```

### With Custom Data
```bash
python bank_marketing_prediction.py --data-path /path/to/bank-full.csv --output-dir outputs
```

### With Row Limit (for testing)
```bash
python bank_marketing_prediction.py --max-rows 1000 --output-dir outputs
```

### Full Command with All Options
```bash
python bank_marketing_prediction.py \
    --data-path /path/to/bank-full.csv \
    --output-dir outputs \
    --max-rows 5000
```

---

## 📋 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-path` | string | None (download) | Path to local `bank-full.csv` file (semicolon-delimited) |
| `--output-dir` | string | `outputs` | Directory to save metrics and visualizations |
| `--max-rows` | integer | None (all) | Limit dataset to first N rows (for faster iteration) |

---

## 📊 Output Files

All outputs are saved in the specified `--output-dir` (default: `outputs/`):

### Visualizations
1. **confusion_matrix_logistic_regression.png**
   - Confusion matrix for Logistic Regression model
   - Shows True/False Positives and Negatives

2. **confusion_matrix_random_forest.png**
   - Confusion matrix for Random Forest model
   - Shows True/False Positives and Negatives

3. **roc_curves.png**
   - ROC curves for both models
   - Includes random baseline
   - Useful for model comparison

### Data Files
4. **shap_top_contributions.csv** (or **lime_explanations.txt**)
   - Explainability data for top 5 predictions
   - Shows which features contributed most to each prediction
   - 10 top features per prediction

---

## 📈 Expected Metrics

The script outputs:
- **F1-Score**: Balance between precision and recall
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: Breakdown of predictions

### Model Comparison
The script identifies the best model by ROC-AUC score and uses it for explainability.

---

## 🎯 Key Features

✅ **Automated Dataset Loading**
- Downloads from UCI Machine Learning Repository if no local path provided

✅ **Comprehensive Preprocessing**
- Automatic detection of categorical vs numerical columns
- One-Hot Encoding for categorical features
- Pipeline-based architecture

✅ **Multiple Models**
- Logistic Regression (interpretable, baseline)
- Random Forest (ensemble, non-linear)

✅ **Evaluation Metrics**
- Confusion Matrix visualization
- F1-Score (handles class imbalance)
- ROC-AUC Score (probability threshold independence)
- ROC Curves (visual model comparison)

✅ **Model Explainability**
- SHAP values (if available)
- LIME explanations (fallback)
- Top 10 features per prediction

---

## 📦 Repository Permissions

Your permissions (User: Rohal004):
- ✅ Admin (full control)
- ✅ Maintain
- ✅ Pull
- ✅ Push
- ✅ Triage

---

## ⚙️ Repository Configuration

### Merge Strategy
- Allow merge commits: ✅
- Allow squash merging: ✅
- Allow rebase merging: ✅
- Allow auto merge: ❌
- Delete branch on merge: ❌

### Repository Features
- ✅ Issues enabled
- ✅ Pull Requests enabled
- ✅ Projects enabled
- ✅ Wiki enabled
- ✅ Downloads enabled
- ❌ Discussions enabled
- ❌ GitHub Pages enabled

### Repository Stats
- Stars: 0
- Forks: 0
- Open Issues: 0
- Watchers: 0
- Network: 0

---

## 🔄 Data Flow

```
UCI Bank Marketing Dataset (or local CSV)
            ↓
    Load & Explore
            ↓
  Feature Engineering
  (Categorical encoding)
            ↓
    Train-Test Split
  (80-20, stratified)
            ↓
    ┌─────────┬──────────────┐
    ↓         ↓
Logistic   Random
Regression Forest
    ↓         ↓
  Model    Model
 Training  Training
    ↓         ↓
    └─────────┬──────────────┘
            ↓
      Model Evaluation
  (F1, ROC-AUC, Confusion Matrix)
            ↓
    Select Best Model
            ↓
   Model Explainability
     (SHAP or LIME)
            ↓
      Generate Outputs
```

---

## 📚 Dataset Information

**UCI Bank Marketing Dataset**
- Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
- File: `bank-full.csv` (semicolon-delimited)
- Target: `y` (yes/no - term deposit subscription)
- Features: Multiple demographic, campaign, and economic indicators

---

## 🚀 Next Steps

1. Run the script: `python bank_marketing_prediction.py --output-dir outputs`
2. Review confusion matrices for model performance
3. Analyze ROC curves for threshold selection
4. Examine SHAP/LIME explanations for feature insights
5. Iterate with different hyperparameters or feature engineering

---

*Generated: June 9, 2026*
*Repository ID: 1263854079*
