# Term-Deposit-Subscription-Prediction-Bank-Marketing-

Predict whether a bank customer will subscribe to a term deposit as a result of a marketing campaign.

## Objective Coverage
The implementation in `bank_marketing_prediction.py` now covers:
- Loading and exploring the UCI Bank Marketing dataset
- Proper encoding of categorical features (One-Hot Encoding)
- Training classification models (Logistic Regression and Random Forest)
- Evaluation with Confusion Matrix, F1-Score, and ROC Curve
- Explainability for at least 5 predictions using SHAP (fallback to LIME)

## Run
```bash
python bank_marketing_prediction.py --output-dir outputs
```

### Optional arguments
- `--data-path <path>`: use a local `bank-full.csv` file (semicolon-delimited)
- `--max-rows <n>`: run on only the first `n` rows for faster iteration

## Outputs
Saved under the selected `--output-dir`:
- `confusion_matrix_logistic_regression.png`
- `confusion_matrix_random_forest.png`
- `roc_curves.png`
- `shap_top_contributions.csv` (or `lime_explanations.txt` if SHAP is unavailable)
