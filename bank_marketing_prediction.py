import argparse
import io
import os
import zipfile
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

UCI_BANK_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"


def load_dataset(data_path: str | None) -> pd.DataFrame:
    if data_path:
        return pd.read_csv(data_path, sep=";")

    import urllib.request

    try:
        with urllib.request.urlopen(UCI_BANK_ZIP_URL, timeout=30) as response:
            zip_content = response.read()
    except Exception as error:
        raise RuntimeError(
            "Failed to download dataset from UCI. "
            "Provide a local CSV path via --data-path."
        ) from error

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
        with zip_file.open("bank-full.csv") as csv_file:
            return pd.read_csv(csv_file, sep=";")


def explore_dataset(data: pd.DataFrame) -> None:
    print("=== Dataset Exploration ===")
    print(f"Shape: {data.shape}")
    print("Columns:", list(data.columns))
    print("\nTarget distribution (y):")
    print(data["y"].value_counts(normalize=True).rename("ratio"))
    print("\nPreview:")
    print(data.head())


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=None),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
                ),
            ]
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained model: {name}")

    return models


def evaluate_models(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)

    roc_scores: Dict[str, float] = {}

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        roc_scores[name] = roc_auc

        print(f"\n=== {name} ===")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"))
        plt.close()

        RocCurveDisplay.from_predictions(y_test, y_proba, name=name)

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("ROC Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()

    return roc_scores


def _extract_binary_class_shap_values(shap_values: np.ndarray | list) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 2:
        return shap_values[:, :, 1]
    return shap_values


def explain_predictions(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str,
    num_predictions: int = 5,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_train_transformed = preprocessor.transform(X_train)
    X_test_sample = X_test.iloc[:num_predictions].copy()
    X_test_sample_transformed = preprocessor.transform(X_test_sample)
    feature_names = preprocessor.get_feature_names_out()

    try:
        import shap

        explainer = shap.TreeExplainer(classifier)
        raw_shap_values = explainer.shap_values(X_test_sample_transformed)
        shap_values = _extract_binary_class_shap_values(raw_shap_values)

        records = []
        for row_idx in range(min(num_predictions, shap_values.shape[0])):
            row_values = shap_values[row_idx]
            top_features_idx = np.argsort(np.abs(row_values))[-10:][::-1]
            for feat_idx in top_features_idx:
                records.append(
                    {
                        "prediction_index": row_idx,
                        "feature": feature_names[feat_idx],
                        "shap_value": float(row_values[feat_idx]),
                    }
                )

        output_path = os.path.join(output_dir, "shap_top_contributions.csv")
        pd.DataFrame(records).to_csv(output_path, index=False)
        return "shap", output_path

    except Exception as shap_error:
        try:
            from lime.lime_tabular import LimeTabularExplainer

            X_train_dense = (
                X_train_transformed.toarray()
                if hasattr(X_train_transformed, "toarray")
                else np.asarray(X_train_transformed)
            )
            X_test_dense = (
                X_test_sample_transformed.toarray()
                if hasattr(X_test_sample_transformed, "toarray")
                else np.asarray(X_test_sample_transformed)
            )

            lime_explainer = LimeTabularExplainer(
                training_data=X_train_dense,
                feature_names=feature_names,
                class_names=["no", "yes"],
                mode="classification",
                discretize_continuous=True,
            )

            output_path = os.path.join(output_dir, "lime_explanations.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                for idx in range(min(num_predictions, X_test_dense.shape[0])):
                    explanation = lime_explainer.explain_instance(
                        data_row=X_test_dense[idx],
                        predict_fn=classifier.predict_proba,
                        num_features=10,
                    )
                    f.write(f"Prediction index: {idx}\n")
                    for feat, weight in explanation.as_list():
                        f.write(f"  {feat}: {weight:.6f}\n")
                    f.write("\n")

            return "lime", output_path

        except Exception as lime_error:
            raise RuntimeError(
                "Unable to generate model explanations with SHAP or LIME. "
                f"SHAP error: {shap_error}; LIME error: {lime_error}"
            ) from lime_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict term-deposit subscription using the UCI Bank Marketing dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to bank-full.csv (semicolon-separated). If omitted, downloads from UCI.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save metrics and plots.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for faster iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_dataset(args.data_path)
    if args.max_rows is not None:
        data = data.head(args.max_rows)

    explore_dataset(data)

    X = data.drop(columns=["y"])
    y = (data["y"] == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)
    models = train_models(X_train, y_train, preprocessor)
    roc_scores = evaluate_models(models, X_test, y_test, args.output_dir)

    best_model_name = max(roc_scores, key=roc_scores.get)
    best_model = models[best_model_name]

    method, explanation_path = explain_predictions(
        best_model,
        X_train,
        X_test,
        args.output_dir,
        num_predictions=5,
    )

    print("\n=== Explainability ===")
    print(f"Method used: {method.upper()}")
    print(f"Saved explanations to: {explanation_path}")


if __name__ == "__main__":
    main()
