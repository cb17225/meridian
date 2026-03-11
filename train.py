"""
Variant pathogenicity prediction: baseline and improved models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def load_data(features_path="data/features.json", target_path="data/target.json"):
    """Load features and target from JSON files."""
    X = pd.read_json(features_path, orient="records")
    y = pd.read_json(target_path, orient="records")["label"]
    return X, y


def encode_features(X_train, X_val, X_test, method="label"):
    """
    Encode categorical features.

    Two methods available:
    - "label": assigns an arbitrary integer per category. Fast and compact,
      but implies a false ordering. Works well with tree-based models that
      split on thresholds. Poor for linear models.
    - "onehot": creates a binary column per category for low-cardinality
      features (Chromosome, alleles). GeneSymbol stays label-encoded to
      avoid creating thousands of sparse columns. Better for linear models
      since each category gets its own independent weight.

    Encoders are fit ONLY on the training set to prevent data leakage.
    """
    categorical_cols = ["GeneSymbol", "Chromosome", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    encoders = {}

    if method == "label":
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])

            for df in [X_val, X_test]:
                df[col] = df[col].map(
                    {label: idx for idx, label in enumerate(le.classes_)}
                ).fillna(-1).astype(int)

            encoders[col] = le

    elif method == "onehot":
        # Label encode GeneSymbol (too many unique values for one-hot)
        le = LabelEncoder()
        X_train["GeneSymbol"] = le.fit_transform(X_train["GeneSymbol"])
        for df in [X_val, X_test]:
            df["GeneSymbol"] = df["GeneSymbol"].map(
                {label: idx for idx, label in enumerate(le.classes_)}
            ).fillna(-1).astype(int)
        encoders["GeneSymbol"] = le

        # One-hot encode low-cardinality columns
        onehot_cols = ["Chromosome", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(X_train[onehot_cols])

        for df in [X_train, X_val, X_test]:
            encoded = pd.DataFrame(
                ohe.transform(df[onehot_cols]),
                columns=ohe.get_feature_names_out(onehot_cols),
                index=df.index,
            )
            df.drop(columns=onehot_cols, inplace=True)
            for c in encoded.columns:
                df[c] = encoded[c]

        encoders["onehot"] = ohe

    return X_train, X_val, X_test, encoders


def split_data(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split into train/val/test with stratification.

    Stratification ensures each split maintains the same ~85/15 class ratio.
    Without it, smaller splits could end up with skewed class distributions
    by random chance.
    """
    # First split: separate out the test set
    test_frac = test_size
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=random_state, stratify=y
    )

    # Second split: separate val from remaining train
    # val_size is relative to the original dataset, so adjust for the remaining data
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, model_type="logistic"):
    """
    Train a classification model.

    - "logistic": simple linear baseline. class_weight='balanced' upweights
      the minority class inversely proportional to its frequency.
    - "random_forest": ensemble of decision trees. Each tree sees a random
      subset of samples (bagging) and features, then votes on the prediction.
      Handles non-linear relationships and categorical features naturally.
      n_estimators=200 means 200 trees vote. More trees = more stable
      predictions but diminishing returns past a point.
    """
    if model_type == "logistic":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, split_name=""):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    auroc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred)

    print(f"\n{'=' * 40}")
    print(f"{split_name} Results")
    print(f"{'=' * 40}")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1 (pathogenic): {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Benign", "Pathogenic"]))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    return {"auroc": auroc, "f1": f1}


def main():
    # Load
    X, y = load_data()
    print(f"Dataset: {len(X):,} samples, {X.shape[1]} features")
    print(f"Class balance: {y.mean():.2%} pathogenic")

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"\nTrain: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # --- Baseline: Logistic Regression with label encoding ---
    print("\n" + "#" * 50)
    print("BASELINE: Logistic Regression (label encoding)")
    print("#" * 50)
    Xt, Xv, Xte, _ = encode_features(
        X_train.copy(), X_val.copy(), X_test.copy(), method="label"
    )
    model = train_model(Xt, y_train, model_type="logistic")
    evaluate(model, Xv, y_val, "Validation")

    # --- Improved: Logistic Regression with one-hot encoding ---
    print("\n" + "#" * 50)
    print("IMPROVED: Logistic Regression (one-hot encoding)")
    print("#" * 50)
    Xt, Xv, Xte, _ = encode_features(
        X_train.copy(), X_val.copy(), X_test.copy(), method="onehot"
    )
    model = train_model(Xt, y_train, model_type="logistic")
    evaluate(model, Xv, y_val, "Validation")

    # --- Random Forest with label encoding ---
    print("\n" + "#" * 50)
    print("RANDOM FOREST (label encoding)")
    print("#" * 50)
    Xt, Xv, Xte, _ = encode_features(
        X_train.copy(), X_val.copy(), X_test.copy(), method="label"
    )
    model = train_model(Xt, y_train, model_type="random_forest")
    evaluate(model, Xv, y_val, "Validation")
    evaluate(model, Xte, y_test, "Test")


if __name__ == "__main__":
    main()
