"""
Baseline model: Logistic Regression for variant pathogenicity prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
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


def encode_features(X_train, X_val, X_test):
    """
    Encode categorical features as integers.

    Label encoding is used here for simplicity. It assigns an arbitrary integer
    to each category. This works fine for tree-based models, but logistic regression
    may interpret these as ordinal (i.e., gene 5 > gene 3). For the baseline this
    is acceptable — we can revisit with target encoding or one-hot later.

    The encoders are fit ONLY on the training set to prevent data leakage. Unseen
    categories in val/test are mapped to -1.
    """
    categorical_cols = ["GeneSymbol", "Chromosome", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])

        # Handle unseen categories in val/test
        for df in [X_val, X_test]:
            df[col] = df[col].map(
                {label: idx for idx, label in enumerate(le.classes_)}
            ).fillna(-1).astype(int)

        encoders[col] = le

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


def train_model(X_train, y_train):
    """
    Train a logistic regression baseline.

    class_weight='balanced' tells the model to upweight the minority class
    (pathogenic) inversely proportional to its frequency. Without this, the
    model could achieve ~85% accuracy by predicting benign for everything.

    max_iter=1000 gives the solver enough iterations to converge on this
    dataset size.
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
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

    # Encode
    X_train, X_val, X_test, encoders = encode_features(X_train, X_val, X_test)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate on val (used for tuning decisions)
    val_metrics = evaluate(model, X_val, y_val, "Validation")

    # Evaluate on test (final, unbiased estimate)
    test_metrics = evaluate(model, X_test, y_test, "Test")


if __name__ == "__main__":
    main()
