"""
Variant pathogenicity prediction: baseline and improved models.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier


def load_data(features_path="data/features.json", target_path="data/target.json"):
    """Load features and target from JSON files."""
    X = pd.read_json(features_path, orient="records")
    y = pd.read_json(target_path, orient="records")["label"]
    return X, y


def engineer_features(X_train, X_val, X_test, y_train):
    """
    Create new features from existing columns.

    Target encoding: replace a categorical value with the mean target (pathogenic
    rate) for that category, computed from the training set only. This converts
    high-cardinality categoricals like GeneSymbol into a single numeric column
    that directly captures the relationship with the target.

    Unseen categories in val/test get the global training mean (prior) as a
    fallback — a reasonable default that doesn't leak information.

    is_transversion: purine (A, G) <-> pyrimidine (C, T) substitutions are
    rarer but more disruptive than transitions (purine <-> purine or
    pyrimidine <-> pyrimidine). The EDA showed transversions have roughly
    double the pathogenic rate.
    """
    global_mean = y_train.mean()

    # Target encode GeneSymbol
    gene_rates = y_train.groupby(X_train["GeneSymbol"]).mean()
    for df in [X_train, X_val, X_test]:
        df["gene_pathogenic_rate"] = df["GeneSymbol"].map(gene_rates).fillna(global_mean)

    # Target encode Chromosome
    chrom_rates = y_train.groupby(X_train["Chromosome"]).mean()
    for df in [X_train, X_val, X_test]:
        df["chrom_pathogenic_rate"] = df["Chromosome"].map(chrom_rates).fillna(global_mean)

    # Target encode Cytogenetic band — finer-grained than chromosome,
    # captures arm (p/q) and band-level pathogenic rates
    if "Cytogenetic" in X_train.columns:
        cyto_rates = y_train.groupby(X_train["Cytogenetic"]).mean()
        for df in [X_train, X_val, X_test]:
            df["cyto_pathogenic_rate"] = df["Cytogenetic"].map(cyto_rates).fillna(global_mean)

    # Transversion flag
    purines = {"A", "G"}
    for df in [X_train, X_val, X_test]:
        ref_is_purine = df["ReferenceAlleleVCF"].isin(purines)
        alt_is_purine = df["AlternateAlleleVCF"].isin(purines)
        df["is_transversion"] = (ref_is_purine != alt_is_purine).astype(int)

    return X_train, X_val, X_test


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
    all_cats = [
        "GeneSymbol", "Chromosome", "Cytogenetic",
        "ReferenceAlleleVCF", "AlternateAlleleVCF",
    ]
    categorical_cols = [c for c in all_cats if c in X_train.columns]
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
        # Label encode high-cardinality columns (too many unique values for one-hot)
        for col in ["GeneSymbol", "Cytogenetic"]:
            if col not in X_train.columns:
                continue
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            for df in [X_val, X_test]:
                df[col] = df[col].map(
                    {label: idx for idx, label in enumerate(le.classes_)}
                ).fillna(-1).astype(int)
            encoders[col] = le

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
    elif model_type == "xgboost":
        # scale_pos_weight handles class imbalance for XGBoost — it's the
        # ratio of negative to positive samples, equivalent to sklearn's
        # class_weight='balanced'
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        model = XGBClassifier(
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, split_name="", threshold=0.5, y_prob_override=None):
    """Evaluate model and print metrics."""
    y_prob = y_prob_override if y_prob_override is not None else model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred)

    print(f"\n{'=' * 40}")
    print(f"{split_name} Results")
    print(f"{'=' * 40}")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1 (pathogenic): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Benign", "Pathogenic"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    return {"auroc": auroc, "f1": f1}


def tune_xgboost(X_train, y_train, X_val, y_val):
    """
    Tune XGBoost hyperparameters using the validation set.

    Instead of cross-validation (which re-splits training data), we evaluate
    each parameter combination directly on our held-out validation set. This
    is simpler, faster, and consistent with how we evaluate all other models.

    The grid is intentionally small — a coarse search over the most impactful
    parameters. In practice you'd follow up with a finer search around the
    best values, or use Bayesian optimization (e.g. Optuna).
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    param_grid = {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 500, 1000],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    best_auroc = 0
    best_params = None
    total = len(list(ParameterGrid(param_grid)))

    print(f"\nSearching {total} parameter combinations...")

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        model = XGBClassifier(
            **params,
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, y_prob)

        if auroc > best_auroc:
            best_auroc = auroc
            best_params = params
            print(f"  [{i}/{total}] AUROC={auroc:.4f} (new best) | {params}")

    print(f"\nBest params: {best_params}")
    print(f"Best validation AUROC: {best_auroc:.4f}")

    # Retrain with best params
    best_model = XGBClassifier(
        **best_params,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    best_model.fit(X_train, y_train)

    return best_model, best_params


def find_best_threshold(y_true, y_prob):
    """
    Find the decision threshold that maximizes F1 on the validation set.

    The default 0.5 threshold assumes balanced classes. With ~85/15 imbalance,
    the model needs a lower threshold to catch more pathogenic variants —
    otherwise it's too conservative and misses true positives.

    We sweep from 0.1 to 0.9 in steps of 0.01 and pick the threshold with
    the highest F1 score.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\nBest threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold


EXPERIMENTS = {
    "logistic": "Logistic Regression (label encoding)",
    "logistic-onehot": "Logistic Regression (one-hot encoding)",
    "rf": "Random Forest (label encoding)",
    "rf-feat": "Random Forest + feature engineering",
    "xgboost": "XGBoost + feature engineering (tuned)",
    "ensemble": "XGBoost + Random Forest ensemble (tuned threshold)",
}


def main():
    parser = argparse.ArgumentParser(description="Train variant pathogenicity models")
    parser.add_argument(
        "models",
        nargs="*",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Models to train (default: all)",
    )
    args = parser.parse_args()

    if "all" in args.models:
        args.models = list(EXPERIMENTS.keys())

    # Load
    X, y = load_data()
    print(f"Dataset: {len(X):,} samples, {X.shape[1]} features")
    print(f"Class balance: {y.mean():.2%} pathogenic")

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"\nTrain: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    for model_key in args.models:
        print("\n" + "#" * 50)
        print(EXPERIMENTS[model_key].upper())
        print("#" * 50)

        if model_key == "logistic":
            Xt, Xv, Xte, _ = encode_features(
                X_train.copy(), X_val.copy(), X_test.copy(), method="label"
            )
            model = train_model(Xt, y_train, model_type="logistic")
            evaluate(model, Xv, y_val, "Validation")

        elif model_key == "logistic-onehot":
            Xt, Xv, Xte, _ = encode_features(
                X_train.copy(), X_val.copy(), X_test.copy(), method="onehot"
            )
            model = train_model(Xt, y_train, model_type="logistic")
            evaluate(model, Xv, y_val, "Validation")

        elif model_key == "rf":
            Xt, Xv, Xte, _ = encode_features(
                X_train.copy(), X_val.copy(), X_test.copy(), method="label"
            )
            model = train_model(Xt, y_train, model_type="random_forest")
            evaluate(model, Xv, y_val, "Validation")

        elif model_key == "rf-feat":
            Xt, Xv, Xte = engineer_features(
                X_train.copy(), X_val.copy(), X_test.copy(), y_train
            )
            Xt, Xv, Xte, _ = encode_features(Xt, Xv, Xte, method="label")
            model = train_model(Xt, y_train, model_type="random_forest")
            evaluate(model, Xv, y_val, "Validation")

        elif model_key == "xgboost":
            Xt, Xv, Xte = engineer_features(
                X_train.copy(), X_val.copy(), X_test.copy(), y_train
            )
            Xt, Xv, Xte, encoders = encode_features(Xt, Xv, Xte, method="label")
            model, best_params = tune_xgboost(Xt, y_train, Xv, y_val)
            evaluate(model, Xv, y_val, "Validation")
            evaluate(model, Xte, y_test, "Test")

            # Save model and preprocessing artifacts for deployment
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/xgboost_best.joblib")
            joblib.dump(list(Xv.columns), "models/feature_names.joblib")
            print("\nModel saved to models/xgboost_best.joblib")

            # Save everything the API needs to preprocess raw input
            # Without these, the API can't replicate the exact same
            # feature engineering and encoding that training used
            pipeline = {
                "model": model,
                "feature_names": list(Xv.columns),
                "gene_rates": y_train.groupby(X_train["GeneSymbol"]).mean().to_dict(),
                "chrom_rates": y_train.groupby(X_train["Chromosome"]).mean().to_dict(),
                "global_mean": y_train.mean(),
                "label_encoders": encoders,
            }
            if "Cytogenetic" in X_train.columns:
                pipeline["cyto_rates"] = (
                    y_train.groupby(X_train["Cytogenetic"]).mean().to_dict()
                )
            joblib.dump(pipeline, "models/pipeline.joblib")
            print("Pipeline saved to models/pipeline.joblib")

        elif model_key == "ensemble":
            Xt, Xv, Xte = engineer_features(
                X_train.copy(), X_val.copy(), X_test.copy(), y_train
            )
            Xt, Xv, Xte, encoders = encode_features(Xt, Xv, Xte, method="label")

            # Train both models
            xgb_model, best_params = tune_xgboost(Xt, y_train, Xv, y_val)
            rf_model = train_model(Xt, y_train, model_type="random_forest")

            # Average probabilities from both models
            xgb_val_prob = xgb_model.predict_proba(Xv)[:, 1]
            rf_val_prob = rf_model.predict_proba(Xv)[:, 1]
            ensemble_val_prob = (xgb_val_prob + rf_val_prob) / 2

            # Find optimal threshold on validation set
            threshold = find_best_threshold(y_val, ensemble_val_prob)

            # Evaluate on validation
            evaluate(
                None, Xv, y_val, "Validation (ensemble, tuned threshold)",
                threshold=threshold, y_prob_override=ensemble_val_prob,
            )

            # Evaluate on test
            xgb_test_prob = xgb_model.predict_proba(Xte)[:, 1]
            rf_test_prob = rf_model.predict_proba(Xte)[:, 1]
            ensemble_test_prob = (xgb_test_prob + rf_test_prob) / 2

            evaluate(
                None, Xte, y_test, "Test (ensemble, tuned threshold)",
                threshold=threshold, y_prob_override=ensemble_test_prob,
            )

            # Save ensemble pipeline
            os.makedirs("models", exist_ok=True)
            pipeline = {
                "xgb_model": xgb_model,
                "rf_model": rf_model,
                "model": xgb_model,  # fallback for API compatibility
                "threshold": threshold,
                "feature_names": list(Xv.columns),
                "gene_rates": y_train.groupby(X_train["GeneSymbol"]).mean().to_dict(),
                "chrom_rates": y_train.groupby(X_train["Chromosome"]).mean().to_dict(),
                "global_mean": y_train.mean(),
                "label_encoders": encoders,
            }
            if "Cytogenetic" in X_train.columns:
                pipeline["cyto_rates"] = (
                    y_train.groupby(X_train["Cytogenetic"]).mean().to_dict()
                )
            joblib.dump(pipeline, "models/pipeline.joblib")
            print("\nEnsemble pipeline saved to models/pipeline.joblib")


if __name__ == "__main__":
    main()
