"""SHAP interpretability analysis for the trained model."""

import matplotlib

matplotlib.use("Agg")

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.train import (
    encode_features,
    engineer_features,
    load_data,
    split_data,
)


def run_shap_analysis(model, X, feature_names, output_dir="figures"):
    """Generate SHAP summary and feature importance plots."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Generating SHAP feature importance plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True
    )
    print("\nFeature importance (mean |SHAP|):")
    for name, importance in feature_importance:
        print(f"  {name:30s} {importance:.4f}")

    return shap_values


def main():
    model_path = "models/xgboost_best.joblib"
    features_path = "models/feature_names.joblib"

    if os.path.exists(model_path) and os.path.exists(features_path):
        print("Loading saved model...")
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
    else:
        print("No saved model found. Run 'python src/train.py xgboost' first.")
        return

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    Xt, Xv, Xte = engineer_features(
        X_train.copy(), X_val.copy(), X_test.copy(), y_train
    )
    Xt, Xv, Xte, _ = encode_features(Xt, Xv, Xte, method="label")

    run_shap_analysis(model, Xv, feature_names)

    print("\nPlots saved to figures/shap_summary.png and figures/shap_importance.png")


if __name__ == "__main__":
    main()
