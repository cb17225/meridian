"""
Model interpretability analysis using SHAP.

SHAP (SHapley Additive exPlanations) assigns each feature an importance value
for each prediction. Unlike global feature importance (which just says "GeneSymbol
is important"), SHAP shows HOW each feature pushes a specific prediction toward
pathogenic or benign, and by how much.

Based on cooperative game theory: each feature is a "player" and the prediction
is the "payout". SHAP calculates the marginal contribution of each feature
across all possible feature combinations, giving a fair attribution.
"""

import matplotlib
matplotlib.use("Agg")

import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from train import (
    load_data,
    split_data,
    engineer_features,
    encode_features,
)


def run_shap_analysis(model, X, feature_names, output_dir="figures"):
    """
    Run SHAP analysis and generate plots.

    TreeExplainer is used because our model is tree-based (XGBoost). It's an
    exact algorithm — not an approximation — that computes SHAP values
    efficiently by exploiting the tree structure.
    """
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

    # Load saved model if available, otherwise retrain
    if os.path.exists(model_path) and os.path.exists(features_path):
        print("Loading saved model...")
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
    else:
        print("No saved model found. Run 'python train.py xgboost' first.")
        return

    # Prepare validation data with same pipeline
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    Xt, Xv, Xte = engineer_features(
        X_train.copy(), X_val.copy(), X_test.copy(), y_train
    )
    Xt, Xv, Xte, _ = encode_features(Xt, Xv, Xte, method="label")

    # Run SHAP on validation set
    shap_values = run_shap_analysis(model, Xv, feature_names)

    print("\nPlots saved to figures/shap_summary.png and figures/shap_importance.png")


if __name__ == "__main__":
    main()
