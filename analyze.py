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

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from train import (
    load_data,
    split_data,
    engineer_features,
    encode_features,
    tune_xgboost,
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

    # Summary plot: shows which features matter most and how they affect predictions.
    # Each dot is one sample. Position on x-axis = SHAP value (impact on prediction).
    # Color = feature value (red = high, blue = low).
    # A red dot on the right means "high feature value pushes prediction toward pathogenic".
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot: global feature importance (mean absolute SHAP value per feature).
    # Simpler view — just shows which features matter most overall, without
    # showing directionality.
    print("Generating SHAP feature importance plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print top features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True
    )
    print("\nFeature importance (mean |SHAP|):")
    for name, importance in feature_importance:
        print(f"  {name:30s} {importance:.4f}")

    return shap_values


def main():
    # Rebuild the best model pipeline
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Feature engineering + encoding (same as train.py)
    Xt, Xv, Xte = engineer_features(
        X_train.copy(), X_val.copy(), X_test.copy(), y_train
    )
    Xt, Xv, Xte, _ = encode_features(Xt, Xv, Xte, method="label")

    # Tune and get best model
    print("Training tuned XGBoost model...")
    model, best_params = tune_xgboost(Xt, y_train, Xv, y_val)

    # Run SHAP on validation set
    feature_names = list(Xv.columns)
    shap_values = run_shap_analysis(model, Xv, feature_names)

    print("\nPlots saved to figures/shap_summary.png and figures/shap_importance.png")


if __name__ == "__main__":
    main()
