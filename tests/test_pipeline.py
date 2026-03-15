"""
Smoke tests for the training pipeline.

These tests use synthetic data that mimics the real dataset's structure.
They verify that the pipeline functions work end-to-end without needing
the actual ClinVar data (which is gitignored and too large for CI).
"""

import numpy as np
import pandas as pd
import pytest

from src.train import encode_features, engineer_features, evaluate, split_data, train_model


@pytest.fixture
def synthetic_data():
    """Generate synthetic data matching the real dataset's schema."""
    np.random.seed(42)
    n = 500

    genes = ["BRCA1", "BRCA2", "MLH1", "TP53", "CFTR", "LDLR", "APC", "NF1"]
    chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]
    alleles = ["A", "C", "G", "T"]

    X = pd.DataFrame({
        "GeneSymbol": np.random.choice(genes, n),
        "Chromosome": np.random.choice(chroms, n),
        "Start": np.random.randint(1, 250_000_000, n),
        "ReferenceAlleleVCF": np.random.choice(alleles, n),
        "AlternateAlleleVCF": np.random.choice(alleles, n),
        "n_phenotypes": np.random.randint(1, 5, n),
    })

    y = pd.Series(np.random.choice([0, 1], n, p=[0.85, 0.15]), name="label")

    return X, y


def test_split_preserves_size(synthetic_data):
    """Train + val + test should equal the original dataset size."""
    X, y = synthetic_data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)


def test_split_stratification(synthetic_data):
    """Each split should approximately maintain the original class ratio."""
    X, y = synthetic_data
    _, _, _, y_train, y_val, y_test = split_data(X, y)

    original_rate = y.mean()
    for split in [y_train, y_val, y_test]:
        assert abs(split.mean() - original_rate) < 0.05


def test_engineer_features_adds_columns(synthetic_data):
    """Feature engineering should add rate and transversion columns."""
    X, y = synthetic_data
    X_train, X_val, X_test, y_train, _, _ = split_data(X, y)

    Xt, Xv, Xte = engineer_features(X_train.copy(), X_val.copy(), X_test.copy(), y_train)

    for df in [Xt, Xv, Xte]:
        assert "gene_pathogenic_rate" in df.columns
        assert "chrom_pathogenic_rate" in df.columns
        assert "is_transversion" in df.columns


def test_encode_features_no_strings(synthetic_data):
    """After encoding, no columns should contain string values."""
    X, y = synthetic_data
    X_train, X_val, X_test, y_train, _, _ = split_data(X, y)

    Xt, Xv, Xte, _ = encode_features(X_train.copy(), X_val.copy(), X_test.copy(), method="label")

    for df in [Xt, Xv, Xte]:
        for col in df.columns:
            assert df[col].dtype != object, f"{col} still contains strings"


def test_train_and_evaluate(synthetic_data):
    """Model should train and return valid metrics."""
    X, y = synthetic_data
    X_train, X_val, X_test, y_train, y_val, _ = split_data(X, y)

    Xt, Xv, _, _ = encode_features(X_train.copy(), X_val.copy(), X_test.copy(), method="label")
    model = train_model(Xt, y_train, model_type="logistic")
    metrics = evaluate(model, Xv, y_val, "Test")

    assert 0 <= metrics["auroc"] <= 1
    assert 0 <= metrics["f1"] <= 1
