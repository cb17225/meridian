"""
FastAPI service for variant pathogenicity prediction.

Loads the trained pipeline (model + preprocessing artifacts) and exposes
a /predict endpoint that accepts raw variant features and returns a
pathogenicity prediction with confidence score.
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "models/pipeline.joblib")
pipeline = None

VALID_ALLELES = ("A", "C", "G", "T")
VALID_CHROMOSOMES = tuple([str(i) for i in range(1, 23)] + ["X", "Y"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not os.path.exists(PIPELINE_PATH):
        raise RuntimeError(
            f"Pipeline not found at {PIPELINE_PATH}. "
            "Run: python train.py xgboost"
        )
    pipeline = joblib.load(PIPELINE_PATH)
    yield


app = FastAPI(
    title="Meridian",
    description="Predict whether a genetic variant is pathogenic or benign",
    lifespan=lifespan,
)


class VariantInput(BaseModel):
    """Raw variant features — same schema as the training data."""
    GeneSymbol: str = Field(min_length=1, max_length=50)
    Chromosome: Literal[*VALID_CHROMOSOMES]
    Start: int = Field(ge=1, le=250_000_000)
    ReferenceAlleleVCF: Literal[*VALID_ALLELES]
    AlternateAlleleVCF: Literal[*VALID_ALLELES]
    n_phenotypes: int = Field(ge=1, le=100)


class PredictionOutput(BaseModel):
    prediction: str
    pathogenic_probability: float
    benign_probability: float


def preprocess(variant: VariantInput) -> pd.DataFrame:
    """Apply the same feature engineering and encoding used during training."""
    row = pd.DataFrame([variant.model_dump()])

    # Target encoding — map gene/chrom to their training pathogenic rates
    gene_rates = pipeline["gene_rates"]
    chrom_rates = pipeline["chrom_rates"]
    global_mean = pipeline["global_mean"]

    row["gene_pathogenic_rate"] = row["GeneSymbol"].map(gene_rates).fillna(global_mean)
    row["chrom_pathogenic_rate"] = row["Chromosome"].map(chrom_rates).fillna(global_mean)

    # Transversion flag
    purines = {"A", "G"}
    ref_is_purine = row["ReferenceAlleleVCF"].isin(purines)
    alt_is_purine = row["AlternateAlleleVCF"].isin(purines)
    row["is_transversion"] = (ref_is_purine != alt_is_purine).astype(int)

    # Label encode categoricals
    label_encoders = pipeline["label_encoders"]
    for col, le in label_encoders.items():
        mapping = {label: idx for idx, label in enumerate(le.classes_)}
        row[col] = row[col].map(mapping).fillna(-1).astype(int)

    # Ensure columns match training order
    row = row[pipeline["feature_names"]]
    return row


@app.post("/predict", response_model=PredictionOutput)
def predict(variant: VariantInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        row = preprocess(variant)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    model = pipeline["model"]
    proba = model.predict_proba(row)[0]
    pred_class = int(proba[1] >= 0.5)

    return PredictionOutput(
        prediction="Pathogenic" if pred_class == 1 else "Benign",
        pathogenic_probability=round(float(proba[1]), 4),
        benign_probability=round(float(proba[0]), 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}
