"""FastAPI service for variant pathogenicity prediction."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "models/pipeline.joblib")
pipeline = None

VALID_ALLELES = ("A", "C", "G", "T")
VALID_CHROMOSOMES = tuple([str(i) for i in range(1, 23)] + ["X", "Y"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not os.path.exists(PIPELINE_PATH):
        raise RuntimeError("Model pipeline not found")
    pipeline = joblib.load(PIPELINE_PATH)
    yield


app = FastAPI(
    title="Meridian",
    description="Predict whether a genetic variant is pathogenic or benign",
    lifespan=lifespan,
)


class VariantInput(BaseModel):
    GeneSymbol: str = Field(min_length=1, max_length=50)
    Chromosome: Literal[*VALID_CHROMOSOMES]
    Cytogenetic: str = Field(min_length=1, max_length=20)
    Start: int = Field(ge=1, le=250_000_000)
    ReferenceAlleleVCF: Literal[*VALID_ALLELES]
    AlternateAlleleVCF: Literal[*VALID_ALLELES]
    n_phenotypes: int = Field(ge=1, le=100)


class PredictionOutput(BaseModel):
    prediction: str
    pathogenic_probability: float
    benign_probability: float


def preprocess(variant: VariantInput) -> pd.DataFrame:
    """Replicate training feature engineering and encoding on a single input."""
    row = pd.DataFrame([variant.model_dump()])

    gene_rates = pipeline["gene_rates"]
    chrom_rates = pipeline["chrom_rates"]
    global_mean = pipeline["global_mean"]

    row["gene_pathogenic_rate"] = row["GeneSymbol"].map(gene_rates).fillna(global_mean)
    row["chrom_pathogenic_rate"] = row["Chromosome"].map(chrom_rates).fillna(global_mean)

    if "cyto_rates" in pipeline:
        cyto_rates = pipeline["cyto_rates"]
        row["cyto_pathogenic_rate"] = row["Cytogenetic"].map(cyto_rates).fillna(global_mean)

    purines = {"A", "G"}
    ref_is_purine = row["ReferenceAlleleVCF"].isin(purines)
    alt_is_purine = row["AlternateAlleleVCF"].isin(purines)
    row["is_transversion"] = (ref_is_purine != alt_is_purine).astype(int)

    label_encoders = pipeline["label_encoders"]
    for col, le in label_encoders.items():
        mapping = {label: idx for idx, label in enumerate(le.classes_)}
        row[col] = row[col].map(mapping).fillna(-1).astype(int)

    row = row[pipeline["feature_names"]]
    return row


@app.post("/predict", response_model=PredictionOutput)
def predict(variant: VariantInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        row = preprocess(variant)
    except Exception:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=422, detail="Invalid input")

    try:
        if "xgb_model" in pipeline and "rf_model" in pipeline:
            xgb_prob = pipeline["xgb_model"].predict_proba(row)[0][1]
            rf_prob = pipeline["rf_model"].predict_proba(row)[0][1]
            prob_pathogenic = (xgb_prob + rf_prob) / 2
        else:
            prob_pathogenic = pipeline["model"].predict_proba(row)[0][1]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction error")

    threshold = pipeline.get("threshold", 0.5)
    pred_class = int(prob_pathogenic >= threshold)
    proba = [1 - prob_pathogenic, prob_pathogenic]

    return PredictionOutput(
        prediction="Pathogenic" if pred_class == 1 else "Benign",
        pathogenic_probability=round(float(proba[1]), 4),
        benign_probability=round(float(proba[0]), 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}
