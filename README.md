# meridian
Binary classification model predicting whether a genetic variant is pathogenic (1) or benign (0), trained on ClinVar data.

## Disclaimer

**This tool is NOT intended for clinical use.** Predictions made by this model should never be used for medical diagnosis, treatment decisions, or clinical reporting. The model is error-prone, and it is simply a project of passion – it does NOT provide medical advice.

## Data Pipeline

1. **`fetch_data.py`** — downloads `variant_summary.txt.gz` from NCBI FTP and extracts it to `data/`
2. **`data_cleaning.ipynb`** — filters to high-confidence SNVs on GRCh38, encodes labels, and saves `data/features.json` and `data/target.json`
3. **`eda.ipynb`** — exploratory analysis of class distribution, top genes, chromosome-level pathogenic rates, and nucleotide substitution patterns
4. **`train.py`** — trains and evaluates models with CLI model selection
5. **`analyze.py`** — SHAP interpretability analysis on the best model

## Features

| Feature | Type | Description |
|---|---|---|
| `GeneSymbol` | categorical | Gene the variant falls in |
| `Chromosome` | categorical | Chromosome (1–22, X, Y) |
| `Start` | numeric | Genomic position |
| `ReferenceAlleleVCF` | categorical | Reference nucleotide |
| `AlternateAlleleVCF` | categorical | Alternate nucleotide |
| `NumberSubmitters` | numeric | Number of labs that submitted evidence |
| `SubmitterCategories` | numeric | Type of submitting organization |
| `n_phenotypes` | numeric | Number of associated phenotypes |

### Engineered Features

| Feature | Description |
|---|---|
| `gene_pathogenic_rate` | Target-encoded GeneSymbol (per-gene pathogenic rate) |
| `chrom_pathogenic_rate` | Target-encoded Chromosome |
| `is_transversion` | Purine ↔ pyrimidine substitution flag |

**Target**: `label` — 1 (Pathogenic / Likely pathogenic), 0 (Benign / Likely benign). Class imbalance ~85% benign / ~15% pathogenic.

## Usage

```bash
# Fetch raw data
python fetch_data.py

# Train specific models
python train.py xgboost
python train.py logistic rf
python train.py              # all models

# Run SHAP analysis
python analyze.py
```

## Data Source

This project uses publicly available data from:
- **ClinVar**: A public archive of reports about relationships between human variations and phenotypes, maintained by NCBI

### Data Attribution
ClinVar: Landrum MJ, et al. (2018). ClinVar: improving access to variant interpretations and supporting evidence. Nucleic Acids Res. 46(D1):D1062-D1067.
