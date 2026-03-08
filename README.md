# meridian
Binary classification model predicting whether a genetic variant is pathogenic (1) or benign (0), trained on ClinVar data.

## Disclaimer

**This tool is NOT intended for clinical use.** Predictions made by this model should never be used for medical diagnosis, treatment decisions, or clinical reporting. The model is error-prone, and it is simply a project of passion – it does NOT provide medical advice.

## Data Pipeline

1. **`fetch_data.py`** — downloads `variant_summary.txt.gz` from NCBI FTP and extracts it to `data/`
2. **`data_cleaning.ipynb`** — filters to high-confidence SNVs on GRCh38, encodes labels, and saves `data/features.json` and `data/target.json`
3. **`eda.ipynb`** — exploratory analysis of class distribution, top genes, chromosome-level pathogenic rates, and nucleotide substitution patterns

## Features

| Feature | Description |
|---|---|
| `GeneSymbol` | Gene the variant falls in |
| `Chromosome` | Chromosome (1–22, X, Y) |
| `Start` / `Stop` | Genomic position |
| `ReferenceAlleleVCF` | Reference nucleotide |
| `AlternateAlleleVCF` | Alternate nucleotide |

**Target**: `label` — 1 (Pathogenic / Likely pathogenic), 0 (Benign / Likely benign). Class imbalance ~85% benign / ~15% pathogenic.

## Data Source

This project uses publicly available data from:
- **ClinVar**: A public archive of reports about relationships between human variations and phenotypes, maintained by NCBI

### Data Attribution
ClinVar: Landrum MJ, et al. (2018). ClinVar: improving access to variant interpretations and supporting evidence. Nucleic Acids Res. 46(D1):D1062-D1067.
