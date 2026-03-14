"""
Streamlit frontend for the Meridian variant pathogenicity predictor.

Provides a form for entering variant features and displays the model's
prediction with confidence scores. Calls the FastAPI backend.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Meridian")
st.markdown("Predict whether a genetic variant is **pathogenic** or **benign**.")
st.caption("For educational purposes only — not for clinical use.")

# --- Input form ---
st.subheader("Variant Features")

col1, col2 = st.columns(2)

with col1:
    gene = st.text_input("Gene Symbol", value="BRCA1", max_chars=50)
    chrom = st.selectbox("Chromosome", [str(i) for i in range(1, 23)] + ["X", "Y"])
    start = st.number_input("Genomic Position (Start)", min_value=1, value=43094464)
    n_phenotypes = st.number_input("Number of Phenotypes", min_value=1, max_value=100, value=1)

with col2:
    ref_allele = st.selectbox("Reference Allele", ["A", "C", "G", "T"])
    alt_allele = st.selectbox("Alternate Allele", ["A", "C", "G", "T"], index=1)
    n_submitters = st.number_input("Number of Submitters", min_value=1, max_value=1000, value=5)
    submitter_cat = st.selectbox("Submitter Categories", [2, 3])

# --- Predict ---
if st.button("Predict", type="primary"):
    payload = {
        "GeneSymbol": gene,
        "Chromosome": chrom,
        "Start": start,
        "ReferenceAlleleVCF": ref_allele,
        "AlternateAlleleVCF": alt_allele,
        "NumberSubmitters": n_submitters,
        "SubmitterCategories": submitter_cat,
        "n_phenotypes": n_phenotypes,
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        st.divider()
        st.subheader("Prediction")

        if result["prediction"] == "Pathogenic":
            st.error(f"**{result['prediction']}**")
        else:
            st.success(f"**{result['prediction']}**")

        col_a, col_b = st.columns(2)
        col_a.metric("Pathogenic Probability", f"{result['pathogenic_probability']:.2%}")
        col_b.metric("Benign Probability", f"{result['benign_probability']:.2%}")

    except requests.ConnectionError:
        st.error(
            f"Cannot connect to API at {API_URL}. "
            "Make sure the API is running: `uvicorn api:app`"
        )
    except requests.Timeout:
        st.error("Request timed out. The API may be overloaded.")
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.text}")
