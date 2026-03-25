"""
Download ClinVar variant summary data from NCBI.
"""

import gzip
import os
import shutil
import urllib.request

import pandas as pd


def download_clinvar(data_dir="data", force=False):
    """Download and extract ClinVar variant summary file."""
    os.makedirs(data_dir, exist_ok=True)

    url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
    gz_file = os.path.join(data_dir, "variant_summary.txt.gz")
    txt_file = os.path.join(data_dir, "variant_summary.txt")

    if os.path.exists(txt_file) and not force:
        print(f"Using existing data at {txt_file}")
        return txt_file

    try:
        urllib.request.urlretrieve(url, gz_file)

        with gzip.open(gz_file, 'rb') as f_in:
            with open(txt_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(gz_file)
        print(f"Download complete. Saved to {txt_file}")
        return txt_file

    except Exception as e:
        print(f"Download failed: {e}")
        raise


def summarize_data(filepath):
    """Load and display summary statistics."""

    print("\nLoading data...")
    df = pd.read_csv(filepath, sep='\t', low_memory=False)

    print(f"\nDataset: {len(df):,} variants, {len(df.columns)} columns")

    key_columns = {
        'ClinicalSignificance': 'Clinical Significance',
        'Type': 'Variant Type',
        'ReviewStatus': 'Review Status'
    }

    for col, label in key_columns.items():
        if col in df.columns:
            print(f"\n{label}:")
            counts = df[col].value_counts().head(5)
            for val, count in counts.items():
                pct = 100 * count / len(df)
                print(f"  {val}: {count:,} ({pct:.1f}%)")

    return df


if __name__ == "__main__":
    filepath = download_clinvar()
    df = summarize_data(filepath)
    print("\nData extraction complete!")
    print(f"Location: {filepath}")
    print(f"\nLoad with: pd.read_csv('{filepath}', sep='\\t')")