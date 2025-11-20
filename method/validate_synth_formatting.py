#!/usr/bin/env python

"""
================================================================================
SYNTHETIC DATA VALIDATOR (NO ARGPARSE)
================================================================================

This script validates a synthetic visit-fact CSV against the schema required
for the evaluation pipeline.

-----------------------------
REQUIRED FILE STRUCTURE
-----------------------------

This script MUST be located inside the `method/` directory:

project_root/
    data/
        real/
            sampled_visit_facts.csv        ← REAL visit-fact table
        synth/
            synth_visit_facts.csv          ← SYNTHETIC table to validate
    method/
        validate_synth.py                  ← YOU ARE HERE

-----------------------------
HOW TO RUN THE SCRIPT
-----------------------------

1. Adjust the CONFIG BLOCK below:

       SYNTH_PATH = "../data/synth/synth_visit_facts.csv"
       REAL_PATH  = "../data/real/sampled_visit_facts.csv"   (optional)

2. From inside the `method/` folder, run:

       python validate_synth.py

3. The script prints PASS/FAIL and detailed explanations of any issues.

================================================================================
"""

from pathlib import Path
import pandas as pd
import textwrap

# ==============================================================================
# CONFIG BLOCK — EDIT THESE PATHS RELATIVE TO method/ DIRECTORY
# ==============================================================================

# REQUIRED — Path to the synthetic CSV you want to validate
SYNTH_PATH = r"C:\Users\Caleb\PycharmProjects\EHR_RandomWalk_SynethticDataSynthesis\data\processed\graphs\samp100\synth\un_alpha30_edg_nodeg_first_comp.csv"

# OPTIONAL — Path to real visit-fact table for vocab checking
REAL_PATH = r"C:\Users\Caleb\PycharmProjects\EHR_RandomWalk_SynethticDataSynthesis\data\processed\graphs\samp100\sampled_visit_facts.csv"  # or "../data/real/sampled_visit_facts.csv"

# OPTIONAL — Explicit expected feature types
# Example: ["DRUG", "COND", "PROC"]
ALLOWED_FEATURE_TYPES = None

# ==============================================================================


REQUIRED_COLUMNS = ["visit_id", "person_id", "provider_id", "feature_type", "concept_id"]


class ValidationError(Exception):
    pass


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise ValidationError(f"{label} CSV not found at: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValidationError(f"Failed to read {label} CSV at {path}: {e}")
    if df.empty:
        raise ValidationError(f"{label} CSV at {path} is EMPTY.")
    return df


def check_required_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Missing required columns: {missing}\nFound: {list(df.columns)}"
        )


def check_nulls(df):
    issues = []
    for col in REQUIRED_COLUMNS:
        n_null = df[col].isna().sum()
        if n_null > 0:
            issues.append(f"  - {col}: {n_null} null values")
    if issues:
        raise ValidationError("Null values found in required columns:\n" + "\n".join(issues))


def check_feature_types(df):
    values = sorted(df["feature_type"].dropna().unique())
    print(f"[INFO] feature_type values in synth: {values}")

    if ALLOWED_FEATURE_TYPES is not None:
        bad = set(values) - set(ALLOWED_FEATURE_TYPES)
        if bad:
            raise ValidationError(
                f"Unexpected feature_type values: {sorted(bad)}\n"
                f"Allowed: {ALLOWED_FEATURE_TYPES}"
            )


def check_duplicates(df):
    dup_mask = df.duplicated(subset=["visit_id", "feature_type", "concept_id"], keep=False)
    n_dup = dup_mask.sum()
    if n_dup > 0:
        print(
            f"[WARN] Found {n_dup} duplicate rows for (visit_id, feature_type, concept_id). "
            "This is allowed but unnecessary."
        )


def check_against_real(synth_df, real_df):
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in real_df.columns]
    if missing_cols:
        raise ValidationError(
            f"Real CSV missing required columns: {missing_cols}"
        )

    synth_ft = set(synth_df["feature_type"].astype(str).unique())
    real_ft = set(real_df["feature_type"].astype(str).unique())

    missing_ft = synth_ft - real_ft
    if missing_ft:
        raise ValidationError(
            f"feature_type values present in synth but not in real: {sorted(missing_ft)}"
        )

    print("[INFO] All synth feature_type values appear in real data.")

    real_pairs = set(zip(real_df["feature_type"].astype(str),
                         real_df["concept_id"].astype(str)))

    synth_pairs = set(zip(synth_df["feature_type"].astype(str),
                          synth_df["concept_id"].astype(str)))

    missing_pairs = synth_pairs - real_pairs
    if missing_pairs:
        sample = list(sorted(missing_pairs))[:20]
        raise ValidationError(
            textwrap.dedent(f"""
                Synth contains feature pairs not present in real data.
                Total missing: {len(missing_pairs)}
                Sample (up to 20):
                {sample}
            """).strip()
        )

    print("[INFO] All (feature_type, concept_id) pairs in synth exist in real.")


def summarize(df):
    print("\n[SUMMARY]")
    print(f"  Rows (facts):          {len(df)}")
    print(f"  Unique visits:         {df['visit_id'].nunique()}")
    print(f"  Unique persons:        {df['person_id'].nunique()}")
    print(f"  Unique providers:      {df['provider_id'].nunique()}")
    print(
        f"  Unique feature tokens: {df[['feature_type', 'concept_id']].drop_duplicates().shape[0]}"
    )


def main():
    try:
        synth_path = Path(SYNTH_PATH)
        print(f"[INFO] Loading synthetic CSV: {synth_path}")
        synth_df = load_csv(synth_path, "Synth")

        # Core checks
        check_required_columns(synth_df)
        check_nulls(synth_df)
        check_feature_types(synth_df)
        check_duplicates(synth_df)
        summarize(synth_df)

        # Real-vs-synth vocab check
        if REAL_PATH is not None:
            real_path = Path(REAL_PATH)
            print(f"[INFO] Loading real CSV: {real_path}")
            real_df = load_csv(real_path, "Real")
            check_against_real(synth_df, real_df)

    except ValidationError as e:
        print("\n[VALIDATION FAILED]")
        print(str(e))
        return

    print("\n[VALIDATION PASSED] Synthetic CSV is valid and compatible with evaluation.")


if __name__ == "__main__":
    main()
