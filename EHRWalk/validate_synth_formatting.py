#!/usr/bin/env python

"""
================================================================================
SYNTHETIC DATA VALIDATOR (PER-VISIT, CAMEL-CASE SCHEMA)
================================================================================

This script validates a synthetic per-visit CSV against the schema required
by the current evaluation pipeline.

The evaluation script expects a *processed* visit-level CSV with:

    ['visitID', 'personID', 'providerID', 'drug_concept_ids', 'condition_concept_ids']

Each row = one visit.
`drug_concept_ids` and `condition_concept_ids` store lists of concept IDs
(e.g., "111;222;333" or "111|222|333").

-----------------------------
REQUIRED PROJECT STRUCTURE
-----------------------------

This script MUST live in the `EHRWalk/` directory:

project_root/
    data/
        real/
            real_visits_processed.csv         ← REAL per-visit table (optional for vocab check)
        synth/
            synth_visits_processed.csv        ← SYNTHETIC per-visit table to validate
    EHRWalk/
        validate_synth.py                     ← YOU ARE HERE

-----------------------------
HOW TO RUN
-----------------------------

1. Edit the CONFIG BLOCK below as needed:

       SYNTH_PATH = "../data/synth/synth_visits_processed.csv"
       REAL_PATH  = "../data/real/real_visits_processed.csv"   # optional

2. From inside the `EHRWalk/` folder, run:

       python validate_synth.py

3. The script prints PASS/FAIL and any issues it finds.

================================================================================
"""

from pathlib import Path
import textwrap

import pandas as pd

# ==============================================================================
# CONFIG BLOCK — EDIT THESE PATHS RELATIVE TO EHRWalk/ DIRECTORY
# ==============================================================================

# REQUIRED — Path to the synthetic per-visit CSV used by the evaluation script
SYNTH_PATH = r"C:\Users\Caleb\PycharmProjects\EHR_RandomWalk_SynethticDataSynthesis\data\processed\graphs\samp100\synth\un_alpha30_edg_nodeg_first_comp.csv"

# Separator pattern for concept ID lists inside *_concept_ids columns.
# The code uses a regex that treats any combo of ; , | whitespace as delimiters,
# so you usually don't need to change this unless your format is very weird.
# Separator pattern for concept ID lists inside *_concept_ids columns.
CONCEPT_SPLIT_PATTERN = r"[;,\s|]+"

# ==============================================================================

# REQUIRED columns in the processed per-visit CSV
REQUIRED_COLUMNS = [
    "visitID",
    "personID",
    "providerID",
    "drug_concept_ids",
    "condition_concept_ids",
]


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


def check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def check_nulls(df: pd.DataFrame) -> None:
    """
    Hard requirement: visitID must not be null.
    personID and providerID ARE allowed to be null, but we report them.
    drug_concept_ids and condition_concept_ids may be null (visits with no drugs/conds).
    """
    # Hard fail for visitID only
    n_null_visit = df["visitID"].isna().sum()
    if n_null_visit > 0:
        raise ValidationError(f"Column 'visitID' has {n_null_visit} null values (not allowed).")

    # Soft info for personID / providerID
    for col in ["personID", "providerID"]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"[INFO] Column '{col}' has {n_null} null values (allowed).")

    # Soft info for concept columns
    for col in ["drug_concept_ids", "condition_concept_ids"]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"[INFO] Column '{col}' has {n_null} null entries (visits with no {col.split('_')[0]}).")


def extract_concept_set(df: pd.DataFrame, col: str) -> set:
    """
    Extract a set of all concept IDs appearing in an *_concept_ids column.
    Assumes IDs are separated by one or more of: ; , | whitespace.
    """
    series = df[col].fillna("").astype(str)
    tokens = series.str.split(CONCEPT_SPLIT_PATTERN, regex=True).explode()
    tokens = tokens[tokens != ""]
    return set(tokens)


def summarize(df: pd.DataFrame) -> None:
    n_rows = len(df)
    n_visits = df["visitID"].nunique()
    n_persons = df["personID"].nunique()
    n_providers = df["providerID"].nunique()

    drug_ids = extract_concept_set(df, "drug_concept_ids")
    cond_ids = extract_concept_set(df, "condition_concept_ids")

    print("\n[SUMMARY]")
    print(f"  Rows (visits):         {n_rows}")
    print(f"  Unique visitID:        {n_visits}")
    print(f"  Unique personID:       {n_persons}")
    print(f"  Unique providerID:     {n_providers}")
    print(f"  Unique drug IDs:       {len(drug_ids)}")
    print(f"  Unique condition IDs:  {len(cond_ids)}")


def check_unique_visit_ids(df: pd.DataFrame) -> None:
    """
    Warn if visitID appears multiple times (usually you expect 1 row per visit).
    """
    dup_mask = df.duplicated(subset=["visitID"], keep=False)
    n_dup_rows = dup_mask.sum()
    if n_dup_rows > 0:
        n_dup_visits = df.loc[dup_mask, "visitID"].nunique()
        print(
            f"[WARN] Found {n_dup_rows} rows with duplicate visitID across "
            f"{n_dup_visits} distinct visits. "
            "Evaluation assumes one row per visit; check this is intentional."
        )


def check_against_real(synth_df: pd.DataFrame, real_df: pd.DataFrame) -> None:
    """
    Ensure that all concept IDs used in the synthetic data are present in
    the real data (for both drugs and conditions).
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in real_df.columns]
    if missing_cols:
        raise ValidationError(
            f"Real CSV is missing required columns for vocab check: {missing_cols}"
        )

    synth_drug_ids = extract_concept_set(synth_df, "drug_concept_ids")
    synth_cond_ids = extract_concept_set(synth_df, "condition_concept_ids")

    real_drug_ids = extract_concept_set(real_df, "drug_concept_ids")
    real_cond_ids = extract_concept_set(real_df, "condition_concept_ids")

    missing_drugs = synth_drug_ids - real_drug_ids
    if missing_drugs:
        sample = sorted(list(missing_drugs))[:20]
        raise ValidationError(
            textwrap.dedent(
                f"""
                Synthetic data uses {len(missing_drugs)} drug concept ID(s)
                that do NOT appear in the real data.

                Example missing drug IDs (up to 20 shown):
                {sample}
                """
            ).strip()
        )

    missing_conds = synth_cond_ids - real_cond_ids
    if missing_conds:
        sample = sorted(list(missing_conds))[:20]
        raise ValidationError(
            textwrap.dedent(
                f"""
                Synthetic data uses {len(missing_conds)} condition concept ID(s)
                that do NOT appear in the real data.

                Example missing condition IDs (up to 20 shown):
                {sample}
                """
            ).strip()
        )

    print("[INFO] All synthetic drug and condition concept IDs are present in real data.")


def main() -> None:
    try:
        synth_path = Path(SYNTH_PATH)
        print(f"[INFO] Loading synthetic CSV: {synth_path}")
        synth_df = load_csv(synth_path, "Synth")

        check_required_columns(synth_df)
        check_nulls(synth_df)
        check_unique_visit_ids(synth_df)
        summarize(synth_df)

        if REAL_PATH is not None:
            real_path = Path(REAL_PATH)
            print(f"\n[INFO] Loading real CSV: {real_path}")
            real_df = load_csv(real_path, "Real")
            check_against_real(synth_df, real_df)

    except ValidationError as e:
        print("\n[VALIDATION FAILED]")
        print(str(e))
        return
    except Exception as e:
        print("\n[UNEXPECTED ERROR]")
        print(repr(e))
        return

    print("\n[VALIDATION PASSED] Synthetic CSV is valid and compatible with the evaluation schema.")


if __name__ == "__main__":
    main()