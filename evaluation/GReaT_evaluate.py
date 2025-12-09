# GReaT_evaluate.py
"""
Evaluate GReaT synthetic rows using the same evaluation pipeline
(Wasserstein, p-test, TRTR/TSTR utility) as rw_synth, by
reconstructing a tabular schema from GReaT's row_text.

Assumes this file lives in the same package as evaluation.py.
"""

from pathlib import Path
import re
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

import evaluation as ev  # your evaluation.py module


# ---------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------

# Reuse project root and REAL paths from evaluation.py if you want,
# or override SAMPLE_ID here.
PROJECT_ROOT: Path = ev.PROJECT_ROOT
SAMPLE_ID = 100

# REAL visit facts (same as evaluation.test_evaluations)
REAL_FACTS_PATH = ev.REAL_FACTS_PATH

# Provider mapping CSV (used only for REAL side)
PROVIDER_CSV_PATH = ev.PROVIDER_CSV_PATH

# GReaT paths
GREAT_DIR = PROJECT_ROOT / "data" / "processed" / "baselines" / "GReaT"
GREAT_REAL_PATH = GREAT_DIR / "great_train_input.csv"
GREAT_SYNTH_PATH = GREAT_DIR / "great_tabularisai_Qwen3-0.3B-distil_ep3_synth.csv"


# ---------------------------------------------------------------------
# Helpers to parse row_text from GReaT
# ---------------------------------------------------------------------

def extract_segment(text: str, field: str) -> str:
    """
    Generic parser to extract 'field = value' from row_text.

    Example expected patterns:
      'provider_specialty = Family Medicine | drug_concept_ids = 123;456 | ...'
      'drug_concept_ids = 123;456 | condition_concept_ids = 789;111'

    Returns the raw value string (no splitting), or '' if not found.
    """
    if not isinstance(text, str):
        return ""
    # match 'field = <anything up to next | or end-of-line>'
    pattern = rf"{re.escape(field)}\s*=\s*([^|]+)"
    m = re.search(pattern, text)
    if not m:
        return ""
    return m.group(1).strip()


def extract_provider_specialty(text: str) -> str:
    """
    Specialized wrapper for provider_specialty so we can handle
    degenerate rows like 'provider_specialty' with no '='.
    """
    if not isinstance(text, str):
        return "Unknown"
    val = extract_segment(text, "provider_specialty")
    if val:
        return val
    # Degenerate rows often look like just 'provider_specialty'
    stripped = text.strip()
    if stripped.startswith("provider_specialty"):
        return "Unknown"
    return "Unknown"


def extract_drug_ids(text: str) -> str:
    """
    Extract drug_concept_ids value. Expect a semicolon-separated list,
    but we return the raw string; parse_concept_list downstream will
    split and handle empty/NA.
    """
    return extract_segment(text, "drug_concept_ids")


def extract_condition_ids(text: str) -> str:
    """
    Extract condition_concept_ids value.
    """
    return extract_segment(text, "condition_concept_ids")


def compute_counts_from_strings(
    df: pd.DataFrame,
    drug_col: str = "drug_concept_ids",
    cond_col: str = "condition_concept_ids",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute #drugs and #conditions using evaluation.parse_concept_list
    from string columns in a GReaT-style DataFrame.
    """
    drug_counts = df[drug_col].apply(ev.parse_concept_list).apply(len).to_numpy()
    cond_counts = df[cond_col].apply(ev.parse_concept_list).apply(len).to_numpy()
    return drug_counts, cond_counts


# ---------------------------------------------------------------------
# Build GReaT synthetic DataFrame in the expected schema
# ---------------------------------------------------------------------

def build_great_synth_df(raw_synth: pd.DataFrame) -> pd.DataFrame:
    """
    From GReaT SYNTH with a single 'row_text' column, build a DataFrame
    matching the normalized schema expected by evaluation.py.

    Columns returned:
      visitID, personID, providerID (dummy),
      provider_specialty (canonicalized),
      drug_concept_ids, condition_concept_ids,
      n_drugs, n_conditions
    """
    if "row_text" not in raw_synth.columns:
        raise ValueError("Expected 'row_text' column in GReaT SYNTH CSV.")

    df = pd.DataFrame()
    df["row_text"] = raw_synth["row_text"].astype(str)

    # Dummy IDs (not used by metrics for GReaT)
    n_rows = df.shape[0]
    df["visitID"] = np.arange(n_rows)
    df["personID"] = -1  # placeholder
    df["providerID"] = "-1"  # placeholder

    # Parse fields
    df["provider_specialty"] = df["row_text"].map(extract_provider_specialty)
    df["drug_concept_ids"] = df["row_text"].map(extract_drug_ids)
    df["condition_concept_ids"] = df["row_text"].map(extract_condition_ids)

    # Canonicalize specialties using the same mapping as REAL
    df["provider_specialty"] = df["provider_specialty"].apply(ev.canonicalize_specialty)

    # Compute counts
    n_drugs, n_conds = compute_counts_from_strings(df)
    df["n_drugs"] = n_drugs
    df["n_conditions"] = n_conds

    # Drop the raw row_text; keep schema compatible with normalize_* output
    df = df[
        [
            "visitID",
            "personID",
            "providerID",
            "provider_specialty",
            "drug_concept_ids",
            "condition_concept_ids",
            "n_drugs",
            "n_conditions",
        ]
    ]
    return df


# ---------------------------------------------------------------------
# Main evaluation function for GReaT
# ---------------------------------------------------------------------

def evaluate_great(
    real_facts_path: Path = REAL_FACTS_PATH,
    great_real_path: Path = GREAT_REAL_PATH,    # not strictly needed, but kept for completeness
    great_synth_path: Path = GREAT_SYNTH_PATH,
    provider_csv_path: Path = PROVIDER_CSV_PATH,
    plot_wasserstein: bool = False,
) -> None:
    """
    Evaluate GReaT synthetic data against REAL visit facts.

    Steps:
      - Load REAL sampled_visit_facts.csv
      - Load provider specialty map
      - Normalize REAL using evaluation.normalize_real_df
      - Load GReaT SYNTH row_text CSV
      - Parse row_text -> tabular schema for SYNTH
      - Build multi-hot features (REAL vs GReaT SYNTH)
      - Run: Wasserstein, p-test, provider-specialty utility
    """
    # ----------------------------
    # REAL side (as in evaluation.test_evaluations)
    # ----------------------------
    print(f"[LOAD] REAL visit facts: {real_facts_path}")
    raw_real = pd.read_csv(real_facts_path)

    print(f"[LOAD] Provider CSV: {provider_csv_path}")
    provider_map = ev.load_provider_specialty_map(provider_csv_path)

    # Normalize REAL schema using your existing helper
    real_df = ev.normalize_real_df(raw_real, provider_map)

    # ----------------------------
    # GReaT SYNTH side
    # ----------------------------
    print(f"[LOAD] GReaT SYNTH: {great_synth_path}")
    raw_synth = pd.read_csv(great_synth_path)

    # Build GReaT synthetic DataFrame in the expected schema
    synth_df = build_great_synth_df(raw_synth)

    # ----------------------------
    # Multi-hot features (drugs+conditions)
    # ----------------------------
    X_real, X_synth, _ = ev.build_multi_hot_features(real_df, synth_df)

    # ----------------------------
    # 1) Wasserstein (#drugs, #conditions)
    # ----------------------------
    w_results = ev.evaluate_wasserstein(real_df, synth_df, plot=plot_wasserstein)

    # ----------------------------
    # 2) p-test (REAL vs SYNTH)
    # ----------------------------
    p_results = ev.evaluate_p_test(real_df, synth_df, X_real, X_synth)

    # ----------------------------
    # 3) Provider specialty utility (TRTR/TSTR/TR+S->R)
    # ----------------------------
    u_results = ev.evaluate_provider_specialty_utility(real_df, synth_df, X_real, X_synth)

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n[SUMMARY: GReaT vs REAL]")
    print("Wasserstein:", w_results)
    print("p-test:", p_results)
    print("Utility (headline):", {
        k: v for k, v in u_results.items() if not k.startswith("per_class")
    })

    print("\n[Per-class F1 (TRTR)]:")
    for cls, f1_val in u_results.get("per_class_f1_trtr", {}).items():
        print(f"  {cls}: {f1_val:.4f}")

    print("\n[Per-class F1 (TSTR)]:")
    for cls, f1_val in u_results.get("per_class_f1_tstr", {}).items():
        print(f"  {cls}: {f1_val:.4f}")

    print("\n[GReaT EVAL COMPLETE]")


if __name__ == "__main__":
    evaluate_great(plot_wasserstein=True)
