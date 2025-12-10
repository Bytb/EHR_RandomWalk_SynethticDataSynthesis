"""
dataset_summary.py

Compute basic stats from the visit-centric table
(visit_fact_table.csv or sampled_visit_facts.csv).

Assumes columns:
  - visit_occurrence_id
  - person_id
  - provider_id
  - drug_concept_ids       (semicolon-separated IDs)
  - condition_concept_ids  (semicolon-separated IDs)
  - n_drugs  (optional; will be derived if missing)
  - n_conditions (optional; will be derived if missing)
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

# This file lives in: PROJECT_ROOT / "evaluation" / dataset_summary.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main real visit-fact table (same contents as samp100/sampled_visit_facts.csv)
VISIT_FACT_PATH = PROJECT_ROOT / "data" / "processed" / "visit_fact_table.csv"
# If you ever want to point to samp100 explicitly, uncomment:
# VISIT_FACT_PATH = PROJECT_ROOT / "data" / "processed" / "graphs" / "samp100" / "sampled_visit_facts.csv"


# =============================================================================
# HELPERS
# =============================================================================

def _split_ids(col: pd.Series) -> pd.Series:
    """Turn semicolon strings into lists; handle NaN/empty safely."""
    def _split(x):
        if pd.isna(x):
            return []
        s = str(x).strip()
        if not s:
            return []
        return [tok for tok in s.split(";") if tok]
    return col.apply(_split)


def _summary(name: str, values: np.ndarray) -> None:
    """Print basic summary statistics for a 1D numeric array."""
    if len(values) == 0:
        print(f"[{name}] no data")
        return
    vals = np.asarray(values, dtype=float)
    print(f"[{name}] n={len(vals):,}")
    print(f"  min   = {vals.min():.3f}")
    print(f"  max   = {vals.max():.3f}")
    print(f"  mean  = {vals.mean():.3f}")
    print(f"  std   = {vals.std(ddof=1):.3f}")
    print(f"  p50   = {np.percentile(vals, 50):.3f}")
    print(f"  p90   = {np.percentile(vals, 90):.3f}")
    print(f"  p99   = {np.percentile(vals, 99):.3f}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print(f"[INFO] Loading visit facts from: {VISIT_FACT_PATH}")
    df = pd.read_csv(VISIT_FACT_PATH)
    print(f"[INFO] rows (visits): {len(df):,}\n")

    # ---- core cardinalities ----
    n_visits = df["visit_occurrence_id"].nunique()
    n_persons = df["person_id"].nunique()
    n_providers = df["provider_id"].nunique()

    drug_lists = _split_ids(df["drug_concept_ids"])
    cond_lists = _split_ids(df["condition_concept_ids"])

    all_drugs = {d for lst in drug_lists for d in lst}
    all_conds = {c for lst in cond_lists for c in lst}

    print("[CARDINALITIES]")
    print(f"  unique visits     = {n_visits:,}")
    print(f"  unique persons    = {n_persons:,}")
    print(f"  unique providers  = {n_providers:,}")
    print(f"  unique drugs      = {len(all_drugs):,}")
    print(f"  unique conditions = {len(all_conds):,}")
    print()

    # ---- per-visit counts ----
    if "n_drugs" in df.columns:
        n_drugs = df["n_drugs"].to_numpy()
    else:
        n_drugs = drug_lists.apply(len).to_numpy()

    if "n_conditions" in df.columns:
        n_conds = df["n_conditions"].to_numpy()
    else:
        n_conds = cond_lists.apply(len).to_numpy()

    n_events = n_drugs + n_conds

    _summary("#drugs per visit", n_drugs)
    _summary("#conditions per visit", n_conds)
    _summary("total events per visit (#drugs + #conditions)", n_events)

    # ---- visits per person ----
    vp = (
        df.groupby("person_id")["visit_occurrence_id"]
          .nunique()
          .to_numpy()
    )
    _summary("visits per person", vp)

    # ---- events per provider ----
    # count how many events (drug + condition entries) each provider sees
    provider_events = (
        df.assign(n_events=n_events)
          .groupby("provider_id")["n_events"]
          .sum()
          .to_numpy()
    )
    _summary("events per provider (drug + condition)", provider_events)

    print("[DONE] dataset summary complete.")


if __name__ == "__main__":
    main()
