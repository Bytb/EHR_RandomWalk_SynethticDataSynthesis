# probe_ablation_results.py
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

# ==========================
# Config (edit as needed)
# ==========================

SAMPLE_ID = 100  # samp{SAMPLE_ID}

# Assume same layout as run_rw_synth_ablation.py
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # adjust if you put this elsewhere
GRAPHS_ROOT = PROJECT_ROOT / "data" / "processed" / "graphs"
SAMPLE_DIR = GRAPHS_ROOT / f"samp{SAMPLE_ID}"
RESULTS_CSV_NAME = "ablation_results_24.csv"
RESULTS_PATH = SAMPLE_DIR / RESULTS_CSV_NAME

# Columns we *know* are knobs / meta (everything else we treat as metrics)
KNOWN_KNOB_COLS: List[str] = [
    "sample_id",
    "synth_path",
    "graph_label",
    "restart_prob",
    "use_edge_weight",
    "inverse_degree",
    "encounter_policy",
    "stop_rule",
    "stop_percentage",
    "max_steps",
    "n_workers",
    "random_state",
]


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find ablation results CSV at:\n  {RESULTS_PATH}"
        )

    print(f"[LOAD] {RESULTS_PATH}")
    df = pd.read_csv(RESULTS_PATH)
    n_rows, n_cols = df.shape
    print(f"[INFO] Shape: {n_rows} rows × {n_cols} columns\n")

    print("[COLUMNS]")
    for col in df.columns:
        print(f"  - {col}")
    print()

    # Identify knob vs metric columns
    knob_cols = [c for c in df.columns if c in KNOWN_KNOB_COLS]
    metric_cols = [c for c in df.columns if c not in KNOWN_KNOB_COLS]

    print("[KNOB / META COLUMNS]")
    for col in knob_cols:
        print(f"  - {col}")
    print()

    print("[METRIC COLUMNS]")
    for col in metric_cols:
        print(f"  - {col}")
    print()

    # Show a couple of example rows to see metric ranges
    print("[HEAD (first 3 rows)]")
    print(df.head(3).to_string(index=False))
    print()

    # Quick sanity: check for NaNs in metric columns
    nan_counts = df[metric_cols].isna().sum()
    if nan_counts.any():
        print("[WARN] NaNs detected in metric columns:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  - {col}: {count} NaNs")
    else:
        print("[INFO] No NaNs detected in metric columns.")


if __name__ == "__main__":
    main()
