# paramdetect_ablation.py
"""
ParamDetect: Scan ablation_results_24.csv and identify best configurations.

- Reads:
    data/processed/graphs/samp{SAMPLE_ID}/ablation_results_24.csv

- Computes per-config scores:
    score_wasserstein  (higher = better)
    score_ptest        (higher = better)
    score_f1           (higher = better)
    score_utility      (higher = better)
    score_overall      (weighted sum)

- Marks best configs (with max_steps tie-break):
    is_best_wasserstein
    is_best_ptest
    is_best_f1
    is_best_utility
    is_best_overall

- Writes:
    data/processed/graphs/samp{SAMPLE_ID}/ablation_paramdetect.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

# ==========================
# Config
# ==========================

SAMPLE_ID = 100  # samp{SAMPLE_ID}

# Sibling to ablation_results_24.csv
RESULTS_CSV_NAME = "ablation_results_24.csv"
PARAMDETECT_CSV_NAME = "ablation_paramdetect.csv"


# ==========================
# Helpers
# ==========================

def _get_project_root(this_file: Path) -> Path:
    """
    Infer project root assuming layout:
      PROJECT_ROOT/
        data/processed/graphs/samp{SAMPLE_ID}/ablation_results_24.csv
        scripts/...
    Adjust parents[...] if you place this file elsewhere.
    """
    return this_file.parents[1]


def _pick_best_with_tiebreak(
    df: pd.DataFrame,
    score_col: str,
    max_steps_col: str = "max_steps",
) -> int:
    """
    Return the index (positional) of the best row for a given score column.

    - Picks the row with maximum score_col.
    - If multiple rows tie on the score, pick the one with smaller max_steps.
    - If still multiple, pick the first among them.
    """
    # Max score
    max_score = df[score_col].max()

    # Filter rows with that max score (use exact equality; scores are simple arithmetic)
    tied = df[df[score_col] == max_score]

    if tied.shape[0] == 1:
        return int(tied.index[0])

    # Tie-break: smaller max_steps
    min_steps = tied[max_steps_col].min()
    tied_steps = tied[tied[max_steps_col] == min_steps]

    # If still multiple, take the first by index
    return int(tied_steps.index[0])


# ==========================
# Main
# ==========================

def main() -> None:
    this_file = Path(__file__).resolve()
    project_root = _get_project_root(this_file)

    graphs_root = project_root / "data" / "processed" / "graphs"
    sample_dir = graphs_root / f"samp{SAMPLE_ID}"

    results_path = sample_dir / RESULTS_CSV_NAME
    out_path = sample_dir / PARAMDETECT_CSV_NAME

    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find ablation results CSV at:\n  {results_path}\n"
            "Run run_rw_synth_ablation.py first."
        )

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Sample dir:   {sample_dir}")
    print(f"[LOAD] Ablation results: {results_path}")

    df = pd.read_csv(results_path)

    required_cols: List[str] = [
        "wasserstein_conditions",
        "wasserstein_drugs",
        "p_test_auc",
        "p_test_accuracy",
        "macro_f1_trtr",
        "macro_f1_trsr",
        "macro_f1_tstr",
        "micro_f1_trtr",
        "micro_f1_trsr",
        "micro_f1_tstr",
        "utility_ratio_trsr_trtr",
        "utility_ratio_tstr_trtr",
        "max_steps",
    ]

    # Also require restart_prob so we can filter by it
    required_cols.append("restart_prob")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in ablation_results_24.csv:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # --------------------------------------------------
    # Filter: only keep rows with restart_prob ~= 0.3
    # --------------------------------------------------
    target_restart = 0.3
    mask = np.isclose(df["restart_prob"].astype(float), target_restart)

    if not mask.any():
        raise ValueError(
            f"No rows found with restart_prob ≈ {target_restart} in {results_path}"
        )

    df = df[mask].reset_index(drop=True)
    print(f"[FILTER] Keeping {len(df)} configs with restart_prob ≈ {target_restart}")

    # ==========================
    # 1) Category scores
    # ==========================

    # Wasserstein: average, then 1 / (1 + avg)
    wasserstein_avg = (df["wasserstein_conditions"] + df["wasserstein_drugs"]) / 2.0
    score_wasserstein = 1.0 / (1.0 + wasserstein_avg)

    # P-test: deviation from 0.5, then 1 / (1 + dev)
    ptest_dev = (
        (df["p_test_auc"] - 0.5).abs() + (df["p_test_accuracy"] - 0.5).abs()
    ) / 2.0
    score_ptest = 1.0 / (1.0 + ptest_dev)

    # F1: average of all six macro/micro F1 values
    f1_cols = [
        "macro_f1_trtr",
        "macro_f1_trsr",
        "macro_f1_tstr",
        "micro_f1_trtr",
        "micro_f1_trsr",
        "micro_f1_tstr",
    ]
    score_f1 = df[f1_cols].mean(axis=1)

    # Utility: average of the two utility ratios
    util_cols = ["utility_ratio_trsr_trtr", "utility_ratio_tstr_trtr"]
    score_utility = df[util_cols].mean(axis=1)

    # ==========================
    # 2) Overall weighted score
    # ==========================

    # Weights:
    #   Wasserstein: 20%
    #   P-test:      30%
    #   F1:          30%
    #   Utility:     20%
    score_overall = (
        0.20 * score_wasserstein
        + 0.30 * score_ptest
        + 0.30 * score_f1
        + 0.20 * score_utility
    )

    # Attach scores as new columns
    df["score_wasserstein"] = score_wasserstein
    df["score_ptest"] = score_ptest
    df["score_f1"] = score_f1
    df["score_utility"] = score_utility
    df["score_overall"] = score_overall

    # ==========================
    # 3) Identify best configs
    # ==========================

    # Initialize flags
    for col in [
        "is_best_wasserstein",
        "is_best_ptest",
        "is_best_f1",
        "is_best_utility",
        "is_best_overall",
    ]:
        df[col] = False

    idx_best_w = _pick_best_with_tiebreak(df, "score_wasserstein")
    idx_best_p = _pick_best_with_tiebreak(df, "score_ptest")
    idx_best_f = _pick_best_with_tiebreak(df, "score_f1")
    idx_best_u = _pick_best_with_tiebreak(df, "score_utility")
    idx_best_o = _pick_best_with_tiebreak(df, "score_overall")

    df.loc[idx_best_w, "is_best_wasserstein"] = True
    df.loc[idx_best_p, "is_best_ptest"] = True
    df.loc[idx_best_f, "is_best_f1"] = True
    df.loc[idx_best_u, "is_best_utility"] = True
    df.loc[idx_best_o, "is_best_overall"] = True

    # ==========================
    # 4) Write output + print summary
    # ==========================

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[SAVE] ParamDetect results written to:\n  {out_path}\n")

    def _summarize(label: str, idx: int) -> None:
        row = df.loc[idx]
        knobs = {
            "graph_label": row.get("graph_label", None),
            "restart_prob": row.get("restart_prob", None),
            "use_edge_weight": row.get("use_edge_weight", None),
            "inverse_degree": row.get("inverse_degree", None),
            "encounter_policy": row.get("encounter_policy", None),
            "stop_rule": row.get("stop_rule", None),
            "stop_percentage": row.get("stop_percentage", None),
            "max_steps": row.get("max_steps", None),
        }
        print(f"[BEST {label}]")
        print("  score_wasserstein:", float(row["score_wasserstein"]))
        print("  score_ptest:      ", float(row["score_ptest"]))
        print("  score_f1:         ", float(row["score_f1"]))
        print("  score_utility:    ", float(row["score_utility"]))
        print("  score_overall:    ", float(row["score_overall"]))
        print("  knobs:", knobs)
        print("  synth_path:", row.get("synth_path", ""))
        print()

    _summarize("WASSERSTEIN", idx_best_w)
    _summarize("P-TEST", idx_best_p)
    _summarize("F1", idx_best_f)
    _summarize("UTILITY", idx_best_u)
    _summarize("OVERALL", idx_best_o)

    # ==========================
    # 5) Interactive console: show raw metrics
    # ==========================

    # Map user letters to indices + labels
    best_map = {
        "W": ("Wasserstein", idx_best_w),
        "P": ("P-test", idx_best_p),
        "F": ("F1", idx_best_f),
        "U": ("Utility", idx_best_u),
        "O": ("Overall", idx_best_o),
    }

    # Metric columns to display (raw values only)
    raw_metric_cols = [
        "wasserstein_conditions",
        "wasserstein_drugs",
        "p_test_accuracy",
        "p_test_auc",
        "macro_f1_trtr",
        "macro_f1_trsr",
        "macro_f1_tstr",
        "micro_f1_trtr",
        "micro_f1_trsr",
        "micro_f1_tstr",
        "utility_ratio_trsr_trtr",
        "utility_ratio_tstr_trtr",
    ]

    print("\n[INTERACTIVE MODE]")
    print("Enter one of: {W, P, F, U, O, exit}")
    print("  W = Best Wasserstein")
    print("  P = Best P-test")
    print("  F = Best F1")
    print("  U = Best Utility")
    print("  O = Best Overall")
    print("  exit = quit the program\n")

    while True:
        choice = input("Enter selection: ").strip().upper()

        if choice == "EXIT":
            print("Exiting.")
            break

        if choice not in best_map:
            print("Invalid input. Enter one of {W, P, F, U, O, exit}.")
            continue

        label, idx = best_map[choice]
        row = df.loc[idx]

        print(f"\n===== RAW METRICS for Best {label} =====")
        print(f"Config index: {idx}")
        print("Knobs:")
        print("  graph_label:", row.get("graph_label"))
        print("  restart_prob:", row.get("restart_prob"))
        print("  use_edge_weight:", row.get("use_edge_weight"))
        print("  inverse_degree:", row.get("inverse_degree"))
        print("  encounter_policy:", row.get("encounter_policy"))
        print("  stop_rule:", row.get("stop_rule"))
        print("  stop_percentage:", row.get("stop_percentage"))
        print("  max_steps:", row.get("max_steps"))
        print("  synth_path:", row.get("synth_path"), "\n")

        print("Raw metrics:")
        for col in raw_metric_cols:
            print(f"  {col}: {row[col]}")
        print("=======================================\n")



if __name__ == "__main__":
    main()
