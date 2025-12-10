# paramdetect_ablation.py
"""
ParamDetect: Scan ablation_results_merge.csv and identify best configurations.

- Reads:
    data/processed/graphs/samp{SAMPLE_ID}/ablation_results_merge.csv

- Computes per-config scores:
    score_wasserstein  (normalized, higher = better; used ONLY for weighted overall)
    score_ptest        (normalized, higher = better; used ONLY for weighted overall)
    score_f1           (higher = better)
    score_utility      (higher = better)
    score_overall      (weighted sum)

- For EACH restart_prob, marks best configs (with max_steps tie-break):
    is_best_wasserstein   (min avg Wasserstein)
    is_best_ptest         (min distance of p-test from 0.5 using AUC + accuracy)
    is_best_f1            (max score_f1)
    is_best_utility       (max score_utility)
    is_best_overall       (max score_overall)

- Also prints GLOBAL best configs across all restart_prob for quick inspection.

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

# Sibling to ablation_results_merge.csv
RESULTS_CSV_NAME = "ablation_results_merge.csv"
PARAMDETECT_CSV_NAME = "ablation_paramdetect.csv"


# ==========================
# Helpers
# ==========================

def _get_project_root(this_file: Path) -> Path:
    """
    Infer project root assuming layout:
      PROJECT_ROOT/
        data/processed/graphs/samp{SAMPLE_ID}/ablation_results_merge.csv
        scripts/...
    Adjust parents[...] if you place this file elsewhere.
    """
    return this_file.parents[1]


def _pick_best_with_tiebreak_max(
    df: pd.DataFrame,
    score_col: str,
    max_steps_col: str = "max_steps",
) -> int:
    """
    Return index (label) of the best row for a given score column (maximize).

    - Picks the row with maximum score_col.
    - If multiple rows tie on the score, pick the one with smaller max_steps.
    - If still multiple, pick the first among them.
    """
    max_score = df[score_col].max()
    tied = df[df[score_col] == max_score]

    if tied.shape[0] == 1:
        return int(tied.index[0])

    min_steps = tied[max_steps_col].min()
    tied_steps = tied[tied[max_steps_col] == min_steps]
    return int(tied_steps.index[0])


def _pick_best_with_tiebreak_min(
    df: pd.DataFrame,
    score_col: str,
    max_steps_col: str = "max_steps",
) -> int:
    """
    Return index (label) of the best row for a given score column (minimize).

    - Picks the row with minimum score_col.
    - If multiple rows tie on the score, pick the one with smaller max_steps.
    - If still multiple, pick the first among them.
    """
    min_score = df[score_col].min()
    tied = df[df[score_col] == min_score]

    if tied.shape[0] == 1:
        return int(tied.index[0])

    min_steps = tied[max_steps_col].min()
    tied_steps = tied[tied[max_steps_col] == min_steps]
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
        "restart_prob",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in ablation_results_merge.csv:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # ==========================
    # 1) Raw aggregates used for selection + reporting
    # ==========================

    # Raw average Wasserstein (lower is better for selection)
    df["wasserstein_avg"] = (
        df["wasserstein_conditions"] + df["wasserstein_drugs"]
    ) / 2.0

    # Raw p-test average (we will REPORT this; selection uses deviation)
    df["p_test_avg"] = (
        df["p_test_auc"] + df["p_test_accuracy"]
    ) / 2.0

    # Deviation of p-test from 0.5 (AUC + accuracy); lower is better for selection
    df["p_test_dev"] = (
        (df["p_test_auc"] - 0.5).abs() + (df["p_test_accuracy"] - 0.5).abs()
    ) / 2.0

    # ==========================
    # 2) Normalized scores for overall weighted score ONLY
    # ==========================

    # Wasserstein normalized: 1 / (1 + avg)
    score_wasserstein = 1.0 / (1.0 + df["wasserstein_avg"])

    # P-test deviation normalized: smaller dev -> larger score
    score_ptest = 1.0 / (1.0 + df["p_test_dev"])

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

    # Weighted overall score (unchanged)
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
    # 3) Initialize flags
    # ==========================

    for col in [
        "is_best_wasserstein",
        "is_best_ptest",
        "is_best_f1",
        "is_best_utility",
        "is_best_overall",
    ]:
        df[col] = False

    # ==========================
    # 4) Mark best configs per restart_prob
    # ==========================

    print("\n[PER-RESTART BEST CONFIGS]")
    for rp, group in df.groupby("restart_prob"):
        sub = group

        idx_best_w = _pick_best_with_tiebreak_min(sub, "wasserstein_avg")
        idx_best_p = _pick_best_with_tiebreak_min(sub, "p_test_dev")
        idx_best_f = _pick_best_with_tiebreak_max(sub, "score_f1")
        idx_best_u = _pick_best_with_tiebreak_max(sub, "score_utility")
        idx_best_o = _pick_best_with_tiebreak_max(sub, "score_overall")

        df.loc[idx_best_w, "is_best_wasserstein"] = True
        df.loc[idx_best_p, "is_best_ptest"] = True
        df.loc[idx_best_f, "is_best_f1"] = True
        df.loc[idx_best_u, "is_best_utility"] = True
        df.loc[idx_best_o, "is_best_overall"] = True

        print(f"\n  [restart_prob = {rp}]")
        print(f"    best Wasserstein idx: {idx_best_w}")
        print(f"    best p-test      idx: {idx_best_p}")
        print(f"    best F1          idx: {idx_best_f}")
        print(f"    best Utility     idx: {idx_best_u}")
        print(f"    best Overall     idx: {idx_best_o}")

    # ==========================
    # 5) Global best configs (across all restart_prob) for quick summary
    # ==========================

    idx_best_w_global = _pick_best_with_tiebreak_min(df, "wasserstein_avg")
    idx_best_p_global = _pick_best_with_tiebreak_min(df, "p_test_dev")
    idx_best_f_global = _pick_best_with_tiebreak_max(df, "score_f1")
    idx_best_u_global = _pick_best_with_tiebreak_max(df, "score_utility")
    idx_best_o_global = _pick_best_with_tiebreak_max(df, "score_overall")

    print("\n[GLOBAL BEST CONFIGS]")

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
        print("  knobs:", knobs)
        print("  synth_path:", row.get("synth_path", ""))
        print("  --- RAW METRICS ---")
        print(f"  wasserstein_drugs:      {float(row['wasserstein_drugs']):.4f}")
        print(f"  wasserstein_conditions: {float(row['wasserstein_conditions']):.4f}")
        print(f"  wasserstein_avg:        {float(row['wasserstein_avg']):.4f}")
        print(f"  p_test_auc:             {float(row['p_test_auc']):.4f}")
        print(f"  p_test_accuracy:        {float(row['p_test_accuracy']):.4f}")
        print(f"  p_test_avg:             {float(row['p_test_avg']):.4f}")
        print("  --- AGGREGATE SCORES (for overall) ---")
        print(f"  score_wasserstein: {float(row['score_wasserstein']):.4f}")
        print(f"  score_ptest:       {float(row['score_ptest']):.4f}")
        print(f"  score_f1:          {float(row['score_f1']):.4f}")
        print(f"  score_utility:     {float(row['score_utility']):.4f}")
        print(f"  score_overall:     {float(row['score_overall']):.4f}")
        print()

    _summarize("WASSERSTEIN", idx_best_w_global)
    _summarize("P-TEST", idx_best_p_global)
    _summarize("F1", idx_best_f_global)
    _summarize("UTILITY", idx_best_u_global)
    _summarize("OVERALL", idx_best_o_global)

    # ==========================
    # 6) Write output CSV
    # ==========================

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[SAVE] ParamDetect results written to:\n  {out_path}\n")

    # ==========================
    # 7) Interactive console: show raw metrics for GLOBAL best configs
    # ==========================

    best_map = {
        "W": ("Wasserstein", idx_best_w_global),
        "P": ("P-test", idx_best_p_global),
        "F": ("F1", idx_best_f_global),
        "U": ("Utility", idx_best_u_global),
        "O": ("Overall", idx_best_o_global),
    }

    raw_metric_cols = [
        "wasserstein_conditions",
        "wasserstein_drugs",
        "wasserstein_avg",
        "p_test_accuracy",
        "p_test_auc",
        "p_test_avg",
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
    print("  W = Best Wasserstein (global)")
    print("  P = Best P-test (global)")
    print("  F = Best F1 (global)")
    print("  U = Best Utility (global)")
    print("  O = Best Overall (global)")
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

        print(f"\n===== RAW METRICS for Best {label} (GLOBAL) =====")
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
