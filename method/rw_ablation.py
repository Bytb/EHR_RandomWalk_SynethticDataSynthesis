"""
run_rw_synth_ablation.py

Ablation driver for the rw_synth random-walk generator.

- Loads undirected + directed graphs for samp{SAMPLE_ID}.
- Runs generate_synthetic_dataset() over a grid of hyperparameters.
- For each synthetic CSV, runs the evaluation pipeline from evaluation.py.
- Writes a single CSV (ablation_results.csv) with:
    - knob settings for each run
    - scalar evaluation metrics

Ablation grid (current):
    graph_label      ∈ {"un", "dir"}
    restart_prob     ∈ {0.1, 0.3, 0.5}
    use_edge_weight  ∈ {True, False}
    inverse_degree   ∈ {False, True}
    encounter_policy ∈ {"first", "second"}
    stop_rule        ∈ {"complete", "percentage"}
        - when stop_rule == "percentage", stop_percentage = 0.8
    max_steps        ∈ {100, 200, 300}
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import csv
import pickle

import networkx as nx
import pandas as pd
from tqdm import tqdm

from rw_synth import generate_synthetic_dataset

# Import evaluation utilities + provider CSV path
from evaluation.evaluation import (
    load_provider_specialty_map,
    normalize_real_df,
    normalize_synth_df,
    build_multi_hot_features,
    evaluate_wasserstein,
    evaluate_p_test,
    evaluate_provider_specialty_utility,
    PROVIDER_CSV_PATH,
)


# ==========================
# User-configurable settings
# ==========================

SAMPLE_ID = 100  # samp{SAMPLE_ID}

# Ablation grid
GRAPH_LABELS = ["un", "dir"]
RESTART_PROBS = [0.1, 0.3, 0.5]
USE_EDGE_WEIGHT_OPTS = [True, False]
INVERSE_DEGREE_OPTS = [False, True]
ENCOUNTER_POLICIES = ["first", "second"]
STOP_CONFIGS: List[Tuple[str, float]] = [
    ("complete", 0.8),     # stop_percentage ignored by logic but kept for logging
    ("percentage", 0.8),   # percentage rule with stop_percentage = 0.8
]
MAX_STEPS_OPTS = [100, 200, 300]

# RW "meta" defaults (kept as columns in results too)
N_WORKERS = 28
BASE_RANDOM_STATE = 42

RESULTS_CSV_NAME = "ablation_results.csv"


# ==========================
# Dataclass for configs
# ==========================

@dataclass
class AblationConfig:
    graph_label: str
    restart_prob: float
    use_edge_weight: bool
    inverse_degree: bool
    encounter_policy: str
    stop_rule: str
    stop_percentage: float
    max_steps: int

    # Meta settings
    n_workers: int = N_WORKERS
    random_state: int = BASE_RANDOM_STATE


# ==========================
# Path / graph helpers
# ==========================

def _find_graph_path(graph_dir: Path, graph_label: str) -> Path:
    """
    Map graph_label to the correct .gpickle filename.
      - 'dir' -> hetero_graph_directed.gpickle
      - 'un' / 'ud' -> hetero_graph_undirected.gpickle
    """
    label_map = {
        "dir": "hetero_graph_directed.gpickle",
        "un": "hetero_graph_undirected.gpickle",
        "ud": "hetero_graph_undirected.gpickle",
    }

    if graph_label not in label_map:
        raise ValueError(
            f"Unknown graph_label '{graph_label}'. "
            f"Expected one of: {list(label_map.keys())}"
        )

    graph_path = graph_dir / label_map[graph_label]

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Expected graph file not found:\n  {graph_path}\n"
            f"Directory contains: {list(graph_dir.glob('*.gpickle'))}"
        )

    print(f"[INFO] Using graph file: {graph_path}")
    return graph_path


def _load_graphs_for_sample(sample_dir: Path) -> Dict[str, nx.Graph]:
    """
    Load both undirected and directed graphs once and cache them.
    Returns dict: {"un": G_undirected, "dir": G_directed}
    """
    graphs: Dict[str, nx.Graph] = {}

    und_path = _find_graph_path(sample_dir, "un")
    with und_path.open("rb") as f:
        graphs["un"] = pickle.load(f)

    dir_path = _find_graph_path(sample_dir, "dir")
    with dir_path.open("rb") as f:
        graphs["dir"] = pickle.load(f)

    return graphs


# ==========================
# Evaluation wrapper
# ==========================

def run_full_evaluation_for_synth(
    real_df: pd.DataFrame,
    provider_map: Dict[str, str],
    synth_path: Path,
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline on one synthetic CSV:

      - Normalize synth schema
      - Build multi-hot features (union vocab with REAL)
      - Wasserstein
      - p-test
      - Provider specialty utility

    Returns a flat dict with scalar metrics only (no per-class dicts).
    """
    print(f"[EVAL] Loading SYNTH data: {synth_path}")
    raw_synth = pd.read_csv(synth_path)

    synth_df = normalize_synth_df(raw_synth, provider_map)

    # Build multi-hot features (union over current REAL + this SYNTH)
    X_real, X_synth, _ = build_multi_hot_features(real_df, synth_df)

    # 1) Wasserstein
    w_results = evaluate_wasserstein(real_df, synth_df, plot=False)

    # 2) p-test
    p_results = evaluate_p_test(real_df, synth_df, X_real, X_synth)

    # 3) Utility (provider specialty)
    u_results = evaluate_provider_specialty_utility(real_df, synth_df, X_real, X_synth)

    # Drop heavy / structured entries from utility dict
    u_scalar = {
        k: v
        for k, v in u_results.items()
        if not k.startswith("per_class_") and k != "retained_specialties"
    }

    # Merge all scalar metrics into one dict
    metrics: Dict[str, Any] = {}
    metrics.update(w_results)
    metrics.update(p_results)
    metrics.update(u_scalar)

    return metrics


# ==========================
# Ablation grid builder
# ==========================

def build_ablation_grid() -> List[AblationConfig]:
    configs: List[AblationConfig] = []

    for graph_label in GRAPH_LABELS:
        for restart_prob in RESTART_PROBS:
            for use_edge_weight in USE_EDGE_WEIGHT_OPTS:
                for inverse_degree in INVERSE_DEGREE_OPTS:
                    for encounter_policy in ENCOUNTER_POLICIES:
                        for stop_rule, stop_pct in STOP_CONFIGS:
                            for max_steps in MAX_STEPS_OPTS:
                                cfg = AblationConfig(
                                    graph_label=graph_label,
                                    restart_prob=restart_prob,
                                    use_edge_weight=use_edge_weight,
                                    inverse_degree=inverse_degree,
                                    encounter_policy=encounter_policy,
                                    stop_rule=stop_rule,
                                    stop_percentage=stop_pct,
                                    max_steps=max_steps,
                                )
                                configs.append(cfg)

    return configs


# ==========================
# Main ablation runner
# ==========================

def main() -> None:
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    graphs_root = project_root / "data" / "processed" / "graphs"
    sample_dir = graphs_root / f"samp{SAMPLE_ID}"

    if not sample_dir.exists():
        raise FileNotFoundError(
            f"Sample directory does not exist: {sample_dir}\n"
            "Run build_graph.py for this SAMPLE_ID first."
        )

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Graphs root:  {graphs_root}")
    print(f"[INFO] Sample dir:   {sample_dir}")

    # REAL data for evaluation
    real_csv_path = sample_dir / "sampled_visit_facts.csv"
    if not real_csv_path.exists():
        raise FileNotFoundError(
            f"Expected REAL sampled visit facts not found:\n  {real_csv_path}\n"
            "Make sure build_graph.py has been run for this SAMPLE_ID."
        )

    print(f"[LOAD] REAL visit facts: {real_csv_path}")
    raw_real = pd.read_csv(real_csv_path)

    # Provider specialties
    provider_csv_path = PROVIDER_CSV_PATH
    print(f"[LOAD] Provider CSV: {provider_csv_path}")
    provider_map = load_provider_specialty_map(provider_csv_path)

    # Normalize REAL once
    real_df = normalize_real_df(raw_real, provider_map)

    # Load graphs once
    print("[LOAD] Loading graphs (un + dir)...")
    graphs = _load_graphs_for_sample(sample_dir)

    # Build ablation grid
    configs = build_ablation_grid()
    print(f"[INFO] Number of ablation configs: {len(configs)}")

    # Prepare results
    results_csv_path = sample_dir / RESULTS_CSV_NAME
    print(f"[INFO] Ablation results will be written to: {results_csv_path}")

    results_rows: List[Dict[str, Any]] = []

    # Main ablation loop
    for cfg in tqdm(configs, desc="Ablation configs", unit="cfg"):
        G = graphs[cfg.graph_label]

        # Generate synthetic dataset for this config
        synth_path = generate_synthetic_dataset(
            G=G,
            graph_label=cfg.graph_label,
            output_dir=sample_dir,
            restart_prob=cfg.restart_prob,
            use_edge_weight=cfg.use_edge_weight,
            inverse_degree=cfg.inverse_degree,
            encounter_policy=cfg.encounter_policy,
            stop_rule=cfg.stop_rule,
            stop_percentage=cfg.stop_percentage,
            max_steps=cfg.max_steps,
            n_workers=cfg.n_workers,
            random_state=cfg.random_state,
            show_progress=False,  # keep only the outer tqdm
        )

        # Run evaluation
        metrics = run_full_evaluation_for_synth(
            real_df=real_df,
            provider_map=provider_map,
            synth_path=synth_path,
        )

        # Build row: knobs + metrics + path + sample_id
        row: Dict[str, Any] = {
            "sample_id": SAMPLE_ID,
            "synth_path": str(synth_path),
        }
        row.update(asdict(cfg))      # all knob values (includes max_steps)
        row.update(metrics)          # evaluation metrics

        results_rows.append(row)

    if not results_rows:
        print("[WARN] No results collected; nothing to write.")
        return

    # Determine CSV header from union of keys
    all_keys = set()
    for row in results_rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    # Write single CSV
    with results_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

    print(f"\n[DONE] Ablation complete. Results saved to:\n  {results_csv_path}")


if __name__ == "__main__":
    main()
