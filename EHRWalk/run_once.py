"""
run_rw_synth_once.py

Single-run sanity check for the rw_synth random-walk generator.

- Loads one graph (e.g., a directed samp{x} graph).
- Runs generate_synthetic_dataset() once with a baseline parameter setting.
- Writes the synthetic CSV into samp{x}/synth/.
- Prints some basic stats and the first few rows.

Adjust SAMPLE_ID, GRAPH_LABEL, and path logic as needed for your project.
"""

from __future__ import annotations

from pathlib import Path
import csv
from collections import Counter

import networkx as nx

from rw_synth import generate_synthetic_dataset
import pickle



# ==========================
# User-configurable settings
# ==========================

# Which sampled graph to use, e.g. samp25
SAMPLE_ID = 100

# 'dir' for directed graph, 'ud' for undirected
GRAPH_LABEL = "un"

# Number of workers for the test run.
# For first sanity checks, keep this at 1 to avoid multiprocessing weirdness.
N_WORKERS = 1

# Baseline random-walk parameters
RESTART_PROB = 0.30
USE_EDGE_WEIGHT = True
INVERSE_DEGREE = False
ENCOUNTER_POLICY = "first"      # 'first' or 'second'
STOP_RULE = "complete"        # 'complete' or 'percentage'
STOP_PERCENTAGE = 0.80
MAX_STEPS = 100
RANDOM_STATE = 42

# How many rows to print from the synthetic CSV for inspection
N_PREVIEW_ROWS = 5


def _find_graph_path(graph_dir: Path, graph_label: str) -> Path:
    """
    Find the correct graph file based on your known naming convention:
      - hetero_graph_directed.gpickle
      - hetero_graph_undirected.gpickle

    GRAPH_LABEL should be:
      - "dir" for directed
      - "un" or "ud" for undirected
    """

    label_map = {
        "dir": "hetero_graph_directed.gpickle",
        "un":  "hetero_graph_undirected.gpickle",
        "ud":  "hetero_graph_undirected.gpickle",
    }

    if graph_label not in label_map:
        raise ValueError(
            f"Unknown GRAPH_LABEL '{graph_label}'. "
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



def _summarize_graph(G: nx.Graph) -> None:
    """
    Print high-level info about the loaded graph.
    """
    print("\n[GRAPH INFO]")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # Count nodes by node_type
    type_counts = Counter()
    for _, data in G.nodes(data=True):
        ntype = data.get("node_type", "UNKNOWN")
        type_counts[ntype] += 1

    print("  Node counts by node_type:")
    for ntype, cnt in sorted(type_counts.items(), key=lambda x: x[0]):
        print(f"    - {ntype}: {cnt:,}")


def _preview_synth_csv(csv_path: Path, n_rows: int = 5) -> None:
    """
    Print the header and first n_rows from the synthetic CSV,
    plus some simple stats.
    """
    print(f"\n[PREVIEW] Synthetic CSV: {csv_path}")

    if not csv_path.exists():
        print("  [ERROR] File does not exist.")
        return

    total_rows = 0
    person_filled = 0
    provider_filled = 0
    drugs_filled = 0
    conds_filled = 0

    print("  First rows:")
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        print("   Header:", reader.fieldnames)

        for row in reader:
            total_rows += 1

            if total_rows <= n_rows:
                print(f"   Row {total_rows}: {row}")

            # Simple stats
            if row.get("personID") not in (None, "", "NA"):
                person_filled += 1
            if row.get("providerID") not in (None, "", "NA"):
                provider_filled += 1
            if row.get("drug_concept_ids") not in (None, "", "NA"):
                drugs_filled += 1
            if row.get("condition_concept_ids") not in (None, "", "NA"):
                conds_filled += 1

    if total_rows == 0:
        print("  [WARN] No data rows found (only header?).")
        return

    print("\n[STATS]")
    print(f"  Total synthetic rows: {total_rows:,}")
    print(f"  personID filled:   {person_filled:,} ({person_filled / total_rows:.2%})")
    print(f"  providerID filled: {provider_filled:,} ({provider_filled / total_rows:.2%})")
    print(f"  drug_concept_ids:  {drugs_filled:,} ({drugs_filled / total_rows:.2%})")
    print(f"  condition_concept_ids: {conds_filled:,} ({conds_filled / total_rows:.2%})")


def main() -> None:
    # ==========================================
    # Resolve project paths (adjust if necessary)
    # ==========================================
    this_file = Path(__file__).resolve()

    # Assumes this script lives in something like <project_root>/src/
    # and data are in <project_root>/data/processed/graphs/samp{SAMPLE_ID}
    project_root = this_file.parents[1]
    graphs_root = project_root / "data" / "processed" / "graphs"
    sample_dir = graphs_root / f"samp{SAMPLE_ID}"

    if not sample_dir.exists():
        raise FileNotFoundError(
            f"Sample directory does not exist: {sample_dir}\n"
            "Adjust SAMPLE_ID or the path logic in main()."
        )

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Graphs root:  {graphs_root}")
    print(f"[INFO] Sample dir:   {sample_dir}")

    # ======================
    # Load the graph
    # ======================
    graph_path = _find_graph_path(sample_dir, GRAPH_LABEL)
    print("[LOAD] Reading graph...")
    with graph_path.open("rb") as f:
        G = pickle.load(f)

    _summarize_graph(G)

    # ======================
    # Run a single synth job
    # ======================
    print("\n[RUN] Generating synthetic dataset (single parameter combo)...")

    synth_path = generate_synthetic_dataset(
        G=G,
        graph_label=GRAPH_LABEL,
        output_dir=sample_dir,
        restart_prob=RESTART_PROB,
        use_edge_weight=USE_EDGE_WEIGHT,
        inverse_degree=INVERSE_DEGREE,
        encounter_policy=ENCOUNTER_POLICY,
        stop_rule=STOP_RULE,
        stop_percentage=STOP_PERCENTAGE,
        max_steps=MAX_STEPS,
        n_workers=N_WORKERS,
        random_state=RANDOM_STATE,
    )

    print(f"\n[DONE] Synthetic dataset written to: {synth_path}")

    # ======================
    # Inspect the output
    # ======================
    _preview_synth_csv(synth_path, n_rows=N_PREVIEW_ROWS)


if __name__ == "__main__":
    main()

