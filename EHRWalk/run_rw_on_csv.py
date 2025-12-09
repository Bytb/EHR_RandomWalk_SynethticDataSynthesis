# run_ehrwalk_for_csv.py
"""
End-to-end runner for EHRWalk on a specific visit-fact CSV.

Steps:
  1) Build heterogeneous (undirected + directed) graphs from a given CSV.
  2) Run a single generate_synthetic_dataset() call on the chosen graph.
  3) Save all outputs under: data/processed/graphs/<RUN_LABEL>/

Inputs configured at the top:
  - CSV_PATH: path to the visit-fact-style CSV (must have
      visit_occurrence_id, person_id, provider_id,
      drug_concept_ids, condition_concept_ids)
  - RUN_LABEL: name for the output subdirectory under graphs/
  - Graph hyperparameters: SAMPLE_FRAC, EXCLUDE_NODE_TYPES, USE_EDGE_WEIGHTS
  - Walk hyperparameters: RESTART_PROB, INVERSE_DEGREE, etc.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
import csv
import pickle
from typing import Any, List, Dict, Tuple

import pandas as pd
import networkx as nx

from rw_synth import generate_synthetic_dataset


# ==========================
# User-configurable settings
# ==========================

# ---- Project + data paths ----

# Project root = one level up from this script (adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input CSV: the 5,000-row visit-fact-style file used for GReaT
# (must contain the required columns listed below)
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "baselines" / "GReaT" / "great_train_visit_fact_subset_5000.csv"

# Label for this run; all graph + synth outputs go under:
#   data/processed/graphs/<RUN_LABEL>/
RUN_LABEL = "great5k"

# Graph label: "dir" for directed graph, "un" or "ud" for undirected
GRAPH_LABEL = "un"

# ---- Graph-building hyperparameters ----

SAMPLE_FRAC = 1.0         # 1.0 = use all rows in CSV_PATH
RANDOM_STATE = 42
EXCLUDE_NODE_TYPES: List[str] = []   # e.g. ["provider", "condition"]
USE_EDGE_WEIGHTS = True

# Valid node types + chain order (same as build_graph.py)
VALID_NODE_TYPES = {"visit", "person", "provider", "drug", "condition"}
CHAIN_TYPES = ["visit", "person", "condition", "provider", "drug"]

# ---- Random-walk hyperparameters (EHRWalk) ----

N_WORKERS = 1

RESTART_PROB = 0.30
USE_RW_EDGE_WEIGHT = True      # separate flag so you can decouple from graph weights if desired
INVERSE_DEGREE = False
ENCOUNTER_POLICY = "first"     # 'first' or 'second'
STOP_RULE = "complete"         # 'complete' or 'percentage'
STOP_PERCENTAGE = 0.80
MAX_STEPS = 300
RW_RANDOM_STATE = 42

# How many rows to print from the synthetic CSV for inspection
N_PREVIEW_ROWS = 5


# ==========================
# Helpers (graph building)
# ==========================

def parse_semicolon_list(value: Any) -> List[str]:
    """
    Parse a semicolon-separated string into a list of non-empty strings.
    Handles NaN/None/empty safely.
    """
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [item for item in s.split(";") if item]


def make_node_id(node_type: str, raw_id: str) -> str:
    """Create a typed node identifier, e.g., 'drug:40227012'."""
    return f"{node_type}:{raw_id}"


def build_graphs_from_csv(
    csv_path: Path,
    graphs_root: Path,
    run_label: str,
    sample_frac: float = 1.0,
    exclude_node_types: List[str] | None = None,
    use_edge_weights: bool = True,
    random_state: int = 42,
) -> Path:
    """
    Build heterogeneous undirected + directed graphs from a visit-fact-style CSV.

    Required columns in CSV:
      - visit_occurrence_id
      - person_id
      - provider_id
      - drug_concept_ids
      - condition_concept_ids

    Outputs are saved to:
      graphs_root / run_label /  (this directory is returned)
    """

    if exclude_node_types is None:
        exclude_node_types = []

    # -------------------------
    # Create output directory
    # -------------------------
    graph_dir = graphs_root / run_label
    graph_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] Output graphs will be saved to: {graph_dir}")

    # Sanitize exclude list
    exclude_set = set(exclude_node_types)
    invalid = exclude_set - VALID_NODE_TYPES
    if invalid:
        print(f"[WARN] Ignoring invalid node types in exclude_node_types: {invalid}")
        exclude_set = exclude_set & VALID_NODE_TYPES

    if "visit" in exclude_set:
        print("[WARN] 'visit' cannot be excluded. Removing.")
        exclude_set.remove("visit")

    print(f"[CONFIG] sample_frac={sample_frac}, use_edge_weights={use_edge_weights}")
    print(f"[CONFIG] exclude_node_types={sorted(exclude_set)}")

    # -------------------------
    # Load CSV
    # -------------------------
    print(f"[LOAD] {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)  # treat everything as string IDs
    print(f"[INFO] visit_fact rows: {len(df):,}")

    # -------------------------
    # Subsample visits + SAVE
    # -------------------------
    if not (0 < sample_frac <= 1.0):
        raise ValueError(f"sample_frac must be in (0,1], got {sample_frac}")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"[INFO] After subsampling: {len(df):,} visits")

    # Save sampled subset for evaluation
    sampled_csv_path = graph_dir / "sampled_visit_facts.csv"
    df.to_csv(sampled_csv_path, index=False)
    print(f"[SAVE] Sampled visit-fact subset -> {sampled_csv_path}")

    # Ensure required columns
    required_cols = [
        "visit_occurrence_id",
        "person_id",
        "provider_id",
        "drug_concept_ids",
        "condition_concept_ids",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV: {csv_path}")

    # -------------------------
    # Node + edge containers
    # -------------------------
    nodes: Dict[str, Tuple[str, str]] = {}  # node_id -> (node_type, raw_id)
    undirected_edges: Dict[Tuple[str, str, str], int] = {}  # (src, dst, type) -> weight
    directed_edges: Dict[Tuple[str, str, str], int] = {}

    # Node registration
    def register_node(node_type: str, raw_id: str) -> str:
        node_id = make_node_id(node_type, raw_id)
        if node_id not in nodes:
            nodes[node_id] = (node_type, raw_id)
        return node_id

    # Edge adders
    def add_undirected_edge(src_id: str, dst_id: str, src_type: str, dst_type: str):
        key = (src_id, dst_id, f"{src_type}-{dst_type}")
        undirected_edges[key] = undirected_edges.get(key, 0) + 1

    def add_directed_edge(src_id: str, dst_id: str, src_type: str, dst_type: str):
        key = (src_id, dst_id, f"{src_type}->{dst_type}")
        directed_edges[key] = directed_edges.get(key, 0) + 1

    # -------------------------
    # Build edges from visits
    # -------------------------
    print("[BUILD] Constructing node + edge dictionaries...")

    for row in df.itertuples(index=False):
        visit_raw = getattr(row, "visit_occurrence_id")
        if pd.isna(visit_raw):
            continue

        # Node sets per type
        nodes_by_type: Dict[str, set] = {t: set() for t in VALID_NODE_TYPES}

        # Visit
        v_id = register_node("visit", visit_raw)
        nodes_by_type["visit"].add(v_id)

        # Person
        person_raw = getattr(row, "person_id", None)
        if "person" not in exclude_set and person_raw and not pd.isna(person_raw):
            p_id = register_node("person", person_raw)
            nodes_by_type["person"].add(p_id)

        # Provider
        provider_raw = getattr(row, "provider_id", None)
        if "provider" not in exclude_set and provider_raw and not pd.isna(provider_raw):
            pr_id = register_node("provider", provider_raw)
            nodes_by_type["provider"].add(pr_id)

        # Condition nodes
        cond_raw = getattr(row, "condition_concept_ids", None)
        if "condition" not in exclude_set:
            for c in parse_semicolon_list(cond_raw):
                c_id = register_node("condition", c)
                nodes_by_type["condition"].add(c_id)

        # Drug nodes
        drug_raw = getattr(row, "drug_concept_ids", None)
        if "drug" not in exclude_set:
            for d in parse_semicolon_list(drug_raw):
                d_id = register_node("drug", d)
                nodes_by_type["drug"].add(d_id)

        # Undirected edges (visit -- others)
        for t in ["person", "provider", "condition", "drug"]:
            if t in exclude_set:
                continue
            for other_id in nodes_by_type[t]:
                add_undirected_edge(v_id, other_id, "visit", t)

        # Directed chain edges
        active_chain: List[str] = []
        for t in CHAIN_TYPES:
            if t not in exclude_set and nodes_by_type[t]:
                active_chain.append(t)

        if len(active_chain) >= 2:
            for i in range(len(active_chain) - 1):
                src_type = active_chain[i]
                dst_type = active_chain[i + 1]
                for src_id in nodes_by_type[src_type]:
                    for dst_id in nodes_by_type[dst_type]:
                        add_directed_edge(src_id, dst_id, src_type, dst_type)

    # -------------------------
    # Build the graphs
    # -------------------------
    print(f"[STATS] Unique nodes: {len(nodes):,}")
    print(f"[STATS] Undirected edge patterns: {len(undirected_edges):,}")
    print(f"[STATS] Directed edge patterns:   {len(directed_edges):,}")

    G_und = nx.Graph()
    G_dir = nx.DiGraph()

    # Add nodes
    for node_id, (node_type, raw_id) in nodes.items():
        attrs = {"node_type": node_type, "raw_id": raw_id}
        G_und.add_node(node_id, **attrs)
        G_dir.add_node(node_id, **attrs)

    # Add edges
    if use_edge_weights:
        for (src, dst, etype), w in undirected_edges.items():
            G_und.add_edge(src, dst, edge_type=etype, weight=w)
        for (src, dst, etype), w in directed_edges.items():
            G_dir.add_edge(src, dst, edge_type=etype, weight=w)
    else:
        for (src, dst, etype), _ in undirected_edges.items():
            G_und.add_edge(src, dst, edge_type=etype)
        for (src, dst, etype), _ in directed_edges.items():
            G_dir.add_edge(src, dst, edge_type=etype)

    # -------------------------
    # Save graphs + CSVs under graph_dir
    # -------------------------
    und_path = graph_dir / "hetero_graph_undirected.gpickle"
    dir_path = graph_dir / "hetero_graph_directed.gpickle"
    nodes_csv_path = graph_dir / "nodes.csv"
    und_edges_csv_path = graph_dir / "edges_undirected.csv"
    dir_edges_csv_path = graph_dir / "edges_directed.csv"

    with open(und_path, "wb") as f:
        pickle.dump(G_und, f)
    with open(dir_path, "wb") as f:
        pickle.dump(G_dir, f)

    print(f"[SAVE] Undirected graph -> {und_path}")
    print(f"[SAVE] Directed graph   -> {dir_path}")

    # Save nodes CSV
    pd.DataFrame(
        [{"node_id": nid, "node_type": nt, "raw_id": rid} for nid, (nt, rid) in nodes.items()]
    ).to_csv(nodes_csv_path, index=False)
    print(f"[SAVE] Node list        -> {nodes_csv_path}")

    # Save edge CSVs
    pd.DataFrame(
        [
            {"src": s, "dst": d, "edge_type": etype, "weight": w}
            if use_edge_weights
            else {"src": s, "dst": d, "edge_type": etype}
            for (s, d, etype), w in undirected_edges.items()
        ]
    ).to_csv(und_edges_csv_path, index=False)

    pd.DataFrame(
        [
            {"src": s, "dst": d, "edge_type": etype, "weight": w}
            if use_edge_weights
            else {"src": s, "dst": d, "edge_type": etype}
            for (s, d, etype), w in directed_edges.items()
        ]
    ).to_csv(dir_edges_csv_path, index=False)

    print(f"[SAVE] Undirected edges -> {und_edges_csv_path}")
    print(f"[SAVE] Directed edges   -> {dir_edges_csv_path}")

    # -------------------------
    # Diagnostics
    # -------------------------
    node_type_counts: Dict[str, int] = {}
    for _, (ntype, _) in nodes.items():
        node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1
    print("[SUMMARY] Node counts by type:")
    for ntype, cnt in sorted(node_type_counts.items()):
        print(f"  - {ntype}: {cnt:,}")

    return graph_dir


# ==========================
# Helpers (random-walk run)
# ==========================

def _find_graph_path(graph_dir: Path, graph_label: str) -> Path:
    """
    Find the correct graph file based on known naming convention.
    GRAPH_LABEL:
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


# ==========================
# Main
# ==========================

def main() -> None:
    # --------------------------
    # Resolve directories
    # --------------------------
    processed_dir = PROJECT_ROOT / "data" / "processed"
    graphs_root = processed_dir / "graphs"
    graphs_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Processed dir: {processed_dir}")
    print(f"[INFO] Graphs root:   {graphs_root}")
    print(f"[INFO] Input CSV:     {CSV_PATH}")
    print(f"[INFO] Run label:     {RUN_LABEL}")

    # --------------------------
    # Step 1: build graphs
    # --------------------------
    graph_dir = build_graphs_from_csv(
        csv_path=CSV_PATH,
        graphs_root=graphs_root,
        run_label=RUN_LABEL,
        sample_frac=SAMPLE_FRAC,
        exclude_node_types=EXCLUDE_NODE_TYPES,
        use_edge_weights=USE_EDGE_WEIGHTS,
        random_state=RANDOM_STATE,
    )

    # --------------------------
    # Step 2: load chosen graph
    # --------------------------
    graph_path = _find_graph_path(graph_dir, GRAPH_LABEL)
    print("[LOAD] Reading graph...")
    with graph_path.open("rb") as f:
        G = pickle.load(f)

    _summarize_graph(G)

    # --------------------------
    # Step 3: run a single synth job
    # --------------------------
    print("\n[RUN] Generating synthetic dataset (single parameter combo)...")

    synth_path = generate_synthetic_dataset(
        G=G,
        graph_label=GRAPH_LABEL,
        output_dir=graph_dir,
        restart_prob=RESTART_PROB,
        use_edge_weight=USE_RW_EDGE_WEIGHT,
        inverse_degree=INVERSE_DEGREE,
        encounter_policy=ENCOUNTER_POLICY,
        stop_rule=STOP_RULE,
        stop_percentage=STOP_PERCENTAGE,
        max_steps=MAX_STEPS,
        n_workers=N_WORKERS,
        random_state=RW_RANDOM_STATE,
    )

    print(f"\n[DONE] Synthetic dataset written to: {synth_path}")

    # --------------------------
    # Step 4: preview the output
    # --------------------------
    _preview_synth_csv(synth_path, n_rows=N_PREVIEW_ROWS)


if __name__ == "__main__":
    main()
