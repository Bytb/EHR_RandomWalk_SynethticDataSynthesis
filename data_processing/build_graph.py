"""
build_graph.py

Construct heterogeneous graphs (undirected + directed) from visit_fact_table.csv.

Node types:
  - visit      (from visit_occurrence_id)
  - person     (from person_id)
  - provider   (from provider_id)
  - drug       (from drug_concept_ids; semicolon-separated)
  - condition  (from condition_concept_ids; semicolon-separated)

Graphs built:
  1) Undirected:
       - visit -- person
       - visit -- provider
       - visit -- each condition
       - visit -- each drug

  2) Directed (chained):
       Canonical chain of types:
         visit -> person -> condition -> provider -> drug

       For each visit, we:
         - build node sets for each type
         - remove excluded types
         - remove types with no nodes for that visit
         - connect all nodes in each adjacent pair of types

       Example chain after exclusions:
         if provider excluded -> visit -> person -> condition -> drug

Options:
  - sample_frac (float in (0, 1]): subsample visits (rows) before building graphs.
  - exclude_node_types: list of types to omit (subset of:
        {"visit", "person", "provider", "drug", "condition"}).
        "visit" will be ignored if provided.
  - use_edge_weights: if True, store edge weights = #visits that produce the edge.
"""

from pathlib import Path
import pandas as pd
import networkx as nx
import pickle

# ----------------- CONFIG DEFAULTS -----------------

SAMPLE_FRAC = 1.0         # 1.0 = use all visits
RANDOM_STATE = 42
EXCLUDE_NODE_TYPES = []   # e.g. ["provider", "condition"]
USE_EDGE_WEIGHTS = True

VALID_NODE_TYPES = {"visit", "person", "provider", "drug", "condition"}
CHAIN_TYPES = ["visit", "person", "condition", "provider", "drug"]


# ----------------- PATHS -----------------


def get_paths():
    base_dir = Path(__file__).resolve().parent
    processed_dir = base_dir.parent / "data" / "processed"
    graph_dir = processed_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir, graph_dir


# ----------------- NODE / EDGE HELPERS -----------------


def make_node_id(node_type: str, raw_id: str) -> str:
    """Create a typed node identifier, e.g., 'drug:40227012'."""
    return f"{node_type}:{raw_id}"


def parse_semicolon_list(value):
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


# ----------------- CORE BUILD FUNCTION -----------------


def build_heterogeneous_graphs(
    visit_fact_path: Path,
    processed_dir: Path,
    sample_frac: float = 1.0,
    exclude_node_types=None,
    use_edge_weights: bool = True,
    random_state: int = 42,
):
    """
    Build heterogeneous undirected + directed graphs from visit_fact_table.csv.
    Save all outputs under: processed/graphs/samp{x}/
    where x = int(sample_frac * 100).
    """

    if exclude_node_types is None:
        exclude_node_types = []

    # -------------------------
    # Create output directory
    # -------------------------
    samp_tag = int(sample_frac * 100)
    graph_dir = processed_dir / "graphs" / f"samp{samp_tag}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] Output graphs will be saved to: {graph_dir}")

    # Sanitize exclude list
    exclude_node_types = set(exclude_node_types)
    invalid = exclude_node_types - VALID_NODE_TYPES
    if invalid:
        print(f"[WARN] Ignoring invalid node types in exclude_node_types: {invalid}")
        exclude_node_types = exclude_node_types & VALID_NODE_TYPES

    if "visit" in exclude_node_types:
        print("[WARN] 'visit' cannot be excluded. Removing.")
        exclude_node_types.remove("visit")

    print(f"[CONFIG] sample_frac={sample_frac}, use_edge_weights={use_edge_weights}")
    print(f"[CONFIG] exclude_node_types={sorted(exclude_node_types)}")

    # -------------------------
    # Load visit_fact_table
    # -------------------------
    print(f"[LOAD] {visit_fact_path}")
    df = pd.read_csv(visit_fact_path, dtype=str)  # treat everything as string IDs
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
            raise ValueError(f"Missing required column '{col}'")

    # -------------------------
    # Node + edge containers
    # -------------------------
    nodes = {}  # node_id -> (node_type, raw_id)
    undirected_edges = {}  # (src, dst, type) -> weight
    directed_edges = {}

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
        nodes_by_type = {t: set() for t in VALID_NODE_TYPES}

        # Visit
        v_id = register_node("visit", visit_raw)
        nodes_by_type["visit"].add(v_id)

        # Person
        person_raw = getattr(row, "person_id", None)
        if "person" not in exclude_node_types and person_raw and not pd.isna(person_raw):
            p_id = register_node("person", person_raw)
            nodes_by_type["person"].add(p_id)

        # Provider
        provider_raw = getattr(row, "provider_id", None)
        if "provider" not in exclude_node_types and provider_raw and not pd.isna(provider_raw):
            pr_id = register_node("provider", provider_raw)
            nodes_by_type["provider"].add(pr_id)

        # Condition nodes
        cond_raw = getattr(row, "condition_concept_ids", None)
        if "condition" not in exclude_node_types:
            for c in parse_semicolon_list(cond_raw):
                c_id = register_node("condition", c)
                nodes_by_type["condition"].add(c_id)

        # Drug nodes
        drug_raw = getattr(row, "drug_concept_ids", None)
        if "drug" not in exclude_node_types:
            for d in parse_semicolon_list(drug_raw):
                d_id = register_node("drug", d)
                nodes_by_type["drug"].add(d_id)

        # Undirected edges (visit -- others)
        for t in ["person", "provider", "condition", "drug"]:
            if t in exclude_node_types:
                continue
            for other_id in nodes_by_type[t]:
                add_undirected_edge(v_id, other_id, "visit", t)

        # Directed chain edges
        active_chain = []
        for t in CHAIN_TYPES:
            if t not in exclude_node_types and nodes_by_type[t]:
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
    # Save graphs + CSVs under samp{x}
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
        [{"src": s, "dst": d, "edge_type": etype, "weight": w} if use_edge_weights else
         {"src": s, "dst": d, "edge_type": etype}
         for (s, d, etype), w in undirected_edges.items()]
    ).to_csv(und_edges_csv_path, index=False)

    pd.DataFrame(
        [{"src": s, "dst": d, "edge_type": etype, "weight": w} if use_edge_weights else
         {"src": s, "dst": d, "edge_type": etype}
         for (s, d, etype), w in directed_edges.items()]
    ).to_csv(dir_edges_csv_path, index=False)

    print(f"[SAVE] Undirected edges -> {und_edges_csv_path}")
    print(f"[SAVE] Directed edges   -> {dir_edges_csv_path}")

    # -------------------------
    # Diagnostics
    # -------------------------
    node_type_counts = {}
    for _, (ntype, _) in nodes.items():
        node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1
    print("[SUMMARY] Node counts by type:")
    for ntype, cnt in sorted(node_type_counts.items()):
        print(f"  - {ntype}: {cnt:,}")



# ----------------- MAIN -----------------


def main():
    processed_dir, _ = get_paths()  # graph_dir is no longer used
    visit_fact_path = processed_dir / "visit_fact_table.csv"

    print(f"[INFO] Processed dir: {processed_dir}")
    print(f"[INFO] Visit-Fact Path: {visit_fact_path}")

    build_heterogeneous_graphs(
        visit_fact_path=visit_fact_path,
        processed_dir=processed_dir,   # <-- changed to processed_dir
        sample_frac=SAMPLE_FRAC,
        exclude_node_types=EXCLUDE_NODE_TYPES,
        use_edge_weights=USE_EDGE_WEIGHTS,
        random_state=RANDOM_STATE,
    )

    print("\n[DONE] Graph construction complete.")



if __name__ == "__main__":
    main()
