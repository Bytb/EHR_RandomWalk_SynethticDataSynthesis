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
    graph_dir: Path,
    sample_frac: float = 1.0,
    exclude_node_types=None,
    use_edge_weights: bool = True,
    random_state: int = 42,
):
    if exclude_node_types is None:
        exclude_node_types = []

    # Sanitize exclude list
    exclude_node_types = set(exclude_node_types)
    invalid = exclude_node_types - VALID_NODE_TYPES
    if invalid:
        print(f"[WARN] Ignoring invalid node types in exclude_node_types: {invalid}")
        exclude_node_types = exclude_node_types & VALID_NODE_TYPES

    if "visit" in exclude_node_types:
        print("[WARN] 'visit' cannot be excluded as it anchors the graph. Ignoring it.")
        exclude_node_types.remove("visit")

    print(f"[CONFIG] sample_frac={sample_frac}, use_edge_weights={use_edge_weights}")
    print(f"[CONFIG] exclude_node_types={sorted(exclude_node_types)}")

    # Load visit_fact_table
    print(f"[LOAD] {visit_fact_path}")
    df = pd.read_csv(visit_fact_path, dtype=str)  # treat everything as string IDs
    print(f"[INFO] visit_fact rows: {len(df):,}")

    # Subsample visits if requested
    if not (0 < sample_frac <= 1.0):
        raise ValueError(f"sample_frac must be in (0,1], got {sample_frac}")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"[INFO] After subsampling: {len(df):,} visits")

    # Ensure required columns exist
    required_cols = [
        "visit_occurrence_id",
        "person_id",
        "provider_id",
        "drug_concept_ids",
        "condition_concept_ids",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from visit_fact_table.csv")

    # Containers for nodes and edges
    # nodes: node_id -> (node_type, raw_id)
    nodes = {}

    # edges_*: (src_id, dst_id, edge_type) -> count
    undirected_edges = {}
    directed_edges = {}

    # ------------- NODE REGISTRATION -------------

    def register_node(node_type: str, raw_id: str) -> str:
        """Return node_id; add to nodes dict if new."""
        node_id = make_node_id(node_type, raw_id)
        if node_id not in nodes:
            nodes[node_id] = (node_type, raw_id)
        return node_id

    # ------------- EDGE ADDERS -------------

    def add_undirected_edge(src_id: str, dst_id: str, src_type: str, dst_type: str):
        edge_type = f"{src_type}-{dst_type}"
        key = (src_id, dst_id, edge_type)
        undirected_edges[key] = undirected_edges.get(key, 0) + 1

    def add_directed_edge(src_id: str, dst_id: str, src_type: str, dst_type: str):
        edge_type = f"{src_type}->{dst_type}"
        key = (src_id, dst_id, edge_type)
        directed_edges[key] = directed_edges.get(key, 0) + 1

    # ------------- MAIN LOOP OVER VISITS -------------

    print("[BUILD] Constructing node and edge dictionaries from visits...")
    for row in df.itertuples(index=False):
        visit_raw = getattr(row, "visit_occurrence_id")

        if pd.isna(visit_raw):
            continue  # skip malformed rows

        # Per-row node sets by type
        nodes_by_type = {t: set() for t in VALID_NODE_TYPES}

        # Visit node
        if "visit" not in exclude_node_types:
            v_id = register_node("visit", visit_raw)
            nodes_by_type["visit"].add(v_id)
        else:
            v_id = None  # shouldn't happen because we blocked it

        # Person node
        person_raw = getattr(row, "person_id", None)
        if "person" not in exclude_node_types and person_raw and not pd.isna(person_raw):
            p_id = register_node("person", person_raw)
            nodes_by_type["person"].add(p_id)

        # Provider node
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

        # ---- Undirected edges: visit -- each other type ----
        if v_id is not None:
            for node_type in ["person", "provider", "condition", "drug"]:
                if node_type in exclude_node_types:
                    continue
                for other_id in nodes_by_type[node_type]:
                    add_undirected_edge(v_id, other_id, "visit", node_type)

        # ---- Directed edges: chained with skipping ----
        # Determine active chain types for this visit:
        #   - not excluded
        #   - has at least one node for that type
        active_chain = []
        for t in CHAIN_TYPES:
            if t in exclude_node_types:
                continue
            if len(nodes_by_type[t]) == 0:
                continue
            active_chain.append(t)

        # Need at least two types to form any directed edge
        if len(active_chain) < 2:
            continue

        # For each adjacent pair in the active chain, connect all-to-all
        for i in range(len(active_chain) - 1):
            src_type = active_chain[i]
            dst_type = active_chain[i + 1]
            for src_id in nodes_by_type[src_type]:
                for dst_id in nodes_by_type[dst_type]:
                    add_directed_edge(src_id, dst_id, src_type, dst_type)

    # ------------- BUILD NETWORKX GRAPHS -------------

    print(f"[STATS] Unique nodes: {len(nodes):,}")
    print(f"[STATS] Undirected edge patterns (src, dst, type): {len(undirected_edges):,}")
    print(f"[STATS] Directed edge patterns (src, dst, type): {len(directed_edges):,}")

    # Undirected graph
    G_und = nx.Graph()
    # Directed graph
    G_dir = nx.DiGraph()

    # Add nodes with attributes
    for node_id, (node_type, raw_id) in nodes.items():
        attrs = {"node_type": node_type, "raw_id": raw_id}
        G_und.add_node(node_id, **attrs)
        G_dir.add_node(node_id, **attrs)

    # Add edges
    if use_edge_weights:
        # Undirected with weights
        for (src, dst, edge_type), count in undirected_edges.items():
            G_und.add_edge(src, dst, edge_type=edge_type, weight=count)

        # Directed with weights
        for (src, dst, edge_type), count in directed_edges.items():
            G_dir.add_edge(src, dst, edge_type=edge_type, weight=count)
    else:
        # Undirected, unweighted
        for (src, dst, edge_type), _ in undirected_edges.items():
            G_und.add_edge(src, dst, edge_type=edge_type)

        # Directed, unweighted
        for (src, dst, edge_type), _ in directed_edges.items():
            G_dir.add_edge(src, dst, edge_type=edge_type)

    # ------------- SAVE GRAPHS -------------

    und_path = graph_dir / "hetero_graph_undirected.gpickle"
    dir_path = graph_dir / "hetero_graph_directed.gpickle"

    # Save undirected graph
    with open(und_path, "wb") as f:
        pickle.dump(G_und, f)

    # Save directed graph
    with open(dir_path, "wb") as f:
        pickle.dump(G_dir, f)

    print(f"[SAVE] Undirected graph -> {und_path}")
    print(f"[SAVE] Directed graph   -> {dir_path}")

    # ------------- SAVE NODE / EDGE LISTS -------------

    # Nodes CSV
    nodes_df = pd.DataFrame(
        [
            {"node_id": nid, "node_type": ntype, "raw_id": rid}
            for nid, (ntype, rid) in nodes.items()
        ]
    )
    nodes_csv_path = graph_dir / "nodes.csv"
    nodes_df.to_csv(nodes_csv_path, index=False)
    print(f"[SAVE] Node list        -> {nodes_csv_path}")

    # Undirected edges CSV
    und_rows = []
    for (src, dst, edge_type), count in undirected_edges.items():
        row = {"src": src, "dst": dst, "edge_type": edge_type}
        if use_edge_weights:
            row["weight"] = count
        und_rows.append(row)

    und_edges_df = pd.DataFrame(und_rows)
    und_edges_csv_path = graph_dir / "edges_undirected.csv"
    und_edges_df.to_csv(und_edges_csv_path, index=False)
    print(f"[SAVE] Undirected edges -> {und_edges_csv_path}")

    # Directed edges CSV
    dir_rows = []
    for (src, dst, edge_type), count in directed_edges.items():
        row = {"src": src, "dst": dst, "edge_type": edge_type}
        if use_edge_weights:
            row["weight"] = count
        dir_rows.append(row)

    dir_edges_df = pd.DataFrame(dir_rows)
    dir_edges_csv_path = graph_dir / "edges_directed.csv"
    dir_edges_df.to_csv(dir_edges_csv_path, index=False)
    print(f"[SAVE] Directed edges   -> {dir_edges_csv_path}")

    # ------------- DIAGNOSTIC SUMMARY -------------

    # Node counts by type
    node_type_counts = {}
    for _, (ntype, _) in nodes.items():
        node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1
    print("[SUMMARY] Node counts by type:")
    for ntype, cnt in sorted(node_type_counts.items()):
        print(f"  - {ntype}: {cnt:,}")

    # Edge counts by type
    def summarize_edge_types(edge_dict, label):
        counts = {}
        for (_, _, etype), cnt in edge_dict.items():
            counts[etype] = counts.get(etype, 0) + (cnt if use_edge_weights else 1)
        print(f"[SUMMARY] {label} edge counts by edge_type:")
        for etype, total in sorted(counts.items()):
            print(f"  - {etype}: {total:,}")

    summarize_edge_types(undirected_edges, "Undirected")
    summarize_edge_types(directed_edges, "Directed")


# ----------------- MAIN -----------------


def main():
    processed_dir, graph_dir = get_paths()
    visit_fact_path = processed_dir / "visit_fact_table.csv"

    print(f"[INFO] Processed dir: {processed_dir}")
    print(f"[INFO] Graph dir:     {graph_dir}")

    build_heterogeneous_graphs(
        visit_fact_path=visit_fact_path,
        graph_dir=graph_dir,
        sample_frac=SAMPLE_FRAC,
        exclude_node_types=EXCLUDE_NODE_TYPES,
        use_edge_weights=USE_EDGE_WEIGHTS,
        random_state=RANDOM_STATE,
    )

    print("\n[DONE] Graph construction complete.")


if __name__ == "__main__":
    main()
