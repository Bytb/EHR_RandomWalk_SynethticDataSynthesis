# subsample_multi_ego_for_gephi.py
#
# Extract:
#   - a multi-ego subgraph around several visit nodes from the undirected
#     heterogeneous EHR graph, and
#   - a directed subgraph around a single randomly sampled person node
# from the directed heterogeneous EHR graph.
#
# Export both as GraphML for visualization in Gephi.

from __future__ import annotations

import pickle
import random
from pathlib import Path

import networkx as nx


# =========================
# CONFIG
# =========================

SAMPLE_ID = 100  # samp{SAMPLE_ID}

# If you put this script inside "evaluation/", change parents[0] -> parents[1].
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_DIR = PROJECT_ROOT / "data" / "processed" / "graphs" / f"samp{SAMPLE_ID}"

UND_GPKL = GRAPH_DIR / "hetero_graph_undirected.gpickle"
DIR_GPKL = GRAPH_DIR / "hetero_graph_directed.gpickle"

OUTPUT_DIR = GRAPH_DIR / "gephi_multi_ego"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Undirected multi-ego parameters (unchanged)
N_VISITS = 1           # number of visit centers to sample
UNDIRECTED_RADIUS = 2  # ego radius for undirected graph
MAX_NODES = 600        # HARD CAP on nodes in each exported subgraph

# Directed person-ego parameters
PERSON_RADIUS = 3      # radius around a sampled person node
RANDOM_SEED = 42       # for reproducibility


# =========================
# Helpers
# =========================

def load_pickle_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    with path.open("rb") as f:
        G = pickle.load(f)
    return G


def get_node_type(data: dict) -> str:
    """Handle either 'node_type' or 'type' as the type key."""
    if "node_type" in data:
        return str(data["node_type"])
    if "type" in data:
        return str(data["type"])
    return ""


def get_visit_nodes(G) -> list:
    """Return list of nodes whose type is 'visit' (case-insensitive)."""
    visits = [
        n for n, d in G.nodes(data=True)
        if get_node_type(d).lower() == "visit"
    ]
    return visits


def get_person_nodes(G) -> list:
    """Return list of nodes whose type is 'person' (case-insensitive)."""
    persons = [
        n for n, d in G.nodes(data=True)
        if get_node_type(d).lower() == "person"
    ]
    return persons


def sanitize_graph_for_graphml(G):
    """
    Convert node IDs and all attributes to strings so GraphML export
    will not choke on mixed types.
    """
    H = nx.DiGraph() if G.is_directed() else nx.Graph()

    for n, data in G.nodes(data=True):
        clean_data = {k: str(v) for k, v in data.items()}
        H.add_node(str(n), **clean_data)

    for u, v, data in G.edges(data=True):
        clean_data = {k: str(v) for k, v in data.items()}
        H.add_edge(str(u), str(v), **clean_data)

    return H


# ---------- UNDIRECTED (unchanged logic) ----------

def build_multi_ego_subgraph_undirected(
    G, n_visits: int, radius: int, max_nodes: int
) -> nx.Graph:
    """
    Sample n_visits visit nodes, take ego-graphs of given radius
    around each, union all nodes, then cap to at most max_nodes.
    """
    rng = random.Random(RANDOM_SEED)

    visits = get_visit_nodes(G)
    if not visits:
        raise ValueError("No visit nodes found in graph (node_type/type != 'visit').")

    # Clamp n_visits
    k = min(n_visits, len(visits))
    centers = rng.sample(visits, k=k)

    nodes_union = set()
    for c in centers:
        ego = nx.ego_graph(G, c, radius=radius)
        nodes_union.update(ego.nodes())

    # Hard cap the number of nodes
    if len(nodes_union) > max_nodes:
        nodes_union = set(rng.sample(list(nodes_union), k=max_nodes))

    sub = G.subgraph(nodes_union).copy()
    print(
        f"[INFO] Undirected subgraph (radius={radius}, centers={k}): "
        f"{sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges"
    )
    return sub


# ---------- DIRECTED (new person-based logic) ----------

def build_person_subgraph_directed(
    G_dir, radius: int, max_nodes: int
) -> nx.DiGraph:
    """
    Sample ONE random person node and build a directed subgraph
    around that person using an ego-graph of given radius.

    We use undirected=True in ego_graph to allow paths that go
    along either in- or out-edges when computing distances, but
    the resulting subgraph keeps the original edge directions.
    """
    rng = random.Random(RANDOM_SEED)

    persons = get_person_nodes(G_dir)
    if not persons:
        raise ValueError("No person nodes found in directed graph.")

    center = rng.choice(persons)

    # Use ego_graph on the directed graph, but with undirected=True so
    # distance ignores direction while the subgraph remains directed.
    ego = nx.ego_graph(G_dir, center, radius=radius, undirected=True)

    nodes = list(ego.nodes())
    if len(nodes) > max_nodes:
        nodes = rng.sample(nodes, k=max_nodes)

    sub = G_dir.subgraph(nodes).copy()

    # === NEW: Remove all isolated nodes (degree-0) ===
    isolated = [n for n in sub.nodes() if sub.in_degree(n) == 0 and sub.out_degree(n) == 0]
    if isolated:
        sub.remove_nodes_from(isolated)
        print(f"[INFO] Removed {len(isolated)} isolated nodes from directed subgraph.")

    print(
        f"[INFO] Directed person subgraph (center={center}, radius={radius}): "
        f"{sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges"
    )
    return sub



def export_graphml(G: nx.Graph, out_path: Path):
    H = sanitize_graph_for_graphml(G)
    nx.write_graphml(H, out_path)
    print(f"[SAVE] GraphML -> {out_path}")


# =========================
# MAIN
# =========================

def main() -> None:
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Graph dir:    {GRAPH_DIR}")

    # Undirected (visit-based multi-ego, unchanged)
    print(f"[INFO] Loading undirected graph: {UND_GPKL}")
    G_un = load_pickle_graph(UND_GPKL)
    sub_un = build_multi_ego_subgraph_undirected(
        G_un,
        n_visits=N_VISITS,
        radius=UNDIRECTED_RADIUS,
        max_nodes=MAX_NODES,
    )
    out_un = OUTPUT_DIR / f"multi_ego_undirected_samp{SAMPLE_ID}.graphml"
    export_graphml(sub_un, out_un)

    # Directed (person-based ego)
    print(f"[INFO] Loading directed graph:   {DIR_GPKL}")
    G_dir = load_pickle_graph(DIR_GPKL)
    sub_dir = build_person_subgraph_directed(
        G_dir,
        radius=PERSON_RADIUS,
        max_nodes=MAX_NODES,
    )
    out_dir = OUTPUT_DIR / f"person_ego_directed_samp{SAMPLE_ID}.graphml"
    export_graphml(sub_dir, out_dir)

    print("\n[DONE] Subgraphs exported. Open the two .graphml files in Gephi.")


if __name__ == "__main__":
    main()
