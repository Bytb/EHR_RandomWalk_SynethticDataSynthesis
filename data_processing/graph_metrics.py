#!/usr/bin/env python
"""
graph_node_stats.py

Load a .gpickle graph via pickle and report:
- Total node count
- Node count per node_type
- Percentage of each node_type out of the whole graph
- Average degree per node_type
- Edge participation percentage per node_type
  (fraction of incident half-edges belonging to nodes of that type)

Usage:
    - Set GRAPH_PATH at the top of this file.
    - Run: python graph_node_stats.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict

import networkx as nx  # only for graph utilities (NOT for reading gpickle)


# ======================================================================================
# CONFIG: set this to the .gpickle you want to analyze
# ======================================================================================

GRAPH_PATH = Path(
    r"C:\Users\Caleb\PycharmProjects\EHR_RandomWalk_SynethticDataSynthesis\data\processed\graphs\samp100\hetero_graph_undirected.gpickle"  # <-- edit this
)


# ======================================================================================
# Helper functions
# ======================================================================================

def load_graph(path: Path) -> nx.Graph:
    """Load a NetworkX graph object from a .gpickle file using pickle."""
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    with open(path, "rb") as f:
        G = pickle.load(f)

    if not isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph):
        raise TypeError(f"Loaded object is not a NetworkX graph: {type(G)}")

    return G


def infer_node_type_key(G: nx.Graph) -> str:
    """
    Infer which node attribute key holds the node type.

    Tries a small set of common keys. Raises if none found.
    """
    candidate_keys = ["node_type", "ntype", "type"]

    # Look at a small sample of nodes
    for _, data in list(G.nodes(data=True))[:50]:
        for key in candidate_keys:
            if key in data:
                return key

    raise KeyError(
        "Could not infer node type key. "
        "Tried 'node_type', 'ntype', and 'type'. "
        "Please standardize your graph node attribute or extend this script."
    )


def compute_node_type_stats(G: nx.Graph, node_type_key: str) -> Dict[str, Any]:
    """
    Compute:
    - total node count
    - total edge count
    - counts per node_type
    - percentages per node_type (of nodes)
    - average degree per node_type
    - edge participation percentage per node_type
    """
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    # Degree (works for both Graph and DiGraph; for DiGraph it's in+out)
    degrees = dict(G.degree())

    counts: Counter = Counter()
    degree_sums = defaultdict(float)

    for node, data in G.nodes(data=True):
        if node_type_key not in data:
            raise KeyError(
                f"Node {node!r} missing '{node_type_key}' attribute. "
                "Ensure all nodes have a type attribute."
            )
        t = data[node_type_key]
        counts[t] += 1
        degree_sums[t] += degrees.get(node, 0.0)

    # Node-type percentages (by node count)
    node_percentages = {
        t: (count / total_nodes) * 100.0 for t, count in counts.items()
    }

    # Average degree per node type
    avg_degree = {
        t: (degree_sums[t] / counts[t]) if counts[t] > 0 else 0.0
        for t in counts.keys()
    }

    # Edge participation per node type:
    # fraction of incident "half-edges" (sum of degrees = 2E)
    if total_edges > 0:
        edge_participation = {
            t: (degree_sums[t] / (2.0 * total_edges)) * 100.0
            for t in counts.keys()
        }
    else:
        edge_participation = {t: 0.0 for t in counts.keys()}

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "counts": counts,
        "node_percentages": node_percentages,
        "avg_degree": avg_degree,
        "edge_participation": edge_participation,
    }


def print_node_type_stats(path: Path, stats: Dict[str, Any]) -> None:
    """Pretty-print node type statistics to console."""
    total_nodes = stats["total_nodes"]
    total_edges = stats["total_edges"]
    counts = stats["counts"]
    node_percentages = stats["node_percentages"]
    avg_degree = stats["avg_degree"]
    edge_participation = stats["edge_participation"]

    print("\n[GRAPH NODE STATISTICS]")
    print(f"  Path:         {path}")
    print(f"  Total nodes:  {total_nodes:,}")
    print(f"  Total edges:  {total_edges:,}\n")

    print("  Node type breakdown:")
    print(
        "    {:20s} {:>12s} {:>12s} {:>14s} {:>18s}".format(
            "node_type", "count", "percent", "avg_degree", "edge_participation"
        )
    )
    print("    " + "-" * 76)

    # Sort by count descending
    for node_type, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        pct_nodes = node_percentages[node_type]
        avg_deg = avg_degree[node_type]
        edge_pct = edge_participation[node_type]
        print(
            f"    {str(node_type):20s} "
            f"{count:12,d} "
            f"{pct_nodes:11.2f}% "
            f"{avg_deg:13.2f} "
            f"{edge_pct:17.2f}%"
        )

    print()


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    print(f"[LOAD] Graph: {GRAPH_PATH}")
    G = load_graph(GRAPH_PATH)

    try:
        node_type_key = infer_node_type_key(G)
        print(f"[INFO] Using node type key: '{node_type_key}'")
    except KeyError as e:
        print(f"[ERROR] {e}")
        return

    stats = compute_node_type_stats(G, node_type_key)
    print_node_type_stats(GRAPH_PATH, stats)


if __name__ == "__main__":
    main()
