"""
images.py

Generate figures for the EHRWalk paper:

1) Graph visualization: undirected vs directed heterogeneous EHR graph
   - multi-ego subgraph around several visit nodes
   - node color by type, size by degree, shared legend
2) P-test AUC heatmaps vs hyperparameters
   - restart_prob × graph_label
   - restart_prob × max_steps
   - restart_prob × use_edge_weight
   - restart_prob × inverse_degree
   - restart_prob × stop_rule
   - restart_prob × encounter_policy
3) Wasserstein distance vs restart probability (best config per restart)
4) Utility comparison line charts (TRTR/TSTR/TR+S->R) for the best config
   at each restart probability
5) LaTeX comparison table for GReaT / GReaTER / TabDDPM / TabSyn / EHRWalk
6) Real-data degree distributions (for context):
   - drug_concept_id
   - condition_concept_id
   - provider_id (events across drug + condition)
   - visits per person
   - events per visit

Run this from the project root (one level above `evaluation/`) with:

    python evaluation/images.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx


# =============================================================================
# CONFIG
# =============================================================================

# Sample ID used in ablation + graph construction
SAMPLE_ID = 100  # samp{SAMPLE_ID}

# This file lives in: PROJECT_ROOT / "evaluation" / images.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths
GRAPHS_ROOT = PROJECT_ROOT / "data" / "processed" / "graphs"
SAMPLE_DIR = GRAPHS_ROOT / f"samp{SAMPLE_ID}"
ABLATION_CSV = SAMPLE_DIR / "ablation_results_merge.csv"

# Real-data CSVs (for distribution plots)
REAL_DATA_DIR = PROJECT_ROOT / "data" / "clean_csvs"
DRUG_CSV = REAL_DATA_DIR / "sampled_drug_exposure.csv"
COND_CSV = REAL_DATA_DIR / "sampled_condition_occurrence.csv"

# Graph filenames (matching run_rw_synth_ablation.py)
UNDIRECTED_GRAPH_FILE = "hetero_graph_undirected.gpickle"
DIRECTED_GRAPH_FILE = "hetero_graph_directed.gpickle"

# Output directory for figures + table
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducible graph subsampling/layout
RANDOM_SEED = 42

# Multi-ego subgraph params for graph visualization
N_VISITS = 5       # number of visit centers to sample
RADIUS = 2         # ego radius around several visits
MAX_NODES = 800    # hard cap on nodes in the plotted subgraph


# =============================================================================
# Utility helpers
# =============================================================================

def load_ablation_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ablation results CSV not found at: {path}")
    df = pd.read_csv(path)
    required_cols = [
        "restart_prob",
        "graph_label",
        "use_edge_weight",
        "inverse_degree",
        "max_steps",
        "p_test_auc",
        "wasserstein_drugs",
        "wasserstein_conditions",
        "macro_f1_trtr",
        "macro_f1_tstr",
        "macro_f1_trsr",
        "micro_f1_trtr",
        "micro_f1_tstr",
        "micro_f1_trsr",
        "utility_ratio_tstr_trtr",
        "utility_ratio_trsr_trtr",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Ablation CSV is missing columns: {missing}")
    return df


def _plot_hist_from_counts(
    counts: np.ndarray,
    title: str,
    filename: str,
    bins: int = 50,
    log_y: bool = True,
) -> None:
    """
    Simple histogram helper for degree/count distributions.
    """
    if counts is None or len(counts) == 0:
        print(f"[WARN] No data to plot for {title}; skipping.")
        return

    plt.figure(figsize=(4.5, 3.5))
    plt.hist(counts, bins=bins)
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    out_path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Real-data distribution -> {out_path}")


def _print_outlier_diagnostics(
    name: str,
    counts_series: pd.Series,
    high_percentiles: Tuple[float, ...] = (0.99, 0.999),
    top_n: int = 10,
) -> None:
    """
    Print diagnostics for a degree/count distribution.

    counts_series: index = ID (drug_concept_id, provider_id, etc.),
                   values = counts (# events).
    """
    if counts_series is None or counts_series.empty:
        print(f"[OUTLIERS] {name}: no data.\n")
        return

    counts = counts_series.astype(float)

    print(f"\n[OUTLIERS] {name}")
    print(f"  # unique IDs: {len(counts):,}")
    print(
        f"  min / median / mean / max: "
        f"{counts.min():.0f} / {counts.median():.2f} / {counts.mean():.2f} / {counts.max():.0f}"
    )

    # High-percentile thresholds
    for p in high_percentiles:
        thr = counts.quantile(p)
        n_ge = int((counts >= thr).sum())
        print(
            f"  >= {p*100:.2f}th pct (threshold ≈ {thr:.1f}): "
            f"{n_ge:,} IDs"
        )

    # Top-N IDs by count
    top = counts.sort_values(ascending=False).head(top_n)
    print(f"  Top {len(top)} IDs by count:")
    for idx, val in top.items():
        print(f"    ID={idx}  count={int(val)}")
    print("")  # blank line for spacing


# =============================================================================
# 1) Graph visualization helpers (multi-ego, colors, legend)
# =============================================================================

def _load_pickle_graph(path: Path) -> nx.Graph:
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    with path.open("rb") as f:
        G = pickle.load(f)
    return G


def _get_node_type(d: Dict) -> str:
    """
    Handle both 'node_type' and 'type' as possible keys.
    """
    if "node_type" in d:
        return str(d["node_type"]).lower()
    if "type" in d:
        return str(d["type"]).lower()
    return "unknown"


def _pick_visit_centers(G: nx.Graph, n_visits: int) -> List:
    """
    Sample up to n_visits nodes of type 'visit' as centers.
    """
    visits = [n for n, data in G.nodes(data=True) if _get_node_type(data) == "visit"]
    if not visits:
        raise ValueError("No visit nodes found in graph (node_type/type != 'visit').")

    rng = np.random.default_rng(RANDOM_SEED)
    k = min(n_visits, len(visits))
    centers = rng.choice(visits, size=k, replace=False).tolist()
    return centers


def _build_multi_ego_subgraph(
    G: nx.Graph,
    n_visits: int = N_VISITS,
    radius: int = RADIUS,
    max_nodes: int = MAX_NODES,
) -> nx.Graph:
    """
    Take ego-graphs around several visit nodes and return their union
    as a single subgraph, capped at max_nodes for readability.
    """
    centers = _pick_visit_centers(G, n_visits)
    nodes_union = set()

    for c in centers:
        ego = nx.ego_graph(G, c, radius=radius)
        nodes_union.update(ego.nodes())

    if len(nodes_union) > max_nodes:
        rng = np.random.default_rng(RANDOM_SEED)
        nodes_union = set(rng.choice(list(nodes_union), size=max_nodes, replace=False))

    sub = G.subgraph(nodes_union).copy()
    return sub


def _node_colors_sizes_types(G: nx.Graph):
    """
    Compute node colors by type and node sizes by degree.
    Also returns list of types and the color map for legend construction.
    """
    color_map = {
        "visit": "#1f77b4",      # blue
        "person": "#ff7f0e",     # orange
        "provider": "#2ca02c",   # green
        "drug": "#d62728",       # red
        "condition": "#9467bd",  # purple
        "unknown": "#7f7f7f",    # gray
    }

    node_types = [_get_node_type(d) for _, d in G.nodes(data=True)]
    colors = [color_map.get(t, color_map["unknown"]) for t in node_types]

    deg = dict(G.degree())
    deg_vals = np.array([deg[n] for n in G.nodes()])
    if len(deg_vals) > 0:
        d_min, d_max = deg_vals.min(), deg_vals.max()
        if d_max == d_min:
            sizes = np.full_like(deg_vals, fill_value=30, dtype=float)
        else:
            sizes = 20 + 80 * (deg_vals - d_min) / max(d_max - d_min, 1)  # scale 20–100
    else:
        sizes = np.array([])

    return colors, sizes, node_types, color_map


# =============================================================================
# 1) Graph visualization: undirected vs directed (pretty)
# =============================================================================

def plot_hetero_graphs(sample_dir: Path) -> None:
    und_path = sample_dir / UNDIRECTED_GRAPH_FILE
    dir_path = sample_dir / DIRECTED_GRAPH_FILE

    print(f"[INFO] Loading graphs from {sample_dir}")
    G_un_full = _load_pickle_graph(und_path)
    G_dir_full = _load_pickle_graph(dir_path)

    # Multi-ego subgraphs
    G_un = _build_multi_ego_subgraph(G_un_full)
    G_dir = _build_multi_ego_subgraph(G_dir_full)

    # Layouts
    pos_un = nx.spring_layout(G_un, seed=RANDOM_SEED)
    pos_dir = nx.spring_layout(G_dir, seed=RANDOM_SEED)

    # Colors, sizes, legend info
    colors_un, sizes_un, types_un, color_map = _node_colors_sizes_types(G_un)
    colors_dir, sizes_dir, types_dir, _ = _node_colors_sizes_types(G_dir)

    present_types = sorted(set(types_un + types_dir))
    legend_handles = [
        mpatches.Patch(color=color_map.get(t, "#7f7f7f"), label=t.capitalize())
        for t in present_types
    ]

    plt.figure(figsize=(10, 5))

    # Undirected
    ax1 = plt.subplot(1, 2, 1)
    nx.draw_networkx_edges(
        G_un,
        pos_un,
        ax=ax1,
        edge_color="lightgray",
        width=0.4,
        alpha=0.5,
    )
    nx.draw_networkx_nodes(
        G_un,
        pos_un,
        ax=ax1,
        node_color=colors_un,
        node_size=sizes_un,
        linewidths=0,
    )
    ax1.set_title("Undirected Heterogeneous EHR Graph")
    ax1.axis("off")

    # Directed
    ax2 = plt.subplot(1, 2, 2)
    nx.draw_networkx_edges(
        G_dir,
        pos_dir,
        ax=ax2,
        edge_color="lightgray",
        width=0.4,
        alpha=0.5,
        arrows=True,
        arrowsize=6,
    )
    nx.draw_networkx_nodes(
        G_dir,
        pos_dir,
        ax=ax2,
        node_color=colors_dir,
        node_size=sizes_dir,
        linewidths=0,
    )
    ax2.set_title("Directed Heterogeneous EHR Graph")
    ax2.axis("off")

    # Shared legend
    plt.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(legend_handles),
        fontsize=8,
        frameon=False,
    )

    out_path = FIG_DIR / f"graph_hetero_un_vs_dir_samp{SAMPLE_ID}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Graph visualization -> {out_path}")


# =============================================================================
# 2) P-test AUC heatmaps vs hyperparameters
# =============================================================================

def _heatmap_from_group(
    df: pd.DataFrame,
    row_key: str,
    col_key: str,
    value_col: str = "p_test_auc",
    title: str = "",
    filename: str = "heatmap.png",
) -> None:
    grouped = (
        df.groupby([row_key, col_key], as_index=False)[value_col]
          .mean()
    )
    if grouped.empty:
        print(f"[WARN] No data for heatmap ({row_key}, {col_key}); skipping.")
        return

    pivot = grouped.pivot(index=row_key, columns=col_key, values=value_col)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis_r",          # inverted: low = bright, high = dark
        cbar_kws={"label": "p-test AUC"},
    )
    plt.title(title)
    plt.xlabel(col_key)
    plt.ylabel(row_key)
    out_path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] P-test heatmap -> {out_path}")


def plot_p_test_heatmaps(df: pd.DataFrame) -> None:
    if "p_test_auc" not in df.columns:
        print("[WARN] 'p_test_auc' not found in ablation results; skipping heatmaps.")
        return

    # 1) restart_prob × graph_label
    _heatmap_from_group(
        df,
        row_key="graph_label",
        col_key="restart_prob",
        value_col="p_test_auc",
        title="p-test AUC vs Graph Type and Restart Probability",
        filename=f"heatmap_p_test_graph_restart_samp{SAMPLE_ID}.png",
    )

    # 2) restart_prob × max_steps
    _heatmap_from_group(
        df,
        row_key="max_steps",
        col_key="restart_prob",
        value_col="p_test_auc",
        title="p-test AUC vs Max Steps and Restart Probability",
        filename=f"heatmap_p_test_maxsteps_restart_samp{SAMPLE_ID}.png",
    )

    # 3) restart_prob × use_edge_weight
    if "use_edge_weight" in df.columns:
        df_ew = df.copy()
        df_ew["use_edge_weight"] = df_ew["use_edge_weight"].astype(str)
        _heatmap_from_group(
            df_ew,
            row_key="use_edge_weight",
            col_key="restart_prob",
            value_col="p_test_auc",
            title="p-test AUC vs Edge Weighting and Restart Probability",
            filename=f"heatmap_p_test_edgeweight_restart_samp{SAMPLE_ID}.png",
        )

    # 4) restart_prob × inverse_degree
    if "inverse_degree" in df.columns:
        df_id = df.copy()
        df_id["inverse_degree"] = df_id["inverse_degree"].astype(str)
        _heatmap_from_group(
            df_id,
            row_key="inverse_degree",
            col_key="restart_prob",
            value_col="p_test_auc",
            title="p-test AUC vs Inverse Degree and Restart Probability",
            filename=f"heatmap_p_test_invdeg_restart_samp{SAMPLE_ID}.png",
        )

    # 5) restart_prob × stop_rule
    if "stop_rule" in df.columns:
        df_sr = df.copy()
        df_sr["stop_rule"] = df_sr["stop_rule"].astype(str)
        _heatmap_from_group(
            df_sr,
            row_key="stop_rule",
            col_key="restart_prob",
            value_col="p_test_auc",
            title="p-test AUC vs Stop Rule and Restart Probability",
            filename=f"heatmap_p_test_stoprule_restart_samp{SAMPLE_ID}.png",
        )

    # 6) restart_prob × encounter_policy
    if "encounter_policy" in df.columns:
        df_ep = df.copy()
        df_ep["encounter_policy"] = df_ep["encounter_policy"].astype(str)
        _heatmap_from_group(
            df_ep,
            row_key="encounter_policy",
            col_key="restart_prob",
            value_col="p_test_auc",
            title="p-test AUC vs Encounter Policy and Restart Probability",
            filename=f"heatmap_p_test_encounter_restart_samp{SAMPLE_ID}.png",
        )


# =============================================================================
# 3) Wasserstein vs restart probability (best configs)
# =============================================================================

def plot_wasserstein_vs_restart(df: pd.DataFrame) -> None:
    """
    For each restart_prob, find the config with the lowest Wasserstein
    (#drugs and #conditions separately), and plot:
      - best W(#drugs) vs restart_prob
      - best W(#conditions) vs restart_prob
      - average of the two vs restart_prob
    """
    required = ["restart_prob", "wasserstein_drugs", "wasserstein_conditions"]
    if any(c not in df.columns for c in required):
        print("[WARN] Wasserstein columns missing; skipping Wasserstein plot.")
        return

    best_records: Dict[float, Tuple[float, float]] = {}

    for rp, group in df.groupby("restart_prob"):
        if group.empty:
            continue
        idx = group["wasserstein_drugs"].idxmin()
        row = df.loc[idx]
        w_d = row["wasserstein_drugs"]
        w_c = row["wasserstein_conditions"]
        best_records[float(rp)] = (float(w_d), float(w_c))

    if not best_records:
        print("[WARN] No Wasserstein records found; skipping plot.")
        return

    restart_vals = sorted(best_records.keys())
    w_drugs_vals = [best_records[rp][0] for rp in restart_vals]
    w_conds_vals = [best_records[rp][1] for rp in restart_vals]
    w_mean_vals = [(d + c) / 2.0 for d, c in zip(w_drugs_vals, w_conds_vals)]

    plt.figure(figsize=(9, 3))

    # (a) W(#drugs)
    plt.subplot(1, 3, 1)
    plt.plot(restart_vals, w_drugs_vals, marker="o")
    plt.title("Best Wasserstein (#drugs)")
    plt.xlabel("Restart probability")
    plt.ylabel("W1 (#drugs)")

    # (b) W(#conditions)
    plt.subplot(1, 3, 2)
    plt.plot(restart_vals, w_conds_vals, marker="o")
    plt.title("Best Wasserstein (#conditions)")
    plt.xlabel("Restart probability")
    plt.ylabel("W1 (#conditions)")

    # (c) mean
    plt.subplot(1, 3, 3)
    plt.plot(restart_vals, w_mean_vals, marker="o")
    plt.title("Best Wasserstein (mean of both)")
    plt.xlabel("Restart probability")
    plt.ylabel("Mean W1")

    out_path = FIG_DIR / f"wasserstein_vs_restart_best_samp{SAMPLE_ID}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Wasserstein vs restart -> {out_path}")


# =============================================================================
# 4) Utility comparison line chart (best config per restart_prob)
# =============================================================================

def _get_best_config_per_restart(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each restart_prob, pick the config with maximum utility_ratio_tstr_trtr.
    If that column is missing or NaN, fall back to macro_f1_tstr.
    Returns a small DataFrame with one row per restart_prob.
    """
    if "utility_ratio_tstr_trtr" in df.columns:
        score_col = "utility_ratio_tstr_trtr"
    else:
        score_col = "macro_f1_tstr"

    rows = []
    for rp, group in df.groupby("restart_prob"):
        g = group.dropna(subset=[score_col])
        if g.empty:
            continue
        idx = g[score_col].idxmax()
        rows.append(df.loc[idx])

    if not rows:
        return pd.DataFrame()

    best_df = pd.DataFrame(rows).reset_index(drop=True)
    return best_df.sort_values("restart_prob")


def plot_utility_vs_restart(df: pd.DataFrame) -> None:
    """
    For each restart_prob, select best config (by utility_ratio_tstr_trtr or macro_f1_tstr),
    and plot TRTR/TSTR/TR+S->R macro-F1, micro-F1, and utility ratios.
    """
    required = [
        "restart_prob",
        "macro_f1_trtr",
        "macro_f1_tstr",
        "macro_f1_trsr",
        "micro_f1_trtr",
        "micro_f1_tstr",
        "micro_f1_trsr",
    ]
    if any(c not in df.columns for c in required):
        print("[WARN] Utility columns missing; skipping utility line chart.")
        return

    best_df = _get_best_config_per_restart(df)
    if best_df.empty:
        print("[WARN] Could not determine best configs per restart; skipping utility plot.")
        return

    restart_vals = best_df["restart_prob"].astype(float).tolist()

    macro_trtr = best_df["macro_f1_trtr"].astype(float).tolist()
    macro_tstr = best_df["macro_f1_tstr"].astype(float).tolist()
    macro_trsr = best_df["macro_f1_trsr"].astype(float).tolist()

    micro_trtr = best_df["micro_f1_trtr"].astype(float).tolist()
    micro_tstr = best_df["micro_f1_tstr"].astype(float).tolist()
    micro_trsr = best_df["micro_f1_trsr"].astype(float).tolist()

    # Utility ratios
    def safe_ratio(num, den):
        return float(num) / float(den) if den and den > 0 else np.nan

    util_tstr = [safe_ratio(t, b) for t, b in zip(macro_tstr, macro_trtr)]
    util_trsr = [safe_ratio(s, b) for s, b in zip(macro_trsr, macro_trtr)]

    plt.figure(figsize=(9, 3))

    # (a) Macro F1
    plt.subplot(1, 3, 1)
    plt.plot(restart_vals, macro_trtr, marker="o", label="TRTR")
    plt.plot(restart_vals, macro_tstr, marker="o", label="TSTR")
    plt.plot(restart_vals, macro_trsr, marker="o", label="TR+S→R")
    plt.title("Macro F1 vs Restart (Best Configs)")
    plt.xlabel("Restart probability")
    plt.ylabel("Macro F1")
    plt.legend()

    # (b) Micro F1
    plt.subplot(1, 3, 2)
    plt.plot(restart_vals, micro_trtr, marker="o", label="TRTR")
    plt.plot(restart_vals, micro_tstr, marker="o", label="TSTR")
    plt.plot(restart_vals, micro_trsr, marker="o", label="TR+S→R")
    plt.title("Micro F1 vs Restart (Best Configs)")
    plt.xlabel("Restart probability")
    plt.ylabel("Micro F1")
    plt.legend()

    # (c) Utility ratios
    plt.subplot(1, 3, 3)
    plt.plot(restart_vals, util_tstr, marker="o", label="TSTR / TRTR")
    plt.plot(restart_vals, util_trsr, marker="o", label="TR+S→R / TRTR")
    plt.title("Utility Ratios vs Restart")
    plt.xlabel("Restart probability")
    plt.ylabel("Utility ratio")
    plt.legend()

    out_path = FIG_DIR / f"utility_vs_restart_best_samp{SAMPLE_ID}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Utility vs restart -> {out_path}")


# =============================================================================
# 5) GReaT / GReaTER / TabDDPM / TabSyn / EHRWalk comparison table
# =============================================================================

def write_comparison_table() -> None:
    """
    Write a LaTeX table comparing GReaT, GReaTER, TabDDPM, TabSyn, and EHRWalk.
    This is conceptual (from literature), not computed from CSVs.
    """
    tex = r"""
\begin{table}[t]
\centering
\caption{Qualitative comparison of synthetic data generators.}
\label{tab:method_comparison}
\begin{tabular}{lccccc}
\toprule
Method & Multi-table & Multi-valued & Training & GPU & Graph-\\
       & EHR support & fields       & required & req. & aware \\
\midrule
GReaT   & \xmark & \(\pm\) & \cmark & \cmark & \xmark \\
GReaTER & \(\pm\) & \(\pm\) & \cmark & \cmark & \xmark \\
TabDDPM & \xmark & \(\pm\) & \cmark & \cmark & \xmark \\
TabSyn  & \xmark & \(\pm\) & \cmark & \cmark & \xmark \\
EHRWalk & \cmark & \cmark & \xmark & \xmark & \cmark \\
\bottomrule
\end{tabular}
\end{table}
""".strip()

    out_path = TABLE_DIR / "method_comparison_table.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"[SAVE] Method comparison table (LaTeX) -> {out_path}")
    print("Note: this table expects \\usepackage{booktabs} and defines \\cmark/\\xmark elsewhere.")


# =============================================================================
# 6) Real-data degree distributions + outlier diagnostics
# =============================================================================

def plot_real_distributions() -> None:
    """
    Plot REAL data distributions for:
      - drug_concept_id degree
      - condition_concept_id degree
      - provider_id degree (events across drug + condition)
      - visits per person
      - events per visit

    Also print outlier diagnostics for each distribution.
    """
    print(f"[INFO] Real data dir: {REAL_DATA_DIR}")
    if not DRUG_CSV.exists():
        print(f"[WARN] Drug CSV not found at {DRUG_CSV}; skipping real-data distributions.")
        return
    if not COND_CSV.exists():
        print(f"[WARN] Condition CSV not found at {COND_CSV}; skipping real-data distributions.")
        return

    print(f"[LOAD] {DRUG_CSV}")
    df_drug = pd.read_csv(
        DRUG_CSV,
        usecols=["person_id", "drug_concept_id", "provider_id", "visit_occurrence_id"],
        low_memory=False,
    )
    print(f"[INFO] sampled_drug_exposure: rows={len(df_drug):,}")

    print(f"[LOAD] {COND_CSV}")
    df_cond = pd.read_csv(
        COND_CSV,
        usecols=["person_id", "condition_concept_id", "provider_id", "visit_occurrence_id"],
        low_memory=False,
    )
    print(f"[INFO] sampled_condition_occurrence: rows={len(df_cond):,}")

    # 1) Drug degree distribution: total occurrences per drug_concept_id
    drug_counts_series = (
        df_drug["drug_concept_id"]
        .dropna()
        .value_counts()
        .astype(int)
    )
    _print_outlier_diagnostics(
        "drug_concept_id degree (real data)",
        drug_counts_series,
    )
    _plot_hist_from_counts(
        drug_counts_series.values,
        "Degree distribution: drug_concept_id (real data)",
        f"real_deg_drug_concept_id_samp{SAMPLE_ID}.png",
        bins=60,
        log_y=True,
    )

    # 2) Condition degree distribution: total occurrences per condition_concept_id
    cond_counts_series = (
        df_cond["condition_concept_id"]
        .dropna()
        .value_counts()
        .astype(int)
    )
    _print_outlier_diagnostics(
        "condition_concept_id degree (real data)",
        cond_counts_series,
    )
    _plot_hist_from_counts(
        cond_counts_series.values,
        "Degree distribution: condition_concept_id (real data)",
        f"real_deg_condition_concept_id_samp{SAMPLE_ID}.png",
        bins=60,
        log_y=True,
    )

    # 3) Provider degree distribution: events across drug + condition
    prov_counts_drug = (
        df_drug["provider_id"]
        .dropna()
        .value_counts()
    )
    prov_counts_cond = (
        df_cond["provider_id"]
        .dropna()
        .value_counts()
    )
    prov_counts_total = prov_counts_drug.add(prov_counts_cond, fill_value=0).astype(int)

    _print_outlier_diagnostics(
        "provider_id degree (events: drug + condition, real data)",
        prov_counts_total,
    )
    _plot_hist_from_counts(
        prov_counts_total.values,
        "Degree distribution: provider_id (events: drug + condition, real data)",
        f"real_deg_provider_id_samp{SAMPLE_ID}.png",
        bins=60,
        log_y=True,
    )

    # 4) Visits per person: unique visits per person across both tables
    pv_drug = (
        df_drug[["person_id", "visit_occurrence_id"]]
        .dropna()
        .drop_duplicates()
    )
    pv_cond = (
        df_cond[["person_id", "visit_occurrence_id"]]
        .dropna()
        .drop_duplicates()
    )
    pv_all = (
        pd.concat([pv_drug, pv_cond], ignore_index=True)
        .drop_duplicates()
    )
    visits_per_person_series = (
        pv_all
        .groupby("person_id")["visit_occurrence_id"]
        .nunique()
        .astype(int)
    )

    _print_outlier_diagnostics(
        "visits per person (real data, drug + condition)",
        visits_per_person_series,
    )
    _plot_hist_from_counts(
        visits_per_person_series.values,
        "Visits per person (real data, drug + condition)",
        f"real_visits_per_person_samp{SAMPLE_ID}.png",
        bins=40,
        log_y=True,
    )

    # 5) Events per visit: total events across drug + condition per visit
    events_vis_drug = (
        df_drug["visit_occurrence_id"]
        .dropna()
        .value_counts()
    )
    events_vis_cond = (
        df_cond["visit_occurrence_id"]
        .dropna()
        .value_counts()
    )
    events_vis_total = events_vis_drug.add(events_vis_cond, fill_value=0).astype(int)

    _print_outlier_diagnostics(
        "events per visit (real data, drug + condition)",
        events_vis_total,
    )
    _plot_hist_from_counts(
        events_vis_total.values,
        "Events per visit (real data, drug + condition)",
        f"real_events_per_visit_samp{SAMPLE_ID}.png",
        bins=40,
        log_y=True,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Sample dir:   {SAMPLE_DIR}")

    if not SAMPLE_DIR.exists():
        raise FileNotFoundError(
            f"SAMPLE_DIR does not exist: {SAMPLE_DIR}\n"
            "Make sure graph construction and ablation have been run."
        )

    # 0) Real-data distributions (context for graph + walk behavior)
    plot_real_distributions()

    # 1) Graph visualization
    plot_hetero_graphs(SAMPLE_DIR)

    # 2–4) Load ablation results
    df = load_ablation_results(ABLATION_CSV)

    # 2) P-test heatmaps
    plot_p_test_heatmaps(df)

    # 3) Wasserstein vs restart
    plot_wasserstein_vs_restart(df)

    # 4) Utility vs restart (best configs)
    plot_utility_vs_restart(df)

    # 5) Method comparison table
    write_comparison_table()

    print("[DONE] All figures/tables generated.")


if __name__ == "__main__":
    main()
