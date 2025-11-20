from __future__ import annotations

from pathlib import Path
import csv
import math
import random
from typing import Dict, List, Tuple, Optional

import networkx as nx
from multiprocessing import Pool, cpu_count
from tqdm import tqdm



# ============================================================
# Core random-walk for a single synthetic row
# ============================================================

def run_random_walk(
    G: nx.Graph,
    start_node: str,
    restart_prob: float = 0.3,
    use_edge_weight: bool = True,
    inverse_degree: bool = False,
    encounter_policy: str = "first",      # 'first' or 'second' (for person/provider)
    stop_rule: str = "percentage",        # 'complete' or 'percentage'
    stop_percentage: float = 0.8,         # only used if stop_rule == 'percentage'
    max_steps: int = 100,
    rng: Optional[random.Random] = None,
) -> Dict[str, str]:
    """
    Run a restart-based random walk on a heterogeneous graph, starting at a visit node,
    and return a synthetic row with:
        - visitID
        - personID
        - providerID
        - drug_concept_ids
        - condition_concept_ids

    Assumptions:
        - Nodes have attributes:
            - 'node_type' in {'visit', 'person', 'provider', 'drug', 'condition'}
            - 'raw_id'      (string)
        - start_node is a 'visit' node.
        - Multi-valued fields (drugs, conditions) aggregate all unique visited nodes of that type.
        - Single-valued fields (person, provider) obey encounter_policy and may remain 'NA'.
    """

    if rng is None:
        rng = random.Random()

    # --- Initialize feature containers ---

    # visitID is just the raw_id of the starting visit
    start_data = G.nodes[start_node]
    visit_id = str(start_data.get("raw_id", start_node))

    person_id = "NA"
    provider_id = "NA"

    drug_set = set()
    condition_set = set()

    # track encounter counts for single-valued types (for "second" policy)
    encounter_counts = {
        "person": 0,
        "provider": 0,
    }

    # which single-valued fields are considered "fillable"
    single_fields = ["person", "provider"]

    # --- Helper: record node into features ---

    def record_node(node: str):
        nonlocal person_id, provider_id

        ndata = G.nodes[node]
        ntype = ndata.get("node_type")
        raw_id = str(ndata.get("raw_id", node))

        if ntype == "person":
            encounter_counts["person"] += 1
            if person_id == "NA":
                if encounter_policy == "first" and encounter_counts["person"] >= 1:
                    person_id = raw_id
                elif encounter_policy == "second" and encounter_counts["person"] >= 2:
                    person_id = raw_id

        elif ntype == "provider":
            encounter_counts["provider"] += 1
            if provider_id == "NA":
                if encounter_policy == "first" and encounter_counts["provider"] >= 1:
                    provider_id = raw_id
                elif encounter_policy == "second" and encounter_counts["provider"] >= 2:
                    provider_id = raw_id

        elif ntype == "drug":
            # multi-valued: always aggregate, no "only-if-empty" constraint
            drug_set.add(raw_id)

        elif ntype == "condition":
            # multi-valued: always aggregate, no "only-if-empty" constraint
            condition_set.add(raw_id)

        # 'visit' type doesn't directly populate any field here.

    # --- Helper: check stopping condition ---

    def should_stop(step_idx: int) -> bool:
        # Hard cap on steps
        if step_idx >= max_steps:
            return True

        # Count how many single-valued fields are filled (person, provider)
        filled = 0
        if person_id != "NA":
            filled += 1
        if provider_id != "NA":
            filled += 1

        total = len(single_fields)

        if stop_rule == "complete":
            return filled == total

        elif stop_rule == "percentage":
            if total == 0:
                return True  # degenerate; nothing to fill
            frac = filled / total
            return frac >= stop_percentage

        else:
            # Unknown stop rule: default to max_steps only
            return False

    # --- Helper: sample next node given current & previous node ---

    def sample_next_node(current: str, previous: Optional[str]) -> str:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            # Dead end; force restart to start_node
            return start_node

        # forbid immediate backtracking if possible
        if previous is not None:
            candidates = [nb for nb in neighbors if nb != previous]
            if candidates:
                neighbors = candidates

        # compute unnormalized weights
        weights = []
        for nb in neighbors:
            # base edge weight
            if use_edge_weight and "weight" in G[current][nb]:
                ew = G[current][nb]["weight"]
                try:
                    ew = float(ew)
                except Exception:
                    ew = 1.0
            else:
                ew = 1.0

            # inverse degree factor (if enabled)
            if inverse_degree:
                deg = G.degree(nb)
                if deg <= 0:
                    deg_factor = 1.0
                else:
                    deg_factor = 1.0 / deg
            else:
                deg_factor = 1.0

            w = ew * deg_factor
            if w <= 0:
                w = 1e-12  # avoid zero/negative weights
            weights.append(w)

        # Normalize
        total_w = sum(weights)
        if total_w <= 0:
            # fallback to uniform
            return rng.choice(neighbors)

        # sample according to weights
        r = rng.random() * total_w
        acc = 0.0
        for nb, w in zip(neighbors, weights):
            acc += w
            if r <= acc:
                return nb

        # fallback (should not happen)
        return neighbors[-1]

    # --- Run the walk ---

    current = start_node
    previous = None

    # Optionally record the starting node as well (usually just visit)
    record_node(current)

    step = 0
    while True:
        if should_stop(step):
            break

        # Decide restart vs normal step
        if rng.random() < restart_prob:
            # restart to start_node
            previous = current
            current = start_node
            record_node(current)
            step += 1
            continue

        # normal transition
        next_node = sample_next_node(current, previous)
        previous, current = current, next_node

        record_node(current)
        step += 1

    # --- Finalize synthetic row ---

    # convert multi-valued sets to semicolon-separated strings
    if drug_set:
        drug_str = ";".join(sorted(drug_set))
    else:
        drug_str = "NA"

    if condition_set:
        cond_str = ";".join(sorted(condition_set))
    else:
        cond_str = "NA"

    row = {
        "visitID": visit_id,
        "personID": person_id,
        "providerID": provider_id,
        "drug_concept_ids": drug_str,
        "condition_concept_ids": cond_str,
    }
    return row


# ============================================================
# Parallel synthetic dataset generator
# ============================================================

def _walk_worker(args: Tuple) -> Dict[str, str]:
    """
    Helper for multiprocessing: unpack args and call run_random_walk.
    """
    (
        G,
        start_node,
        restart_prob,
        use_edge_weight,
        inverse_degree,
        encounter_policy,
        stop_rule,
        stop_percentage,
        max_steps,
        seed,
    ) = args

    rng = random.Random(seed)
    return run_random_walk(
        G=G,
        start_node=start_node,
        restart_prob=restart_prob,
        use_edge_weight=use_edge_weight,
        inverse_degree=inverse_degree,
        encounter_policy=encounter_policy,
        stop_rule=stop_rule,
        stop_percentage=stop_percentage,
        max_steps=max_steps,
        rng=rng,
    )


def _build_synth_filename(
    graph_label: str,
    restart_prob: float,
    use_edge_weight: bool,
    inverse_degree: bool,
    encounter_policy: str,
    stop_rule: str,
    stop_percentage: float,
) -> str:
    """
    Build filename encoding the ablation parameters.

    graph_label: 'dir' or 'ud'
    restart_prob: e.g. 0.3 -> alpha30
    use_edge_weight: edg / nedg
    inverse_degree: invdeg / nodeg
    encounter_policy: 'first' or 'second'
    stop_rule: 'complete' -> comp, 'percentage' -> pct{int(100*stop_percentage)}
    """
    alpha_int = int(round(100 * restart_prob))
    edge_flag = "edg" if use_edge_weight else "nedg"
    deg_flag = "invdeg" if inverse_degree else "nodeg"

    if stop_rule == "complete":
        stop_part = "comp"
    else:
        pct_int = int(round(100 * stop_percentage))
        stop_part = f"pct{pct_int}"

    fname = f"{graph_label}_alpha{alpha_int}_{edge_flag}_{deg_flag}_{encounter_policy}_{stop_part}.csv"
    return fname


def generate_synthetic_dataset(
    G: nx.Graph,
    graph_label: str,
    output_dir: Path,
    restart_prob: float,
    use_edge_weight: bool,
    inverse_degree: bool,
    encounter_policy: str,
    stop_rule: str,
    stop_percentage: float,
    max_steps: int = 100,
    n_workers: Optional[int] = None,
    random_state: int = 42,
) -> Path:
    """
    Generate a synthetic dataset by running one random walk per visit node.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        The heterogeneous graph (directed or undirected).
    graph_label : str
        'dir' for directed, 'ud' for undirected. Used only in filename.
    output_dir : Path
        Path to the samp{x} directory, e.g. processed/graphs/samp25.
        The synthetic file will be saved under output_dir / 'synth' / <filename>.csv
    restart_prob : float
    use_edge_weight : bool
    inverse_degree : bool
    encounter_policy : str
        'first' or 'second' (applies to person/provider).
    stop_rule : str
        'complete' or 'percentage'.
    stop_percentage : float
        Fraction of single-valued fields that must be filled if stop_rule == 'percentage'.
    max_steps : int
        Maximum number of steps per walk.
    n_workers : int or None
        Number of worker processes. If None, use all available cores.
    random_state : int
        Base seed for reproducibility.

    Returns
    -------
    Path
        Path to the synthetic CSV file.
    """

    output_dir = Path(output_dir)
    synth_dir = output_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Collect visit nodes
    visit_nodes = [
        n for n, data in G.nodes(data=True)
        if data.get("node_type") == "visit"
    ]
    visit_nodes = sorted(visit_nodes)  # deterministic order

    if not visit_nodes:
        raise ValueError("No visit nodes found in graph (node_type == 'visit').")

    # Build filename
    synth_fname = _build_synth_filename(
        graph_label=graph_label,
        restart_prob=restart_prob,
        use_edge_weight=use_edge_weight,
        inverse_degree=inverse_degree,
        encounter_policy=encounter_policy,
        stop_rule=stop_rule,
        stop_percentage=stop_percentage,
    )
    synth_path = synth_dir / synth_fname

    print(f"[INFO] Generating synthetic data for {len(visit_nodes):,} visit nodes.")
    print(f"[INFO] Output synthetic CSV: {synth_path}")

    # Prepare worker args
    if n_workers is None or n_workers <= 0:
        n_workers = cpu_count()

    base_seed = random_state
    worker_args = []
    for i, v in enumerate(visit_nodes):
        seed = base_seed + i
        worker_args.append(
            (
                G,
                v,
                restart_prob,
                use_edge_weight,
                inverse_degree,
                encounter_policy,
                stop_rule,
                stop_percentage,
                max_steps,
                seed,
            )
        )

    # Run in parallel
    # Run in parallel with tqdm tracking
    results = []

    if n_workers == 1:
        # Serial loop with tqdm
        for res in tqdm(map(_walk_worker, worker_args),
                        total=len(worker_args),
                        desc="Random walks"):
            results.append(res)
    else:
        # Parallel pool with tqdm
        with Pool(processes=n_workers) as pool:
            # imap_unordered yields results as they complete
            for res in tqdm(pool.imap_unordered(_walk_worker, worker_args),
                            total=len(worker_args),
                            desc=f"Random walks ({n_workers} workers)"):
                results.append(res)

    # Write CSV
    fieldnames = [
        "visitID",
        "personID",
        "providerID",
        "drug_concept_ids",
        "condition_concept_ids",
    ]

    with synth_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[SAVE] Synthetic dataset -> {synth_path}")
    return synth_path
