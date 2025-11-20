"""
compare_distributions.py

Compares the distribution of #drugs and #conditions per visit
between REAL sampled_visit_facts.csv and SYNTHETIC rw_synth output.

Now includes a matplotlib chart to visually compare distributions.
"""

from pathlib import Path
import csv
from collections import Counter

import matplotlib.pyplot as plt


# ===========================
# User-configurable paths
# ===========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ID = 100

REAL_FACTS_PATH = (
    PROJECT_ROOT /
    "data" / "processed" / "graphs" /
    f"samp{SAMPLE_ID}" /
    "sampled_visit_facts.csv"
)

SYNTH_PATH = (
    PROJECT_ROOT /
    "data" / "processed" / "graphs" /
    f"samp{SAMPLE_ID}" /
    "synth" /
    "un_alpha30_edg_nodeg_first_comp.csv"
)


# ===========================
# Helpers
# ===========================

def parse_multi(value):
    """Convert semicolon-separated list to count."""
    if value == "NA" or value.strip() == "":
        return 0
    return len(value.split(";"))


def load_counts(csv_path: Path):
    """Load drug/condition counts from facts or synthetic."""
    drug_counts = []
    cond_counts = []

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_counts.append(parse_multi(row["drug_concept_ids"]))
            cond_counts.append(parse_multi(row["condition_concept_ids"]))

    return drug_counts, cond_counts


def summarize(name, data):
    c = Counter(data)
    print(f"\n===== {name} =====")
    print("Total rows:", len(data))
    print("Mean:", sum(data) / len(data))
    print("Min:", min(data))
    print("Max:", max(data))
    print("\nMost common counts:")
    for k, v in c.most_common(10):
        print(f"  {k} → {v}")


# ===========================
# Plotting
# ===========================

def plot_distributions(real_drug, real_cond, synth_drug, synth_cond):
    plt.figure(figsize=(12, 6))

    # ---- Drugs ----
    plt.subplot(1, 2, 1)
    plt.hist(real_drug, bins=40, alpha=0.6, label="Real")
    plt.hist(synth_drug, bins=40, alpha=0.6, label="Synthetic")
    plt.title("Distribution of #Drugs per Visit")
    plt.xlabel("# Drugs")
    plt.ylabel("Frequency")
    plt.legend()

    # ---- Conditions ----
    plt.subplot(1, 2, 2)
    plt.hist(real_cond, bins=40, alpha=0.6, label="Real")
    plt.hist(synth_cond, bins=40, alpha=0.6, label="Synthetic")
    plt.title("Distribution of #Conditions per Visit")
    plt.xlabel("# Conditions")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ===========================
# Main
# ===========================

def main():
    print(f"[LOAD] Real visit facts: {REAL_FACTS_PATH}")
    real_drug, real_cond = load_counts(REAL_FACTS_PATH)

    print(f"[LOAD] Synthetic data: {SYNTH_PATH}")
    synth_drug, synth_cond = load_counts(SYNTH_PATH)

    summarize("REAL #drugs per visit", real_drug)
    summarize("REAL #conditions per visit", real_cond)

    summarize("SYNTH #drugs per synthetic row", synth_drug)
    summarize("SYNTH #conditions per synthetic row", synth_cond)

    # ---- Show comparison chart ----
    plot_distributions(real_drug, real_cond, synth_drug, synth_cond)

    print("\n[Done]")


if __name__ == "__main__":
    main()
