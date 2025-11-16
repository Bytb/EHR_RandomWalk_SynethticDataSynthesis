"""
analyze_distributions.py

Preliminary EHR EDA:
- Load sampled_drug_exposure, sampled_condition_occurrence, sampled_person, provider
- Compute frequency distributions for:
    * drug_concept_id
    * condition_concept_id
    * provider_id (from both tables)
    * person_id
    * visit_occurrence_id
    * age (derived from year_of_birth; see REFERENCE_YEAR)

Outputs under ../data/diagnostics/:

  diagnostics/basic/
    - freq_*.csv for all core IDs
    - granular histograms:
        * value-count distributions (how many times each ID appears per table)
        * per-visit and per-person counts
        * per-provider patients / visits
        * age distribution

  diagnostics/final/
    - "Total degree" distributions per feature:
        * drug_concept_id: total count across dataset
        * condition_concept_id: total count
        * provider_id: merged counts from drug + condition tables
        * person_id: merged counts from drug + condition tables
        * visit_occurrence_id: merged counts from drug + condition tables
        * age estimates: overall age distribution

This script is intentionally simple and modular (no argparse).
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Config ----------

REFERENCE_YEAR = 2022  # used to approximate age from year_of_birth


# ---------- Utility functions ----------

def get_paths():
    """
    Return base paths for raw CSVs and diagnostics output.

    diagnostics/
        basic/ : raw freq tables and basic histograms
        final/ : final degree distributions for main features
    """
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir.parent / "data" / "clean_csvs"

    diag_root = base_dir.parent / "data" / "diagnostics"
    basic_dir = diag_root / "basic"
    final_dir = diag_root / "final"

    basic_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, basic_dir, final_dir


def compute_and_save_freq(series: pd.Series, name: str, out_dir: Path) -> pd.DataFrame:
    """
    Compute value counts for a Series and save as CSV.

    Parameters
    ----------
    series : pd.Series
        The data column to summarize.
    name : str
        Short name used in filenames (e.g., 'drug_concept_id').
    out_dir : Path
        Output directory for CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [value, count, fraction].
    """
    series = series.dropna()

    counts = series.value_counts()
    total = counts.sum()
    freq_df = (
        counts.rename("count")
        .to_frame()
        .reset_index()
        .rename(columns={"index": name})
    )
    freq_df["fraction"] = freq_df["count"] / total

    csv_path = out_dir / f"freq_{name}.csv"
    freq_df.to_csv(csv_path, index=False)
    print(f"[SAVE] Frequency table for {name} -> {csv_path}")

    return freq_df


def plot_hist(values, title: str, out_path: Path, log_y: bool = True, bins: int = 50):
    """
    Plot a simple histogram and save as PNG.

    Parameters
    ----------
    values : array-like
        Numeric values to plot.
    title : str
        Plot title.
    out_path : Path
        Path to save image.
    log_y : bool
        Whether to use log-scale on y-axis.
    bins : int
        Number of histogram bins.
    """
    plt.figure()
    plt.hist(values, bins=bins)
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVE] Histogram -> {out_path}")


# ---------- Main analysis functions (BASIC) ----------

def analyze_drug_exposure(raw_dir: Path, basic_dir: Path) -> None:
    """Analyze sampled_drug_exposure.csv distribution metrics."""
    path = raw_dir / "sampled_drug_exposure.csv"
    if not path.exists():
        print(f"[WARN] {path} not found, skipping drug exposure analysis.")
        return

    print(f"[LOAD] {path}")
    use_cols = [
        "person_id",
        "drug_concept_id",
        "provider_id",
        "visit_occurrence_id",
    ]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)

    print(f"[INFO] sampled_drug_exposure: rows={len(df):,}")

    # --- Frequencies for core IDs (BASIC) ---
    drug_freq = compute_and_save_freq(df["drug_concept_id"], "drug_concept_id", basic_dir)
    provider_freq_drug = compute_and_save_freq(df["provider_id"], "provider_id_from_drug", basic_dir)
    person_freq_drug = compute_and_save_freq(df["person_id"], "person_id_from_drug", basic_dir)
    visit_freq_drug = compute_and_save_freq(df["visit_occurrence_id"], "visit_occurrence_id_from_drug", basic_dir)

    # --- Visualize value-count distributions (how often each ID appears in this table) ---
    plot_hist(
        drug_freq["count"].values,
        "Frequency of drug_concept_id (value counts, drug_exposure)",
        basic_dir / "hist_freq_drug_concept_id.png",
        log_y=True,
    )

    plot_hist(
        provider_freq_drug["count"].values,
        "Frequency of provider_id (from drug_exposure)",
        basic_dir / "hist_freq_provider_id_from_drug.png",
        log_y=True,
    )

    plot_hist(
        person_freq_drug["count"].values,
        "Frequency of person_id (from drug_exposure)",
        basic_dir / "hist_freq_person_id_from_drug.png",
        log_y=True,
    )

    plot_hist(
        visit_freq_drug["count"].values,
        "Frequency of visit_occurrence_id (from drug_exposure)",
        basic_dir / "hist_freq_visit_occurrence_id_from_drug.png",
        log_y=True,
    )

    # --- Drugs per visit ---
    drugs_per_visit = df.groupby("visit_occurrence_id")["drug_concept_id"].nunique()
    print("[STATS] # unique drugs per visit (from drug_exposure):")
    print(drugs_per_visit.describe())

    plot_hist(
        drugs_per_visit.values,
        "Unique drugs per visit (drug_exposure)",
        basic_dir / "hist_drugs_per_visit.png",
        log_y=True,
    )

    # --- Visits per person (drug-based) ---
    visits_per_person_drug = df.groupby("person_id")["visit_occurrence_id"].nunique()
    print("[STATS] # unique visits per person (from drug_exposure):")
    print(visits_per_person_drug.describe())

    plot_hist(
        visits_per_person_drug.values,
        "Unique visits per person (drug_exposure)",
        basic_dir / "hist_visits_per_person_from_drug.png",
        log_y=True,
    )

    # --- Provider-centric stats: patients & visits per provider (drug table) ---
    patients_per_provider_drug = df.groupby("provider_id")["person_id"].nunique()
    print("[STATS] # unique patients per provider (from drug_exposure):")
    print(patients_per_provider_drug.describe())

    plot_hist(
        patients_per_provider_drug.values,
        "Unique patients per provider (drug_exposure)",
        basic_dir / "hist_patients_per_provider_from_drug.png",
        log_y=True,
    )

    visits_per_provider_drug = df.groupby("provider_id")["visit_occurrence_id"].nunique()
    print("[STATS] # unique visits per provider (from drug_exposure):")
    print(visits_per_provider_drug.describe())

    plot_hist(
        visits_per_provider_drug.values,
        "Unique visits per provider (drug_exposure)",
        basic_dir / "hist_visits_per_provider_from_drug.png",
        log_y=True,
    )


def analyze_condition_occurrence(raw_dir: Path, basic_dir: Path) -> None:
    """Analyze sampled_condition_occurrence.csv distribution metrics."""
    path = raw_dir / "sampled_condition_occurrence.csv"
    if not path.exists():
        print(f"[WARN] {path} not found, skipping condition occurrence analysis.")
        return

    print(f"[LOAD] {path}")
    use_cols = [
        "person_id",
        "condition_concept_id",
        "provider_id",
        "visit_occurrence_id",
    ]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)

    print(f"[INFO] sampled_condition_occurrence: rows={len(df):,}")

    # --- Frequencies for core IDs (BASIC) ---
    cond_freq = compute_and_save_freq(df["condition_concept_id"], "condition_concept_id", basic_dir)
    provider_freq_cond = compute_and_save_freq(df["provider_id"], "provider_id_from_condition", basic_dir)
    person_freq_cond = compute_and_save_freq(df["person_id"], "person_id_from_condition", basic_dir)
    visit_freq_cond = compute_and_save_freq(df["visit_occurrence_id"], "visit_occurrence_id_from_condition", basic_dir)

    # --- Visualize value-count distributions ---
    plot_hist(
        cond_freq["count"].values,
        "Frequency of condition_concept_id (value counts, condition_occurrence)",
        basic_dir / "hist_freq_condition_concept_id.png",
        log_y=True,
    )

    plot_hist(
        provider_freq_cond["count"].values,
        "Frequency of provider_id (from condition_occurrence)",
        basic_dir / "hist_freq_provider_id_from_condition.png",
        log_y=True,
    )

    plot_hist(
        person_freq_cond["count"].values,
        "Frequency of person_id (from condition_occurrence)",
        basic_dir / "hist_freq_person_id_from_condition.png",
        log_y=True,
    )

    plot_hist(
        visit_freq_cond["count"].values,
        "Frequency of visit_occurrence_id (from condition_occurrence)",
        basic_dir / "hist_freq_visit_occurrence_id_from_condition.png",
        log_y=True,
    )

    # --- Conditions per visit ---
    conds_per_visit = df.groupby("visit_occurrence_id")["condition_concept_id"].nunique()
    print("[STATS] # unique conditions per visit:")
    print(conds_per_visit.describe())

    plot_hist(
        conds_per_visit.values,
        "Unique conditions per visit",
        basic_dir / "hist_conditions_per_visit.png",
        log_y=True,
    )

    # --- Visits per person (condition-based) ---
    visits_per_person_cond = df.groupby("person_id")["visit_occurrence_id"].nunique()
    print("[STATS] # unique visits per person (from condition_occurrence):")
    print(visits_per_person_cond.describe())

    plot_hist(
        visits_per_person_cond.values,
        "Unique visits per person (condition_occurrence)",
        basic_dir / "hist_visits_per_person_from_condition.png",
        log_y=True,
    )

    # --- Provider-centric stats: patients & visits per provider (condition table) ---
    patients_per_provider_cond = df.groupby("provider_id")["person_id"].nunique()
    print("[STATS] # unique patients per provider (from condition_occurrence):")
    print(patients_per_provider_cond.describe())

    plot_hist(
        patients_per_provider_cond.values,
        "Unique patients per provider (condition_occurrence)",
        basic_dir / "hist_patients_per_provider_from_condition.png",
        log_y=True,
    )

    visits_per_provider_cond = df.groupby("provider_id")["visit_occurrence_id"].nunique()
    print("[STATS] # unique visits per provider (from condition_occurrence):")
    print(visits_per_provider_cond.describe())

    plot_hist(
        visits_per_provider_cond.values,
        "Unique visits per provider (condition_occurrence)",
        basic_dir / "hist_visits_per_provider_from_condition.png",
        log_y=True,
    )


def analyze_person(raw_dir: Path, basic_dir: Path) -> None:
    """Analyze sampled_person.csv for age distributions."""
    path = raw_dir / "sampled_person.csv"
    if not path.exists():
        print(f"[WARN] {path} not found, skipping person analysis.")
        return

    print(f"[LOAD] {path}")
    use_cols = ["person_id", "year_of_birth"]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)

    print(f"[INFO] sampled_person: rows={len(df):,}")

    # Approximate age; REFERENCE_YEAR can be tuned later
    valid = df["year_of_birth"].dropna()
    ages = REFERENCE_YEAR - valid.astype(int)

    age_df = pd.DataFrame({"age_estimate": ages})
    age_df.to_csv(basic_dir / "age_estimates_from_year_of_birth.csv", index=False)
    print(f"[SAVE] Age estimates -> {basic_dir / 'age_estimates_from_year_of_birth.csv'}")

    print("[STATS] Age estimate distribution (REFERENCE_YEAR-based):")
    print(ages.describe())

    plot_hist(
        ages.values,
        f"Age estimates (reference year {REFERENCE_YEAR})",
        basic_dir / "hist_age_estimates.png",
        log_y=False,
    )


def analyze_provider_table(raw_dir: Path, basic_dir: Path) -> None:
    """Optional: analyze provider table for metadata like specialty."""
    path = raw_dir / "provider.csv"
    if not path.exists():
        print(f"[WARN] {path} not found, skipping provider metadata analysis.")
        return

    print(f"[LOAD] {path}")
    use_cols = ["provider_id", "specialty_concept_id", "specialty_source_value"]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)

    print(f"[INFO] provider: rows={len(df):,}")

    # Frequency of specialties (concept-based and source-based)
    spec_concept_freq = compute_and_save_freq(df["specialty_concept_id"], "specialty_concept_id", basic_dir)
    spec_source_freq = compute_and_save_freq(df["specialty_source_value"], "specialty_source_value", basic_dir)

    plot_hist(
        spec_concept_freq["count"].values,
        "Frequency of specialty_concept_id",
        basic_dir / "hist_freq_specialty_concept_id.png",
        log_y=True,
    )

    plot_hist(
        spec_source_freq["count"].values,
        "Frequency of specialty_source_value",
        basic_dir / "hist_freq_specialty_source_value.png",
        log_y=True,
    )


# ---------- FINAL: total degree distributions per feature ----------

def build_final_degree_distributions(basic_dir: Path, final_dir: Path) -> None:
    """
    Build "total degree" distributions for main feature types and save
    histograms in final_dir.

    Features:
      - drug_concept_id: freq_drug_concept_id.csv (already total)
      - condition_concept_id: freq_condition_concept_id.csv
      - provider_id: merged counts from drug + condition tables
      - person_id: merged counts from drug + condition tables
      - visit_occurrence_id: merged counts from drug + condition tables
      - age_estimate: distribution over patients
    """

    # --- Drugs: total degree distribution ---
    path_drug = basic_dir / "freq_drug_concept_id.csv"
    if path_drug.exists():
        df_drug = pd.read_csv(path_drug)
        plot_hist(
            df_drug["count"].values,
            "Degree distribution: drug_concept_id (total occurrences)",
            final_dir / "deg_drug_concept_id.png",
            log_y=True,
        )
    else:
        print(f"[WARN] {path_drug} not found; skipping drug degree distribution.")

    # --- Conditions: total degree distribution ---
    path_cond = basic_dir / "freq_condition_concept_id.csv"
    if path_cond.exists():
        df_cond = pd.read_csv(path_cond)
        plot_hist(
            df_cond["count"].values,
            "Degree distribution: condition_concept_id (total occurrences)",
            final_dir / "deg_condition_concept_id.png",
            log_y=True,
        )
    else:
        print(f"[WARN] {path_cond} not found; skipping condition degree distribution.")

    # --- Providers: total degree across drug + condition ---
    path_prov_drug = basic_dir / "freq_provider_id_from_drug.csv"
    path_prov_cond = basic_dir / "freq_provider_id_from_condition.csv"
    if path_prov_drug.exists() and path_prov_cond.exists():
        df_prov_drug = pd.read_csv(path_prov_drug).rename(
            columns={
                "provider_id_from_drug": "provider_id",
                "count": "count_drug",
            }
        )
        df_prov_cond = pd.read_csv(path_prov_cond).rename(
            columns={
                "provider_id_from_condition": "provider_id",
                "count": "count_cond",
            }
        )
        prov_combo = pd.merge(df_prov_drug, df_prov_cond, on="provider_id", how="outer")
        prov_combo["count_drug"] = prov_combo["count_drug"].fillna(0)
        prov_combo["count_cond"] = prov_combo["count_cond"].fillna(0)
        prov_combo["count_total"] = prov_combo["count_drug"] + prov_combo["count_cond"]

        prov_combo.to_csv(final_dir / "freq_provider_combined.csv", index=False)
        plot_hist(
            prov_combo["count_total"].values,
            "Degree distribution: provider_id (total events: drug + condition)",
            final_dir / "deg_provider_id.png",
            log_y=True,
        )
    else:
        print("[WARN] Provider freq tables missing; skipping provider degree distribution.")

    # --- Persons: total degree across drug + condition ---
    path_person_drug = basic_dir / "freq_person_id_from_drug.csv"
    path_person_cond = basic_dir / "freq_person_id_from_condition.csv"
    if path_person_drug.exists() and path_person_cond.exists():
        df_person_drug = pd.read_csv(path_person_drug).rename(
            columns={
                "person_id_from_drug": "person_id",
                "count": "count_drug",
            }
        )
        df_person_cond = pd.read_csv(path_person_cond).rename(
            columns={
                "person_id_from_condition": "person_id",
                "count": "count_cond",
            }
        )
        person_combo = pd.merge(df_person_drug, df_person_cond, on="person_id", how="outer")
        person_combo["count_drug"] = person_combo["count_drug"].fillna(0)
        person_combo["count_cond"] = person_combo["count_cond"].fillna(0)
        person_combo["count_total"] = person_combo["count_drug"] + person_combo["count_cond"]

        person_combo.to_csv(final_dir / "freq_person_id_combined.csv", index=False)
        plot_hist(
            person_combo["count_total"].values,
            "Degree distribution: person_id (total events: drug + condition)",
            final_dir / "deg_person_id.png",
            log_y=True,
        )
    else:
        print("[WARN] Person freq tables missing; skipping person degree distribution.")

    # --- Visits: total degree across drug + condition ---
    path_visit_drug = basic_dir / "freq_visit_occurrence_id_from_drug.csv"
    path_visit_cond = basic_dir / "freq_visit_occurrence_id_from_condition.csv"
    if path_visit_drug.exists() and path_visit_cond.exists():
        df_visit_drug = pd.read_csv(path_visit_drug).rename(
            columns={
                "visit_occurrence_id_from_drug": "visit_occurrence_id",
                "count": "count_drug",
            }
        )
        df_visit_cond = pd.read_csv(path_visit_cond).rename(
            columns={
                "visit_occurrence_id_from_condition": "visit_occurrence_id",
                "count": "count_cond",
            }
        )
        visit_combo = pd.merge(df_visit_drug, df_visit_cond, on="visit_occurrence_id", how="outer")
        visit_combo["count_drug"] = visit_combo["count_drug"].fillna(0)
        visit_combo["count_cond"] = visit_combo["count_cond"].fillna(0)
        visit_combo["count_total"] = visit_combo["count_drug"] + visit_combo["count_cond"]

        visit_combo.to_csv(final_dir / "freq_visit_occurrence_id_combined.csv", index=False)
        plot_hist(
            visit_combo["count_total"].values,
            "Degree distribution: visit_occurrence_id (total events: drug + condition)",
            final_dir / "deg_visit_occurrence_id.png",
            log_y=True,
        )
    else:
        print("[WARN] Visit freq tables missing; skipping visit degree distribution.")

    # --- Age: overall distribution across patients ---
    path_age = basic_dir / "age_estimates_from_year_of_birth.csv"
    if path_age.exists():
        df_age = pd.read_csv(path_age)
        plot_hist(
            df_age["age_estimate"].values,
            "Age distribution (patients)",
            final_dir / "age_distribution.png",
            log_y=False,
        )
    else:
        print(f"[WARN] {path_age} not found; skipping age distribution in final.")


# ---------- Main ----------

def main():
    raw_dir, basic_dir, final_dir = get_paths()
    print(f"[INFO] Raw CSV dir: {raw_dir}")
    print(f"[INFO] Basic diagnostics dir: {basic_dir}")
    print(f"[INFO] Final diagnostics dir: {final_dir}\n")

    analyze_drug_exposure(raw_dir, basic_dir)
    print("\n" + "-" * 80 + "\n")

    analyze_condition_occurrence(raw_dir, basic_dir)
    print("\n" + "-" * 80 + "\n")

    analyze_person(raw_dir, basic_dir)
    print("\n" + "-" * 80 + "\n")

    analyze_provider_table(raw_dir, basic_dir)
    print("\n" + "-" * 80 + "\n")

    # Build the main "degree distributions" for the final folder
    build_final_degree_distributions(basic_dir, final_dir)

    print("\n[DONE] Distribution analysis complete.")


if __name__ == "__main__":
    main()
