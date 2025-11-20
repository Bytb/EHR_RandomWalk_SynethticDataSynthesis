"""
build_visit_fact_table.py

Construct a unified, visit-centric fact table from cleaned OMOP CSVs.

Inputs (from ../data/clean_csvs/):
  - sampled_drug_exposure.csv
  - sampled_condition_occurrence.csv
  - sampled_person.csv
  - provider.csv

Output (to ../data/processed/):
  - visit_fact_table.csv

Each row corresponds to a single visit_occurrence_id and aggregates:
  - person_id           (prefers person from drug table, else condition table)
  - provider_id         (prefers provider from drug table, else condition table)
  - provider_specialty  (joined from provider.csv, does NOT affect graph/walk)
  - drug_concept_ids    (unique per visit, semicolon-separated)
  - condition_concept_ids (unique per visit, semicolon-separated)
  - n_drugs, n_conditions

Optionally, we can enforce that all node types are present for every row:
  - require_all_node_types = True:
      keep only rows where:
        - person_id and provider_id are non-null
        - n_drugs > 0 and drug_concept_ids is non-empty
        - n_conditions > 0 and condition_concept_ids is non-empty
"""

from pathlib import Path
import pandas as pd


# Toggle you can edit if you want the saved table to only contain
# rows that have ALL node types present (visit, person, provider, drug, condition).
REQUIRE_ALL_NODE_TYPES = True


# ---------- Paths ----------

def get_paths():
    base_dir = Path(__file__).resolve().parent
    clean_dir = base_dir.parent / "data" / "clean_csvs"
    processed_dir = base_dir.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return clean_dir, processed_dir


# ---------- Helpers ----------

def _aggregate_str_list(series: pd.Series) -> str:
    """
    Aggregate a column of IDs into a semicolon-separated string of unique values.
    Handles numeric or string; returns empty string if all NaN.
    """
    vals = series.dropna().unique()
    if len(vals) == 0:
        return ""
    # convert to string and sort for reproducibility
    vals_str = sorted(str(v) for v in vals)
    return ";".join(vals_str)


def build_provider_specialty_view(clean_dir: Path) -> pd.DataFrame:
    """
    Build a provider -> specialty lookup from provider.csv.

    Uses 'specialty_source_value' as the specialty column.

    Returns:
      provider_id, provider_specialty
    """
    path = clean_dir / "provider.csv"
    if not path.exists():
        print(f"[WARN] {path} not found; provider specialty will be missing.")
        return pd.DataFrame(columns=["provider_id", "provider_specialty"])

    print(f"[LOAD] {path}")
    # Adjusted column name here:
    use_cols = ["provider_id", "specialty_source_value"]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)

    # Deduplicate providers
    df = df.drop_duplicates(subset=["provider_id"]).copy()
    df = df.rename(columns={"specialty_source_value": "provider_specialty"})

    print(f"[BUILD] provider_specialty view: {len(df):,} providers")
    return df



# ---------- Builders ----------

def build_visit_drug_view(clean_dir: Path) -> pd.DataFrame:
    """
    Build per-visit aggregates from sampled_drug_exposure:
      - person_id_from_drug
      - provider_id_from_drug
      - drug_concept_ids
      - n_drugs
    """
    path = clean_dir / "sampled_drug_exposure.csv"
    if not path.exists():
        print(f"[WARN] {path} not found; drug view will be empty.")
        return pd.DataFrame(columns=[
            "visit_occurrence_id",
            "person_id_from_drug",
            "provider_id_from_drug",
            "drug_concept_ids",
            "n_drugs",
        ])

    print(f"[LOAD] {path}")
    use_cols = ["visit_occurrence_id", "person_id", "provider_id", "drug_concept_id"]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)
    print(f"[INFO] drug_exposure rows: {len(df):,}")

    grp = df.groupby("visit_occurrence_id")

    # aggregate scalar stats
    agg = grp.agg(
        person_id_from_drug=("person_id", "first"),
        provider_id_from_drug=("provider_id", "first"),
        n_drugs=("drug_concept_id", "nunique"),
    ).reset_index()

    # aggregate list of drugs
    drug_lists = grp["drug_concept_id"].apply(_aggregate_str_list).reset_index()
    drug_lists = drug_lists.rename(columns={"drug_concept_id": "drug_concept_ids"})

    # merge them
    visit_drug = agg.merge(drug_lists, on="visit_occurrence_id", how="left")

    print(f"[BUILD] visit_drug view: {len(visit_drug):,} visits")
    return visit_drug


def build_visit_condition_view(clean_dir: Path) -> pd.DataFrame:
    """
    Build per-visit aggregates from sampled_condition_occurrence:
      - person_id_from_condition
      - provider_id_from_condition
      - condition_concept_ids
      - n_conditions
    """
    path = clean_dir / "sampled_condition_occurrence.csv"
    if not path.exists():
        print(f"[WARN] {path} not found; condition view will be empty.")
        return pd.DataFrame(columns=[
            "visit_occurrence_id",
            "person_id_from_condition",
            "provider_id_from_condition",
            "condition_concept_ids",
            "n_conditions",
        ])

    print(f"[LOAD] {path}")
    use_cols = ["visit_occurrence_id", "person_id", "provider_id", "condition_concept_id"]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)
    print(f"[INFO] condition_occurrence rows: {len(df):,}")

    grp = df.groupby("visit_occurrence_id")

    # aggregate scalar stats
    agg = grp.agg(
        person_id_from_condition=("person_id", "first"),
        provider_id_from_condition=("provider_id", "first"),
        n_conditions=("condition_concept_id", "nunique"),
    ).reset_index()

    # aggregate list of conditions
    cond_lists = grp["condition_concept_id"].apply(_aggregate_str_list).reset_index()
    cond_lists = cond_lists.rename(columns={"condition_concept_id": "condition_concept_ids"})

    # merge them
    visit_cond = agg.merge(cond_lists, on="visit_occurrence_id", how="left")

    print(f"[BUILD] visit_condition view: {len(visit_cond):,} visits")
    return visit_cond


# ---------- Main builder ----------

def build_visit_fact_table(clean_dir: Path,
                           processed_dir: Path,
                           require_all_node_types: bool = False) -> None:
    """
    Build the unified visit_fact_table and save as CSV.

    If require_all_node_types is True, only keep rows where:
      - visit_occurrence_id, person_id, provider_id are non-null
      - n_drugs > 0 and drug_concept_ids is non-empty
      - n_conditions > 0 and condition_concept_ids is non-empty
    """
    visit_drug = build_visit_drug_view(clean_dir)
    visit_cond = build_visit_condition_view(clean_dir)
    provider_specialty = build_provider_specialty_view(clean_dir)

    # Outer merge of visits from drug and condition tables
    print("[MERGE] Combining visit_drug and visit_condition views...")
    visits = pd.merge(
        visit_drug,
        visit_cond,
        on="visit_occurrence_id",
        how="outer",
    )
    print(f"[INFO] Combined visits: {len(visits):,}")

    # Resolve person_id: prefer from drug, else from condition
    visits["person_id"] = visits["person_id_from_drug"].combine_first(
        visits["person_id_from_condition"]
    )

    # Resolve provider_id: prefer from drug, else from condition
    visits["provider_id"] = visits["provider_id_from_drug"].combine_first(
        visits["provider_id_from_condition"]
    )

    # Basic consistency check: person_id_from_drug vs person_id_from_condition
    mask_both_person = visits["person_id_from_drug"].notna() & visits["person_id_from_condition"].notna()
    if mask_both_person.any():
        mismatches = (visits.loc[mask_both_person, "person_id_from_drug"] !=
                      visits.loc[mask_both_person, "person_id_from_condition"]).sum()
        if mismatches > 0:
            print(f"[WARN] Person ID mismatch in {mismatches} visits where both sources present.")

    # Fill NaNs for counts where appropriate
    for col in ["n_drugs", "n_conditions"]:
        if col in visits.columns:
            visits[col] = visits[col].fillna(0).astype(int)

    # --- NEW: join provider_specialty (does NOT affect graph / walk) ---
    if not provider_specialty.empty:
        print("[MERGE] Joining provider_specialty from provider.csv...")
        visits = visits.merge(provider_specialty, on="provider_id", how="left")
    else:
        visits["provider_specialty"] = pd.NA

    # Optional: require that all node types are present
    if require_all_node_types:
        print("[FILTER] Enforcing presence of all node types (visit, person, provider, drug, condition)...")
        before = len(visits)

        # require non-null IDs
        mask = (
            visits["visit_occurrence_id"].notna()
            & visits["person_id"].notna()
            & visits["provider_id"].notna()
        )

        # require non-zero counts and non-empty ID lists for drugs/conditions
        # drug_concept_ids / condition_concept_ids may be NaN or empty string
        for col_ids, col_n in [
            ("drug_concept_ids", "n_drugs"),
            ("condition_concept_ids", "n_conditions"),
        ]:
            ids = visits[col_ids]
            mask_ids = ids.notna() & (ids != "")
            mask = mask & mask_ids & (visits[col_n] > 0)

        visits = visits[mask].copy()
        after = len(visits)
        print(f"[FILTER] Kept {after:,} / {before:,} visits after requiring all node types.")

    # Reorder columns for readability and drop internal debug columns
    col_order = [
        "visit_occurrence_id",
        "person_id",
        "provider_id",
        "provider_specialty",   # NEW: included for evaluation
        "drug_concept_ids",
        "n_drugs",
        "condition_concept_ids",
        "n_conditions",
    ]
    # Keep only columns that actually exist
    col_order = [c for c in col_order if c in visits.columns]
    visits = visits[col_order]

    out_path = processed_dir / "visit_fact_table.csv"
    visits.to_csv(out_path, index=False)
    print(f"[SAVE] visit_fact_table -> {out_path}")
    print(f"[STATS] visits: {len(visits):,}")


def main():
    clean_dir, processed_dir = get_paths()
    print(f"[INFO] Clean CSV dir:   {clean_dir}")
    print(f"[INFO] Processed dir:   {processed_dir}\n")

    build_visit_fact_table(
        clean_dir,
        processed_dir,
        require_all_node_types=REQUIRE_ALL_NODE_TYPES,
    )

    print("\n[DONE] Visit fact table construction complete.")


if __name__ == "__main__":
    main()
