"""
clean_ehr_tables.py

Clean core EHR tables and write cleaned CSVs to ../data/clean_csvs.

Tables:
  - sampled_drug_exposure.csv
  - sampled_condition_occurrence.csv
  - sampled_person.csv
  - provider.csv

Cleaning rules (core IDs):
  - Coerce core ID columns to numeric (invalid -> NaN)
  - Drop rows with NaN in any core ID
  - Drop exact duplicate rows
  - For year_of_birth: keep only [1900, REFERENCE_YEAR]
"""

from pathlib import Path
import pandas as pd

REFERENCE_YEAR = 2022


# ---------- Path helpers ----------

def get_paths():
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir.parent / "data" / "raw_csvs"
    clean_dir = base_dir.parent / "data" / "clean_csvs"
    clean_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, clean_dir


def _summary_print(name: str, n_before: int, n_after: int):
    dropped = n_before - n_after
    print(f"[CLEAN] {name}: {n_before:,} -> {n_after:,} rows "
          f"(dropped {dropped:,})")


# ---------- Cleaning functions ----------

def clean_sampled_drug_exposure(raw_dir: Path, clean_dir: Path) -> None:
    name = "sampled_drug_exposure"
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        print(f"[WARN] {path} not found, skipping.")
        return

    print(f"[LOAD] {path}")
    df = pd.read_csv(path, low_memory=False)
    n_before = len(df)

    core_cols = ["person_id", "drug_concept_id", "provider_id",
                 "visit_occurrence_id"]

    # Coerce core IDs to numeric
    for col in core_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"[WARN] Column {col} missing in {name}; "
                  f"cleaning may be incomplete.")

    # Drop rows with missing core IDs
    present_core_cols = [c for c in core_cols if c in df.columns]
    if present_core_cols:
        df = df.dropna(subset=present_core_cols)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Cast core IDs to int64 where present
    for col in present_core_cols:
        df[col] = df[col].astype("int64")

    n_after = len(df)
    _summary_print(name, n_before, n_after)

    out_path = clean_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Cleaned {name} -> {out_path}\n")


def clean_sampled_condition_occurrence(raw_dir: Path, clean_dir: Path) -> None:
    name = "sampled_condition_occurrence"
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        print(f"[WARN] {path} not found, skipping.")
        return

    print(f"[LOAD] {path}")
    df = pd.read_csv(path, low_memory=False)
    n_before = len(df)

    core_cols = ["person_id", "condition_concept_id", "provider_id",
                 "visit_occurrence_id"]

    # Coerce core IDs to numeric
    for col in core_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"[WARN] Column {col} missing in {name}; "
                  f"cleaning may be incomplete.")

    # Drop rows with missing core IDs
    present_core_cols = [c for c in core_cols if c in df.columns]
    if present_core_cols:
        df = df.dropna(subset=present_core_cols)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Cast core IDs to int64 where present
    for col in present_core_cols:
        df[col] = df[col].astype("int64")

    n_after = len(df)
    _summary_print(name, n_before, n_after)

    out_path = clean_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Cleaned {name} -> {out_path}\n")


def clean_sampled_person(raw_dir: Path, clean_dir: Path) -> None:
    name = "sampled_person"
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        print(f"[WARN] {path} not found, skipping.")
        return

    print(f"[LOAD] {path}")
    df = pd.read_csv(path, low_memory=False)
    n_before = len(df)

    # Coerce IDs and year_of_birth
    if "person_id" in df.columns:
        df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")
    else:
        print(f"[WARN] person_id missing in {name}.")

    if "year_of_birth" in df.columns:
        df["year_of_birth"] = pd.to_numeric(
            df["year_of_birth"], errors="coerce"
        )
        # Filter to plausible years
        df = df[
            (df["year_of_birth"] >= 1900)
            & (df["year_of_birth"] <= REFERENCE_YEAR)
        ]
    else:
        print(f"[WARN] year_of_birth missing in {name}.")

    # Drop rows missing core fields
    subset_cols = [c for c in ["person_id", "year_of_birth"] if c in df.columns]
    if subset_cols:
        df = df.dropna(subset=subset_cols)

    # Drop duplicates on person_id
    if "person_id" in df.columns:
        df = df.drop_duplicates(subset=["person_id"])

    # Cast person_id to int64
    if "person_id" in df.columns:
        df["person_id"] = df["person_id"].astype("int64")

    n_after = len(df)
    _summary_print(name, n_before, n_after)

    out_path = clean_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Cleaned {name} -> {out_path}\n")


def clean_provider(raw_dir: Path, clean_dir: Path) -> None:
    name = "provider"
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        print(f"[WARN] {path} not found, skipping.")
        return

    print(f"[LOAD] {path}")
    df = pd.read_csv(path, low_memory=False)
    n_before = len(df)

    # Coerce provider_id to numeric
    if "provider_id" in df.columns:
        df["provider_id"] = pd.to_numeric(df["provider_id"], errors="coerce")
        df = df.dropna(subset=["provider_id"])
        df["provider_id"] = df["provider_id"].astype("int64")
    else:
        print(f"[WARN] provider_id missing in {name}.")

    # Drop duplicate provider_id rows (keep first)
    if "provider_id" in df.columns:
        df = df.drop_duplicates(subset=["provider_id"])

    n_after = len(df)
    _summary_print(name, n_before, n_after)

    out_path = clean_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Cleaned {name} -> {out_path}\n")


# ---------- Main ----------

def main():
    raw_dir, clean_dir = get_paths()
    print(f"[INFO] Raw CSV dir:   {raw_dir}")
    print(f"[INFO] Clean CSV dir: {clean_dir}\n")

    clean_sampled_drug_exposure(raw_dir, clean_dir)
    clean_sampled_condition_occurrence(raw_dir, clean_dir)
    clean_sampled_person(raw_dir, clean_dir)
    clean_provider(raw_dir, clean_dir)

    print("[DONE] Cleaning complete.")


if __name__ == "__main__":
    main()
