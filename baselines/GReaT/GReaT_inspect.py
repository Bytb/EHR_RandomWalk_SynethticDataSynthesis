from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "baselines" / "GReaT"

REAL_PATH = OUT_DIR / "great_train_input.csv"
SYNTH_PATH = OUT_DIR / "great_tabularisai_Qwen3-0.3B-distil_ep3_synth.csv"


def compute_counts(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a Series of #items per row for a semicolon-separated column."""
    # empty string -> 0, otherwise split on ';'
    return df[col].fillna("").astype(str).apply(
        lambda s: 0 if s == "" else len(s.split(";"))
    )


def main():
    print(f"[INFO] Loading REAL from:   {REAL_PATH}")
    real = pd.read_csv(REAL_PATH)
    print(f"[INFO] Loading SYNTH from:  {SYNTH_PATH}")
    synth = pd.read_csv(SYNTH_PATH)

    print("\n[SHAPE]")
    print("  REAL :", real.shape)
    print("  SYNTH:", synth.shape)

    print("\n[COLUMNS]")
    print("  REAL :", list(real.columns))
    print("  SYNTH:", list(synth.columns))

    # Head
    print("\n[REAL HEAD]")
    print(real.head(5))
    print("\n[SYNTH HEAD]")
    print(synth.head(5))

    # Basic provider_specialty stats
    print("\n[provider_specialty stats]")
    for name, df in [("REAL", real), ("SYNTH", synth)]:
        vc = df["provider_specialty"].value_counts(normalize=True).head(10)
        print(f"\n  {name} top 10 specialties (fraction):")
        print(vc.round(3))

    # Drugs / conditions per visit
    for name, df in [("REAL", real), ("SYNTH", synth)]:
        n_drugs = compute_counts(df, "drug_concept_ids")
        n_conds = compute_counts(df, "condition_concept_ids")
        print(f"\n[{name} length stats]")
        print(f"  #drugs      -> mean={n_drugs.mean():.2f}, median={n_drugs.median():.0f}, max={n_drugs.max()}")
        print(f"  #conditions -> mean={n_conds.mean():.2f}, median={n_conds.median():.0f}, max={n_conds.max()}")

    # Check for obviously broken rows
    for name, df in [("REAL", real), ("SYNTH", synth)]:
        empty_drugs = (df["drug_concept_ids"].fillna("") == "").mean()
        empty_conds = (df["condition_concept_ids"].fillna("") == "").mean()
        unknown_spec = (df["provider_specialty"] == "UNKNOWN_SPECIALTY").mean()
        print(f"\n[{name} missing-ish fractions]")
        print(f"  empty drug_concept_ids      : {empty_drugs:.3f}")
        print(f"  empty condition_concept_ids : {empty_conds:.3f}")
        print(f"  provider_specialty == UNKNOWN_SPECIALTY: {unknown_spec:.3f}")


if __name__ == "__main__":
    main()
