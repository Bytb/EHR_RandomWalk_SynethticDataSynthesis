# baselines/GReaT/GReaT_inspect.py

from pathlib import Path
import re
from typing import Optional, Tuple

import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REAL_PATH = PROJECT_ROOT / "data" / "processed" / "baselines" / "GReaT" / "great_train_input.csv"
SYNTH_PATH = PROJECT_ROOT / "data" / "processed" / "baselines" / "GReaT" / "great_tabularisai_Qwen3-0.3B-distil_ep3_synth.csv"


# =============================================================================
# HELPERS
# =============================================================================

def extract_provider_specialty(text: str) -> Optional[str]:
    """
    Parse provider_specialty out of a GReaT row_text like:

        "provider_specialty = Family Practice | drug_concept_ids = [...] | ..."

    Returns the specialty string or None if we can't find it.
    """
    if not isinstance(text, str):
        return None

    # Normal case: "provider_specialty = <value> | ..."
    m = re.search(r"provider_specialty\s*=\s*([^|]+)", text)
    if m:
        return m.group(1).strip()

    # Degenerate generations like "provider_specialty" with no value
    stripped = text.strip()
    if stripped.startswith("provider_specialty"):
        # You can choose a label here; I mark it as missing.
        return None

    return None


def add_provider_specialty_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure df has a 'provider_specialty' column.
    If not, derive it from 'row_text'.
    """
    if "provider_specialty" in df.columns:
        return df

    if "row_text" not in df.columns:
        raise ValueError(
            f"{name} dataframe has no 'row_text' column; columns={list(df.columns)}"
        )

    df = df.copy()
    df["provider_specialty"] = df["row_text"].map(extract_provider_specialty)
    return df


def print_basic_info(label: str, df: pd.DataFrame) -> None:
    print(f"[{label} SHAPE]")
    print(" ", df.shape)
    print(f"[{label} COLUMNS]")
    print(" ", list(df.columns))
    print(f"[{label} HEAD]")
    print(df.head())
    print()


def print_provider_stats(label: str, df: pd.DataFrame) -> None:
    print(f"[provider_specialty stats - {label}]")

    if "provider_specialty" not in df.columns:
        print("  (no provider_specialty column found)")
        return

    # Missing rate
    missing_frac = df["provider_specialty"].isna().mean()
    print(f"  Missing provider_specialty: {missing_frac:.3%}")

    # Top 10 distribution (normalized)
    vc = df["provider_specialty"].value_counts(normalize=True, dropna=True).head(10)
    print("  Top 10 provider_specialty (relative freq):")
    for spec, frac in vc.items():
        print(f"    {spec!r:50s}  {frac:8.4f}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print(f"[INFO] Loading REAL from:   {REAL_PATH}")
    print(f"[INFO] Loading SYNTH from:  {SYNTH_PATH}")

    df_real = pd.read_csv(REAL_PATH)
    df_synth = pd.read_csv(SYNTH_PATH)

    print_basic_info("REAL", df_real)
    print_basic_info("SYNTH", df_synth)

    # Derive provider_specialty from row_text
    df_real = add_provider_specialty_column(df_real, "REAL")
    df_synth = add_provider_specialty_column(df_synth, "SYNTH")

    print_provider_stats("REAL", df_real)
    print_provider_stats("SYNTH", df_synth)


if __name__ == "__main__":
    main()
