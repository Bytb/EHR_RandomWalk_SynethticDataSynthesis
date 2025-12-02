from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from be_great import GReaT


# ======================================================================================
# CONFIG (edit these if you want to change behavior)
# ======================================================================================

# Project root is two levels up: baselines/ -> ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input visit-fact table
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "visit_fact_table.csv"

# Where to write outputs
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "baselines" / "GReaT"

# LLM + training settings
DEFAULT_LLM = "tabularisai/Qwen3-0.3B-distil"
BATCH_SIZE = 8
EPOCHS = 3          # keep small for now; bump later if needed
MAX_ROWS: Optional[int] = 5000  # set to None to use all rows

# Sampling settings
N_SAMPLES: Optional[int] = None   # None => same as number of training rows
SAMPLE_MAX_LENGTH = 512           # needs to be >= typical input length

# Columns we need from visit_fact_table to build the row_text
REQUIRED_COLUMNS = [
    "provider_specialty",
    "drug_concept_ids",
    "condition_concept_ids",
]


# ======================================================================================
# Data loading / preprocessing
# ======================================================================================

def load_visit_fact_table(input_path: Path) -> pd.DataFrame:
    """Load the full visit_fact_table.csv."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Loading visit_fact_table from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded visit_fact_table with shape: {df.shape}")
    return df


def _clean_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the three core columns exist and are cleaned:
      - provider_specialty: string, UNKNOWN_SPECIALTY for missing
      - drug_concept_ids / condition_concept_ids: string, "" for missing
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in visit_fact_table: {missing}")

    core = df[REQUIRED_COLUMNS].copy()

    # provider_specialty: string, UNKNOWN for missing
    core["provider_specialty"] = (
        core["provider_specialty"]
        .fillna("UNKNOWN_SPECIALTY")
        .astype(str)
        .str.strip()
    )

    # drug_concept_ids / condition_concept_ids: ensure string; empty string for missing
    for col in ["drug_concept_ids", "condition_concept_ids"]:
        core[col] = (
            core[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return core


def build_great_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the GReaT training table from the visit_fact_table subset.

    Instead of 3 separate columns, we collapse them into a single text field
    with a strict template, e.g.:

        provider_specialty = Internal Medicine |
        drug_concept_ids = 254591;318096 |
        condition_concept_ids = 197917;199754

    GReaT will then model this one 'row_text' column.
    """
    core = _clean_core_columns(df)

    def to_row_text(row) -> str:
        spec = row["provider_specialty"]
        drugs = row["drug_concept_ids"]
        conds = row["condition_concept_ids"]

        # Keep the original semicolon-separated list format
        return (
            f"provider_specialty = {spec} | "
            f"drug_concept_ids = {drugs} | "
            f"condition_concept_ids = {conds}"
        )

    row_text = core.apply(to_row_text, axis=1)

    great_df = pd.DataFrame({"row_text": row_text})
    print(f"[INFO] GReaT input table shape (single 'row_text' column): {great_df.shape}")
    return great_df


# ======================================================================================
# GReaT model training / sampling
# ======================================================================================

def train_great_model(
    train_df: pd.DataFrame,
    llm: str = DEFAULT_LLM,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
) -> GReaT:
    """Instantiate and fit a GReaT model on the given DataFrame."""
    print(f"[INFO] Initializing GReaT with llm={llm}, batch_size={batch_size}, epochs={epochs}")
    model = GReaT(llm=llm, batch_size=batch_size, epochs=epochs)
    model.fit(train_df)
    print("[INFO] Finished training GReaT model.")
    return model


def sample_synthetic(model: GReaT, n_samples: int) -> pd.DataFrame:
    """
    Sample synthetic rows from a trained GReaT model.

    Uses a larger max_length and guided_sampling to avoid generation failures.
    """
    print(f"[INFO] Sampling {n_samples} synthetic rows from GReaT...")

    try:
        synth_df = model.sample(
            n_samples=n_samples,
            max_length=SAMPLE_MAX_LENGTH,
            guided_sampling=True,
        )
    except Exception as e:
        print(f"[ERROR] GReaT sampling failed: {e}")
        print("[HINT] You can try increasing SAMPLE_MAX_LENGTH in the config.")
        synth_df = pd.DataFrame(columns=["row_text"])

    print(f"[INFO] Synthetic sample shape: {synth_df.shape}")
    return synth_df


# ======================================================================================
# I/O helpers
# ======================================================================================

def ensure_output_dir(output_dir: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using output directory: {output_dir}")


def sanitize_llm_name(llm: str) -> str:
    """Make llm name safe for filenames."""
    return llm.replace("/", "_").replace(":", "_")


def save_outputs(
    great_input_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    output_dir: Path,
    llm: str,
    epochs: int,
) -> Tuple[Path, Path]:
    """
    Save the GReaT training input (row_text) and the synthetic sample.

    Returns:
        (path_to_train_input, path_to_synth)
    """
    ensure_output_dir(output_dir)

    train_path = output_dir / "great_train_input.csv"
    great_input_df.to_csv(train_path, index=False)
    print(f"[INFO] Saved GReaT training input (row_text) to: {train_path}")

    llm_tag = sanitize_llm_name(llm)
    synth_filename = f"great_{llm_tag}_ep{epochs}_synth.csv"
    synth_path = output_dir / synth_filename
    synth_df.to_csv(synth_path, index=False)
    print(f"[INFO] Saved synthetic sample (row_text) to: {synth_path}")

    return train_path, synth_path


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    # Make sure output dir exists early so we can save subset info
    ensure_output_dir(OUTPUT_DIR)

    # 1) Load full visit_fact_table
    visit_df = load_visit_fact_table(INPUT_PATH)

    # 2) Optional subsampling for faster experiments (on FULL table)
    if MAX_ROWS is not None and len(visit_df) > MAX_ROWS:
        visit_df_sub = (
            visit_df.sample(n=MAX_ROWS, random_state=0).reset_index(drop=True)
        )
        print(f"[INFO] Subsampled visit_fact_table to {len(visit_df_sub)} rows "
              f"for GReaT and EHRWalk (MAX_ROWS={MAX_ROWS}).")
    else:
        visit_df_sub = visit_df.copy()
        print(f"[INFO] Using all {len(visit_df_sub)} rows (no subsampling).")

    # 2a) Save the subset info so EHRWalk can use the SAME visits
    if "visit_occurrence_id" in visit_df_sub.columns:
        ids_path = OUTPUT_DIR / f"great_train_visit_ids_{len(visit_df_sub)}.csv"
        visit_df_sub[["visit_occurrence_id"]].to_csv(ids_path, index=False)
        print(f"[INFO] Saved visit_occurrence_id subset to: {ids_path}")

    subset_full_path = OUTPUT_DIR / f"great_train_visit_fact_subset_{len(visit_df_sub)}.csv"
    visit_df_sub.to_csv(subset_full_path, index=False)
    print(f"[INFO] Saved full visit_fact subset to: {subset_full_path}")

    # 3) Build 1-column GReaT input table (row_text) FROM THE SUBSET
    great_input_df = build_great_input(visit_df_sub)

    # 4) Train GReaT on (possibly subsampled) table
    model = train_great_model(
        great_input_df,
        llm=DEFAULT_LLM,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    # 5) Decide how many synthetic rows to sample
    n_samples = N_SAMPLES or len(great_input_df)
    synth_df = sample_synthetic(model, n_samples=n_samples)

    # 6) Save cleaned input + synthetic output for evaluation script
    train_path, synth_path = save_outputs(
        great_input_df,
        synth_df,
        OUTPUT_DIR,
        llm=DEFAULT_LLM,
        epochs=EPOCHS,
    )

    print("\n[SUMMARY]")
    print(f"  Real (GReaT input, row_text) path : {train_path}")
    print(f"  Synthetic output (row_text) path  : {synth_path}")
    print(f"  Subset visit_fact_table path      : {subset_full_path}")
    if 'ids_path' in locals():
        print(f"  Subset visit_occurrence_id path   : {ids_path}")
    print("  Next step: parse row_text back into columns for evaluation and "
          "use the subset files for EHRWalk to ensure a fair comparison.")


if __name__ == "__main__":
    main()
