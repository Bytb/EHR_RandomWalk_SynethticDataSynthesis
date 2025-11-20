"""
evaluation.py

Evaluation utilities for synthetic EHR random walks.

Implements:
  1) Wasserstein distance on #drugs / #conditions
  2) p-test: real vs synthetic classification
  3) Predictive utility: provider specialty prediction (TRTR / TSTR)

Use test_evaluations() at the bottom to sanity-check everything.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance, norm

# ---------------------------------------------------------------------
# Paths & defaults (adjust SAMPLE_ID / SYNTH_FILENAME as needed)
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_ID = 100

REAL_FACTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
    / f"samp{SAMPLE_ID}"
    / "sampled_visit_facts.csv"
)

SYNTH_FILENAME = "un_alpha30_edg_nodeg_first_comp.csv"
SYNTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
    / f"samp{SAMPLE_ID}"
    / "synth"
    / SYNTH_FILENAME
)

PROVIDER_CSV_PATH = PROJECT_ROOT / "data" / "clean_csvs" / "provider.csv"


# ---------------------------------------------------------------------
# Provider specialty mapping
# ---------------------------------------------------------------------

def load_provider_specialty_map(provider_csv: Path) -> Dict[str, str]:
    """
    Build providerID -> specialty_source_value mapping from provider.csv.
    Keys and values are strings, with any trailing '.0' stripped off IDs.
    """
    if not provider_csv.exists():
        print(f"[WARN] Provider CSV not found at {provider_csv}. "
              f"Provider specialty mapping will be empty.")
        return {}

    use_cols = ["provider_id", "specialty_source_value"]
    df = pd.read_csv(provider_csv, usecols=use_cols, low_memory=False)
    df = df.drop_duplicates(subset=["provider_id"])

    # Normalize provider_id to string without trailing '.0'
    df["provider_id"] = (
        df["provider_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    df["specialty_source_value"] = df["specialty_source_value"].astype(str)

    mapping = dict(zip(df["provider_id"], df["specialty_source_value"]))
    print(f"[INFO] Loaded provider specialty map with {len(mapping):,} providers.")
    return mapping



def attach_provider_specialty(
    df: pd.DataFrame,
    provider_map: Dict[str, str],
    provider_col: str = "providerID",
) -> pd.DataFrame:
    """
    Attach provider_specialty to a visit-level DataFrame using providerID.
    Does NOT drop rows; rows without mapping get 'Unknown'.
    """
    df = df.copy()
    if provider_col not in df.columns:
        raise ValueError(f"Column '{provider_col}' not found in DataFrame.")

    df["providerID_str"] = df[provider_col].astype(str)
    df["provider_specialty"] = df["providerID_str"].map(provider_map)
    df["provider_specialty"] = df["provider_specialty"].fillna("Unknown")
    df = df.drop(columns=["providerID_str"])
    return df


# ---------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------

def normalize_real_df(raw_real: pd.DataFrame,
                      provider_map: Dict[str, str]) -> pd.DataFrame:
    """
    Normalize REAL sampled_visit_facts.csv to a unified camelCase schema and
    attach provider_specialty from provider.csv via provider_map.
    """
    df = raw_real.copy()

    required = [
        "visit_occurrence_id",
        "person_id",
        "provider_id",
        "drug_concept_ids",
        "condition_concept_ids",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"REAL data missing required column '{col}'")

    # Rename IDs to camelCase
    df = df.rename(
        columns={
            "visit_occurrence_id": "visitID",
            "person_id": "personID",
            "provider_id": "providerID",
        }
    )

    # Normalize providerID (strip trailing '.0')
    df["providerID"] = (
        df["providerID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    # Attach provider_specialty using the same map as for SYNTH
    df = attach_provider_specialty(df, provider_map, provider_col="providerID")

    # Ensure n_drugs / n_conditions exist (not strictly needed for eval)
    for col in ["n_drugs", "n_conditions"]:
        if col not in df.columns:
            df[col] = np.nan

    return df[
        [
            "visitID",
            "personID",
            "providerID",
            "provider_specialty",
            "drug_concept_ids",
            "condition_concept_ids",
            "n_drugs",
            "n_conditions",
        ]
    ]


def normalize_synth_df(raw_synth: pd.DataFrame,
                       provider_map: Dict[str, str]) -> pd.DataFrame:
    """
    Normalize SYNTH rw_synth output to unified camelCase schema and
    attach provider_specialty via provider_map.
    """
    df = raw_synth.copy()

    required = [
        "visitID",
        "personID",
        "providerID",
        "drug_concept_ids",
        "condition_concept_ids",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"SYNTH data missing required column '{col}'")

    # Normalize providerID (strip trailing '.0')
    df["providerID"] = (
        df["providerID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    # Attach provider_specialty from provider_map
    df = attach_provider_specialty(df, provider_map, provider_col="providerID")

    # Compute counts from lists
    n_drugs, n_conds = compute_counts_from_lists(df)
    df["n_drugs"] = n_drugs
    df["n_conditions"] = n_conds

    return df[
        [
            "visitID",
            "personID",
            "providerID",
            "provider_specialty",
            "drug_concept_ids",
            "condition_concept_ids",
            "n_drugs",
            "n_conditions",
        ]
    ]


# ---------------------------------------------------------------------
# Concept list parsing and counts
# ---------------------------------------------------------------------

def parse_concept_list(value: Any) -> List[str]:
    """
    Parse a semicolon-separated concept ID string into a list of tokens.
    Handles 'NA' or empty strings as empty list.
    """
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "" or s.upper() == "NA":
        return []
    return [x for x in s.split(";") if x != ""]


def compute_counts_from_lists(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute #drugs and #conditions from semicolon-separated lists
    in columns 'drug_concept_ids' and 'condition_concept_ids'.
    """
    drug_counts = df["drug_concept_ids"].apply(parse_concept_list).apply(len).to_numpy()
    cond_counts = df["condition_concept_ids"].apply(parse_concept_list).apply(len).to_numpy()
    return drug_counts, cond_counts


# ---------------------------------------------------------------------
# Multi-hot feature builder (drugs + conditions)
# ---------------------------------------------------------------------

def build_multi_hot_features(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> Tuple[Any, Any, MultiLabelBinarizer]:
    """
    Build sparse multi-hot feature matrices for REAL and SYNTH rows
    based on the union vocabulary of drug + condition concept IDs.

    Tokens:
      d_<drug_concept_id>
      c_<condition_concept_id>
    """

    def row_tokens(row) -> List[str]:
        drugs = ["d_" + cid for cid in parse_concept_list(row["drug_concept_ids"])]
        conds = ["c_" + cid for cid in parse_concept_list(row["condition_concept_ids"])]
        return drugs + conds

    real_tokens = real_df.apply(row_tokens, axis=1).tolist()
    synth_tokens = synth_df.apply(row_tokens, axis=1).tolist()
    all_tokens = real_tokens + synth_tokens

    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(all_tokens)

    X_real = mlb.transform(real_tokens)
    X_synth = mlb.transform(synth_tokens)

    print(f"[INFO] Multi-hot vocab size: {len(mlb.classes_):,}")
    print(f"[INFO] X_real shape: {X_real.shape}, X_synth shape: {X_synth.shape}")

    return X_real, X_synth, mlb


# ---------------------------------------------------------------------
# 1) Wasserstein evaluation
# ---------------------------------------------------------------------

def evaluate_wasserstein(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    plot: bool = False,
) -> Dict[str, float]:
    """
    Wasserstein distances between REAL and SYNTH distributions of:
      - #drugs per visit
      - #conditions per visit
    """
    real_drugs, real_conds = compute_counts_from_lists(real_df)
    synth_drugs, synth_conds = compute_counts_from_lists(synth_df)

    w_drugs = wasserstein_distance(real_drugs, synth_drugs)
    w_conds = wasserstein_distance(real_conds, synth_conds)

    print("\n[WASSERSTEIN]")
    print(f"  W(#drugs):      {w_drugs:.4f}")
    print(f"  W(#conditions): {w_conds:.4f}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(real_drugs, bins=40, alpha=0.6, label="Real")
        axes[0].hist(synth_drugs, bins=40, alpha=0.6, label="Synthetic")
        axes[0].set_title("#Drugs per Visit")
        axes[0].set_xlabel("#Drugs")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        axes[1].hist(real_conds, bins=40, alpha=0.6, label="Real")
        axes[1].hist(synth_conds, bins=40, alpha=0.6, label="Synthetic")
        axes[1].set_title("#Conditions per Visit")
        axes[1].set_xlabel("#Conditions")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return {
        "wasserstein_drugs": float(w_drugs),
        "wasserstein_conditions": float(w_conds),
    }


# ---------------------------------------------------------------------
# 2) p-test: real vs synthetic classification
# ---------------------------------------------------------------------

def evaluate_p_test(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    X_real,
    X_synth,
) -> Dict[str, float]:
    """
    p-test: train a classifier to distinguish REAL vs SYNTH using
    multi-hot drug+condition features.

    Returns:
      - auc: ROC AUC on held-out test
      - accuracy: test accuracy
      - p_value: approximate p-value for acc > 0.5 under null
    """
    from scipy.sparse import vstack

    y_real = np.ones(real_df.shape[0], dtype=int)
    y_synth = np.zeros(synth_df.shape[0], dtype=int)

    X = vstack([X_real, X_synth])
    y = np.concatenate([y_real, y_synth])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # Approximate p-value vs 0.5 accuracy (binomial, normal approx)
    n = len(y_test)
    p0 = 0.5
    se = np.sqrt(p0 * (1 - p0) / n)
    z = (acc - p0) / se
    p_val = 2 * (1 - norm.cdf(abs(z)))

    print("\n[P-TEST: REAL vs SYNTH CLASSIFICATION]")
    print(f"  Test AUC:       {auc:.4f}")
    print(f"  Test Accuracy:  {acc:.4f}")
    print(f"  Approx p-value: {p_val:.4e}")

    return {
        "p_test_auc": float(auc),
        "p_test_accuracy": float(acc),
        "p_test_p_value": float(p_val),
    }


# ---------------------------------------------------------------------
# 3) Predictive utility: provider specialty prediction (TRTR / TSTR)
# ---------------------------------------------------------------------
def evaluate_provider_specialty_utility(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    X_real,
    X_synth,
    min_class_size: int = 5,
) -> Dict[str, Any]:
    """
    Predict provider_specialty from multi-hot drug+condition features.

    TRTR = Train on REAL, Test on REAL
    TSTR = Train on SYNTH, Test on REAL

    Rare specialties (< min_class_size in REAL) are removed for stability.
    """
    # Mask Unknown specialties
    real_mask = (real_df["provider_specialty"] != "Unknown")
    synth_mask = (synth_df["provider_specialty"] != "Unknown")

    real_mask_np = real_mask.to_numpy()
    synth_mask_np = synth_mask.to_numpy()

    if real_mask_np.sum() == 0 or synth_mask_np.sum() == 0:
        print("[WARN] No known provider_specialty found. Skipping utility evaluation.")
        return {}

    # Filter known-only
    y_real_raw = real_df.loc[real_mask, "provider_specialty"].astype(str)
    y_synth_raw = synth_df.loc[synth_mask, "provider_specialty"].astype(str)

    # -----------------------------
    # Remove rare REAL specialties
    # -----------------------------
    vc = y_real_raw.value_counts()
    keep_classes = vc[vc >= min_class_size].index.tolist()

    if len(keep_classes) < 2:
        print(f"[WARN] After filtering, only {len(keep_classes)} specialty class(es) remain. "
              f"Skipping utility evaluation.")
        return {}

    # Build masks for retained classes
    keep_real = y_real_raw.isin(keep_classes).to_numpy()
    keep_synth = y_synth_raw.isin(keep_classes).to_numpy()

    # Apply masks
    X_real_known = X_real[real_mask_np][keep_real]
    X_synth_known = X_synth[synth_mask_np][keep_synth]

    y_real = y_real_raw[keep_real].to_numpy()
    y_synth = y_synth_raw[keep_synth].to_numpy()

    # Encode labels on retained specialties only
    le = LabelEncoder()
    le.fit(np.concatenate([y_real, y_synth]))
    y_real_enc = le.transform(y_real)
    y_synth_enc = le.transform(y_synth)

    # -----------------------------
    # TRTR: train on REAL, test on REAL
    # -----------------------------
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_real_known,
        y_real_enc,
        test_size=0.2,
        stratify=y_real_enc,
        random_state=42,
    )

    clf_trtr = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        multi_class="ovr",
    )
    clf_trtr.fit(Xr_train, yr_train)
    yr_pred_trtr = clf_trtr.predict(Xr_test)

    macro_f1_trtr = f1_score(yr_test, yr_pred_trtr, average="macro")
    report_trtr = classification_report(
        yr_test,
        yr_pred_trtr,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    # -----------------------------
    # TSTR: train on SYNTH, test on REAL
    # -----------------------------
    clf_tstr = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        multi_class="ovr",
    )
    clf_tstr.fit(X_synth_known, y_synth_enc)
    yr_pred_tstr = clf_tstr.predict(Xr_test)

    macro_f1_tstr = f1_score(yr_test, yr_pred_tstr, average="macro")
    report_tstr = classification_report(
        yr_test,
        yr_pred_tstr,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    utility_ratio = macro_f1_tstr / macro_f1_trtr if macro_f1_trtr > 0 else np.nan

    print("\n[UTILITY: PROVIDER SPECIALTY PREDICTION]")
    print(f"  Retained provider specialties: {len(keep_classes)}")
    print(f"  Macro F1 (TRTR, real->real):   {macro_f1_trtr:.4f}")
    print(f"  Macro F1 (TSTR, synth->real):  {macro_f1_tstr:.4f}")
    print(f"  Utility ratio (TSTR/TRTR):     {utility_ratio:.4f}")

    per_class_f1_trtr = {
        cls: report_trtr[cls]["f1-score"]
        for cls in le.classes_
        if cls in report_trtr
    }
    per_class_f1_tstr = {
        cls: report_tstr[cls]["f1-score"]
        for cls in le.classes_
        if cls in report_tstr
    }

    return {
        "macro_f1_trtr": float(macro_f1_trtr),
        "macro_f1_tstr": float(macro_f1_tstr),
        "utility_ratio": float(utility_ratio),
        "per_class_f1_trtr": per_class_f1_trtr,
        "per_class_f1_tstr": per_class_f1_tstr,
        "retained_specialties": keep_classes,
    }



# ---------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------

def test_evaluations(
    real_facts_path: Path = REAL_FACTS_PATH,
    synth_path: Path = SYNTH_PATH,
    provider_csv_path: Path = PROVIDER_CSV_PATH,
    plot_wasserstein: bool = False,
) -> None:
    """
    Quick test function to verify that all evaluation components run.

    Steps:
      - Load REAL sampled_visit_facts.csv
      - Load SYNTH rw_synth CSV
      - Load provider specialty map
      - Normalize schemas to camelCase
      - Build multi-hot drug+condition features
      - Run: Wasserstein, p-test, provider-specialty utility
    """
    print(f"[LOAD] REAL visit facts: {real_facts_path}")
    raw_real = pd.read_csv(real_facts_path)

    print(f"[LOAD] SYNTH data: {synth_path}")
    raw_synth = pd.read_csv(synth_path)

    provider_map = load_provider_specialty_map(provider_csv_path)

    # Normalize schemas (both use the same provider map)
    real_df = normalize_real_df(raw_real, provider_map)
    synth_df = normalize_synth_df(raw_synth, provider_map)

    # Build multi-hot features
    X_real, X_synth, _ = build_multi_hot_features(real_df, synth_df)

    # 1) Wasserstein
    w_results = evaluate_wasserstein(real_df, synth_df, plot=plot_wasserstein)

    # 2) p-test
    p_results = evaluate_p_test(real_df, synth_df, X_real, X_synth)

    # 3) Utility (provider specialty)
    u_results = evaluate_provider_specialty_utility(real_df, synth_df, X_real, X_synth)

    print("\n[SUMMARY]")
    print("Wasserstein:", w_results)
    print("p-test:", p_results)
    print("Utility (headline):", {
        k: v for k, v in u_results.items() if not k.startswith("per_class")
    })

    print("\n[Per-class F1 (TRTR)]:")
    for cls, f1_val in u_results.get("per_class_f1_trtr", {}).items():
        print(f"  {cls}: {f1_val:.4f}")

    print("\n[Per-class F1 (TSTR)]:")
    for cls, f1_val in u_results.get("per_class_f1_tstr", {}).items():
        print(f"  {cls}: {f1_val:.4f}")

    print("\n[TEST COMPLETE]")


if __name__ == "__main__":
    # Run a quick test with defaults
    test_evaluations(plot_wasserstein=False)
