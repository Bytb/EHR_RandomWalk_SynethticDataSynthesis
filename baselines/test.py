import pandas as pd
from pathlib import Path
import textwrap

def analyze_csv(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return

    print(f"[INFO] Loading CSV from: {path.resolve()}")
    df = pd.read_csv(path)
    n_rows, n_cols = df.shape
    print(f"\n[SHAPE] Rows: {n_rows:,}  |  Columns: {n_cols:,}\n")

    summary_rows = []

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        non_null = s.notna().sum()
        nulls = s.isna().sum()
        nunique = s.nunique(dropna=True)
        pct_unique = (nunique / non_null * 100) if non_null > 0 else 0.0

        min_val = max_val = None
        if pd.api.types.is_numeric_dtype(s):
            min_val = s.min()
            max_val = s.max()

        examples = list(s.dropna().unique()[:5])
        examples_str = ", ".join(map(str, examples))

        summary_rows.append({
            "column": col,
            "dtype": dtype,
            "non_null": int(non_null),
            "nulls": int(nulls),
            "nunique": int(nunique),
            "pct_unique": round(pct_unique, 2),
            "min": min_val,
            "max": max_val,
            "examples": examples_str,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="pct_unique", ascending=False)

    with pd.option_context("display.max_rows", None,
                           "display.max_colwidth", 80,
                           "display.width", 200):
        print("[COLUMN SUMMARY]\n")
        print(summary_df[[
            "column", "dtype", "non_null", "nulls",
            "nunique", "pct_unique", "min", "max"
        ]])

    print("\n[EXAMPLE VALUES BY COLUMN]\n")
    for row in summary_rows:
        col = row["column"]
        ex_str = textwrap.shorten(row["examples"], width=120, placeholder="...")
        print(f"- {col}: {ex_str}")


if __name__ == "__main__":
    csv_path = r"C:\Users\Caleb\PycharmProjects\EHR_RandomWalk_SynethticDataSynthesis\data\processed\visit_fact_table.csv"
    analyze_csv(csv_path)
