from pathlib import Path
import pandas as pd

def inspect_csv(file_path: Path) -> None:
    """Load a CSV and print its basic summary."""
    print("=" * 60)
    print(f"FILE: {file_path.name}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        print("=" * 60)
        return

    # Print shape
    print(f"ROWS: {df.shape[0]:,}   COLUMNS: {df.shape[1]:,}")

    # Print column names
    print("\nCOLUMNS:")
    for col in df.columns:
        print(f"  - {col}")

    # Print head
    print("\nHEAD:")
    print(df.head())

    print("=" * 60)
    print("\n")


def run_inspection():
    """Find all CSV files under ../data/raw_csvs and inspect them."""
    # Folder path relative to this script's location
    base_dir = Path(__file__).resolve().parent
    raw_csv_folder = base_dir.parent / "data" / "raw_csvs"

    if not raw_csv_folder.exists():
        print(f"ERROR: Folder not found -> {raw_csv_folder}")
        return

    csv_files = sorted(raw_csv_folder.glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {raw_csv_folder}:\n")
    for f in csv_files:
        print(f" - {f.name}")
    print("\n")

    for csv_file in csv_files:
        inspect_csv(csv_file)


if __name__ == "__main__":
    run_inspection()
