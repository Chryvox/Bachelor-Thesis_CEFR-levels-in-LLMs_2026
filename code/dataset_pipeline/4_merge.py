"""
CEFR Dataset Merger
===================
Merges two standardized CEFR CSV files.
"""

from pathlib import Path
import pandas as pd


STANDARD_COLUMNS = [
    "title",        # Title / unique ID of the text
    "lang",         # Language, e.g. "en", "de"
    "source_name",  # Dataset name / source
    "format",       # Text format, e.g. "essay", "dialogue", "news"
    "category",     # Category, e.g. "grammar", "vocabulary"
    "cefr_level",   # A1 – C2
    "license",      # License
    "text",         # The actual text
]

LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def print_stats(df: pd.DataFrame, label: str) -> None:
    counts = df["cefr_level"].value_counts()
    total = len(df)
    print(f"\n{label}")
    print(f"{'─' * 40}")
    for level in LEVEL_ORDER:
        count = counts.get(level, 0)
        bar = "█" * int(count / total * 30) if total > 0 else ""
        print(f"  {level:>3}  {count:>6}  ({count / total * 100:5.1f}%)  {bar}")
    print(f"{'─' * 40}")
    print(f"  {'Total':>3}  {total:>6}")


def merge_datasets(
    input_path_a: str,
    input_path_b: str,
    output_path: str,
    reset_text_id: bool = True,
) -> pd.DataFrame:
    """Merges two standardized CEFR CSVs.

    Args:
        input_path_a:  Path to the first CSV.
        input_path_b:  Path to the second CSV.
        output_path:   Target path for the merged CSV.
        reset_text_id: If True, text_id is renumbered (0, 1, 2 …)
                       to avoid collisions between the datasets.

    Returns:
        The merged DataFrame.
    """
    for path in [input_path_a, input_path_b]:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    df_a = pd.read_csv(input_path_a)
    df_b = pd.read_csv(input_path_b)

    for name, df in [(input_path_a, df_a), (input_path_b, df_b)]:
        missing = [col for col in STANDARD_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    # Show individual distributions
    print_stats(df_a, label=f"Dataset A: {Path(input_path_a).name}")
    print_stats(df_b, label=f"Dataset B: {Path(input_path_b).name}")

    # Merge
    df_merged = pd.concat([df_a, df_b], ignore_index=True)

    if reset_text_id:
        df_merged["text_id"] = range(len(df_merged))

    # Show merged distribution
    print_stats(df_merged, label=f"Merged: {Path(output_path).name}")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out, index=False)
    print(f"\n[OK] Merged CSV saved -> {out}\n")

    return df_merged


if __name__ == "__main__":
    merge_datasets(
        input_path_a="/data/dataset_a.csv",
        input_path_b="/data/dataset_b.csv",
        output_path="/data/dataset_a+b.csv",
        reset_text_id=True,
    )