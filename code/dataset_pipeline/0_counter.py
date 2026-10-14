"""
CEFR Level Counter
==================
Reads a standardized CSV and shows the distribution of CEFR levels
as well as statistics on text length (characters and words).
"""

from pathlib import Path
import pandas as pd


def count_cefr_levels(csv_path: str) -> None:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    df = pd.read_csv(p)

    if "cefr_level" not in df.columns:
        raise ValueError("Column 'cefr_level' not found. Is this a standardized CSV?")

    # Always all 6 levels in correct order, even if count = 0
    level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    counts = df["cefr_level"].value_counts()
    total = len(df)

    print(f"\nFile: {p.name}")
    print(f"{'─' * 40}")
    for level in level_order:
        count = counts.get(level, 0)
        bar = "█" * int(count / total * 30) if total > 0 else ""
        print(f"  {level:>3}  {count:>6}  ({count / total * 100:5.1f}%)  {bar}")
    print(f"{'─' * 40}")
    print(f"  {'Total':>3}  {total:>6}\n")

    # Append text length statistics
    print_text_stats(df)


def print_text_stats(df: pd.DataFrame) -> None:
    if "text" not in df.columns:
        raise ValueError("Column 'text' not found. Is this a standardized CSV?")

    # Treat NaN values as empty strings
    texts = df["text"].fillna("").astype(str)

    char_counts = texts.str.len()
    word_counts = texts.str.split().str.len()

    def _fmt_row(label: str, series: pd.Series) -> str:
        return (
            f"  {label:<12}"
            f"  min={series.min():>8.0f}"
            f"  max={series.max():>8.0f}"
            f"  Ø={series.mean():>8.1f}"
            f"  median={series.median():>8.1f}"
        )

    print("Text length statistics")
    print(f"{'─' * 60}")
    print(_fmt_row("Characters", char_counts))
    print(_fmt_row("Words", word_counts))
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    count_cefr_levels("/data/dataset_input-std.csv")

    # Multiple CSVs at once:
    #for path in Path("/data/datasets/_mix/en+de+hi").glob("*.csv"):
    #     count_cefr_levels(path)