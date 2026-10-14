"""
CEFR Dataset Balancer
=====================
Truncates all CEFR levels to the size of the smallest level (undersampling).
The rows kept are chosen randomly.
"""

from pathlib import Path
import pandas as pd


def balance_cefr_dataset(
    input_path: str,
    output_path: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Balances all CEFR levels to match the smallest level count.

    Args:
        input_path: Path to the standardized CSV.
        output_path: Target path for the balanced CSV.
        random_seed: Seed for the random selection (reproducibility).

    Returns:
        The balanced DataFrame.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    df = pd.read_csv(p)

    if "cefr_level" not in df.columns:
        raise ValueError("Column 'cefr_level' not found. Is this a standardized CSV?")

    level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    counts = df["cefr_level"].value_counts()
    min_count = counts.min()
    min_level = counts.idxmin()

    # Print distribution before
    total_before = len(df)
    print(f"\nBefore: {p.name}")
    print(f"{'─' * 40}")
    for level in level_order:
        count = counts.get(level, 0)
        bar = "█" * int(count / total_before * 30) if total_before > 0 else ""
        print(f"  {level:>3}  {count:>6}  ({count / total_before * 100:5.1f}%)  {bar}")
    print(f"{'─' * 40}")
    print(f"  {'Total':>3}  {total_before:>6}")
    print(f"\n  → Smallest level: {min_level} with {min_count} entries")
    print(f"  → All levels will be truncated to {min_count}\n")

    # Balance: randomly sample each level down to min_count
    balanced_parts = []
    for level in level_order:
        level_df = df[df["cefr_level"] == level]
        if len(level_df) == 0:
            continue
        sampled = level_df.sample(n=min_count, random_state=random_seed)
        balanced_parts.append(sampled)

    df_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Print distribution after
    counts_after = df_balanced["cefr_level"].value_counts()
    total_after = len(df_balanced)
    print(f"After: {Path(output_path).name}")
    print(f"{'─' * 40}")
    for level in level_order:
        count = counts_after.get(level, 0)
        bar = "█" * int(count / total_after * 30) if total_after > 0 else ""
        print(f"  {level:>3}  {count:>6}  ({count / total_after * 100:5.1f}%)  {bar}")
    print(f"{'─' * 40}")
    print(f"  {'Total':>3}  {total_after:>6}\n")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(out, index=False)
    print(f"[OK] Balanced CSV saved -> {out}")

    return df_balanced


if __name__ == "__main__":
    balance_cefr_dataset(
        input_path="/data/dataset_input.csv",
        output_path="/data/dataset_input-balanced.csv",
        random_seed=42,     # change for a different random selection
    )