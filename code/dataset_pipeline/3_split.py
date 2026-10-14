"""
CEFR Dataset Splitter
=====================
Splits a standardized CEFR CSV into train/test sets, stratified by
CEFR level so that every level is equally represented in both splits.
Optionally filters to a single language before splitting.
"""


import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      required=True,                         help="Path to the CSV file")
    parser.add_argument("--split",    default="80/20", choices=["80/20", "70/30"], help="Split ratio")
    parser.add_argument("--out",      required=True,                         help="Output prefix, e.g. data/cefr")
    parser.add_argument("--language", default=None,                          help="Optional: only one language, e.g. 'en'")
    args = parser.parse_args()

    test_size = 0.2 if args.split == "80/20" else 0.3

    df = pd.read_csv(args.csv)

    if args.language:
        df = df[df["lang"] == args.language].reset_index(drop=True)
        print(f"Language filter '{args.language}': {len(df)} rows")

    # Split per level individually and then merge
    # -> guarantees the same count per level in both train AND test
    n_per_level = df["cefr_level"].value_counts().min()  # smallest level as upper bound
    n_test  = int(n_per_level * test_size)
    n_train = n_per_level - n_test

    train_parts, test_parts = [], []
    for level, group in df.groupby("cefr_level"):
        group = group.sample(n=n_per_level, random_state=42)  # truncate to n_per_level
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:n_train + n_test])  # ← correct

    df_train = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test  = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    out_train = f"{args.out}_train.csv"
    out_test  = f"{args.out}_test.csv"
    df_train.to_csv(out_train, index=False)
    df_test.to_csv(out_test,   index=False)

    print(f"Split {args.split}: train={len(df_train)}, test={len(df_test)}")
    print(f"Saved: {out_train}")
    print(f"Saved: {out_test}")
    print(f"\nLevel distribution train:\n{df_train['cefr_level'].value_counts().sort_index()}")
    print(f"\nLevel distribution test:\n{df_test['cefr_level'].value_counts().sort_index()}")

if __name__ == "__main__":
    main()