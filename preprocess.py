"""
Preprocess the HS3D dataset into donor/acceptor train/test splits.

Reads a tab-separated file with columns including 'classification' (one of
EI_true, EI_false, IE_true, IE_false) and 'seq' (140-nt sequence), then writes
four output files:
    - donors_train.txt
    - donors_test.txt
    - acceptors_train.txt
    - acceptors_test.txt

Each split is stratified on the classification column so that the class
distribution is preserved.

Usage:
    python preprocess.py --input data/allSeqs.txt --output-dir data/
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.25
RANDOM_STATE = 24


def split_and_save(df, label_prefix, output_dir):
    """Split a dataframe into train/test and save both as TSV."""
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["classification"],
        random_state=RANDOM_STATE,
    )
    train_path = os.path.join(output_dir, f"{label_prefix}_train.txt")
    test_path = os.path.join(output_dir, f"{label_prefix}_test.txt")
    train.to_csv(train_path, index=False, sep="\t")
    test.to_csv(test_path, index=False, sep="\t")
    print(f"  Wrote {train_path} ({len(train)} rows)")
    print(f"  Wrote {test_path} ({len(test)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the combined HS3D sequences file (TSV with 'seq' and 'classification' columns).",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write the four split files (default: ./data).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading {args.input}...")
    data = pd.read_table(
        args.input,
        sep="\t",
        dtype={"index": int, "exon_num": str, "intron_num": str},
    )

    donors = data[data["classification"].isin(["EI_true", "EI_false"])]
    acceptors = data[data["classification"].isin(["IE_true", "IE_false"])]

    print(f"Splitting donors ({len(donors)} rows)...")
    split_and_save(donors, "donors", args.output_dir)

    print(f"Splitting acceptors ({len(acceptors)} rows)...")
    split_and_save(acceptors, "acceptors", args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()