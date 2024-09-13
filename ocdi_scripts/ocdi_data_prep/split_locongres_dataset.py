"""Script to split the dataset from Lo Congres organization into train/dev/test splits
(news / websites / SoftwaresOccitanTranslations)"""
import argparse
import glob
import os
import re

import pandas as pd

from ocdi.split import split_train_dev_test

argparser = argparse.ArgumentParser()
argparser.add_argument("in_folder", help="Path to the parent folder "
                                        "where all CSV files for the LoCongres dataset are stored")
argparser.add_argument("out_prefix", help="Prefix to the output dataset splits")
args = argparser.parse_args()


def get_dialect_label(full_label):
    m = re.search("^oc-(?P<dia>[a-z]+)-gr[a-z]+$", full_label.lower())
    assert m, f"Dialect label not found in LoCongres label {full_label}"
    return m["dia"]


# Load all CSV files of the dataset
in_folder = args.in_folder
assert os.path.exists(in_folder), f"Input folder does not exist: {in_folder}"
print(f"Load LoCongres dataset files from {in_folder}")
in_files = sorted(glob.glob("oc-*.csv", root_dir=in_folder))

dia_dfs = []
for filename in in_files:
    path = os.path.join(in_folder, filename)
    df = pd.read_csv(
        path,
        sep="§",
        engine="python",
        header=None,
        usecols=[0, 1],
        na_filter=False,
        quoting=3  # csv.QUOTE_NONE
    )
    # Rename columns
    df.rename(columns={0: "text", 1: "dialect"}, inplace=True)
    dia_dfs.append(df)

# Concatenate all
df = pd.concat(dia_dfs, axis=0, ignore_index=True)

# Reformat dialect labels (also enables merging spelling variants as we ignore this aspect in our experiments)
df["dialect"] = df["dialect"].apply(get_dialect_label)

# Split the full dataset into train/dev/test
splits = split_train_dev_test(df)

# Store full dataset and each split
export_options = {"header": True}
path_full = args.out_prefix + "full.csv"
df.to_csv(path_full, **export_options)
for split_name, df_split in splits.items():
    path_split = args.out_prefix + split_name + ".csv"
    df_split.to_csv(path_split, **export_options)

print(f"Saved all splits (incl. full) to {path_full} (or train/dev/test)")
