"""Script to merge and split LoCongresNews and LoCongresWebsites datasets"""

import argparse
import glob
import os
import re

import pandas as pd

from ocdi.split import split_train_dev_test

argparser = argparse.ArgumentParser()
argparser.add_argument("folder_news",
                       help="Path to the parent folder "
                       "where all CSV files for LoCongresNews dataset are stored")
argparser.add_argument("folder_web",
                       help="Path to the parent folder "
                       "where all CSV files for LoCongresWebsites dataset are stored")
argparser.add_argument("out_prefix", help="Prefix to the output dataset splits")
args = argparser.parse_args()


def get_dialect_label(full_label):
    m = re.search("^oc-(?P<dia>[a-z]+)-gr[a-z]+$", full_label.lower())
    assert m, f"Dialect label not found in LoCongres label {full_label}"
    return m["dia"]


def load_dataset_folder(folder):
    # Load all CSV files of the dataset
    assert os.path.exists(folder), f"Input folder does not exist: {folder}"
    print(f"Load LoCongres dataset files from {folder}")
    in_files = sorted(glob.glob("oc-*.csv", root_dir=folder))

    dia_dfs = []
    for filename in in_files:
        path = os.path.join(folder, filename)
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
        # Reformat dialect labels (also enables merging spelling variants as we ignore this aspect in our experiments)
        df["dialect"] = df["dialect"].apply(get_dialect_label)
        # Store
        dia_dfs.append(df)

    # Concatenate all
    df = pd.concat(dia_dfs, axis=0, ignore_index=True)
    return df


# Load LoCongresNews and LoCongresWebsites dataset folders
df_news = load_dataset_folder(args.folder_news)
df_web = load_dataset_folder(args.folder_web)

# Merge both datasets
df_merged = pd.concat([df_news, df_web], axis=0, ignore_index=True)

# Remove duplicates
df_merged.drop_duplicates(keep="first", inplace=True, ignore_index=True)

# Remove samples with cisaup label (only 20 samples)
df_merged = df_merged[df_merged["dialect"] != "cisaup"].copy()

# Split the full dataset into train/dev/test
splits = split_train_dev_test(df_merged)

# Store full dataset and each split
export_options = {"header": True}
path_full = args.out_prefix + "full.csv"
df_merged.to_csv(path_full, **export_options)
for split_name, df_split in splits.items():
    path_split = args.out_prefix + split_name + ".csv"
    df_split.to_csv(path_split, **export_options)

print(f"Saved all splits (incl. full) to {path_full} (or train/dev/test)")


