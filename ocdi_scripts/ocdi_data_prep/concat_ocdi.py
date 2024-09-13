"""Script to concatenate individual subcorpora and their train/dev/test splits, for the ocDI task."""

import argparse
import os.path

import pandas as pd

from ocdi.labels import SOURCE_TO_SCHEME, LABEL_SCHEME_TO_MAPPER


def get_dataset_name_from_path(path):
    basename = os.path.basename(path)
    for source in SOURCE_TO_SCHEME:
        if source in basename:
            return source


argparser = argparse.ArgumentParser()
argparser.add_argument("-s", "--sources", nargs="+",
                       help="Path prefixes to the ocDI source files to concatenate")
argparser.add_argument("-o", "--out_prefix", help="Output path prefix")
args = argparser.parse_args()

splits = ["train", "dev", "test"]

# CSV export_options
export_options = {"header": True, "index": False}

splits_dfs = {"train": None, "dev": None, "test": None}
for split_name in splits:
    df_split = None
    for source in args.sources:
        path_source = source + split_name + ".csv"
        # Get dataset name
        dataset_name = get_dataset_name_from_path(path_source)
        assert dataset_name, f"Dataset name not found for source {source}"
        # Load CSV
        df_source = pd.read_csv(
            path_source,
            header=0,
            index_col=False,
            usecols=["text", "dialect"]
        )
        # Normalize model labels
        label_scheme = LABEL_SCHEME_TO_MAPPER[SOURCE_TO_SCHEME[dataset_name]]
        df_source["dialect"] = df_source["dialect"].map(label_scheme)
        if df_split is None:
            df_split = df_source
        else:
            df_split = pd.concat([df_split, df_source], axis=0, ignore_index=True)
    # Save split to file
    path_split = args.out_prefix + split_name + ".csv"
    df_split.to_csv(path_split, **export_options)
    splits_dfs[split_name] = df_split

# Create full split
df_full = pd.concat(splits_dfs.values(), axis=0, ignore_index=True)
path_full = args.out_prefix + "full.csv"
df_full.to_csv(path_full, **export_options)

print(f"Saved all concatenated splits (incl. full) to {path_full} (or train/dev/test)")
