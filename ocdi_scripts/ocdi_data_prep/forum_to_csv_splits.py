"""Script to convert JSONL ForumOccitania splits into CSV ones for the ocDI task"""

import argparse
import glob
import os

import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument("in_folder", help="Path to the folder with the JSONL train/dev/test splits")
argparser.add_argument("out_prefix", help="Prefix to the output dataset splits")

args = argparser.parse_args()

# CSV export_options
export_options = {"header": True, "index": False}

# Load splits
splits = {"train": None, "dev": None, "test": None}
for split_name in splits:
    # Get corresponding file
    file = glob.glob(os.path.join(args.in_folder, f"*-{split_name}.jsonl"))
    assert file, f"Split {split_name} not found in folder {args.in_folder}"
    file = file[0]
    # Load data
    df_split = pd.read_json(file, orient="records", lines=True)
    # Reduce columns
    df_split = df_split[["id", "text", "author_dialect"]]
    # Rename columns
    df_split.rename(columns={"author_dialect": "dialect"}, inplace=True)
    # Preprocess text: remove newlines
    df_split["text"] = df_split["text"].apply(lambda text: text.replace("\n", " "))
    # Save to file
    path_split = args.out_prefix + split_name + ".csv"
    df_split.to_csv(path_split, **export_options)
    splits[split_name] = df_split

# Create full split
df_full = pd.concat(splits.values(), axis=0, ignore_index=True)
path_full = args.out_prefix + "full.csv"
df_full.to_csv(path_full, **export_options)

print(f"Saved all splits (incl. full) to {path_full} (or train/dev/test)")
