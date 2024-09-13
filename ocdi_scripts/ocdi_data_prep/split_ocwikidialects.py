"""Script to split the OcWikiDialects dataset into train/dev/test splits"""

import argparse
import re

import pandas as pd

from ocdi.split import split_train_dev_test


def normalize_whitespace(text):
    """Converts all newlines to single whitespaces (+ handle potential duplicated whitespaces)"""
    text = re.sub(r"\s+", " ", text)
    return text


def expand_df_col(df, col_name):
    df_expanded = df.explode(col_name, ignore_index=True)
    return df_expanded


argparser = argparse.ArgumentParser()
argparser.add_argument("path_jsonl", help="Path to JSONL OcWikiDialects dataset")
argparser.add_argument("out_prefix", help="Prefix to the output dataset splits")
argparser.add_argument("--unit", choices=["segment", "paragraph", "article"], default="paragraph",
                       help="Text unit to use for each article")
args = argparser.parse_args()

unit = args.unit

# Load JSONL
df = pd.read_json(args.path_jsonl, orient="records", lines=True)
df.rename(columns={"id": "article_id"}, inplace=True)
# Reduce columns
cols = ["article_id", "dialect", "text_finewiki"]
if unit in ["segment", "paragraph"]:
    cols.append(unit+"s")
df = df[cols]

# Remove samples from dialects aguiainés and marchés
df = df[~df["dialect"].isin(["aguiainés", "marchés"])]

# Split at article-level, by taking into account the size once expanded
print("Split articles between train/dev/test")
df["n_units"] = df[unit+"s"].apply(lambda pars: len(pars) if pars else 0)
splits_article = split_train_dev_test(df, col_text="text_finewiki", unit_multiplier_col="n_units")

# Expand articles at the chosen unit level
print(f"Expand at level {unit}")
for split_name, df_split in splits_article.items():
    if unit == "article":
        df_split["text"] = df_split["text_finewiki"].apply(normalize_whitespace)
    elif unit == "paragraph":
        df_split = expand_df_col(df_split, "paragraphs")
        df_split["text"] = df_split["paragraphs"].apply(normalize_whitespace)
        df_split.drop(columns="paragraphs", inplace=True)
    elif unit == "segment":
        df_split = expand_df_col(df, "segments")
        df_split["text"] = df_split["segments"].apply(normalize_whitespace)
        df_split.drop(columns="segments", inplace=True)
    else:
        raise ValueError(f"Unit not recognized: {unit}")
    df_split.drop(columns=["text_finewiki", "n_units"], inplace=True)
    splits_article[split_name] = df_split

# Concatenate to get df_full
df_full = pd.concat(splits_article.values(), axis=0, ignore_index=True)

# Save splits to files (incl. full dataset)
print(f"Save splits to files")
export_options = {"header": True, "index": False}  # Make index=False but there will still be
path_full = args.out_prefix + "full.csv"
df_full.to_csv(path_full, **export_options)
for split_name, df_split in splits_article.items():
    path_split = args.out_prefix + split_name + ".csv"
    df_split.to_csv(path_split, **export_options)

print(f"Saved all splits (incl. full) to {path_full} (or train/dev/test)")
