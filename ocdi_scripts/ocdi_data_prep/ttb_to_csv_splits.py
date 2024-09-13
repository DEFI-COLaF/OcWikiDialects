"""Script to convert UD conllu files into CSV with the same train/dev/test + dialect labels"""

import argparse
import os
import re

import conllu
import pandas as pd


def get_dialect_from_sentid(sentid):
    m = re.search(r"_(?P<dia>[a-z]+)\.conllu\.s\d+", sentid)
    assert m, f"Dialect not found in sent_id {sentid}"
    return m["dia"]


argparser = argparse.ArgumentParser()
argparser.add_argument("in_folder", help="Path to the parent folder "
                                        "where all CSV files for the LoCongres dataset are stored")
argparser.add_argument("out_prefix", help="Prefix to the output dataset splits")
args = argparser.parse_args()

# CSV export_options
export_options = {"header": True, "index": False}

# Load splits
splits = {"train": None, "dev": None, "test": None}
for split_name in splits.keys():
    # Get corresponding file
    path = os.path.join(args.in_folder, f"oc_ttb-ud-{split_name}.conllu")
    assert os.path.exists(path), f"File not found: {path}"
    # Load data
    labels = []
    sents = []
    with open(path) as f:
        for conllu_sent in conllu.parse_incr(f):
            sent = conllu_sent.metadata["text"]
            sents.append(sent)
            dialect = get_dialect_from_sentid(conllu_sent.metadata["sent_id"])
            labels.append(dialect)
    # Make DataFrame and save to CSV
    df_split = pd.DataFrame({"text": sents, "dialect": labels})
    splits[split_name] = df_split
    path_split = args.out_prefix + split_name + ".csv"
    df_split.to_csv(path_split, **export_options)

# Create full CSV
df_full = pd.concat(splits.values(), axis=0, ignore_index=True)
path_full = args.out_prefix + "full.csv"
df_full.to_csv(path_full, **export_options)

print(f"Saved all splits (incl. full) to {path_full} (or train/dev/test)")
