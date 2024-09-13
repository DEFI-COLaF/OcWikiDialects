"""Script to convert LIDoc CSV format
to the input format to train FastText classifiers"""
import argparse

import pandas as pd
from tqdm.auto import tqdm

from ocdi.labels import LABEL_SCHEME_TO_MAPPER, FASTTEXT_PREFIX


argparser = argparse.ArgumentParser()
argparser.add_argument("csv_path",
                       help="Path to the CSV file to convert")
argparser.add_argument("label_scheme",
                       choices=list(LABEL_SCHEME_TO_MAPPER),
                       help="Label scheme used in the CSV, to be converted to the FastText scheme")
argparser.add_argument("--col_label", default="dialect",
                       help="Name of the column containing the dialect label")
argparser.add_argument("--col_text", default="text",
                       help="Name of the column containing the text to use")
args = argparser.parse_args()

col_text = args.col_text
col_label = args.col_label

# Load CSV
df = pd.read_csv(
    args.csv_path,
    header=0,
    na_filter=False,
    usecols=[col_text, col_label]
)

# Convert labels to FastText format
scheme_mapper = LABEL_SCHEME_TO_MAPPER[args.label_scheme]
df[col_label] = df[col_label].apply(lambda label: FASTTEXT_PREFIX + scheme_mapper[label])

# Print each line starting with the label
for _, row in tqdm(df.iterrows(), desc="Samples converted", total=len(df)):
    line = f"{row[col_label]} {row[col_text]}"
    print(line)
