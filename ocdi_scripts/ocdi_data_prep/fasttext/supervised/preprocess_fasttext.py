"""Script to produce the preprocessed input files for FastText experiments"""

import argparse
import glob
import os
import warnings

import yaml

from tqdm import tqdm

from ocdi.svm.preprocessing import preprocess

argparser = argparse.ArgumentParser()
argparser.add_argument("dataset_name",
                       help="Name of the dataset (main folder). File paths should follow the pattern "
                       "DNAME/DNAME_SPLITSUFFIX/DNAME_[train/dev/test]-fasttext.txt")
argparser.add_argument("splitsuffix", help="Name of the split (e.g. splitv1)")
argparser.add_argument("preprocess_config", help="Path to a YAML file with the preprocessing options")
args = argparser.parse_args()


# Load config
with open(args.preprocess_config) as f:
    config = yaml.safe_load(f)
prep_name = config["prep_name"]
prep_options = config["prep_options"]

# Get files to transform
in_suffix = "-fasttext"
in_extension = ".txt"

files = glob.glob(os.path.join(
    args.dataset_name,
    f"{args.dataset_name}_{args.splitsuffix}",
    f"{args.dataset_name}_*{in_suffix}{in_extension}")
)
assert files, f"No files found. args: {args}"

# Transform files one by one
for path in files:
    print(f"Preprocessing {path}")
    path_out = path.replace(in_suffix, f"{in_suffix}-{prep_name}")
    with open(path) as f_in:
        in_lines = [line.strip() for line in f_in]
    with open(path_out, "w") as f_out:
        for line_in in tqdm(in_lines, desc="Lines preprocessed"):
            try:
                label, text = line_in.split(maxsplit=1)
            except ValueError:
                warnings.warn(f"Skipping malformed line: {line_in}")
            text_preprocessed = preprocess(text, **prep_options)
            f_out.write(f"{label} {text_preprocessed}\n")

print("All done !")
