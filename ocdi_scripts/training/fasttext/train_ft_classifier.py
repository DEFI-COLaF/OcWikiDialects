"""Script to train FastText classifiers with multiple hyperparams combinations,
including based on pretrained FastText embeddings"""

import argparse
import itertools
import os
import yaml

import fasttext
import tqdm
from slugify import slugify


def get_vectors(emb_model, out_vectors_path):
    words = emb_model.get_words()  # all words in the dictionary
    dim = emb_model.get_dimension()  # vector dimension

    with open(out_vectors_path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {dim}\n")  # optional header
        for w in tqdm.tqdm(words):
            vec = emb_model.get_word_vector(w)
            vec_str = " ".join(map(str, vec))
            f.write(f"{w} {vec_str}\n")


def params_to_filename(params: dict, prefix: str = "", suffix: str = ".bin") -> str:
    parts = []
    for key, value in sorted(params.items()):
        safe_key = slugify(str(key), lowercase=False, separator="")[:4]
        # safe_key = str(key).replace(" ", "_")[:4]
        safe_val = str(value)
        if os.path.exists(safe_val):
            safe_val = os.path.splitext(os.path.basename(safe_val))[0]
        safe_val = slugify(safe_val, lowercase=False, separator="")
        parts.append(f"{safe_key}-{safe_val}")
    core = "_".join(parts)
    return f"{prefix}{core}{suffix}"


# CLI arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("path_train_data", default="fasttext_lang_train.txt",
                       help="Path to the labelled training data")
argparser.add_argument("config",
                       help="Path to a YAML config file with model params and preprocessing params")
argparser.add_argument("--out_prefix", default="models/classifier_",
                       help="Path prefix to the saved models (folders should exist)")
args = argparser.parse_args()


# Load config
if args.config:
    with open(args.config) as f:
        config = yaml.safe_load(f)
else:
    config = {}


# Extract vectors from the embeddings model
emb_model_path = config["emb_model_path"]
vec_path = config["vectors_path"]
if not os.path.exists(vec_path):
    model = fasttext.load_model(emb_model_path)
    get_vectors(model, vec_path)


# Fixed hyperparameters
hyperparams = config.get("model_params", {})
print(f"Fixed hyperparameters: {hyperparams}\n")

# Hyperparameters to optimize
ngrams = config.get("ngrams", [1])
pretrained = [vec_path]  # [args.vectors, ""]
print(f"Hyperparameters to optimize:\n\t- ngrams {ngrams}\n\t- pretrainedVectors {pretrained}\n")

# Create all possible inter
moving_hyperparams = [
    dict(zip(["wordNgrams", "pretrainedVectors"], combination))
    for combination in itertools.product(ngrams, pretrained)
]

for moving_config in moving_hyperparams:
    out_path_model = params_to_filename(moving_config, prefix=args.out_prefix)
    if os.path.exists(out_path_model):
        print(f"Model found, skipping: {out_path_model}")
        continue
    print(f"Train with params {moving_config}")
    model = fasttext.train_supervised(input=args.path_train_data, **hyperparams, **moving_config)
    # Save model
    model_folder = os.path.dirname(out_path_model)
    if model_folder:
        os.makedirs(model_folder, exist_ok=True)
    model.save_model(out_path_model)
    print(f"Saved model {out_path_model}")
