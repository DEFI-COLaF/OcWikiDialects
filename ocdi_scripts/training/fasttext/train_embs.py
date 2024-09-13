"""Script to train unsupervised FastText embeddings"""
import argparse

import fasttext

argparser = argparse.ArgumentParser()
argparser.add_argument("input", help="Path to the input TXT file")
argparser.add_argument("out", default="fasttext.bin",
                       help="Path where to save the trained embedding model")

args = argparser.parse_args()

hyperparams = {
    "model": "skipgram",  # or 'cbow'
    "dim": 300,  # embedding size
    "lr": 0.05,
    "epoch": 5,
    "minn": 3,  # char n-grams (subword info)
    "maxn": 6,
    "thread": 40
}

print(f"Run unsupervised training based on dataset at {args.input}")
print(f"Hyperparameters: {hyperparams}")
model = fasttext.train_unsupervised(args.input, **hyperparams)
print("Training successful")

model.save_model(args.out)
print(f"Model saved to {args.out}")
