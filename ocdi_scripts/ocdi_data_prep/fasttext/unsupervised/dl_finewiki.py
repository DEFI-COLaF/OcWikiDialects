"""Script to download FineWiki (to TXT)"""
import re

from datasets import load_dataset
from tqdm.auto import tqdm

FINEWIKI_VERSION = "8bd13e72e6a002407649b3e898535f42ceb1aeb9"  # Commit SHA from the HuggingFace repository

ds = load_dataset(
    "HuggingFaceFW/finewiki",
    revision=FINEWIKI_VERSION,
    name="oc",
    split="train",
    streaming=True
)

for sample in tqdm(ds, "Samples exported"):
    sample_text = sample["text"]
    sample_text = re.sub(r"\s+", " ", sample_text)
    print(sample_text)
