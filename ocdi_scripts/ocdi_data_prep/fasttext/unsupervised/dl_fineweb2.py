"""Script to download FineWeb-2 (to TXT)"""
import re

from datasets import load_dataset
from tqdm.auto import tqdm

FINEWIKI_VERSION = "af9c13333eb981300149d5ca60a8e9d659b276b9"

ds = load_dataset(
    "HuggingFaceFW/fineweb-2",
    revision=FINEWIKI_VERSION,
    name="oci_Latn",
    split="train",
    streaming=True
)

for sample in tqdm(ds, "Samples exported"):
    sample_text = sample["text"]
    sample_text = re.sub(r"\s+", " ", sample_text)
    print(sample_text)
