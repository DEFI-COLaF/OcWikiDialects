"""Script to download NLLB Occitan data, to be added to the truecaser training data"""
import re

from datasets import load_dataset
from tqdm.auto import tqdm

dataset = load_dataset("allenai/nllb", "fra_Latn-oci_Latn", trust_remote_code=True, split="train")

# Filter using LID score on the Occitan text
lid_threshold = 0.8
dataset_filtered = dataset.filter(lambda sample: sample["target_sentence_lid"] >= lid_threshold)

# Export to TXT
for sample in tqdm(dataset_filtered, desc="Samples exported"):
    sample_text = sample["translation"]["oci_Latn"]
    sample_text = re.sub(r"\s+", " ", sample_text)
    print(sample_text)
