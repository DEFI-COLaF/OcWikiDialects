import argparse
import logging
import os
import random
from datetime import datetime

import yaml

import numpy as np
import torch

from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, EarlyStoppingCallback)

from ocdi.labels import LABEL_SCHEME_TO_MAPPER
from ocdi.bert.data import DidDataset
from ocdi.bert.evaluate import evaluate_bert


def config_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        datefmt='%d/%m/%Y %I:%M %p'
    )
    return logger


def set_deterministic_experiment(seed):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed = seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


argparser = argparse.ArgumentParser()
argparser.add_argument("config", help="Path to the YAML config")
argparser.add_argument("--root_folder", default=".",
                       help="Path to the root folder, to prefix all paths indicated in the config")
argparser.add_argument("--path_train", help="Path to the training dataset in CSV format")
argparser.add_argument("--path_val", help="Path to the validation dataset in CSV format")
argparser.add_argument("--path_test", help="Path to the test dataset in CSV format")
argparser.add_argument("--label_scheme", choices=list(LABEL_SCHEME_TO_MAPPER),
                       help="Label scheme for the input datasets (esp. training)")
argparser.add_argument("--suffix",
                       help="Suffix to add to the run_name from the config (eg. name of the training dataset)")
argparser.add_argument("--version_suffix", default=".1",
                       help="Suffix to add to the run_name from the config")
args = argparser.parse_args()

logger = config_logger()
logger.info(f"Args: {args}")

# Load config
with open(args.config) as f:
    config = yaml.safe_load(f)
run_name = config["run_name"] + "-" + config["version"] + args.version_suffix
run_name_full = run_name + args.suffix if args.suffix else run_name

logger.info(f"Config passed: {config}")

# Check that input datasets exist
path_train = os.path.join(args.root_folder, args.path_train)
path_val = os.path.join(args.root_folder, args.path_val)
path_test = os.path.join(args.root_folder, args.path_test)
for file in [path_train, path_val, path_test]:
    assert os.path.exists(file), f"File not found: {file}"

# Set deterministic experiment
set_deterministic_experiment(config["seed"])

# Create output folder
output_folder = str(os.path.join(args.root_folder, config["out_folder"], run_name, run_name_full))
os.makedirs(output_folder, exist_ok=False)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["pretrained_name"])
logger.info(f"Loaded tokenizer from pretrained model {config['pretrained_name']}")

# Make label2id mapper with normalized labels
labels_mapper = LABEL_SCHEME_TO_MAPPER[args.label_scheme]
label2id = {labels_mapper[label]: i for i, label in enumerate(labels_mapper)}

# Load train/dev datasets
prep_opts = config.get("data_prep")
ds_train = DidDataset(path_train, tokenizer,
                      labels_mapper=labels_mapper, label2id=label2id,
                      prep_opts=prep_opts)
logger.info(f"Training dataset: {len(ds_train)} samples")
ds_val = DidDataset(path_val, tokenizer,
                    labels_mapper=labels_mapper, label2id=label2id,
                    prep_opts=prep_opts)
logger.info(f"Validation dataset: {len(ds_train)} samples")

# Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    config["pretrained_name"],
    num_labels=len(label2id),
    label2id=label2id,
    id2label=ds_train.id2label
)

# Define training settings
training_args = TrainingArguments(
    output_dir=output_folder,
    run_name=run_name_full,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    eval_on_start=True,
    log_level="info",
    logging_first_step=True,
    disable_tqdm=True,
    **config["training_args"]
)

# Define callbacks
if "early_stopping" in config:
    stop_params = config["early_stopping"] if isinstance(config["early_stopping"], dict) else {}
    early_stopping_callback = EarlyStoppingCallback(**stop_params)
    callbacks = [early_stopping_callback]
else:
    callbacks = None

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=callbacks
)

# Launch training
logger.info("Launch training")
time_start = datetime.now()
trainer.train()
time_end = datetime.now()
duration = time_end - time_start
logger.info(f"Training completed in {duration}")

# Save best model
logger.info("Saving best model")
model_dir = os.path.join(output_folder, f"{run_name_full}_model")
trainer.save_model(model_dir)

# Run evaluation on the test dataset
print(f"Run evaluation on {path_test}")
evaluate_bert(model, path_test, tokenizer,
              data_prep_opts=prep_opts,
              test_label_scheme=args.label_scheme,
              train_label_scheme=args.label_scheme,
              batch_size=config["training_args"].get("per_device_eval_batch_size", 1),
              num_workers=config["training_args"].get("dataloader_num_workers", 1))
