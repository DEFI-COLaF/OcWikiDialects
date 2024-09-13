"""Module with Dataset to use for training models with PyTorch"""
import logging

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset

from ocdi.split import load_csv
from ocdi.svm.preprocessing import preprocess

logger = logging.getLogger(__name__)


class DidDataset(TorchDataset):

    tokenization_params = {
        # "return_tensors": "pt",
        "max_length": 256,
        "truncation": True,
        "padding": "max_length"
    }

    def __init__(self, csv_path, tokenizer,
                 col_text="text", col_label="dialect",
                 labels_mapper=None, label2id=None,
                 prep_opts=None):
        # Load data
        self.data = self.load_csv(csv_path)
        self.col_text = col_text
        self.col_label = col_label
        if labels_mapper:
            self.convert_labels(labels_mapper)

        # Preprocess data
        if prep_opts:
            self.apply_prep_opts(prep_opts)
        self.tokenizer = tokenizer
        self.tokenize_dataset()

        self.label2id = self.get_label2id(label2id)
        self.id2label = {i: label for label, i in self.label2id.items()}
        if col_label in self.data.column_names:
            self.convert_labels_to_id()

        logger.info(f"Dataset contains the following columns: {self.data.column_names}")

    @staticmethod
    def load_csv(path):
        df = load_csv(path)
        ds = HFDataset.from_pandas(df)
        return ds

    def apply_prep_opts(self, opts):
        self.data = self.data.map(
            lambda text: {self.col_text: preprocess(text, **opts)},
            input_columns=self.col_text,
            batched=False,
            desc="Text preprocessing"
        )

    def tokenize_dataset(self):
        self.data = self.data.map(
            lambda x: self.tokenizer(x[self.col_text], **self.tokenization_params),
            batched=True,
            desc="Tokenizing"
        )

    def convert_labels(self, mapper):
        self.data = self.data.map(
            lambda label: {self.col_label: mapper.get(label, label)},
            input_columns=self.col_label,
            batched=False,
            desc="Normalizing labels"
        )

    def get_label2id(self, label2id=None):
        data_labels = sorted(self.data.unique(self.col_label))
        # Check if extra labels in existing label2id
        if label2id:
            extra_labels = [lab for lab in data_labels if lab not in label2id]
            last_id = max(label2id.values())
            label2id.update({lab: i for i, lab in enumerate(extra_labels, last_id+1)})
        else:
            label2id = {lab: i for i, lab in enumerate(data_labels)}
        return label2id

    def convert_labels_to_id(self):
        self.data = self.data.map(
            lambda label: {"label": self.label2id[label], "label_str": label},
            input_columns=self.col_label,
            batched=False,
            desc="Converting labels to IDs"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        # Fetch item
        item = self.data[index]
        # Filter to keep only the necessary columns for BERT
        item = {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'label': item['label']
        }
        return item
