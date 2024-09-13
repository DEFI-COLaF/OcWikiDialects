# OcWikiDialects: A Wikipedia Dataset with Rich Metadata for Occitan Dialect Identification

This repository contains the code, dataset, models and analyses described in the following paper (accepted at VarDial 2026):

```
@inproceedings{
	author = {Nédey, Oriane and Bawden, Rachel and Clérice, Thibault and Sagot, Benoît},
	title = {OcWikiDialects: A Wikipedia Dataset with Rich Metadata for Occitan Dialect Identification},
	url = {TBA},
	booktitle = {Proceedings of the 13th {Workshop} on {NLP} for {Similar} {Languages}, {Varieties} and {Dialects}},
	year = {2026}
}
```

# Released assets
We will release the OcWikiDialects dataset (JSONL + CSV splits) as well as the SVM, FastText and BERT models trained
on the concatenated datasets without text preprocessing, and our custom FastText embedding model
as assets in the [releases](https://github.com/DEFI-COLaF/OcWikiDialects/releases) page.

# The OcWikiDialects dataset
The OcWikiDialects dataset contains 7,247 articles, each split into paragraphs (57k in total) and into sentences (288k), for a total of approximately 4M tokens.
10 varieties of (or closely-related) Occitan are represented: Auvergnat, Gascon, Limousine, Languedocian, Provençal, Vivaroalpine, Aranese, Niçard, Aguianese and Marchese.

This dataset was built from two sources:
- metadata from the XML dump at date 20250901, taking into account all revisions until 2025-08-20
- clean Markdown texts from [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki), built from the HTML dump at date 20250820

The dataset is available in JSONL format. Each article is associated to the following elements:
- id
- title
- dialect: dialect category found in the XML dump
- text_xml: raw text from the XML dump, in [WikiText](https://en.wikipedia.org/wiki/Help:Wikitext) format
- text_finewiki: clean Markdown text from FineWiki
- paragraphs: list of lightly cleaned paragraphs extracted from text_finewiki
- segments: list of lightly cleaned segments extracted from text_finewiki
- len_bytes_creation: size of the article (in bytes) in its first revision
- len_bytes_latest: size of the article (in bytes) in the latest revision
- date_creation: date of the first revision
- date_latest_xml: date of the latest revision in the XML dump
- date_latest_finewiki: date of the latest revision in FineWiki
- latest_revision_id
- nb_revisions
- nb_contributors
- user_creator_id: ID of the user who made the first revision
- user_latest_id: ID of the user who made the latest revision
- user_most_contrib_id: user with the highest number of contributions to this article
- user_biggest_contrib_id: user who made the most changes on this article (in bytes, negative contributions are counted positively with a 0.5 factor)
- user_rank_freq: ranking of users by the number of contributions made to the article
- user_rank_size: ranking of users by the size of their contributions made to the article (negative contributions are counted positively with a 0.5 factor)
- oc_level_first: Occitan proficiency level declared by the user of the first revision
- oc_level_max: highest Occitan proficiency level declared among all contributors of the article
- oc_level_rank_freq: ranking of Occitan levels by the number of contributions from users declaring each level
- oc_level_rank_size: ranking of Occitan levels by the size of contributions from users declaring each level (negative contributions are counted positively with a 0.5 factor)
- bot_created: True if the user of the first contribution was detected as a bot (username and/or tag in user page)
- bot_first_author: True if the user of the first contribution with more than +100 (or +200 ??) bytes
- bot_nb_revisions: Number of revisions authored by a user detected as a bot (username and/or tag in user page)
- user_dialects: list of dialects declared among all contributors of the article (whitespace separated)

## Code to build the dataset
The dataset was built using the script `ocwikidialects/ocwikidialects-pipeline.py`, with all default options.

The code depends on the internal package `wisteps` as well as 3rd-party dependencies which can be installed as follows:
```bash
git clone git@github.com:DEFI-COLaF/OcWikiDialects.git
cd OcWikiDialects
pip install .
```

Note that the script will take some time to download the XML dump as well as to parse it, 
and that it creates an intermediary SQLite database that was used in our analyses.

# Occitan Dialect Identification (ocDI) experiments

Our experiments rely on the internal `ocDI` package which can be installed by running `pip install .` at the root of this repository.

## Data preparation
We experimented with the OcWikiDialects dataset as well as other datasets labelled with Occitan varieties.

The data preprocessing steps are described in [`ocdi_scripts/ocdi_data_prep/README.md`](ocdi_scripts/ocdi_data_prep/README.md).

Note that the experiments were conducted on an early version of OcWikiDialects 
that was based on the XML dump dated 20250801, where articles created after that date were not included. 
The 14 missing paragraphs were appended afterwards at the end of the released _train_ and _full_ splits.

## Experiments with SVM
We trained SVM models using scikit-learn. The training and evaluation pipelines are available in the following notebooks:
- models without text preprocessing: `experiments/svm/ocDI_svm_noprep.ipynb`
- models with text preprocessing: `experiments/svm/ocDI_svm_prep.ipynb`

We release the SVM model trained on the concatenated datasets without text preprocessing.

Example of model usage:
```python
from ocdi.svm.save import load_sklearn_pipeline
from ocdi.svm.predict import predict_with_svm

model = load_sklearn_pipeline(model_path)

predict_with_svm(model, ["Adishatz tot lo monde !"])
```

## Experiments with FastText
We trained custom FastText embeddings using the script `ocdi_scripts/training/fasttext/train_embs.py`, 
then trained various classifiers with the script `ocdi_scripts/training/fasttexttrain_ft_classifer.py`. 

Training configs can be found under `experiments/fasttext/configs`:
- Config `fasttext_did_config_ccembs.yaml` enables training a classifier on top of the official FastText vectors trained on CommonCrawl ([source](https://fasttext.cc/docs/en/crawl-vectors.html), [paper](https://aclanthology.org/L18-1550/))
- Config `fasttext_did_config_ocdiembs.yaml` enables training a classifier on top of our custom embeddings
- Config `fasttext_preprocess_lapduen.yaml` is meant to preprocess a dataset before training (cf. [`ocdi_scripts/ocdi_data_prep/fasttext/README.md`](ocdi_scripts/ocdi_data_prep/fasttext/README.md))

We release our custom FastText embedding model as well as the FastText ocDI model trained with these embeddings 
on the concatenated datasets, without text preprocessing.

Predict with ocDI-fasttext:
```python
import fasttext

model = fasttext.load_model(model_path)
label, score = model.predict("Adishatz tot lo monde !")
```

Evaluate:
```python
import pandas as pd
from ocdi.fasttext.evaluate import evaluate_fasttext

df_test = pd.DataFrame({"text": ["Adishatz tot lo monde !"], "label": ["gas"]})
results = evaluate_fasttext(model_path, df_test)
```

## Experiments with BERT
We fine-tuned pretrained BERT models using the script `ocdi_scripts/training/bert/finetune.py` and training configs in `experiments/bert/configs`.

We compared performance of the following models:
- [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- [google-bert/bert-base-multilingual-uncased](https://huggingface.co/google-bert/bert-base-multilingual-uncased)
- [zhopto3/oc_mbert](https://huggingface.co/zhopto3/oc_mbert)

The suffix `_prep` in the config name indicates that the data is preprocessed before passing to the model.

We release the oc-mBERT model fine-tuned for the ocDI task on the concatenated datasets, without text preprocessing.

Predict with ocDI-bert:
```python
from transformers import pipeline

pipe = pipeline("text-classification", model=model_path)
pipe("Adishatz tot lo monde!")
```

Evaluate:
```python
from transformers import BertTokenizer, BertForSequenceClassification

from ocdi.bert.evaluate import evaluate_bert

model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)

model_results, preds, true_labels = evaluate_bert(
    model,
    test_path,  # CSV file
    tokenizer,
    test_label_scheme="fasttext",  # label scheme of the test set (see ocdi.labels.SOURCE_TO_SCHEME)
    train_label_scheme="fasttext",  # label scheme of the model (for CONCAT, use 'fasttext')
    batch_size=1,
    num_workers=1,
    only_report=False,  # Only show the classification summary, do not return scores
    show_confusion_matrix=False,  # Set to True to display the confusion matrix   
)
```
