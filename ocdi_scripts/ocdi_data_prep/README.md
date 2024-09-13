# HOWTO: Prepare data for ocDI experiments

## Subcorpora

For our experiments, we organized subcorpora using the following folder structure:

```
ocdi_data/
├─ WIKI/
│  ├─ WIKI_splitv1/
│  │  ├─ WIKI_full.csv
│  │  ├─ WIKI_train.csv
│  │  ├─ WIKI_dev.csv
│  │  ├─ WIKI_test.csv
├─ .../
```

### OcWikiDialects (WIKI)
We split the OcWikiDialects dataset into train/dev/test using the script `split_ocwikidialects.py`, with option `--unit paragraph`.

Note that the experiments were conducted on an early version of OcWikiDialects 
that was based on the XML dump dated 20250801, where articles created after that date were not included. 
The 14 missing paragraphs were appended afterwards at the end of the released _train_ and _full_ splits.

### Tolosa Treebank (TTB)
We use the existing train/dev/test splits of the Tolosa Treebank dataset [(Miletić et al, 2020)](https://aclanthology.org/2020.vardial-1.13/) 
available in [Universal Dependencies v2.17](https://lindat.mff.cuni.cz/repository/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0). 
Each sample consists of a single sentence. We convert the CONLLU files to CSV using the script `ttb_to_csv_splits.py`.

### ForumOccitania (FORUM)
We use the existing train/dev/test splits of the ForumOccitania dataset [(Nédey et al, 2025)](https://inria.hal.science/hal-05413035), 
which can be made available on request for research purposes. 
Each sample consists of a full forum post. We convert the JSONL files to CSV using the script `forum_to_csv_splits.py`.

### CONGRES (Lo Congres News + Websites)
As the dataset [Lo Congres Websites](zenodo.org/records/12192029) has a large overlap with [Lo Congres News](https://zenodo.org/records/8411197), 
we merge both and split them into train/dev/test using the script `merge_locongres_newsweb.py`. 
Each sample consists of a single sentence.

### SoftwaresOccitanTranslations (SOFT)
We split the dataset [SoftwaresOccitanTranslations](https://zenodo.org/records/8411351) into train/dev/test using the script `split_locongres_dataset.py`. 
Each sample consists of a single sentence.

## CONCAT
We merge all subcorpora into a dataset called CONCAT (train/dev/test) using the script `concat_ocdi.py`. 
The script also normalizes labels of subcorpora into the same label scheme as used for FastText experiments 
(mappers can be found in `ocdi/labels.py`). 

### CONCAT-nowiki
In order to evaluate the impact of OcWikiDialects on the ocDI task, we created an alternative concatenation using the script `concat_ocdi.py`, 
to which we passed all subcorpora except WIKI.

## FastText-specific steps
Data preparation steps needed to train FastText (unsupervised and supervised) models are described in [`fasttext/README.md`](fasttext/README.md).
