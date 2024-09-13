# HOWTO: Prepare data for ocDI experiments with FastText

## Data for unsupervised training

The training data for our custom FastText embeddings model was built with scripts under the `unsupervised` folder:
1. Download Occitan samples from NLLB French-Occitan, with a LID threshold (0.8): `python dl_nllb.py > nllb_froc_oc.txt`
2. Download [OcWikiDisc](https://zenodo.org/records/7079580) and convert it to a TXT file with one Occitan sample per line (remove newlines): `python ocwikidisc_to_txt.py ocwikidisc_balanced.csv > ocwikidisc_balanced_v1.0.txt`
3. Download Occitan samples from FineWiki: `python dl_finewiki.py > finewiki_oc.txt`
4. Download Occitan samples from FineWeb2: `python dl_fineweb2.py > fineweb2_oc.txt`
5. Download Occitan samples from Tatoeba v2023-04-12 from [OPUS](https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/oc.txt.gz) (`tatoeba_oc.txt`)
6. Convert the ocDI concat-train corpus to plain text: `python ft-supervised-to-plaintext.py concat_train-fasttext.txt concat_train-plaintext.txt`
7. Concatenate all subcorpora: `cat nllb_froc_oc.txt ocwikidisc_balanced_v1.0.txt finewiki_oc.txt fineweb2_oc.txt tatoeba_oc.txt concat_train-plaintext.txt`

Documentation on how to build the ocDI concat-train subset can be found [here](../README.md).

## Data for supervised training
As FastText requires a specific format to train classification models (`__label__LABEL TEXT`), 
we converted our CSV files to this format with the bash script `convert_splits_to_fasttext.sh`, 
that calls `csv_to_fasttext.py` on all (train/dev/test) CSV files of a given folder.
For example: `bash convert_splits_to_fasttext.sh ocdi_data/concat/concat_splitv1/`

For experiments with preprocessing, we used the script `preprocess_fasttext.py` to apply preprocessing steps 
before running the training script. The config used in our experiments is in 
`/experiments/fasttext/configs/fasttext_preprocess_lapduen.yaml`.
