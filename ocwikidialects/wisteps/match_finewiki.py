"""Module with a function to match articles parsed from the XML dump with articles of the FineWiki dataset"""
import re
import warnings

from datasets import load_dataset
from tqdm.auto import tqdm

from wisteps.defaults import FINEWIKI_VERSION
from wisteps.db import WikiDB


def match_db_articles(wiki_db: WikiDB, lang_subset="oc", wikiname="ocwiki"):

    def get_article_id(sample, wikiname):
        id_match = re.search(f"{wikiname}/(?P<id>.*)", sample["id"])
        assert id_match, f"Article ID not found in sample ID '{sample['id']}'"
        article_id = id_match["id"]
        return article_id

    ds = load_dataset("HuggingFaceFW/finewiki",
                      revision=FINEWIKI_VERSION,
                      name=lang_subset, split="train",
                      streaming=True)
    pbar = tqdm(desc="FineWiki matches", total=wiki_db.get_nb_articles())
    n_timestamps_errors = 0
    for sample in ds:
        article_id = get_article_id(sample, wikiname)
        db_article = wiki_db.get_article(article_id)
        if db_article:
            # Check that the revision timestamps match
            if sample["date_modified"] != db_article["date_latest"]:
                n_timestamps_errors += 1
                warnings.warn(
                    f"Revision timestamps don't match: "
                    f"(FineWiki) {sample['date_modified']} vs. (DB) {db_article['date_latest']}"
                )
            # Add FineWiki sample text to DB
            wiki_db.update_article(article_id, "content_finewiki", sample["text"])
            wiki_db.update_article(article_id, "date_modified_finewiki", sample["date_modified"])
            pbar.update()
    pbar.close()
    if n_timestamps_errors:
        warnings.warn(f"Found {n_timestamps_errors} articles with non-matching timestamps")
