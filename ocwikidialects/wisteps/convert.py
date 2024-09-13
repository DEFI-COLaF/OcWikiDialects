"""Script and functions to convert wikipedia articles
previously parsed and stored into a SQLite DB into HTML and/or TEI"""
import json
import os
import warnings

import pandas as pd
import regex as re
from tqdm.auto import tqdm
from slugify import slugify

from wisteps.db import WikiDB

import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize


def contains_letters(text, required=2):
    found = 0
    for _ in re.finditer(r"\p{L}", text):
        found += 1
        if found >= required:
            break
    return found >= required


def split_into_paragraphs(article_text):
    """Splits the text into paragraphs, by splitting around empty lines"""

    def finalize_par(par):
        # Check that the paragraph contains letters
        if not contains_letters(par):
            return ""
        # Strip whitespaces
        par = par.strip()
        # Remove duplicated whitespaces
        par = re.sub(r"[^\S\n]+", " ", par)
        return par

    if article_text is None:
        return None
    elif not article_text.strip():
        return ""
    pars = []
    cur_par = ""
    for line in article_text.splitlines(keepends=True):
        if not line.strip():
            if cur_par:
                cur_par = finalize_par(cur_par)
                if cur_par:
                    pars.append(cur_par)
                cur_par = ""  # Reset
        else:
            cur_par += line
    if cur_par:
        cur_par = finalize_par(cur_par)
        if cur_par:
            pars.append(cur_par)
    return pars


def split_into_segments(article_text):
    """Splits the text into segments, using NLTK"""

    def finalize_seg(seg):
        # Strip whitespaces
        seg = seg.strip()
        # Remove duplicated whitespaces
        seg = re.sub(r"[^\S\n]+", " ", seg)
        return seg

    if article_text is None:
        return None
    elif not article_text.strip():
        return ""
    segments = []
    for line in article_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split line into sentences
        for seg in sent_tokenize(line):
            if not contains_letters(seg):  # Skip segments that contain only symbols but no text
                continue
            # Normalize whitespaces
            seg = finalize_seg(seg)
            segments.append(seg)
    return segments


def export_one_txt_per_article(db_path, folder_individual_files):
    """Export articles of the DB by creating one TXT file per article,
    and organizing files by subfolders for each dialect"""
    db = WikiDB(db_path, read_only=True)

    # Process each article
    for article_row in tqdm(
        db.iter_all_articles(),
        total=db.get_nb_articles(),
        desc="Articles converted to TXT"
    ):
        # Get article text to export
        article_raw_text = article_row["content_finewiki"]
        if article_raw_text is None:
            warnings.warn(f"FineWiki text not found, skipping article {article_row['id']}")
            continue
        elif not article_raw_text.strip():
            raise ValueError(f"Article {article_row['id']} is empty")

        # Set output dialect folder
        dialect_folder = os.path.join(folder_individual_files, slugify(article_row["dialect"]))
        os.makedirs(dialect_folder, exist_ok=True)
        txt_path = os.path.join(dialect_folder, article_row["title_slugify"] + ".txt")

        # Write to output file(s)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(article_raw_text)
    print(f"All articles in DB exported to individual TXT files inside {folder_individual_files}")
    del db


def export_to_jsonl(db_path, output_file):

    def format_date(date):
        if not date:
            return date
        return date.split("T")[0]

    db = WikiDB(db_path, read_only=True)
    q = """SELECT * FROM article"""
    df_articles = pd.read_sql(q, db.con)

    # Split text into paragraphs
    df_articles["paragraphs"] = df_articles["content_finewiki"].apply(split_into_paragraphs)

    # Split text into segments
    df_articles["segments"] = df_articles["content_finewiki"].apply(split_into_segments)

    # Format dates
    df_articles["date_creation"] = df_articles["date_creation"].apply(format_date)
    df_articles["date_latest"] = df_articles["date_latest"].apply(format_date)
    df_articles["date_modified_finewiki"] = df_articles["date_modified_finewiki"].apply(format_date)

    # Make user dialects a Python list instead of space-separated list
    df_articles["user_dialects"] = df_articles["user_dialects"].apply(lambda el: el.split() if el else None)

    # Convert JSON dumped dicts to Python dicts (to be dumped again correctly in the final JSONL)
    df_articles["user_rank_freq"] = df_articles["user_rank_freq"].apply(json.loads)
    df_articles["user_rank_size"] = df_articles["user_rank_size"].apply(json.loads)
    df_articles["oc_level_rank_freq"] = df_articles["oc_level_rank_freq"].apply(json.loads)
    df_articles["oc_level_rank_size"] = df_articles["oc_level_rank_size"].apply(json.loads)

    # Rename columns
    df_articles.rename(columns={
        "content": "text_xml",
        "content_finewiki": "text_finewiki",
        "date_latest": "date_latest_xml",
        "date_modified_finewiki": "date_latest_finewiki"
    }, inplace=True)

    # Delete unwanted columns ?
    df_articles.drop(columns=["title_slugify"], inplace=True)

    # Reorder columns (esp. dialect, paragraphs and segments closer to the beginning)
    columns_order = [
        "id",
        "title",
        "dialect",
        "text_xml",
        "text_finewiki",
        "paragraphs",
        "segments",
        "len_bytes_creation",
        "len_bytes_latest",
        "date_creation",
        "date_latest_xml",
        "date_latest_finewiki",
        "latest_revision_id",
        "nb_revisions",
        "nb_contributors",
        "user_creator_id",
        "user_latest_id",
        "user_most_contrib_id",
        "user_biggest_contrib_id",
        "user_rank_freq",
        "user_rank_size",
        "oc_level_first",
        "oc_level_max",
        "oc_level_rank_freq",
        "oc_level_rank_size",
        "bot_created",
        "bot_first_author",
        "bot_nb_revisions",
        "user_dialects"
    ]
    df_articles = df_articles[columns_order]

    # Export to JSONL
    df_articles.to_json(output_file, orient="records", lines=True)
    print(f"Dataset in JSONL format available at {output_file}")
