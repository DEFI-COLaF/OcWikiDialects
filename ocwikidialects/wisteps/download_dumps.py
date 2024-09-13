"""Script to download some dumps from Wikipedia"""
import bz2
import os

import requests

from wisteps.defaults import WIKI_NAME, DUMP_DATE


def download_dump_articles(wiki_name=WIKI_NAME, dump_date=DUMP_DATE, output_folder="."):
    # Download articles dump
    filename_articles = f"{wiki_name}-{dump_date}-pages-meta-history.xml"
    filename_zip_articles = f"{filename_articles}.bz2"
    filepath_articles = os.path.join(output_folder, filename_articles)
    filepath_zip_articles = os.path.join(output_folder, filename_zip_articles)
    if not os.path.exists(filepath_zip_articles) and not os.path.exists(filepath_articles):
        articles_dump_url = f"https://dumps.wikimedia.org/{wiki_name}/{dump_date}/{filename_zip_articles}"
        print("Download all pages with complete history...")
        with requests.get(articles_dump_url) as r:
            r.raise_for_status()
            with open(filepath_zip_articles, "wb") as f:
                for chunk in r.iter_content(chunk_size=10000):
                    f.write(chunk)
    if not os.path.exists(filepath_articles):
        print("Unzip pages-meta-history file...")
        with open(filepath_articles, "wb") as f, \
                bz2.BZ2File(filepath_zip_articles, "rb") as z:
            for data in iter(lambda: z.read(100*1024), b''):
                f.write(data)

    return filepath_articles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_name", default=WIKI_NAME)
    parser.add_argument("--dump_date", default=DUMP_DATE)
    parser.add_argument("--output_folder", default=".")

    args = parser.parse_args()

    download_dump_articles(args.wiki_name, args.dump_date, args.output_folder)
