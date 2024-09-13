"""Main script to (re)build the OcWikiDialects dataset.
Set PYTHONPATH to this directory to avoid installing the wisteps package.
"""
import os

from wisteps import download_dumps, parse_articles, convert, match_finewiki
from wisteps.defaults import WIKI_NAME, WIKI_LANG_CODE, DUMP_DATE, DB_PREFIX, REVISIONS_UNTIL
from wisteps.db import WikiDB


def main(wiki_name=WIKI_NAME, dump_date=DUMP_DATE, db_prefix=DB_PREFIX, wiki_lang_code=WIKI_LANG_CODE,
         output_folder=".", do_parse=True, do_export=True, parse_kwargs=None):
    db_name = f"{db_prefix}-{wiki_name}-{dump_date}"
    db_path = os.path.join(output_folder, db_name + ".sqlite")

    if do_parse is True:
        # Download Wikipedia dumps
        path_xml_articles = download_dumps.download_dump_articles(wiki_name, dump_date, output_folder)

        # Parse articles
        wiki_db = WikiDB(db_path)
        parse_articles.parse_pages(path_xml_articles, wiki_db, **parse_kwargs)

        # Extract clean articles from FineWiki dataset
        # TODO Remove bibliography section from articles (and similar unwanted sections)
        match_finewiki.match_db_articles(wiki_db, lang_subset=wiki_lang_code, wikiname=wiki_name)

        del wiki_db

    if do_export is True:
        export_folder = os.path.join(output_folder, "exports")
        os.makedirs(export_folder, exist_ok=True)

        # Export to 1 file per article, in dialect subfolders - uncomment to run it
        # out_folder_txt_ind = os.path.join(export_folder, "TXT-INDIVIDUAL")
        # os.makedirs(out_folder_txt_ind, exist_ok=True)
        # convert.export_one_txt_per_article(db_path, out_folder_txt_ind)

        # Export to JSONL, with rich metadata + splits into paragraphs and segments
        out_path_jsonl = os.path.join(export_folder, f"ocwikidialects-{dump_date}.jsonl")
        convert.export_to_jsonl(db_path, out_path_jsonl)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        default="data"
    )
    parser.add_argument(
        "--wiki_name",
        default=WIKI_NAME,
        help=f"Name of the wikipedia dump prefix (for Occitan: {WIKI_NAME})"
    )
    parser.add_argument(
        "--dump_date",
        default=DUMP_DATE,
        help="Date of the dump (or 'latest')"
    )
    parser.add_argument(
        "--wiki_lang_code",
        default=WIKI_LANG_CODE,
        help="Wikipedia language code used in the URL (e.g. oc for oc.wikipedia.org)"
    )
    parser.add_argument(
        "--db_prefix",
        default=DB_PREFIX,
        help="Prefix to the SQLite DB file name to be loaded or created (without prefix folder nor extension)"
    )
    parser.add_argument(
        "--skip_parsing",
        action="store_true",
        help="Option to skip parsing - expect the DB to exist and will start directly at the export phase"
    )
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Option to skip export - only downloads dump, parses articles and stores them in the DB"
    )
    parser.add_argument(
        "--keep_no_dialect",
        action="store_true",
        default=False,
        help="Option to disable skipping articles when no dialect is declared in its contents"
    )
    parser.add_argument(
        "--skip_dialects",
        nargs="+",
        default=[],
        help="Option to skip articles with some dialect tags: list of dialects to skip"
    )
    parser.add_argument(
        "--keep_lang_levels",
        nargs="+",
        default=["all"],
        help="Option to filter out articles with some estimated language levels ('all' to keep all levels incl. empty)"
    )
    parser.add_argument(
        "--revisions_until",
        default=REVISIONS_UNTIL,
        help="Option to parse revisions of the XML dump only until the given date"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Option to parse only a defined number of pages from the XML file - for dev purposes"
    )

    args = parser.parse_args()

    parse_kwargs = {
        "skip_no_dialect": not args.keep_no_dialect,
        "skip_dialects": args.skip_dialects,
        "keep_lang_levels": args.keep_lang_levels,
        "revisions_until": args.revisions_until,
        "limit": args.limit
    }

    main(
        args.wiki_name,
        args.dump_date,
        args.db_prefix,
        args.wiki_lang_code,
        args.output_folder,
        do_parse=not args.skip_parsing,
        do_export=not args.skip_export,
        parse_kwargs=parse_kwargs
    )
