"""Script/Module to parse and filter wikipedia articles from its XML dump"""

import json
import re
import uuid
import warnings
from collections import defaultdict, Counter

from lxml import etree as ET
from tqdm.auto import tqdm
from slugify import slugify

from wisteps.defaults import DB_PREFIX, WIKI_NAME, DUMP_DATE, DIALECTS_NORMALIZED
from wisteps.db import Article, User, UserPage, WikiDB

LANG_LEVELS_SORTED = ["NULL", "0", "1", "2", "3", "4", "5", "N"]


def reweight_negative_contribution(len_diff):
    """Makes the contribution count as positive but only half of its length."""
    return max(1, round(abs(len_diff) / 2))


def parse_article(page_element, wiki_db: WikiDB, skip_no_dialect=True, skip_dialects=[], revisions_until=None):
    title = page_element.find("{*}title").text
    page_id = int(page_element.find("{*}id").text)

    # Check if article exists in DB
    db_article = wiki_db.get_article(page_id)
    if db_article:
        return db_article

    # Check if slugified title exists in DB
    title_slugify = slugify(title)
    if not title_slugify:
        title_slugify = f"article-{page_id}"
    else:
        title_slugify += f"-{page_id}"

    page_revisions = page_element.findall("{*}revision")

    # Filter revisions with end date
    if revisions_until:
        page_revisions = [rev for rev in page_revisions
                          if rev.find("{*}timestamp").text[:10] <= revisions_until]
        if not page_revisions:
            warnings.warn(f"Page has no revisions before {revisions_until} ({page_id} - {title})")
            return None

    page_text = page_revisions[-1].find("{*}text").text
    if not page_text:
        warnings.warn(f"No text found in article id {page_id} - {title}")
        return None

    page_len_bytes_creation = int(page_revisions[0].find("{*}text").get("bytes"))
    page_len_bytes_latest = int(page_revisions[-1].find("{*}text").get("bytes"))
    latest_revision_id = int(page_revisions[-1].find("{*}id").text)

    # Look for dialect tag
    m_dialect = re.search(r"{{Dialècte (?P<dia>\w+)}}", page_text)
    dialect = m_dialect.group("dia").lower() if m_dialect else None
    if dialect:
        # Normalize (for vivaroalpenc)
        dialect = DIALECTS_NORMALIZED.get(dialect, dialect)
    if skip_no_dialect and dialect is None:
        return None
    if dialect in skip_dialects:
        return None

    # Get info on revisions and contributors (in the XML but maybe also DB or else):
    # - User IDs: creator, last revision, most frequent and biggest contribution
    # - Number of contributors
    # - Date of creation
    # - Date of last revision
    # - Number of revisions
    date_creation = page_revisions[0].find("{*}timestamp").text
    date_latest = page_revisions[-1].find("{*}timestamp").text
    nb_revisions = len(page_revisions)
    # Connect contributions with contributors
    user_contributions_len = defaultdict(int)
    user_contributions_nb = defaultdict(int)
    contributions_selection = []
    len_last = 0
    for i_rev, tag_revision in enumerate(page_revisions):
        # Get contributor ID: User ID if logged, else IP address
        tag_contributor = tag_revision.find("{*}contributor")
        tag_contributor_id = tag_contributor.find("{*}id")
        tag_contributor_ip = None
        if tag_contributor_id is not None:
            contributor_id = int(tag_contributor_id.text)
            # Add user to DB if there is a username and it is not already in the DB
            tag_username = tag_contributor.find("{*}username")
            if tag_username is not None:
                username = tag_username.text
                user = User(
                    id=contributor_id,
                    username=username,
                    dialect=None,
                    oc_level=None,
                    is_bot=1 if "bot" in username.lower() else 0
                )
                wiki_db.insert_user(user, if_new=True)
        else:
            tag_contributor_ip = tag_contributor.find("{*}ip")
            if tag_contributor_ip is not None:
                contributor_id = tag_contributor_ip.text
            else:
                if tag_contributor.get("deleted", "") == "deleted":
                    contributor_id = f"deleted_{uuid.uuid4()}"
                else:
                    raise ValueError(
                        f"Could not find contributor ID for revision ID {tag_revision.find('{*}id').text}: "
                        f"{ET.tostring(tag_contributor)}"
                    )
        if i_rev == 0:
            # Get page creator ID
            user_creator = contributor_id
        if i_rev == len(page_revisions) - 1:
            # Get user ID for the latest revision
            user_latest = contributor_id
        # Get estimated amount of contribution based on the text bytes length
        # Make the negative contributions (deletions) count for half of the positive ones (insertions)
        len_new = int(tag_revision.find("{*}text").get("bytes"))
        len_diff = len_new - len_last
        len_contrib = len_diff if len_diff > 0 else reweight_negative_contribution(len_diff)
        user_contributions_len[contributor_id] += len_contrib
        user_contributions_nb[contributor_id] += 1
        len_last = len_new
        # Update contributions_selection
        contributions_selection.append(
            {"page_id": page_id,
             "pos_revision": i_rev,
             "user_id": contributor_id if tag_contributor_id is not None else None,
             "len_bytes": len_new,
             "diff_bytes": len_diff}
        )

    # Get user with most contributions in terms of number and length
    user_most_contrib = max(user_contributions_nb, key=user_contributions_nb.get)
    user_biggest_contrib = max(user_contributions_len, key=user_contributions_len.get)
    # Get number of unique contributors
    nb_contributors = len(user_contributions_nb)

    # Store in intermediary SQLite DB
    article = Article(
        title=title,
        title_slugify=title_slugify,
        id=page_id,
        content=page_text,
        content_finewiki=None,
        len_bytes_creation=page_len_bytes_creation,
        len_bytes_latest=page_len_bytes_latest,
        date_creation=date_creation,
        date_latest=date_latest,
        date_modified_finewiki=None,
        latest_revision_id=latest_revision_id,
        nb_revisions=nb_revisions,
        nb_contributors=nb_contributors,
        user_creator_id=user_creator,
        user_latest_id=user_latest,
        user_most_contrib_id=user_most_contrib,
        user_biggest_contrib_id=user_biggest_contrib,
        dialect=dialect,
        oc_level_first=None,
        oc_level_max=None,
        oc_level_rank_freq=None,
        oc_level_rank_size=None,
        bot_created=None,
        bot_first_author=None,
        bot_nb_revisions=None,
        user_dialects=None
    )
    wiki_db.insert_article(article)

    # Store contributions selection (needed for computing the article's language level)
    wiki_db.insert_article_contributions(contributions_selection)

    return article


def parse_user(page_element, wiki_db: WikiDB):
    title = page_element.find("{*}title").text
    page_id = int(page_element.find("{*}id").text)
    user_page = UserPage(page_id=page_id, page_title=title, user_id=None)

    # Check if article exists in DB
    db_user_page = wiki_db.get_user_page(page_id)
    if db_user_page:
        return wiki_db.get_user(db_user_page["user_id"])

    username = re.search(r"Utilizaire:(?P<username>.+)", title).group("username")
    db_user = wiki_db.get_user_by_username(username)
    if db_user:
        user_page["user_id"] = db_user["id"]
        if db_user["oc_level"] and db_user["dialect"]:
            wiki_db.insert_user_page(user_page)
            return db_user

    page_rev = page_element.findall("{*}revision")[-1]
    page_text = page_rev.find("{*}text").text
    if page_text is None:
        wiki_db.insert_user_page(user_page)
        return db_user

    # Get user ID
    if not db_user:
        user_id = None
        tag_contributor = page_rev.find("{*}contributor")
        tag_username = tag_contributor.find("{*}username")
        tag_userid = tag_contributor.find("{*}id")
        if None not in [tag_username, tag_userid] and tag_username.text == username:
            user_id = tag_userid.text
            user_page["user_id"] = user_id
    else:
        user_id = db_user["id"]

    if not user_id:  # e.g. if no contributor was found with the same username
        wiki_db.insert_user_page(user_page)
        return db_user

    # Look for Babel template
    oc_level = db_user["oc_level"] if db_user else None
    if oc_level is None:
        m_oc_level = re.search(
            r"\{\{#?(Babel|Boita d'utilizaire|Boita Babel).*?[|:\s]oc(-(?P<oc_level>\d|N|M))?(\s*(\|)|(}}))",
            page_text,
            flags=re.I | re.DOTALL
        )
        if not m_oc_level or not m_oc_level["oc_level"]:  # second condition is if matched only "oc" and not "oc-2" e.g.
            m_oc_level_2 = re.search(
                r"\{\{(Utilizaire|user) oc(-(?P<oc_level>[\dNM]))?}}",
                page_text,
                flags=re.I | re.DOTALL
            )
            if m_oc_level_2:
                if m_oc_level and m_oc_level_2["oc_level"]:
                    m_oc_level = m_oc_level_2
                else:
                    m_oc_level = m_oc_level_2

        if m_oc_level:
            oc_level = m_oc_level["oc_level"] or "N"
        else:
            oc_level = None

    # Look for dialect template
    dialect = db_user["dialect"] if db_user else None
    if dialect is None:
        m_dialect = re.search(  # e.g. {{Boita d'utilizaire|Wikipedian provençau}} or {{Boita d'utilizaire|demòra Lengadòc | fr | oc-2|Wikipedian lengadocian| es-3| en-2 | Commons}}
            r"\{\{#?(Babel|Boita d'utilizaire|Boita Babel).*?[|:\s]Wikipedian(\s+(?P<dialect>\w+))?(\s*(\|)|(}}))",
            page_text
        )
        dialect = m_dialect.group("dialect") if m_dialect else None

    # Check for bot in username or in template: {{Boita d'utilizaire|bòt}} or {{Babel|oc-0|en-3|it-3|bòt}}
    is_bot = db_user["is_bot"] if db_user else 0
    if is_bot == 0:
        m_bot = re.search(
            r"\{\{#?(Babel|Boita d'utilizaire|Boita Babel).*?[|:\s]bòt(\s*(\|)|(}}))",
            page_text
        )
        if m_bot is not None:
            is_bot = 1
        elif "bot" in username.lower():
            is_bot = 1

    # Store or update user in DB
    if not db_user:
        db_user = User(id=user_id, username=username, dialect=dialect, oc_level=oc_level, is_bot=is_bot)
        wiki_db.insert_user(db_user)
    else:
        if oc_level is not None:
            wiki_db.update_user(user_id, "oc_level", oc_level)
        if dialect is not None:
            wiki_db.update_user(user_id, "dialect", dialect)
        if is_bot != 0:
            wiki_db.update_user(user_id, "is_bot", is_bot)

    # Store user page in DB
    wiki_db.insert_user_page(user_page)

    return db_user


def aggregate_article_metadata(wiki_db: WikiDB):

    def get_article_lang_level(article_contributions):
        """Get the highest level found in revisions where the size of the article is within 10% of the latest revision"""
        len_threshold_first = 100  # Consider the first significant contribution (i.e. ignore if a bot create a page with no text or only one word)
        len_threshold_any = 1  # Filter empty contributions (maybe only title or metadata)

        oc_level_first = False
        highest_level = None
        for contrib in article_contributions:
            # Check if the contribution is not empty
            contrib_len = contrib["len_bytes"]
            if contrib_len < len_threshold_any:
                continue

            # Get Occitan level of user
            db_user = wiki_db.get_user(contrib["user_id"])
            user_level = db_user["oc_level"] if db_user else None

            # Set oc_level of the 1st contribution
            if oc_level_first is False:
                # Check length of article for this contribution
                if contrib_len < len_threshold_first:
                    continue
                oc_level_first = user_level

            # Update highest_level if new highest
            if user_level is None:
                continue
            if highest_level is None:
                highest_level = user_level
            elif LANG_LEVELS_SORTED.index(user_level) > LANG_LEVELS_SORTED.index(highest_level):
                highest_level = user_level

        if oc_level_first is False:
            oc_level_first = None

        return oc_level_first, highest_level

    def is_bot_created(user_creator_id):
        if not user_creator_id:
            return False
        db_user = wiki_db.get_user(user_creator_id)
        if not db_user:
            return False
        return bool(db_user["is_bot"])

    def is_bot_first_authored(article_contributions):
        len_threshold_first = 100  # Consider the first significant contribution (i.e. ignore if a bot create a page with no text or only one word)
        res = False
        for contrib in article_contributions:
            # Check length of article for this contribution
            contrib_len = contrib["len_bytes"]
            if contrib_len < len_threshold_first:
                continue
            # Get user of the contribution
            db_user = wiki_db.get_user(contrib["user_id"])
            if not db_user:
                res = False
                break
            res = db_user["is_bot"]
            break
        return res

    def count_bot_revisions(article_contributions):
        count = 0
        for contrib in article_contributions:
            # Get user of the contribution
            db_user = wiki_db.get_user(contrib["user_id"])
            if db_user and db_user["is_bot"]:
                count += 1
        return count

    def get_user_dialects(article_contributions):
        dialects = set()
        for contrib in article_contributions:
            db_user = wiki_db.get_user(contrib["user_id"])
            if not db_user:  # happens with IP addresses as user_id
                continue
            user_dialect = db_user["dialect"]
            if user_dialect:
                dialects.add(user_dialect)
        return dialects

    def get_users_rank_by_freq(article_contributions):
        user_freqs = Counter([contrib["user_id"] for contrib in article_contributions])
        user_freqs = {user_id: freq for user_id, freq in user_freqs.most_common()}
        return user_freqs

    def get_users_rank_by_size(article_contributions):
        user_sizes = Counter()
        for contrib in article_contributions:
            user_id = contrib["user_id"]
            contrib_size = contrib["diff_bytes"]
            if contrib_size < 0:
                contrib_size = reweight_negative_contribution(contrib_size)
            user_sizes[user_id] += contrib_size
        user_sizes = {user_id: size for user_id, size in user_sizes.most_common()}
        return user_sizes

    def get_levels_rank_by_freq(article_contributions):
        level_freqs = Counter()
        for contrib in article_contributions:
            db_user = wiki_db.get_user(contrib["user_id"])
            if not db_user:
                level_freqs[None] += 1
            else:
                level_freqs[db_user["oc_level"]] += 1
        level_freqs = {level: freq for level, freq in level_freqs.most_common()}
        return level_freqs

    def get_levels_rank_by_size(article_contributions):
        level_sizes = Counter()
        for contrib in article_contributions:
            contrib_size = contrib["diff_bytes"]
            if contrib_size < 0:
                contrib_size = reweight_negative_contribution(contrib_size)
            db_user = wiki_db.get_user(contrib["user_id"])
            if not db_user:
                level_sizes[None] += contrib_size
            else:
                level_sizes[db_user["oc_level"]] += contrib_size
        level_sizes = {level: size for level, size in level_sizes.most_common()}
        return level_sizes

    n_articles = wiki_db.get_nb_articles()
    for article in tqdm(wiki_db.iter_all_articles(), desc="Aggregate metadata in articles", total=n_articles):
        update_dict = {}
        article_id = article["id"]
        article_contributions = wiki_db.get_article_contributions(article_id)
        # Compute oc_level metadata (oc_level_first, oc_level_max)
        if article["oc_level_first"] is None and article["oc_level_max"] is None:
            update_dict["oc_level_first"], update_dict["oc_level_max"] = get_article_lang_level(article_contributions)
        # Find if the article was created by a bot
        if article["bot_created"] is None:
            update_dict["bot_created"] = is_bot_created(article["user_creator_id"])
        # Find if the article was first authored (non-empty contribution) by a bot
        if article["bot_first_author"] is None:
            update_dict["bot_first_author"] = is_bot_first_authored(article_contributions)
        # Count the number of revisions made by bots
        if article["bot_nb_revisions"] is None:
            update_dict["bot_nb_revisions"] = count_bot_revisions(article_contributions)
        # Get set of user dialects
        if article["user_dialects"] is None:
            user_dialects = sorted(get_user_dialects(article_contributions))
            if user_dialects:
                update_dict["user_dialects"] = " ".join(user_dialects)
        # Rank users by number/size of contributions
        if article["user_rank_freq"] is None:
            update_dict["user_rank_freq"] = json.dumps(get_users_rank_by_freq(article_contributions))
        if article["user_rank_size"] is None:
            update_dict["user_rank_size"] = json.dumps(get_users_rank_by_size(article_contributions))
        # Rank oc levels by number/size of contributions
        if article["oc_level_rank_freq"] is None:
            update_dict["oc_level_rank_freq"] = json.dumps(get_levels_rank_by_freq(article_contributions))
        if article["oc_level_rank_size"] is None:
            update_dict["oc_level_rank_size"] = json.dumps(get_levels_rank_by_size(article_contributions))

        # Remove None values in update_dict
        update_dict = {k: v for k, v in update_dict.items() if v is not None}
        # Update article in DB
        for k, v in update_dict.items():
            wiki_db.update_article(article_id, k, v)


def filter_lang_level(wiki_db: WikiDB, keep_levels="all"):
    if "all" in keep_levels:
        return
    assert all([level in LANG_LEVELS_SORTED for level in keep_levels])
    delete_levels = [level for level in LANG_LEVELS_SORTED if level not in keep_levels]
    if delete_levels:
        wiki_db.delete_articles_lang_levels(delete_levels)


def parse_pages(xml_path,
                wiki_db: WikiDB,
                skip_no_dialect=True,
                skip_dialects=[],
                keep_lang_levels="all",
                revisions_until=None,
                limit=None):
    def fast_iter_pages(context, limit):
        """
        http://lxml.de/parsing.html#modifying-the-tree
        Based on Liza Daly's fast_iter
        http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
        See also http://effbot.org/zone/element-iterparse.htm
        """
        t_pages = tqdm(desc="Pages found", mininterval=5)
        t_results = tqdm(desc="Pages parsed", mininterval=5)
        for event, elem in context:
            t_pages.update(1)
            # Route to parsing of article or user page depending on the namespace ns value
            ns_value = int(elem.find("{*}ns").text)
            if ns_value == 0:
                res = parse_article(
                    elem,
                    wiki_db,
                    skip_no_dialect=skip_no_dialect,
                    skip_dialects=skip_dialects,
                    revisions_until=revisions_until
                )
            elif ns_value == 2:
                res = parse_user(elem, wiki_db)
            else:
                res = None
            if res:
                t_results.update(1)
            # It's safe to call clear() here because no descendants will be accessed
            elem.clear()
            # Also eliminate now-empty references from the root node to elem
            for ancestor in elem.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]
            if limit and t_results.n == limit:
                break
        n_found = t_pages.n
        n_parsed = t_results.n
        t_results.close()
        t_pages.close()
        del context
        return n_found, n_parsed

    print("Start parsing...")
    context = ET.iterparse(xml_path, tag="{*}page", events=('end',))
    n_found, n_parsed = fast_iter_pages(context, limit)
    n_articles = wiki_db.get_nb_articles()

    print(f"Finished parsing XML file. Found {n_found} pages, parsed {n_parsed} (articles, users), "
          f"incl. {n_articles} articles.")

    print(f"Aggregate articles metadata...")
    aggregate_article_metadata(wiki_db)

    if keep_lang_levels and "all" not in keep_lang_levels:
        print(f"Filter articles based on language levels (keep {keep_lang_levels})")
        filter_lang_level(wiki_db, keep_lang_levels)

    print(f"DB contains {wiki_db.get_nb_articles()} articles.")
