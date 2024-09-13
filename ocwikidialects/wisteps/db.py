"""
Module with classes, methods and functions to interact with a temporary SQLite DB to store articles and users
before the final export.
"""

import sqlite3
from typing import TypedDict


class Article(TypedDict):
    title: str
    title_slugify: str
    id: int
    content: str
    content_finewiki: str | None
    len_bytes_creation: int
    len_bytes_latest: int
    date_creation: str
    date_latest: str
    date_modified_finewiki: str | None
    latest_revision_id: int
    nb_revisions: int
    nb_contributors: int
    user_creator_id: int
    user_latest_id: int
    user_most_contrib_id: int
    user_biggest_contrib_id: int
    user_rank_freq: str | None  # JSON dump of dict[int, int]
    user_rank_size: str | None  # JSON dump of dict[int, int]
    dialect: str | None
    oc_level_max: str | None
    oc_level_first: str | None
    oc_level_rank_freq: str | None
    oc_level_rank_size: str | None
    bot_created: bool | None  # First contribution
    bot_first_author: bool | None  # First non-empty contribution
    bot_nb_revisions: int | None
    user_dialects: str | list | None


class User(TypedDict):
    id: int
    username: str
    dialect: str | None
    oc_level: str | None
    is_bot: int


class UserPage(TypedDict):
    page_id: int
    page_title: str
    user_id: int | None


class WikiDB:

    ARTICLE_FIELDS = list(Article.__annotations__.keys())

    USER_FIELDS = list(User.__annotations__.keys())

    CONTRIBUTIONS_FIELDS = ["page_id", "pos_revision", "user_id", "len_bytes", "diff_bytes"]
    USERPAGE_FIELDS = list(UserPage.__annotations__.keys())

    def __init__(self, db_path, read_only=False):
        self.db_path = db_path
        self.read_only = read_only
        self.con = self.create_new_connection()
        self._init_db()

    def __exit__(self, *args):
        self.con.close()

    def create_new_connection(self):
        db_path = f"file:{self.db_path}?mode=ro" if self.read_only else self.db_path
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self):
        cur = self.con.cursor()

        # Create table for articles
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS article(
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                title_slugify TEXT NOT NULL,
                content TEXT NOT NULL,
                content_finewiki TEXT,
                len_bytes_creation INT NOT NULL,
                len_bytes_latest INT NOT NULL,                
                date_creation TEXT NOT NULL,
                date_latest TEXT NOT NULL,
                date_modified_finewiki TEXT,
                latest_revision_id INT NOT NULL,
                nb_revisions INT NOT NULL,
                nb_contributors INT NOT NULL,
                user_creator_id INT NOT NULL,
                user_latest_id INT NOT NULL,
                user_most_contrib_id INT NOT NULL,
                user_biggest_contrib_id INT NOT NULL,
                user_rank_freq TEXT,
                user_rank_size TEXT,
                dialect TEXT,
                oc_level_first TEXT,
                oc_level_max TEXT,
                oc_level_rank_freq TEXT,
                oc_level_rank_size TEXT,
                bot_created INT,
                bot_first_author INT,
                bot_nb_revisions INT,
                user_dialects TEXT,
                CONSTRAINT unique_slug_title UNIQUE (title_slugify)
            )
            """
        )

        # Create table for users
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user(
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                oc_level TEXT,
                dialect TEXT,
                is_bot INT,
                CONSTRAINT unique_username UNIQUE (username)
            )
            """
        )

        # Create table to associate user page titles with user IDs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_page(
                page_id INTEGER PRIMARY KEY,
                page_title TEXT NOT NULL,
                user_id INT
            )
            """
        )

        # Create table with lengths of contributions needed to compute the estimated language level of article
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS contribution(
                page_id INT NOT NULL,
                pos_revision INT NOT NULL,
                user_id INT,
                len_bytes INT NOT NULL,
                diff_bytes INT NOT NULL,
                CONSTRAINT unique_page_rev UNIQUE (page_id, pos_revision)
            )
            """
        )

        self.con.commit()

    def insert_article(self, article):
        cur = self.con.cursor()

        fields = [f for f in article.keys() if f in self.ARTICLE_FIELDS]
        fields_str = ", ".join(fields)
        values_str = ", ".join([":"+f for f in fields])
        try:
            cur.execute(
                f"INSERT INTO article ({fields_str}) VALUES ({values_str})",
                article
            )
        except sqlite3.IntegrityError:
            raise sqlite3.IntegrityError(f"IntegrityError with the following article: {article}")
        self.con.commit()

    def insert_user(self, user, if_new=False):
        cur = self.con.cursor()

        ignore_str = "OR IGNORE" if if_new is True else ""

        fields = [f for f in user.keys() if f in self.USER_FIELDS]
        fields_str = ", ".join(fields)
        values_str = ", ".join([":" + f for f in fields])

        cur.execute(
            f"INSERT {ignore_str} INTO user ({fields_str}) VALUES ({values_str})",
            user
        )
        self.con.commit()

    def insert_article_contributions(self, contributions):
        cur = self.con.cursor()

        fields_str = ", ".join(self.CONTRIBUTIONS_FIELDS)
        values_str = ", ".join([":"+f for f in self.CONTRIBUTIONS_FIELDS])

        cur.executemany(
            f"INSERT INTO contribution ({fields_str}) VALUES ({values_str})",
            contributions
        )
        self.con.commit()

    def insert_user_page(self, user_page):
        cur = self.con.cursor()

        fields_str = ", ".join(self.USERPAGE_FIELDS)
        values_str = ", ".join([":" + f for f in self.USERPAGE_FIELDS])

        cur.execute(
            f"INSERT INTO user_page ({fields_str}) VALUES ({values_str})",
            user_page
        )
        self.con.commit()

    def get_article(self, page_id):
        cur = self.con.cursor()

        cur.execute("SELECT * FROM article WHERE id = ?", [page_id])
        results = [dict(r) for r in cur.fetchall()]
        if len(results) == 1:
            return results[0]

    def get_article_by_slug_title(self, title_slugify):
        cur = self.con.cursor()

        cur.execute("SELECT * FROM article WHERE title_slugify = ?", [title_slugify])
        results = [dict(r) for r in cur.fetchall()]
        if len(results) == 1:
            return results[0]

    def get_user(self, user_id):
        cur = self.con.cursor()

        cur.execute("SELECT * FROM user WHERE id = ?", [user_id])
        results = [dict(f) for f in cur.fetchall()]
        if len(results) == 1:
            return results[0]

    def get_user_by_username(self, username):
        cur = self.con.cursor()

        cur.execute("SELECT * FROM user WHERE username = ?", [username])
        results = [dict(f) for f in cur.fetchall()]
        if len(results) == 1:
            return results[0]

    def get_user_page(self, page_id):
        cur = self.con.cursor()

        cur.execute("SELECT * FROM user_page WHERE page_id = ?", [page_id])
        results = [dict(f) for f in cur.fetchall()]
        if len(results) == 1:
            return results[0]

    def get_article_contributions(self, article_id, limit=None, sort_ascending=False):
        # Prepare query
        sort_ascending = "ASC" if sort_ascending else "DESC"
        q = f"""
        SELECT * FROM contribution
        WHERE page_id = :page_id
        ORDER BY pos_revision {sort_ascending}
        """
        params = {"page_id": article_id}
        if limit is not None:
            q += "LIMIT :limit"
            params["limit"] = limit

        # Run query
        cur = self.con.cursor()
        cur.execute(q, params)

        # Format results
        results = [dict(r) for r in cur.fetchall()]
        return results

    def get_nb_articles(self):
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM article")
        return cur.fetchone()[0]

    def iter_all_articles(self):
        cur = self.con.cursor()
        cur.execute("SELECT * FROM article")
        while True:
            r = cur.fetchone()
            if r is None:
                break
            yield dict(r)

    def iter_articles_no_lang_level(self):
        cur = self.con.cursor()
        cur.execute("SELECT * FROM article WHERE oc_level_max is NULL")
        while True:
            r = cur.fetchone()
            if r is None:
                break
            yield dict(r)

    def update_article(self, article_id, field, value):
        cur = self.con.cursor()
        assert field in self.ARTICLE_FIELDS
        cur.execute(
            f"""UPDATE article SET {field}=:{field} WHERE id=:id""",
            {field: value, "id": article_id}
        )
        self.con.commit()

    def update_user(self, user_id, field, value):
        cur = self.con.cursor()
        assert field in self.USER_FIELDS
        cur.execute(
            f"""UPDATE user SET {field}=:{field} WHERE id=:id""",
            {field: value, "id": user_id}
        )
        self.con.commit()

    def delete_articles_lang_levels(self, delete_levels):
        cur = self.con.cursor()
        delete_null = "NULL" in delete_levels
        if delete_null:
            delete_levels.remove("NULL")
            delete_null_str = "oc_level_max IS NULL OR "
        else:
            delete_null_str = ""
        query_fields = ", ".join(["?"] * len(delete_levels))
        cur.execute(
            f"""
            DELETE FROM article 
            WHERE {delete_null_str} oc_level_max IN ({query_fields})
            """,
            delete_levels
        )
        self.con.commit()
