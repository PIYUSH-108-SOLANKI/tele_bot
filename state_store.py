"""
state_store.py
SQLite helper module to deduplicate tech news URLs.
Stores successfully sent article URLs to prevent resending them in future digests.
"""

import os
import sqlite3
from typing import Iterable, Set

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sent.db")


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize SQLite database and ensure the sent_articles table exists.
    Uses 'url' as the PRIMARY KEY to enforce uniqueness at the database level.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sent_articles (
                url TEXT PRIMARY KEY,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def get_sent_urls(db_path: str = DEFAULT_DB_PATH) -> Set[str]:
    """
    Retrieve all article URLs that have been previously sent.
    If the database file does not exist yet, returns an empty set without creating the file.
    """
    if not os.path.exists(db_path):
        return set()

    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM sent_articles")
        rows = cursor.fetchall()
        return {row[0] for row in rows}


def record_sent_urls(urls: Iterable[str], db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Record successfully delivered article URLs in SQLite.
    Uses 'INSERT OR IGNORE' to prevent duplicate errors.
    Returns the count of URLs processed.
    """
    url_list = [u.strip() for u in urls if u and u.strip()]
    if not url_list:
        return 0

    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO sent_articles (url) VALUES (?)",
            [(url,) for url in url_list],
        )
        conn.commit()

    return len(url_list)

