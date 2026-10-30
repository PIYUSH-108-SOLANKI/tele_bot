"""
chat_memory.py
SQLite-backed persistent conversation memory for the conversational Telegram AI assistant.
Stores per-chat message history in chat_memory.db with thread isolation.
Does not create the database upon module import.
"""

import os
import sqlite3
from typing import Dict, List, Union

DEFAULT_CHAT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_memory.db")


def init_chat_memory_db(db_path: str = DEFAULT_CHAT_DB_PATH) -> None:
    """
    Initialize SQLite database and ensure the conversation_messages table exists.
    Isolates chats by chat_id and maintains chronological message ordering.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_lookup 
            ON conversation_messages(chat_id, id)
            """
        )
        conn.commit()


def get_chat_history(
    chat_id: Union[str, int],
    limit: int = 10,
    db_path: str = DEFAULT_CHAT_DB_PATH,
) -> List[Dict[str, str]]:
    """
    Retrieve the most recent `limit` messages for a specific chat_id in chronological order.
    Returns an empty list if the database file does not exist without creating the file.
    """
    if not os.path.exists(db_path):
        return []

    str_chat_id = str(chat_id)
    init_chat_memory_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Fetch the latest N messages, then order them chronologically (oldest to newest)
        cursor.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content
                FROM conversation_messages
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (str_chat_id, limit),
        )
        rows = cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]


def add_chat_message(
    chat_id: Union[str, int],
    role: str,
    content: str,
    db_path: str = DEFAULT_CHAT_DB_PATH,
) -> None:
    """
    Append a single user or assistant message to the chat history.
    """
    if not content or not str(content).strip():
        return

    str_chat_id = str(chat_id)
    init_chat_memory_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversation_messages (chat_id, role, content)
            VALUES (?, ?, ?)
            """,
            (str_chat_id, role.strip().lower(), str(content).strip()),
        )
        conn.commit()


def clear_chat_history(
    chat_id: Union[str, int],
    db_path: str = DEFAULT_CHAT_DB_PATH,
) -> int:
    """
    Clear all stored messages for a specific chat_id (e.g. for /reset command).
    Returns the number of deleted records.
    """
    if not os.path.exists(db_path):
        return 0

    str_chat_id = str(chat_id)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM conversation_messages WHERE chat_id = ?",
            (str_chat_id,),
        )
        deleted = cursor.rowcount
        conn.commit()
        return deleted

