"""
tech-news-mcp Server
A beginner-friendly Model Context Protocol (MCP) server providing tools to:
1. Fetch tech news from Hacker News, TechCrunch, and The Verge.
2. Send chunked HTML messages to Telegram.
"""

import asyncio
import html
import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import feedparser
import httpx
from mcp.server.fastmcp import FastMCP

# Load local .env if present (never logs or prints credentials)
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("tech-news-mcp")

# RSS Feed URLs
TECHCRUNCH_FEED_URL = "https://techcrunch.com/feed/"
THE_VERGE_FEED_URL = "https://www.theverge.com/rss/index.xml"
HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL_TEMPLATE = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"

DEFAULT_USER_AGENT = "tech-news-mcp/1.0 (College Mini-Project)"


def split_message(text: str, max_chunk_size: int = 4000) -> List[str]:
    """
    Split text into chunks of at most max_chunk_size characters.
    Prefers splitting along line boundaries (newlines).
    If a line exceeds max_chunk_size, it is cleanly hard-split.
    """
    if not text:
        return []

    chunks: List[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chunk_size:
            chunks.append(remaining)
            break

        # Look for the last newline within the allowed chunk limit
        split_idx = remaining.rfind("\n", 0, max_chunk_size)

        # If no newline is found within the limit (or it's at index 0), hard-split
        if split_idx <= 0:
            split_idx = max_chunk_size
            chunk = remaining[:split_idx]
            remaining = remaining[split_idx:]
        else:
            # Include the newline at the end of the current chunk
            chunk = remaining[:split_idx + 1]
            remaining = remaining[split_idx + 1:]

        if chunk:
            chunks.append(chunk)

    return chunks


def _strip_html(raw_html: str) -> str:
    """Helper to remove HTML tags and unescape entities for clean summaries."""
    if not raw_html:
        return ""
    clean = re.sub(r"<[^>]+>", "", raw_html)
    return html.unescape(clean).strip()


async def _fetch_hn_item(client: httpx.AsyncClient, item_id: int) -> Optional[Dict[str, Any]]:
    """Fetch individual story details from Hacker News Firebase API."""
    url = HN_ITEM_URL_TEMPLATE.format(item_id=item_id)
    try:
        response = await client.get(url, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if data and data.get("type") == "story":
                return {
                    "title": data.get("title", "Untitled"),
                    "url": data.get("url") or f"https://news.ycombinator.com/item?id={item_id}",
                    "source": "Hacker News",
                    "score": data.get("score", 0),
                }
    except Exception:
        pass
    return None


async def _fetch_hn_news(limit: int) -> List[Dict[str, Any]]:
    """Fetch top tech stories concurrently from Hacker News Firebase API."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
        response = await client.get(HN_TOP_STORIES_URL)
        if response.status_code != 200:
            return []

        story_ids: List[int] = response.json()[:limit]
        # Concurrently fetch details for all selected story IDs
        tasks = [_fetch_hn_item(client, item_id) for item_id in story_ids]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        return [item for item in results if item is not None]


def _parse_rss_content(xml_text: str, source_name: str, limit: int) -> List[Dict[str, Any]]:
    """Synchronous parsing of RSS feed XML using feedparser."""
    feed = feedparser.parse(xml_text)
    articles: List[Dict[str, Any]] = []

    for entry in feed.entries[:limit]:
        title = entry.get("title", "Untitled")
        link = entry.get("link", "")
        summary_raw = entry.get("summary", "") or entry.get("description", "")
        clean_summary = _strip_html(summary_raw)

        # Truncate overly long summaries to keep responses concise
        if len(clean_summary) > 300:
            clean_summary = clean_summary[:297] + "..."

        articles.append({
            "title": title,
            "url": link,
            "source": source_name,
            "summary": clean_summary,
        })

    return articles


async def _fetch_rss_news(source_name: str, feed_url: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch RSS XML asynchronously and parse it in a background thread."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    async with httpx.AsyncClient(headers=headers, timeout=15.0, follow_redirects=True) as client:
        response = await client.get(feed_url)
        if response.status_code != 200:
            return []
        xml_text = response.text

    # Offload blocking feedparser execution to a separate worker thread
    return await asyncio.to_thread(_parse_rss_content, xml_text, source_name, limit)


@mcp.tool()
async def fetch_tech_news(source: str = "hackernews", limit: int = 10) -> str:
    """
    Fetch the latest tech news stories from Hacker News, TechCrunch, or The Verge.

    Args:
        source: News source to fetch from ("hackernews", "techcrunch", "theverge"). Default: "hackernews".
        limit: Number of articles to fetch (clamped to 1-30). Default: 10.

    Returns:
        JSON string containing the list of articles or an error message.
    """
    # Clamp limit to range 1-30
    try:
        clamped_limit = max(1, min(int(limit), 30))
    except (ValueError, TypeError):
        clamped_limit = 10

    normalized_source = source.strip().lower()

    if normalized_source in ("hackernews", "hacker news", "hn"):
        articles = await _fetch_hn_news(clamped_limit)
        return json.dumps(articles, indent=2)

    elif normalized_source in ("techcrunch", "tc"):
        articles = await _fetch_rss_news("TechCrunch", TECHCRUNCH_FEED_URL, clamped_limit)
        return json.dumps(articles, indent=2)

    elif normalized_source in ("theverge", "the verge", "verge"):
        articles = await _fetch_rss_news("The Verge", THE_VERGE_FEED_URL, clamped_limit)
        return json.dumps(articles, indent=2)

    else:
        return json.dumps({
            "error": f"Unknown source '{source}'. Supported sources are: 'hackernews', 'techcrunch', 'theverge'."
        }, indent=2)


@mcp.tool()
async def send_telegram_message(text: str) -> str:
    """
    Send a message to a Telegram chat using Telegram Bot API.
    Splits long messages (>4000 characters) into chunks along line boundaries.
    Credentials are read securely at runtime from environment variables.

    Args:
        text: The message content in HTML format.

    Returns:
        Status message describing success or error.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variable is missing."

    if not text or not text.strip():
        return "Error: Message text cannot be empty."

    chunks = split_message(text, max_chunk_size=4000)
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    async with httpx.AsyncClient(timeout=15.0) as client:
        for idx, chunk in enumerate(chunks, start=1):
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            try:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    # Never reveal bot_token in error messages
                    return (
                        f"Telegram API error on chunk {idx}/{len(chunks)} "
                        f"(HTTP status {response.status_code}): {response.text}"
                    )
            except Exception as e:
                return f"Network error sending chunk {idx}/{len(chunks)} to Telegram: {type(e).__name__}"

    chunk_desc = f"{len(chunks)} chunk" if len(chunks) == 1 else f"{len(chunks)} chunks"
    return f"Successfully sent message to Telegram ({chunk_desc})."


if __name__ == "__main__":
    mcp.run(transport="stdio")

