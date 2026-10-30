"""
Unit tests for tech-news-mcp server.
All network operations and credentials are fully mocked.
No real API keys, internet connection, or Telegram credentials required.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from server import fetch_tech_news, send_telegram_message, split_message


# ============================================================================
# Tests for Chunk-Splitting Logic (split_message)
# ============================================================================

def test_split_message_empty():
    """Empty string returns empty list."""
    assert split_message("") == []


def test_split_message_under_limit():
    """Text shorter than limit should not be split."""
    text = "Hello world! This is a simple test."
    chunks = split_message(text, max_chunk_size=100)
    assert chunks == [text]


def test_split_message_exact_limit():
    """Text exactly equal to the limit should be a single chunk."""
    text = "a" * 100
    chunks = split_message(text, max_chunk_size=100)
    assert chunks == [text]
    assert len(chunks[0]) == 100


def test_split_message_prefer_line_boundary():
    """Text longer than limit should prefer splitting at the last newline before limit."""
    line1 = "Line 1: Important tech news.\n"
    line2 = "Line 2: Another article details.\n"
    line3 = "Line 3: Final conclusions."
    full_text = line1 + line2 + line3

    # Limit set so line1 + line2 fits in chunk 1, but adding line3 exceeds limit
    limit = len(line1) + len(line2) + 5
    chunks = split_message(full_text, max_chunk_size=limit)

    assert len(chunks) == 2
    assert chunks[0] == line1 + line2
    assert chunks[1] == line3
    assert "".join(chunks) == full_text
    assert all(len(c) <= limit for c in chunks)


def test_split_message_hard_split_without_newlines():
    """If no newlines exist, text must be hard-split without dropping characters."""
    long_line = "x" * 250
    chunks = split_message(long_line, max_chunk_size=100)

    assert len(chunks) == 3
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 50
    assert "".join(chunks) == long_line


def test_split_message_default_4000_char_limit():
    """Verify behavior with default 4000-character Telegram chunk size."""
    paragraph = "Short sentence.\n" * 300  # ~4800 characters
    chunks = split_message(paragraph, max_chunk_size=4000)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 4000
    assert "".join(chunks) == paragraph


# ============================================================================
# Tests for Unknown-Source Error in fetch_tech_news
# ============================================================================

@pytest.mark.asyncio
async def test_fetch_tech_news_unknown_source():
    """Requesting an unsupported news source returns a descriptive JSON error."""
    result = await fetch_tech_news(source="unknown_source_xyz", limit=5)
    data = json.loads(result)

    assert isinstance(data, dict)
    assert "error" in data
    assert "unknown_source_xyz" in data["error"]
    assert "hackernews" in data["error"]


@pytest.mark.asyncio
async def test_fetch_tech_news_empty_source():
    """Blank or whitespace source also returns a JSON error."""
    result = await fetch_tech_news(source="   ", limit=5)
    data = json.loads(result)

    assert isinstance(data, dict)
    assert "error" in data


# ============================================================================
# Tests for Clamping and Mocked Fetching in fetch_tech_news
# ============================================================================

@pytest.mark.asyncio
async def test_fetch_tech_news_mocked_hn():
    """Verify Hacker News fetching returns formatted JSON when mocked."""
    fake_top_stories = [101, 102]
    fake_item_1 = {
        "id": 101,
        "type": "story",
        "title": "Show HN: My Cool Project",
        "url": "https://example.com/show",
        "score": 150,
    }
    fake_item_2 = {
        "id": 102,
        "type": "story",
        "title": "Ask HN: What is your favorite tool?",
        "url": None,
        "score": 42,
    }

    async def mock_get(self, url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        if "topstories.json" in str(url):
            mock_resp.json.return_value = fake_top_stories
        elif "101.json" in str(url):
            mock_resp.json.return_value = fake_item_1
        elif "102.json" in str(url):
            mock_resp.json.return_value = fake_item_2
        else:
            mock_resp.status_code = 404
        return mock_resp

    with patch("httpx.AsyncClient.get", new=mock_get):
        result_json = await fetch_tech_news(source="hackernews", limit=10)
        articles = json.loads(result_json)

        assert isinstance(articles, list)
        assert len(articles) == 2
        assert articles[0]["title"] == "Show HN: My Cool Project"
        assert articles[0]["source"] == "Hacker News"
        assert articles[0]["score"] == 150
        assert articles[1]["title"] == "Ask HN: What is your favorite tool?"
        assert articles[1]["url"] == "https://news.ycombinator.com/item?id=102"


@pytest.mark.asyncio
async def test_fetch_tech_news_mocked_rss():
    """Verify RSS feed parsing returns formatted JSON when mocked."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>TechCrunch</title>
        <item>
          <title>AI Startup Raises Seed Round</title>
          <link>https://techcrunch.com/seed-round</link>
          <description>&lt;p&gt;Startup announces new funding.&lt;/p&gt;</description>
        </item>
      </channel>
    </rss>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_xml

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        result_json = await fetch_tech_news(source="techcrunch", limit=5)
        articles = json.loads(result_json)

        assert isinstance(articles, list)
        assert len(articles) == 1
        assert articles[0]["title"] == "AI Startup Raises Seed Round"
        assert articles[0]["url"] == "https://techcrunch.com/seed-round"
        assert articles[0]["source"] == "TechCrunch"
        assert "Startup announces new funding." in articles[0]["summary"]


@pytest.mark.asyncio
async def test_fetch_tech_news_limit_clamping():
    """Verify limit is clamped to the range [1, 30]."""
    captured_limits = []

    async def mock_fetch_hn(limit):
        captured_limits.append(limit)
        return []

    with patch("server._fetch_hn_news", side_effect=mock_fetch_hn):
        # Below minimum -> clamped to 1
        await fetch_tech_news(source="hackernews", limit=-5)
        # Above maximum -> clamped to 30
        await fetch_tech_news(source="hackernews", limit=100)
        # Invalid type -> fallback to 10
        await fetch_tech_news(source="hackernews", limit="invalid")

    assert captured_limits == [1, 30, 10]


# ============================================================================
# Tests for send_telegram_message (Mocked / No Real Credentials)
# ============================================================================

@pytest.mark.asyncio
async def test_send_telegram_missing_env_vars(monkeypatch):
    """Missing bot token or chat ID returns a clean error string."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    res = await send_telegram_message("Test message")
    assert "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID" in res


@pytest.mark.asyncio
async def test_send_telegram_empty_text(monkeypatch):
    """Sending empty text returns an error string."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock_dummy_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456789")

    res = await send_telegram_message("   ")
    assert "Error: Message text cannot be empty." in res


@pytest.mark.asyncio
async def test_send_telegram_success_mocked(monkeypatch):
    """Successful API response returns confirmation without leaking credentials."""
    dummy_token = "dummy_token_never_printed"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", dummy_token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456789")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '{"ok": true, "result": {}}'

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        res = await send_telegram_message("<b>Hello</b> Telegram!")
        assert "Successfully sent message to Telegram" in res
        assert dummy_token not in res


@pytest.mark.asyncio
async def test_send_telegram_http_error_mocked(monkeypatch):
    """Non-200 HTTP response returns clear error without leaking the secret token."""
    dummy_token = "secret_token_123"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", dummy_token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456789")

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = '{"ok": false, "description": "Forbidden: bot was blocked by the user"}'

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        res = await send_telegram_message("Test message")
        assert "Telegram API error on chunk 1/1 (HTTP status 403)" in res
        assert "Forbidden" in res
        # Ensure secret token is never echoed in the output
        assert dummy_token not in res
