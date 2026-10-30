"""
tests/test_phase2.py
Unit tests for Phase 2: state_store.py SQLite deduplication and graph.py LangGraph workflow.
All network, MCP tools, and LLM calls are completely mocked.
Tests run 100% offline with zero cost and zero external calls.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock
import pytest

from graph import (
    compiled_graph,
    extract_tool_result_text,
    fetch_news_node,
    filter_new_node,
    parse_articles_json,
    route_after_fetch,
    route_after_filter,
    sanitize_telegram_html,
    select_source_balanced,
    send_node,
    summarize_node,
)
import state_store


# ============================================================================
# Tests for SQLite state_store.py
# ============================================================================

def test_state_store_nonexistent_db_returns_empty(tmp_path):
    """If DB does not exist, get_sent_urls returns empty set without creating file."""
    temp_db = str(tmp_path / "nonexistent.db")
    assert not os.path.exists(temp_db)
    urls = state_store.get_sent_urls(temp_db)
    assert urls == set()
    assert not os.path.exists(temp_db)


def test_state_store_record_and_get(tmp_path):
    """Recording URLs persists them into the SQLite database."""
    temp_db = str(tmp_path / "test_sent.db")
    to_record = ["https://example.com/art1", "https://example.com/art2"]

    count = state_store.record_sent_urls(to_record, db_path=temp_db)
    assert count == 2
    assert os.path.exists(temp_db)

    stored = state_store.get_sent_urls(temp_db)
    assert stored == {"https://example.com/art1", "https://example.com/art2"}


def test_state_store_duplicate_prevention(tmp_path):
    """Duplicate URLs are ignored due to the PRIMARY KEY constraint."""
    temp_db = str(tmp_path / "test_dedup.db")
    batch1 = ["https://example.com/item1", "https://example.com/item2"]
    batch2 = ["https://example.com/item2", "https://example.com/item3"]

    state_store.record_sent_urls(batch1, db_path=temp_db)
    state_store.record_sent_urls(batch2, db_path=temp_db)

    stored = state_store.get_sent_urls(temp_db)
    assert stored == {
        "https://example.com/item1",
        "https://example.com/item2",
        "https://example.com/item3",
    }


def test_state_store_empty_inputs(tmp_path):
    """Passing empty or blank URLs handles cleanly."""
    temp_db = str(tmp_path / "test_empty.db")
    count = state_store.record_sent_urls(["", "   ", None], db_path=temp_db)
    assert count == 0
    assert state_store.get_sent_urls(temp_db) == set()


# ============================================================================
# Tests for graph.py Helpers
# ============================================================================

def test_extract_tool_result_text():
    """Extract plain text from string, list of blocks, or text objects."""
    # Plain string
    assert extract_tool_result_text("simple text") == "simple text"

    # List of strings
    assert extract_tool_result_text(["line 1", "line 2"]) == "line 1\nline 2"

    # Content block object with .text attribute
    class DummyBlock:
        def __init__(self, text):
            self.text = text

    assert extract_tool_result_text([DummyBlock("block text")]) == "block text"

    # Dict with text key
    assert extract_tool_result_text([{"text": "dict text"}]) == "dict text"


def test_parse_articles_json():
    """Safely parse JSON article payloads and handle errors/malformed data."""
    valid_json = json.dumps([
        {"title": "Story 1", "url": "https://a.com"},
        {"title": "Story 2", "url": "https://b.com"}
    ])
    parsed = parse_articles_json(valid_json)
    assert len(parsed) == 2
    assert parsed[0]["title"] == "Story 1"

    # Error JSON response
    error_json = json.dumps({"error": "Unknown source"})
    assert parse_articles_json(error_json) == []

    # Malformed text
    assert parse_articles_json("Not JSON at all") == []


def test_select_source_balanced():
    """Verify deterministic round-robin interleaving while preserving source order."""
    hn = [{"title": f"HN {i}", "source": "Hacker News"} for i in range(1, 6)]
    tc = [{"title": f"TC {i}", "source": "TechCrunch"} for i in range(1, 6)]
    verge = [{"title": f"Verge {i}", "source": "The Verge"} for i in range(1, 6)]

    all_articles = hn + tc + verge
    selected = select_source_balanced(all_articles, max_count=5)

    assert len(selected) == 5
    # Sources sorted: Hacker News, TechCrunch, The Verge
    assert selected[0]["title"] == "HN 1"
    assert selected[1]["title"] == "TC 1"
    assert selected[2]["title"] == "Verge 1"
    assert selected[3]["title"] == "HN 2"
    assert selected[4]["title"] == "TC 2"


def test_sanitize_telegram_html():
    """Sanitize HTML: escape ampersands, convert markdown bold and links."""
    raw = 'Visit [Example](https://example.com) & check **Big Tech**!'
    cleaned = sanitize_telegram_html(raw)
    assert '<a href="https://example.com">Example</a>' in cleaned
    assert '<b>Big Tech</b>' in cleaned
    assert '&amp;' in cleaned
    assert ' & ' not in cleaned

    # Pre-existing entity remains unchanged
    assert sanitize_telegram_html('Tom &amp; Jerry') == 'Tom &amp; Jerry'


def test_sanitize_telegram_html_valid_urls():
    """Verify that only valid article URLs are preserved in HTML links."""
    valid_urls = {"https://legit-news.com/ai"}
    html_with_fake = (
        'Check <b><a href="https://legit-news.com/ai">Real News</a></b> and '
        '<a href="https://hallucinated-link.com">Hallucinated News</a>.'
    )
    cleaned = sanitize_telegram_html(html_with_fake, valid_urls=valid_urls)
    assert '<a href="https://legit-news.com/ai">Real News</a>' in cleaned
    assert 'https://hallucinated-link.com' not in cleaned
    # The title text of the hallucinated link is preserved
    assert 'Hallucinated News' in cleaned


# ============================================================================
# Tests for Node B: filter_new_node
# ============================================================================

def test_filter_new_node_excludes_sent(tmp_path):
    """filter_new_node excludes already-sent URLs and keeps fresh ones."""
    temp_db = str(tmp_path / "filter_test.db")
    state_store.record_sent_urls(["https://news.com/old1", "https://news.com/old2"], db_path=temp_db)

    state = {
        "news": [
            {"title": "Old Article 1", "url": "https://news.com/old1"},
            {"title": "New Article", "url": "https://news.com/fresh"},
        ],
        "db_path": temp_db,
    }

    result = filter_new_node(state)
    assert len(result["fresh_news"]) == 1
    assert result["fresh_news"][0]["url"] == "https://news.com/fresh"
    assert "Found 1 fresh stories" in result["status"]


def test_filter_new_node_no_new_stories(tmp_path):
    """When all stories are already in sent_articles, status is 'No new stories'."""
    temp_db = str(tmp_path / "filter_empty.db")
    state_store.record_sent_urls(["https://news.com/old"], db_path=temp_db)

    state = {
        "news": [{"title": "Old Article", "url": "https://news.com/old"}],
        "db_path": temp_db,
    }

    result = filter_new_node(state)
    assert result["fresh_news"] == []
    assert result["status"] == "No new stories"


def test_filter_new_node_never_records_urls(tmp_path):
    """filter_new_node must never write to SQLite database."""
    temp_db = str(tmp_path / "never_write.db")

    state = {
        "news": [{"title": "Fresh 1", "url": "https://news.com/fresh1"}],
        "db_path": temp_db,
    }

    filter_new_node(state)
    # Database file shouldn't even be created if get_sent_urls returned empty
    assert state_store.get_sent_urls(temp_db) == set()


# ============================================================================
# Tests for Node A: fetch_news_node (Mocked MCP)
# ============================================================================

@pytest.mark.asyncio
async def test_fetch_news_node_success():
    """fetch_news_node queries tools and deduplicates URLs across sources."""
    mock_tool = MagicMock()
    mock_tool.name = "fetch_tech_news"

    async def mock_ainvoke(params):
        src = params.get("source")
        if src == "hackernews":
            return json.dumps([{"title": "HN 1", "url": "https://a.com", "source": "Hacker News"}])
        elif src == "techcrunch":
            # Returns a shared URL to test cross-source deduplication
            return json.dumps([
                {"title": "TC 1", "url": "https://a.com", "source": "TechCrunch"},
                {"title": "TC 2", "url": "https://b.com", "source": "TechCrunch"}
            ])
        else:
            return json.dumps([{"title": "Verge 1", "url": "https://c.com", "source": "The Verge"}])

    mock_tool.ainvoke = AsyncMock(side_effect=mock_ainvoke)

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[mock_tool])

    state = {}
    result = await fetch_news_node(state, mcp_client=mock_client)

    assert "news" in result
    # Total unique URLs: https://a.com, https://b.com, https://c.com -> 3
    assert len(result["news"]) == 3
    urls = [item["url"] for item in result["news"]]
    assert urls == ["https://a.com", "https://b.com", "https://c.com"]
    assert "Fetched 3 articles" in result["status"]


@pytest.mark.asyncio
async def test_fetch_news_node_tool_missing():
    """If fetch_tech_news is missing from MCP tools, returns error status without crash."""
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    result = await fetch_news_node({}, mcp_client=mock_client)
    assert result["news"] == []
    assert "Error: Fetch failed" in result["status"]


# ============================================================================
# Tests for Node C: summarize_node (Mocked LLM)
# ============================================================================

@pytest.mark.asyncio
async def test_summarize_node_success():
    """summarize_node formats prompt and returns sanitized HTML."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '<b><a href="https://news.com/1">AI Breakthrough</a></b> - A new model was released.'
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    state = {
        "fresh_news": [
            {"title": "AI Breakthrough", "url": "https://news.com/1", "source": "Hacker News", "summary": "Great model"}
        ]
    }

    result = await summarize_node(state, llm_instance=mock_llm)
    assert "AI Breakthrough" in result["digest"]
    assert "<b><a href=" in result["digest"]
    assert "Generated digest" in result["status"]


@pytest.mark.asyncio
async def test_summarize_node_quota_exhaustion():
    """When Gemini quota is exceeded, gracefully reports error without crashing or paying."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("429 RESOURCE_EXHAUSTED: quota limit reached"))

    state = {
        "fresh_news": [{"title": "News 1", "url": "https://news.com/1", "source": "TechCrunch"}]
    }

    result = await summarize_node(state, llm_instance=mock_llm)
    assert result["digest"] == ""
    assert "quota exceeded" in result["status"].lower()
    assert "paid usage" in result["status"].lower()


# ============================================================================
# Tests for Node D: send_node & URL Recording Verification
# ============================================================================

@pytest.mark.asyncio
async def test_send_node_success_records_urls(tmp_path):
    """URLs are recorded in SQLite ONLY AFTER Telegram confirms successful delivery."""
    temp_db = str(tmp_path / "send_success.db")

    mock_tool = MagicMock()
    mock_tool.name = "send_telegram_message"
    mock_tool.ainvoke = AsyncMock(return_value="Successfully sent message to Telegram (1 chunk).")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[mock_tool])

    state = {
        "digest": "<b><a href=\"https://news.com/fresh\">Title</a></b> - Explanation.",
        "fresh_news": [{"title": "Title", "url": "https://news.com/fresh"}],
        "db_path": temp_db,
    }

    result = await send_node(state, mcp_client=mock_client)
    assert "Successfully sent digest" in result["status"]

    # Verify URL is now stored in SQLite
    stored = state_store.get_sent_urls(temp_db)
    assert "https://news.com/fresh" in stored


@pytest.mark.asyncio
async def test_send_node_failure_never_records_urls(tmp_path):
    """If Telegram returns an error, URLs are NOT recorded in SQLite."""
    temp_db = str(tmp_path / "send_fail.db")

    mock_tool = MagicMock()
    mock_tool.name = "send_telegram_message"
    mock_tool.ainvoke = AsyncMock(return_value="Telegram API error on chunk 1/1 (HTTP status 400): Bad Request")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[mock_tool])

    state = {
        "digest": "<b><a href=\"https://news.com/fresh\">Title</a></b> - Explanation.",
        "fresh_news": [{"title": "Title", "url": "https://news.com/fresh"}],
        "db_path": temp_db,
    }

    result = await send_node(state, mcp_client=mock_client)
    assert "Telegram delivery failed" in result["status"]

    # Verify URL is NOT recorded
    stored = state_store.get_sent_urls(temp_db)
    assert "https://news.com/fresh" not in stored


@pytest.mark.asyncio
async def test_send_node_error_alert():
    """If status has an Error and no digest, sends safe error alert to Telegram."""
    mock_tool = MagicMock()
    mock_tool.name = "send_telegram_message"
    mock_tool.ainvoke = AsyncMock(return_value="Successfully sent message to Telegram (1 chunk).")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[mock_tool])

    state = {
        "status": "Error: Fetch failed - network timeout",
    }

    result = await send_node(state, mcp_client=mock_client)
    assert "Sent error alert to Telegram" in result["status"]
    # Check payload sent to tool
    sent_payload = mock_tool.ainvoke.call_args[0][0]["text"]
    assert "Tech News Bot Alert" in sent_payload
    assert "Error: Fetch failed" in sent_payload


# ============================================================================
# Tests for Graph Routing & Structure
# ============================================================================

def test_routing_after_fetch():
    """route_after_fetch sends errors to send node and successes to filter_new."""
    assert route_after_fetch({"status": "Error: Fetch failed"}) == "send"
    assert route_after_fetch({"status": "Fetched 10 articles"}) == "filter_new"


def test_routing_after_filter():
    """route_after_filter routes to END if no new stories, else summarize."""
    assert route_after_filter({"status": "No new stories", "fresh_news": []}) == "__end__"
    assert route_after_filter({"status": "Found 5 stories", "fresh_news": [{"url": "a"}]}) == "summarize"


def test_compiled_graph_structure():
    """Ensure compiled graph contains all 4 nodes."""
    nodes = compiled_graph.get_graph().nodes
    assert "fetch_news" in nodes
    assert "filter_new" in nodes
    assert "summarize" in nodes
    assert "send" in nodes

