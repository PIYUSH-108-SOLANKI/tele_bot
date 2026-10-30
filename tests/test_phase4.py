"""
tests/test_phase4.py
Unit tests for Phase 4 fixes:
1. Extracting clean text from MCP content blocks and LangChain/Gemini wrapper objects.
2. Stripping unwanted wrapper representations like content=[{'type': 'text', ...}].
3. Converting escaped newline sequences (\\n\\n) to real paragraph breaks.
4. Limiting digests to at most 10 stories and tracking selected stories.
5. Recording ONLY the URLs of stories actually included in the delivered digest.
6. Ensuring zero URLs are recorded when Telegram delivery fails.
"""

from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
import pytest

from graph import (
    build_telegram_digest,
    extract_text,
    parse_numbered_explanations,
    send_node,
    summarize_node,
)
import state_store


# ============================================================================
# 1 & 2: Text Extraction from MCP Blocks, Gemini Wrappers & Dicts
# ============================================================================

def test_extract_text_from_mcp_content_blocks():
    """Extract plain text from MCP content block objects and lists of blocks."""
    class DummyMCPTextBlock:
        def __init__(self, text: str):
            self.text = text

    blocks = [DummyMCPTextBlock("Hacker News Item 1"), DummyMCPTextBlock("TechCrunch Item 2")]
    extracted = extract_text(blocks)
    assert extracted == "Hacker News Item 1\nTechCrunch Item 2"


def test_extract_text_from_gemini_string_and_blocks():
    """Extract plain text from AIMessage with string content or content-block list."""
    # Case A: Plain string AIMessage
    msg_str = AIMessage(content="Mistral released a new model.")
    assert extract_text(msg_str) == "Mistral released a new model."

    # Case B: List of content block dicts (used by modern google-genai SDK)
    msg_blocks = AIMessage(content=[
        {"type": "text", "text": "1. Explanation for story one."},
        {"type": "text", "text": "2. Explanation for story two."}
    ])
    extracted = extract_text(msg_blocks)
    assert extracted == "1. Explanation for story one.\n2. Explanation for story two."


def test_extract_text_removes_unwanted_wrapper_string():
    """If a raw stringified wrapper is received, extract the clean inner text."""
    # Simulates what happened when str(response) was called on AIMessage
    wrapper_repr = "content=[{'type': 'text', 'text': 'Mistral AI releases weights of Mistral Large 4'}]"
    clean = extract_text(wrapper_repr)
    assert "content=[" not in clean
    assert clean == "Mistral AI releases weights of Mistral Large 4"


def test_extract_text_converts_escaped_newlines():
    """Convert escaped \\n sequences into actual line breaks."""
    raw = "Headline 1\\n\\nHeadline 2\\n\\nHeadline 3"
    clean = extract_text(raw)
    assert "\\n" not in clean
    assert clean == "Headline 1\n\nHeadline 2\n\nHeadline 3"


# ============================================================================
# 3: Deterministic HTML Digest Construction
# ============================================================================

def test_build_telegram_digest_deterministic_format():
    """Build clean Telegram HTML digest from verified articles with proper line breaks."""
    articles = [
        {"title": "C++ & Rust <Innovations>", "url": "https://example.com/rust", "source": "Hacker News"},
        {"title": "AI Chip Breakthrough", "url": "https://example.com/chip", "source": "TechCrunch"}
    ]
    explanations = [
        "Memory-safe languages continue to gain adoption in systems programming.",
        "New silicon design drastically cuts inference power requirements."
    ]

    digest = build_telegram_digest(articles, explanations)

    # Paragraph breaks check
    assert "\n\n" in digest
    # Escaping check for & and < >
    assert "C++ &amp; Rust &lt;Innovations&gt;" in digest
    # Verified link check
    assert '<a href="https://example.com/rust">' in digest
    assert '<a href="https://example.com/chip">' in digest
    # Clean text without wrapper prefixes
    assert "content=" not in digest


def test_parse_numbered_explanations():
    """Parse numbered items from model output cleanly."""
    raw = "1. First sentence explaining story one.\n2. Second sentence explaining story two."
    parsed = parse_numbered_explanations(raw, expected_count=2)
    assert len(parsed) == 2
    assert parsed[0] == "First sentence explaining story one."
    assert parsed[1] == "Second sentence explaining story two."


# ============================================================================
# 4 & 5: Story Selection Limit (At Most 10) & URL Recording
# ============================================================================

@pytest.mark.asyncio
async def test_summarize_node_clamps_to_10_stories():
    """Ensure summarize_node selects at most 10 stories even when 30 are available."""
    # Create 30 mock fresh articles (10 from each source)
    mock_30_articles = []
    for i in range(1, 11):
        mock_30_articles.append({"title": f"HN {i}", "url": f"https://news.ycombinator.com/{i}", "source": "Hacker News"})
        mock_30_articles.append({"title": f"TC {i}", "url": f"https://techcrunch.com/{i}", "source": "TechCrunch"})
        mock_30_articles.append({"title": f"Verge {i}", "url": f"https://theverge.com/{i}", "source": "The Verge"})

    assert len(mock_30_articles) == 30

    mock_llm = MagicMock()
    # Return 10 numbered explanations
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=[
        {"type": "text", "text": "\n".join(f"{i}. Explanation for story {i}." for i in range(1, 11))}
    ]))

    state = {
        "fresh_news": mock_30_articles,
    }

    result = await summarize_node(state, llm_instance=mock_llm)

    # Verify at most 10 stories are selected and tracked
    assert len(result["selected_news"]) == 10
    # Verify digest contains exactly 10 links
    assert result["digest"].count("<a href=") == 10
    assert "content=[" not in result["digest"]


@pytest.mark.asyncio
async def test_send_node_records_only_selected_urls(tmp_path):
    """Ensure send_node records ONLY the 10 selected URLs, not all 30 fresh URLs."""
    temp_db = str(tmp_path / "test_only_10.db")

    # 30 fresh articles exist in state
    all_30_fresh = [{"title": f"Story {i}", "url": f"https://example.com/{i}"} for i in range(1, 31)]
    # Exactly 10 were selected for the digest
    selected_10 = all_30_fresh[:10]

    mock_tool = MagicMock()
    mock_tool.name = "send_telegram_message"
    mock_tool.ainvoke = AsyncMock(return_value="Successfully sent message to Telegram (1 chunk).")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[mock_tool])

    state = {
        "digest": "<b><a href=\"https://example.com/1\">Story 1</a></b>\nExplanation.",
        "fresh_news": all_30_fresh,
        "selected_news": selected_10,
        "db_path": temp_db,
    }

    result = await send_node(state, mcp_client=mock_client)

    # Status must report 10 URLs recorded, NOT 30
    assert "recorded 10 URLs" in result["status"]

    # Verify database contains exactly the 10 selected URLs
    recorded_urls = state_store.get_sent_urls(temp_db)
    assert len(recorded_urls) == 10
    assert "https://example.com/1" in recorded_urls
    assert "https://example.com/10" in recorded_urls
    assert "https://example.com/11" not in recorded_urls
    assert "https://example.com/30" not in recorded_urls


@pytest.mark.asyncio
async def test_send_node_failure_never_records_urls(tmp_path):
    """When Telegram delivery fails, zero URLs are recorded in the database."""
    temp_db = str(tmp_path / "test_failure.db")

    selected_10 = [{"title": f"Story {i}", "url": f"https://example.com/{i}"} for i in range(1, 11)]

    mock_tool = MagicMock()
    mock_tool.name = "send_telegram_message"
    mock_tool.ainvoke = AsyncMock(return_value="Telegram API error on chunk 1/1 (HTTP status 400): Bad Request")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[mock_tool])

    state = {
        "digest": "<b><a href=\"https://example.com/1\">Story 1</a></b>\nExplanation.",
        "fresh_news": selected_10,
        "selected_news": selected_10,
        "db_path": temp_db,
    }

    result = await send_node(state, mcp_client=mock_client)

    assert "Telegram delivery failed" in result["status"]
    # Database must remain completely empty
    recorded_urls = state_store.get_sent_urls(temp_db)
    assert recorded_urls == set()

