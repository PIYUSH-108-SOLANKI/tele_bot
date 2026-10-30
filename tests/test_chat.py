"""
tests/test_chat.py
Offline unit tests for Phase 5 Conversational AI Assistant:
1. SQLite-backed conversation memory persistence and thread isolation.
2. LangGraph conversational state machine with context and news intent.
3. Telegram message processing, authorization guards, and /reset command.
4. Error handling: quota exhaustion, Telegram delivery failures, and active webhooks.
All LLM and Telegram operations are 100% mocked. No real network or credentials required.
"""

from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
import pytest

import chat_bot
import chat_graph
import chat_memory


# ============================================================================
# 1. Tests for chat_memory.py
# ============================================================================

def test_chat_memory_empty_database(tmp_path):
    """If database file does not exist, get_chat_history returns empty list without creating file."""
    temp_db = str(tmp_path / "nonexistent_chat.db")
    history = chat_memory.get_chat_history("12345", db_path=temp_db)
    assert history == []


def test_chat_memory_add_and_retrieve(tmp_path):
    """Add user and assistant messages, and verify chronological retrieval."""
    temp_db = str(tmp_path / "memory_test.db")
    chat_id = "test_user_100"

    chat_memory.add_chat_message(chat_id, "user", "Hello there!", db_path=temp_db)
    chat_memory.add_chat_message(chat_id, "assistant", "Hi! How can I help today?", db_path=temp_db)

    history = chat_memory.get_chat_history(chat_id, limit=10, db_path=temp_db)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello there!"}
    assert history[1] == {"role": "assistant", "content": "Hi! How can I help today?"}


def test_chat_memory_sliding_window_limit(tmp_path):
    """Limit parameter retrieves only the most recent N messages in chronological order."""
    temp_db = str(tmp_path / "sliding_test.db")
    chat_id = "test_user_200"

    # Add 10 messages
    for i in range(1, 11):
        chat_memory.add_chat_message(chat_id, "user", f"Question {i}", db_path=temp_db)
        chat_memory.add_chat_message(chat_id, "assistant", f"Answer {i}", db_path=temp_db)

    # Fetch last 4 messages (limit=4)
    history = chat_memory.get_chat_history(chat_id, limit=4, db_path=temp_db)
    assert len(history) == 4
    assert history[0]["content"] == "Question 9"
    assert history[1]["content"] == "Answer 9"
    assert history[2]["content"] == "Question 10"
    assert history[3]["content"] == "Answer 10"


def test_chat_memory_thread_isolation(tmp_path):
    """Different chat IDs maintain strictly isolated message histories."""
    temp_db = str(tmp_path / "isolation_test.db")

    chat_memory.add_chat_message("user_A", "user", "Message for A", db_path=temp_db)
    chat_memory.add_chat_message("user_B", "user", "Message for B", db_path=temp_db)

    history_a = chat_memory.get_chat_history("user_A", db_path=temp_db)
    history_b = chat_memory.get_chat_history("user_B", db_path=temp_db)

    assert len(history_a) == 1
    assert history_a[0]["content"] == "Message for A"

    assert len(history_b) == 1
    assert history_b[0]["content"] == "Message for B"


def test_chat_memory_clear_history(tmp_path):
    """clear_chat_history wipes conversation for target chat_id while preserving others."""
    temp_db = str(tmp_path / "clear_test.db")

    chat_memory.add_chat_message("target_user", "user", "To be cleared", db_path=temp_db)
    chat_memory.add_chat_message("other_user", "user", "To be kept", db_path=temp_db)

    deleted_count = chat_memory.clear_chat_history("target_user", db_path=temp_db)
    assert deleted_count == 1

    assert chat_memory.get_chat_history("target_user", db_path=temp_db) == []
    assert len(chat_memory.get_chat_history("other_user", db_path=temp_db)) == 1


# ============================================================================
# 2. Tests for chat_graph.py
# ============================================================================

def test_news_intent_detection():
    """Verify is_asking_for_news correctly detects news queries vs general chat."""
    assert chat_graph.is_asking_for_news("What is the latest tech news?") is True
    assert chat_graph.is_asking_for_news("tell me today's news headlines") is True
    assert chat_graph.is_asking_for_news("any trending news about AI?") is True

    assert chat_graph.is_asking_for_news("hello, good morning!") is False
    assert chat_graph.is_asking_for_news("can you explain what a binary search tree is?") is False
    assert chat_graph.is_asking_for_news("how do I prepare for a Python interview?") is False


@pytest.mark.asyncio
async def test_chat_graph_general_conversation(tmp_path):
    """Generate conversational answer and persist user + assistant turns into SQLite."""
    temp_db = str(tmp_path / "graph_chat.db")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Hello! I am doing great, how can I help you today?"))

    response = await chat_graph.answer_user_message(
        chat_id="user_123",
        user_message="Hey, how are you?",
        db_path=temp_db,
        llm_instance=mock_llm,
    )

    assert "Hello! I am doing great" in response

    # Check conversation history was saved to SQLite
    history = chat_memory.get_chat_history("user_123", db_path=temp_db)
    assert len(history) == 2
    assert history[0]["content"] == "Hey, how are you?"
    assert history[1]["content"] == "Hello! I am doing great, how can I help you today?"


@pytest.mark.asyncio
async def test_chat_graph_follow_up_context(tmp_path):
    """Follow-up questions receive previous conversation turns in the model prompt."""
    temp_db = str(tmp_path / "context_test.db")
    chat_id = "user_456"

    # Pre-populate history in memory
    chat_memory.add_chat_message(chat_id, "user", "My name is Piyush.", db_path=temp_db)
    chat_memory.add_chat_message(chat_id, "assistant", "Nice to meet you, Piyush!", db_path=temp_db)

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Your name is Piyush!"))

    await chat_graph.answer_user_message(
        chat_id=chat_id,
        user_message="What is my name?",
        db_path=temp_db,
        llm_instance=mock_llm,
    )

    # Inspect messages passed into mock_llm.ainvoke
    call_args = mock_llm.ainvoke.call_args[0][0]
    # Check that previous turns were included in prompt
    user_msgs_in_prompt = [m.content for m in call_args if m.type == "human"]
    assert "My name is Piyush." in user_msgs_in_prompt
    assert "What is my name?" in user_msgs_in_prompt


@pytest.mark.asyncio
async def test_chat_graph_news_intent_with_tool(tmp_path):
    """When news is requested, the news tool is invoked and its headlines are passed to prompt."""
    temp_db = str(tmp_path / "news_tool_test.db")

    async def mock_fetch_news(source="hackernews", limit=5):
        return '[{"title": "Breakthrough in Quantum Computing", "url": "https://example.com/quantum"}]'

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Here is the latest: Quantum Computing reached a milestone."))

    response = await chat_graph.answer_user_message(
        chat_id="user_789",
        user_message="What is the latest tech news today?",
        db_path=temp_db,
        llm_instance=mock_llm,
        fetch_tool_func=mock_fetch_news,
    )

    assert "Quantum Computing" in response
    # Verify verified news context was injected into system message
    call_args = mock_llm.ainvoke.call_args[0][0]
    system_msg = next(m.content for m in call_args if m.type == "system")
    assert "Breakthrough in Quantum Computing" in system_msg
    assert "https://example.com/quantum" in system_msg


@pytest.mark.asyncio
async def test_chat_graph_quota_error(tmp_path):
    """When Gemini quota is exhausted, responds with a polite free-tier explanation."""
    temp_db = str(tmp_path / "quota_test.db")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("429 RESOURCE_EXHAUSTED: quota limit reached"))

    response = await chat_graph.answer_user_message(
        chat_id="user_quota",
        user_message="Explain recursion",
        db_path=temp_db,
        llm_instance=mock_llm,
    )

    assert "free-tier" in response.lower()
    assert "quota" in response.lower()


# ============================================================================
# 3. Tests for chat_bot.py (Telegram Polling & Security)
# ============================================================================

def test_is_authorized_chat():
    """Only configured chat ID is authorized; others are strictly rejected."""
    authorized_id = "12345678"
    assert chat_bot.is_authorized_chat(12345678, authorized_id) is True
    assert chat_bot.is_authorized_chat("12345678", authorized_id) is True
    assert chat_bot.is_authorized_chat("99999999", authorized_id) is False
    assert chat_bot.is_authorized_chat(None, authorized_id) is False


@pytest.mark.asyncio
async def test_process_telegram_unauthorized_user_dropped(tmp_path):
    """Messages from unauthorized users are silently dropped without invoking AI or replying."""
    temp_db = str(tmp_path / "unauth_test.db")
    update = {
        "update_id": 1,
        "message": {
            "chat": {"id": 99999999},  # Unauthorized chat ID
            "text": "Hello stranger!"
        }
    }
    mock_client = MagicMock()

    processed = await chat_bot.process_telegram_update(
        update=update,
        bot_token="dummy_token",
        authorized_chat_id="12345678",
        client=mock_client,
        db_path=temp_db,
    )

    assert processed is False
    # No reply sent
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_process_telegram_authorized_user_success(tmp_path):
    """Messages from authorized user are routed to graph and replied to."""
    temp_db = str(tmp_path / "auth_test.db")
    update = {
        "update_id": 2,
        "message": {
            "chat": {"id": 12345678},
            "text": "Explain Python lists"
        }
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Lists in Python are ordered, mutable collections."))

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client.post = AsyncMock(return_value=mock_resp)

    processed = await chat_bot.process_telegram_update(
        update=update,
        bot_token="dummy_token",
        authorized_chat_id="12345678",
        client=mock_client,
        db_path=temp_db,
        llm_instance=mock_llm,
    )

    assert processed is True
    # Verify message was posted to Telegram
    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["chat_id"] == "12345678"
    assert "Lists in Python" in payload["text"]


@pytest.mark.asyncio
async def test_process_telegram_reset_command(tmp_path):
    """The /reset command clears conversational history and sends confirmation."""
    temp_db = str(tmp_path / "reset_test.db")
    chat_id = "12345678"

    # Seed history
    chat_memory.add_chat_message(chat_id, "user", "Prior conversation", db_path=temp_db)

    update = {
        "update_id": 3,
        "message": {
            "chat": {"id": int(chat_id)},
            "text": "/reset"
        }
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client.post = AsyncMock(return_value=mock_resp)

    processed = await chat_bot.process_telegram_update(
        update=update,
        bot_token="dummy_token",
        authorized_chat_id=chat_id,
        client=mock_client,
        db_path=temp_db,
    )

    assert processed is True
    # History must now be empty
    assert chat_memory.get_chat_history(chat_id, db_path=temp_db) == []
    # Confirmation message sent
    sent_text = mock_client.post.call_args[1]["json"]["text"]
    assert "reset" in sent_text.lower()


@pytest.mark.asyncio
async def test_check_webhook_active():
    """Detect when a Telegram webhook is currently active."""
    mock_client = MagicMock()
    # Webhook set
    active_resp = MagicMock()
    active_resp.status_code = 200
    active_resp.json.return_value = {"ok": True, "result": {"url": "https://example.com/webhook"}}
    mock_client.get = AsyncMock(return_value=active_resp)

    url = await chat_bot.check_webhook_active(mock_client, "token")
    assert url == "https://example.com/webhook"

    # Webhook not set
    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {"ok": True, "result": {"url": ""}}
    mock_client.get = AsyncMock(return_value=empty_resp)

    url_none = await chat_bot.check_webhook_active(mock_client, "token")
    assert url_none is None


@pytest.mark.asyncio
async def test_send_chat_reply_chunking():
    """Long replies exceeding 4000 characters are safely split into chunks."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client.post = AsyncMock(return_value=mock_resp)

    long_message = "Paragraph of code explanation.\n" * 200  # ~6200 chars

    success = await chat_bot.send_chat_reply(mock_client, "token", "123", long_message)
    assert success is True
    assert mock_client.post.call_count >= 2

