"""
chat_graph.py
LangGraph conversational workflow for the personal Telegram AI assistant.
Supports general conversation, programming assistance, beginner explanations,
and integrates verified tech news retrieval when users request current headlines.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

import chat_memory
from graph import extract_text
import server

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

SYSTEM_PERSONALITY_PROMPT = (
    "You are a friendly, helpful, and natural AI assistant on Telegram.\n"
    "You can chat about everyday life, answer general knowledge questions, explain programming "
    "and technology concepts to beginners using simple examples, and offer thoughtful educational and career advice.\n"
    "Guidelines:\n"
    "1. Speak naturally and warmly, like a knowledgeable, supportive friend.\n"
    "2. Be concise and approachable. Avoid unnecessary filler or excessively long walls of text.\n"
    "3. Do not force technology topics into ordinary conversations where they do not belong (e.g. greetings, general chat).\n"
    "4. If you are uncertain about something, admit it honestly rather than inventing facts.\n"
    "5. When teaching, use brief illustrative examples.\n"
    "6. If the user asks about current tech news, refer strictly to the verified news context provided below."
)


class ChatState(TypedDict, total=False):
    chat_id: str
    user_message: str
    history: List[Dict[str, str]]
    news_context: Optional[str]
    ai_response: str
    status: str
    db_path: str  # Configurable SQLite path for tests
    # Optional dependency injection for testing
    llm_instance: Any
    fetch_tool_func: Any


def is_asking_for_news(message: str) -> bool:
    """Detect if the user is explicitly inquiring about current tech news."""
    msg = message.strip().lower()
    triggers = [
        "latest news", "tech news", "news today", "trending news",
        "what's new in tech", "what is new in tech", "news headlines",
        "today's news", "recent news", "latest in tech", "ai news",
        "hacker news", "techcrunch", "the verge", "news updates"
    ]
    return any(trigger in msg for trigger in triggers)


# ============================================================================
# Graph Nodes
# ============================================================================

def load_context_node(state: ChatState) -> ChatState:
    """
    Node 1: Retrieve recent conversation history from SQLite chat_memory.db.
    Limits context to the last 10 messages to keep token usage within free-tier limits.
    """
    chat_id = state.get("chat_id", "default_chat")
    db_path = state.get("db_path") or chat_memory.DEFAULT_CHAT_DB_PATH

    history = chat_memory.get_chat_history(chat_id, limit=10, db_path=db_path)
    return {
        "history": history,
        "status": f"Loaded {len(history)} historical messages."
    }


async def check_news_intent_node(state: ChatState) -> ChatState:
    """
    Node 2: If the user asks about current tech developments, invoke the
    fetch_tech_news tool to retrieve verified headlines rather than hallucinating.
    """
    user_msg = state.get("user_message", "")
    fetch_func = state.get("fetch_tool_func") or server.fetch_tech_news

    if is_asking_for_news(user_msg):
        try:
            # Fetch top verified tech stories from Hacker News
            raw_result = await fetch_func(source="hackernews", limit=5)
            text_result = extract_text(raw_result)
            articles = json.loads(text_result)
            if isinstance(articles, list) and articles:
                lines = [f"- {a.get('title')} ({a.get('url')})" for a in articles[:5] if isinstance(a, dict)]
                news_context = "\n".join(lines)
                return {
                    "news_context": news_context,
                    "status": "Fetched verified news context."
                }
        except Exception as e:
            return {
                "news_context": None,
                "status": f"News fetch error: {type(e).__name__}"
            }

    return {
        "news_context": None,
        "status": "No news context required."
    }


async def generate_response_node(state: ChatState) -> ChatState:
    """
    Node 3: Build the conversational prompt with history and call Google Gemini.
    Gracefully handles quota limits without initiating paid usage.
    """
    user_msg = state.get("user_message", "").strip()
    history = state.get("history") or []
    news_ctx = state.get("news_context")
    llm_instance = state.get("llm_instance")

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL

    if not api_key and llm_instance is None:
        return {
            "ai_response": "I cannot respond because GOOGLE_API_KEY is not configured.",
            "status": "Error: Missing GOOGLE_API_KEY."
        }

    # Construct system prompt
    system_text = SYSTEM_PERSONALITY_PROMPT
    if news_ctx:
        system_text += (
            f"\n\n[VERIFIED CURRENT NEWS CONTEXT]:\n{news_ctx}\n"
            "Use only these verified headlines to answer the user's inquiry. "
            "Do not invent facts or external URLs."
        )

    messages: List[BaseMessage] = [SystemMessage(content=system_text)]

    # Append sliding-window conversation history
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Append current user prompt
    messages.append(HumanMessage(content=user_msg))

    try:
        llm = llm_instance or ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )
        response = await llm.ainvoke(messages)
        clean_text = extract_text(response)

        return {
            "ai_response": clean_text,
            "status": "Generated response successfully."
        }

    except Exception as e:
        err_str = str(e)
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower():
            err_reply = (
                "I've temporarily reached the daily free-tier quota limit with Google Gemini. "
                "Please try again in a little while!"
            )
            return {
                "ai_response": err_reply,
                "status": "Error: Gemini free-tier quota exhausted."
            }
        else:
            err_reply = f"I ran into an issue while generating a reply: {type(e).__name__}."
            return {
                "ai_response": err_reply,
                "status": f"Error: {type(e).__name__}: {e}"
            }


def save_context_node(state: ChatState) -> ChatState:
    """
    Node 4: Persist the current user prompt and assistant response into SQLite chat_memory.db.
    """
    chat_id = state.get("chat_id", "default_chat")
    user_msg = state.get("user_message", "").strip()
    ai_resp = state.get("ai_response", "").strip()
    db_path = state.get("db_path") or chat_memory.DEFAULT_CHAT_DB_PATH
    status = state.get("status", "")

    # Save to SQLite only if generation succeeded and wasn't a fatal config error
    if user_msg and ai_resp and not status.startswith("Error: Missing GOOGLE_API_KEY"):
        chat_memory.add_chat_message(chat_id, "user", user_msg, db_path=db_path)
        chat_memory.add_chat_message(chat_id, "assistant", ai_resp, db_path=db_path)

    return state


# ============================================================================
# Graph Construction
# ============================================================================

def build_chat_graph() -> StateGraph:
    """Construct the StateGraph for the conversational Telegram AI."""
    builder = StateGraph(ChatState)

    builder.add_node("load_context", load_context_node)
    builder.add_node("check_news_intent", check_news_intent_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("save_context", save_context_node)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "check_news_intent")
    builder.add_edge("check_news_intent", "generate_response")
    builder.add_edge("generate_response", "save_context")
    builder.add_edge("save_context", END)

    return builder


compiled_chat_graph = build_chat_graph().compile()


async def answer_user_message(
    chat_id: Union[str, int],
    user_message: str,
    db_path: Optional[str] = None,
    llm_instance: Optional[Any] = None,
    fetch_tool_func: Optional[Any] = None,
) -> str:
    """
    Convenience function to execute the conversational graph for an incoming message.
    """
    initial_state: ChatState = {
        "chat_id": str(chat_id),
        "user_message": user_message,
        "history": [],
        "news_context": None,
        "ai_response": "",
        "status": "Starting conversation turn",
        "db_path": db_path or chat_memory.DEFAULT_CHAT_DB_PATH,
        "llm_instance": llm_instance,
        "fetch_tool_func": fetch_tool_func,
    }
    result = await compiled_chat_graph.ainvoke(initial_state)
    return result.get("ai_response", "")
