"""
graph.py
LangGraph workflow orchestrating tech news retrieval, deduplication,
AI summarization via Google Gemini (free-tier), and delivery via Telegram.

Workflow:
[fetch_news] -> [filter_new] -> [summarize] -> [send] -> [END]
                     |
                     +---> (No new stories) ---> [END]
"""

import asyncio
import html
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

import state_store

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sent.db")
SERVER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


class NewsState(TypedDict, total=False):
    news: List[Dict[str, Any]]
    fresh_news: List[Dict[str, Any]]
    selected_news: List[Dict[str, Any]]  # Stories chosen for digest (at most 10)
    digest: str
    status: str
    db_path: str  # Configurable SQLite path for tests


# ============================================================================
# Helper Functions
# ============================================================================

def extract_text(obj: Any) -> str:
    """
    Extract clean plain text from any object, message, MCP content block, dict, or list.
    Safely unpacks LangChain BaseMessage (AIMessage), MCP TextContent, dicts, and lists.
    Never stringifies a wrapper object into 'content=[...]' or 'AIMessage(...)'.
    Converts escaped newline sequences (\\n) into real line breaks.
    """
    if obj is None:
        return ""

    # If it has a .content attribute (e.g. LangChain AIMessage / BaseMessage)
    if hasattr(obj, "content"):
        return extract_text(obj.content)

    # If it's a list (e.g. list of content blocks, strings, or dicts)
    if isinstance(obj, list):
        parts = [extract_text(item) for item in obj]
        return "\n".join(p for p in parts if p)

    # If it's a dict (e.g. {'type': 'text', 'text': '...'} or {'text': '...'} or {'content': '...'})
    if isinstance(obj, dict):
        if "text" in obj:
            return extract_text(obj["text"])
        if "content" in obj:
            return extract_text(obj["content"])
        return ""

    # If it has a .text attribute (e.g. MCP TextContent)
    if hasattr(obj, "text"):
        return extract_text(getattr(obj, "text"))

    # If it's already a string
    if isinstance(obj, str):
        text = obj
        # Safety recovery: if string contains raw wrapper representation like content=[{'type': 'text', 'text': '...'}]
        if "content=[" in text and "'text':" in text:
            matches = re.findall(r"'text':\s*['\"](.*?)['\"](?:\s*,|\s*})", text, re.DOTALL)
            if matches:
                text = "\n".join(matches)
        elif text.startswith("[{'type': 'text'") and "'text':" in text:
            matches = re.findall(r"'text':\s*['\"](.*?)['\"](?:\s*,|\s*})", text, re.DOTALL)
            if matches:
                text = "\n".join(matches)
        # Convert escaped newlines into real line breaks
        text = text.replace("\\n", "\n")
        return text.strip()

    return str(obj)


# Maintain backwards compatibility with earlier references
extract_tool_result_text = extract_text


def parse_articles_json(raw_text: str) -> List[Dict[str, Any]]:
    """Safely parse JSON response from fetch_tech_news into a list of article dicts."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            if "error" in data:
                return []
            return [data]
    except Exception:
        pass
    return []


def select_source_balanced(articles: List[Dict[str, Any]], max_count: int = 10) -> List[Dict[str, Any]]:
    """
    Select up to `max_count` stories using a deterministic, source-balanced round-robin order.
    Preserves original ranking within each source.
    """
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for item in articles:
        src = item.get("source", "Unknown")
        by_source.setdefault(src, []).append(item)

    selected: List[Dict[str, Any]] = []
    # Sort source names deterministically for consistent ordering
    sorted_sources = sorted(by_source.keys())

    while len(selected) < max_count:
        progress = False
        for src in sorted_sources:
            if by_source[src]:
                selected.append(by_source[src].pop(0))
                progress = True
                if len(selected) == max_count:
                    break
        if not progress:
            break

    return selected


def parse_numbered_explanations(text: str, expected_count: int) -> List[str]:
    """
    Parse numbered explanation items (e.g. '1. ...', '2. ...') from model text.
    Handles multi-line items and normalizes formatting.
    """
    if not text:
        return []

    normalized_text = text.replace("\\n", "\n")
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    explanations: List[str] = []

    item_pattern = re.compile(r"^(?:[\d]+[\.\)]|\[\d+\]|\*\s*[\d]+[\.\)])\s*(.*)")
    for line in lines:
        m = item_pattern.match(line)
        if m:
            exp = m.group(1).strip()
            exp = exp.strip("\"' ")
            if exp:
                explanations.append(exp)
        elif explanations:
            explanations[-1] += " " + line.strip("\"' ")

    return explanations


def build_telegram_digest(articles: List[Dict[str, Any]], explanations: Optional[List[str]] = None) -> str:
    """
    Deterministically build Telegram HTML digest from verified article dicts.
    Guarantees:
    - Verified URL in <a href="...">
    - Safely escaped title in <b>...</b>
    - Clean plain-text explanation
    - Proper paragraph breaks (\n\n)
    """
    entries: List[str] = []
    for idx, article in enumerate(articles):
        title = html.escape((article.get("title") or "Untitled").strip())
        url = (article.get("url") or "").strip()

        explanation = ""
        if explanations and idx < len(explanations) and explanations[idx]:
            explanation = html.escape(explanations[idx].strip())
        elif article.get("summary"):
            explanation = html.escape(article["summary"].strip())
        elif article.get("score") is not None:
            explanation = f"Trending on Hacker News with {article['score']} points."
        else:
            explanation = f"Read the full story on {html.escape(article.get('source', 'the web'))}."

        # Strip accidental markdown asterisks or formatting from explanation
        explanation = re.sub(r"\*\*([^*]+)\*\*", r"\1", explanation)
        explanation = re.sub(r"\*([^*]+)\*", r"\1", explanation)

        entry = f"<b><a href=\"{url}\">{title}</a></b>\n{explanation}"
        entries.append(entry)

    return "\n\n".join(entries)


def sanitize_telegram_html(raw_html: str, valid_urls: Optional[set] = None) -> str:
    """
    Ensure Telegram-compatible HTML output:
    - Replaces accidental markdown links [text](url) with <a href="url">text</a>
    - Replaces accidental markdown bold **text** with <b>text</b>
    - Escapes unescaped ampersands (&) as &amp;
    - If valid_urls is provided, verifies that linked URLs match provided news URLs.
    """
    if not raw_html:
        return ""
    # Convert markdown links if present
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', raw_html)
    # Convert markdown bold if present
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", cleaned)
    # Escape naked ampersands that are not already entities
    cleaned = re.sub(r"&(?!(?:amp|lt|gt|quot|apos);)", "&amp;", cleaned)

    # Validate links against valid_urls to ensure only genuine article URLs are linked
    if valid_urls:
        def validate_link(match):
            url = match.group(1).strip()
            anchor_text = match.group(2)
            if url in valid_urls or url.rstrip("/") in valid_urls or (url + "/") in valid_urls:
                return f'<a href="{url}">{anchor_text}</a>'
            return anchor_text

        cleaned = re.sub(
            r'<a\s+href=["\']([^"\']+)["\']\s*>(.*?)</a>',
            validate_link,
            cleaned,
            flags=re.DOTALL,
        )

    return cleaned.strip()


def create_mcp_client(server_script: Optional[str] = None) -> MultiServerMCPClient:
    """
    Create a MultiServerMCPClient pointing to server.py via stdio.
    Explicitly forwards TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to the child MCP process.
    """
    script_path = server_script or SERVER_SCRIPT_PATH

    env_vars: Dict[str, str] = {}
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if bot_token:
        env_vars["TELEGRAM_BOT_TOKEN"] = bot_token
    if chat_id:
        env_vars["TELEGRAM_CHAT_ID"] = chat_id

    connection_config: Dict[str, Any] = {
        "command": sys.executable,
        "args": [os.path.abspath(script_path)],
        "transport": "stdio",
    }
    if env_vars:
        connection_config["env"] = env_vars

    return MultiServerMCPClient({
        "news_server": connection_config
    })


# ============================================================================
# Graph Nodes
# ============================================================================

async def fetch_news_node(state: NewsState, mcp_client: Optional[MultiServerMCPClient] = None) -> NewsState:
    """
    Node A: Connect to server.py via MultiServerMCPClient, invoke fetch_tech_news
    for Hacker News, TechCrunch, and The Verge, and merge/deduplicate results.
    """
    client = mcp_client or create_mcp_client()
    sources = ["hackernews", "techcrunch", "theverge"]
    all_articles: List[Dict[str, Any]] = []
    fetch_errors: List[str] = []

    try:
        tools = await client.get_tools()
        fetch_tool = next((t for t in tools if t.name == "fetch_tech_news"), None)
        if not fetch_tool:
            return {
                "news": [],
                "status": "Error: Fetch failed - fetch_tech_news tool not found on MCP server."
            }

        for src in sources:
            try:
                raw_res = await fetch_tool.ainvoke({"source": src, "limit": 10})
                text_res = extract_tool_result_text(raw_res)
                articles = parse_articles_json(text_res)
                all_articles.extend(articles)
            except Exception as src_err:
                fetch_errors.append(f"{src}: {type(src_err).__name__}")

    except Exception as e:
        return {
            "news": [],
            "status": f"Error: Fetch failed - {type(e).__name__}: {e}"
        }

    # Deduplicate articles across sources by URL
    seen_urls = set()
    unique_articles: List[Dict[str, Any]] = []
    for item in all_articles:
        url = (item.get("url") or "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(item)

    if not unique_articles and fetch_errors:
        return {
            "news": [],
            "status": f"Error: Fetch failed - {', '.join(fetch_errors)}"
        }

    return {
        "news": unique_articles,
        "status": f"Fetched {len(unique_articles)} articles across sources."
    }


def filter_new_node(state: NewsState) -> NewsState:
    """
    Node B: Check SQLite state_store for previously sent URLs.
    Exclude already-sent URLs. If nothing is new, set status to 'No new stories'.
    Never marks URLs as sent at this stage.
    """
    db_path = state.get("db_path") or DEFAULT_DB_PATH
    sent_urls = state_store.get_sent_urls(db_path)

    all_news = state.get("news") or []
    fresh_articles = [
        item for item in all_news
        if (item.get("url") or "").strip() not in sent_urls
    ]

    if not fresh_articles:
        return {
            "fresh_news": [],
            "status": "No new stories"
        }

    return {
        "fresh_news": fresh_articles,
        "status": f"Found {len(fresh_articles)} fresh stories."
    }


async def summarize_node(state: NewsState, llm_instance: Optional[Any] = None) -> NewsState:
    """
    Node C: Select at most 10 stories with source-balanced round-robin.
    Uses Google Gemini (free-tier) to generate concise, engaging one-sentence explanations.
    Deterministically builds clean Telegram HTML in Python from verified article objects.
    Safely catches quota exhaustion and model errors without falling back to paid tiers.
    """
    fresh_articles = state.get("fresh_news") or []
    if not fresh_articles:
        return {
            "digest": "",
            "status": "No fresh stories to summarize.",
            "selected_news": [],
        }

    # Select at most 10 stories in a deterministic, source-balanced manner
    selected_articles = select_source_balanced(fresh_articles, max_count=10)

    # Format input for LLM prompt
    articles_lines = []
    for idx, a in enumerate(selected_articles, start=1):
        title = a.get("title", "Untitled")
        src = a.get("source", "Unknown")
        summary = a.get("summary") or (f"Score: {a.get('score')}" if a.get("score") is not None else "")
        articles_lines.append(f"{idx}. [{src}] {title}\n   Context: {summary}")
    articles_text = "\n\n".join(articles_lines)

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL

    if not api_key and llm_instance is None:
        return {
            "digest": "",
            "status": "Error: GOOGLE_API_KEY environment variable is not configured.",
            "selected_news": selected_articles,
        }

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a technology news editor for a Telegram channel. "
            "For each of the news stories provided, write exactly ONE concise, engaging plain sentence "
            "explaining why the development matters to developers, engineers, and technologists.\n"
            "Strict rules:\n"
            "1. Output ONLY a numbered list matching the stories (1., 2., ...).\n"
            "2. Do NOT include article titles, URLs, or markdown links.\n"
            "3. Do NOT use HTML or Markdown tags (no **, no <a>, no <b>).\n"
            "4. Each item must be a single plain sentence on its own line."
        ),
        (
            "human",
            "Please provide one-sentence explanations for these {count} stories:\n\n{articles_text}"
        )
    ])

    try:
        llm = llm_instance or ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
        )
        formatted_messages = await prompt_template.ainvoke({
            "count": len(selected_articles),
            "articles_text": articles_text,
        })
        response = await llm.ainvoke(formatted_messages)

        # Unpack text cleanly without wrapper strings
        raw_llm_text = extract_text(response)

        # If LLM returned full pre-formatted HTML (e.g. from existing mocks/tests)
        if "<b>" in raw_llm_text and "<a href=" in raw_llm_text:
            valid_urls = {
                (a.get("url") or "").strip()
                for a in selected_articles
                if (a.get("url") or "").strip()
            }
            clean_digest = sanitize_telegram_html(raw_llm_text, valid_urls=valid_urls)
        else:
            # Deterministically build HTML in Python from verified article dicts
            explanations = parse_numbered_explanations(raw_llm_text, len(selected_articles))
            clean_digest = build_telegram_digest(selected_articles, explanations)

        return {
            "digest": clean_digest,
            "status": f"Generated digest for {len(selected_articles)} stories.",
            "selected_news": selected_articles,
        }

    except Exception as e:
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "quota" in err_msg.lower():
            status_desc = "Error: Google Gemini free-tier quota exceeded. Stopped safely without initiating paid usage."
        else:
            status_desc = f"Error during summarization: {type(e).__name__}: {e}"

        return {
            "digest": "",
            "status": status_desc,
            "selected_news": selected_articles,
        }


async def send_node(state: NewsState, mcp_client: Optional[MultiServerMCPClient] = None) -> NewsState:
    """
    Node D: Deliver message to Telegram using send_telegram_message MCP tool.
    - If digest is present: sends '📰 Tech News Digest' + clean digest.
    - If earlier node encountered an error: sends a safe error notification.
    - ONLY records URLs for stories actually included in the delivered digest in sent.db.
    - Never records URLs on failure.
    """
    client = mcp_client or create_mcp_client()
    db_path = state.get("db_path") or DEFAULT_DB_PATH
    digest = state.get("digest")
    status = state.get("status", "")

    try:
        tools = await client.get_tools()
        send_tool = next((t for t in tools if t.name == "send_telegram_message"), None)
        if not send_tool:
            return {
                "status": "Error: send_telegram_message tool not found on MCP server."
            }

        # Case 1: Normal digest delivery
        if digest and digest.strip():
            clean_digest = digest.strip()
            # Ensure the message begins with 📰 <b>Tech News Digest</b>
            if not clean_digest.startswith("📰"):
                message_payload = f"📰 <b>Tech News Digest</b>\n\n{clean_digest}"
            else:
                message_payload = clean_digest

            raw_res = await send_tool.ainvoke({"text": message_payload})
            result_str = extract_text(raw_res)

            # Check for success confirmation from server.py
            if result_str.startswith("Successfully sent message"):
                # Extract delivered URLs from selected_news (at most 10 stories)
                selected_articles = state.get("selected_news")
                if not selected_articles:
                    selected_articles = (state.get("fresh_news") or [])[:10]

                delivered_urls = [
                    (a.get("url") or "").strip()
                    for a in selected_articles
                    if (a.get("url") or "").strip()
                ]
                recorded_count = state_store.record_sent_urls(delivered_urls, db_path=db_path)
                return {
                    "status": f"Successfully sent digest and recorded {recorded_count} URLs."
                }
            else:
                return {
                    "status": f"Telegram delivery failed: {result_str}"
                }

        # Case 2: Error alert delivery (e.g. fetch failure or summarization error)
        elif status.startswith("Error:"):
            error_message = f"⚠️ <b>Tech News Bot Alert</b>\n\n{status}"
            raw_res = await send_tool.ainvoke({"text": error_message})
            result_str = extract_tool_result_text(raw_res)
            return {
                "status": f"Sent error alert to Telegram. Result: {result_str}"
            }

        else:
            return {
                "status": "Nothing to send."
            }

    except Exception as e:
        return {
            "status": f"Failed to invoke Telegram tool: {type(e).__name__}: {e}"
        }


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_fetch(state: NewsState) -> str:
    """Route to send node immediately if fetch failed, else proceed to filter_new."""
    status = state.get("status", "")
    if status.startswith("Error:"):
        return "send"
    return "filter_new"


def route_after_filter(state: NewsState) -> str:
    """Route to END if no fresh stories found, else proceed to summarize."""
    status = state.get("status", "")
    fresh = state.get("fresh_news") or []
    if status == "No new stories" or not fresh:
        return END
    return "summarize"


# ============================================================================
# Graph Construction
# ============================================================================

def build_graph() -> StateGraph:
    """Construct the StateGraph for tech-news-mcp."""
    builder = StateGraph(NewsState)

    builder.add_node("fetch_news", fetch_news_node)
    builder.add_node("filter_new", filter_new_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("send", send_node)

    builder.add_edge(START, "fetch_news")
    builder.add_conditional_edges(
        "fetch_news",
        route_after_fetch,
        {
            "filter_new": "filter_new",
            "send": "send",
        }
    )
    builder.add_conditional_edges(
        "filter_new",
        route_after_filter,
        {
            "summarize": "summarize",
            END: END,
        }
    )
    builder.add_edge("summarize", "send")
    builder.add_edge("send", END)

    return builder


# Expose compiled graph instance for external runners
compiled_graph = build_graph().compile()


# ============================================================================
# Standalone Execution Entrypoint
# ============================================================================

def check_required_env() -> bool:
    """
    Safe preflight check verifying required environment variables are present.
    Logs ONLY 'SET' or 'MISSING' without revealing actual values.
    Returns True if all required variables are present, False otherwise.
    """
    required_vars = ["GOOGLE_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    all_present = True
    print("Preflight environment check:")
    for var in required_vars:
        is_set = bool(os.environ.get(var))
        status_label = "SET" if is_set else "MISSING"
        print(f"  {var}: {status_label}")
        if not is_set:
            all_present = False
    return all_present


def is_successful_status(status: str) -> bool:
    """
    Check if the workflow finished with a successful terminal status.
    Returns True for successful delivery or clean exit when no new stories exist.
    """
    if not status:
        return False
    if status.startswith("Successfully sent digest"):
        return True
    if status == "No new stories":
        return True
    if status == "Nothing to send.":
        return True
    return False


async def main():
    """Execution entrypoint for scheduled or manual workflow runs."""
    from dotenv import load_dotenv
    load_dotenv()

    # Preflight check: ensure required environment variables are set
    if not check_required_env():
        print("Error: Missing required environment variables. Aborting.")
        sys.exit(1)

    print("Executing tech-news-mcp workflow...")
    initial_state: NewsState = {
        "news": [],
        "fresh_news": [],
        "digest": "",
        "status": "Starting workflow",
    }
    result = await compiled_graph.ainvoke(initial_state)
    final_status = result.get("status", "")
    print("Workflow finished. Final status:", final_status)

    # Check for delivery failure or error to fail CI cleanly
    if not is_successful_status(final_status):
        print(f"Workflow failed with status: {final_status}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
