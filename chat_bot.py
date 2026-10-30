"""
chat_bot.py
Telegram long-polling conversational listener for the personal AI assistant.
Receives ordinary user messages, manages conversation context, and replies naturally.
Includes security checks to ignore unauthorized users and prevent webhook conflicts.
Does not start polling upon module import.
"""

import asyncio
import os
import sys
from typing import Any, Dict, Optional, Union

import httpx

import chat_graph
import chat_memory
import server


def is_authorized_chat(incoming_chat_id: Union[str, int], authorized_chat_id: Union[str, int]) -> bool:
    """
    Verify whether the incoming message is from the authorized user or chat.
    Prevents unauthorized access and protects privacy and API quotas.
    """
    if not incoming_chat_id or not authorized_chat_id:
        return False
    return str(incoming_chat_id).strip() == str(authorized_chat_id).strip()


async def check_webhook_active(client: httpx.AsyncClient, bot_token: str) -> Optional[str]:
    """
    Check if a Telegram webhook is currently registered.
    If a webhook is active, long-polling cannot run.
    Returns the webhook URL if active, or None if no webhook is set.
    """
    url = f"https://api.telegram.org/bot{bot_token}/getWebhookInfo"
    try:
        response = await client.get(url, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                webhook_url = data.get("result", {}).get("url", "").strip()
                return webhook_url if webhook_url else None
    except Exception:
        pass
    return None


async def send_chat_reply(
    client: httpx.AsyncClient,
    bot_token: str,
    chat_id: Union[str, int],
    text: str,
) -> bool:
    """
    Deliver response text to Telegram with automatic message chunking (<= 4000 chars).
    Never logs or exposes tokens or secrets.
    """
    if not text or not str(text).strip():
        return False

    chunks = server.split_message(str(text).strip(), max_chunk_size=4000)
    send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    all_ok = True
    for chunk in chunks:
        payload = {
            "chat_id": str(chat_id),
            "text": chunk,
        }
        try:
            res = await client.post(send_url, json=payload, timeout=15.0)
            if res.status_code != 200:
                all_ok = False
        except Exception:
            all_ok = False

    return all_ok


async def process_telegram_update(
    update: Dict[str, Any],
    bot_token: str,
    authorized_chat_id: str,
    client: Optional[httpx.AsyncClient] = None,
    db_path: Optional[str] = None,
    llm_instance: Optional[Any] = None,
    fetch_tool_func: Optional[Any] = None,
) -> bool:
    """
    Process a single Telegram update:
    - Verifies chat authorization (silently drops unauthorized messages)
    - Supports /reset command to wipe conversation history
    - Routes user text to chat_graph for conversational response
    - Sends answer back to Telegram
    Returns True if a reply was sent, False otherwise.
    """
    message = update.get("message")
    if not message or not isinstance(message, dict):
        return False

    incoming_chat = message.get("chat", {})
    incoming_chat_id = incoming_chat.get("id")
    user_text = message.get("text", "")

    # Security check: Ignore unauthorized users
    if not is_authorized_chat(incoming_chat_id, authorized_chat_id):
        return False

    if not user_text or not user_text.strip():
        return False

    user_text_clean = user_text.strip()

    # Create local client if not provided
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=15.0)
        close_client = True

    try:
        # Handle /reset command
        if user_text_clean.lower() in ("/reset", "/reset@botname", "reset"):
            chat_memory.clear_chat_history(incoming_chat_id, db_path=db_path)
            reset_msg = "🔄 Conversation history has been reset. Fresh start!"
            await send_chat_reply(client, bot_token, incoming_chat_id, reset_msg)
            return True

        # Conversational generation via LangGraph
        reply_text = await chat_graph.answer_user_message(
            chat_id=incoming_chat_id,
            user_message=user_text_clean,
            db_path=db_path,
            llm_instance=llm_instance,
            fetch_tool_func=fetch_tool_func,
        )

        if reply_text:
            await send_chat_reply(client, bot_token, incoming_chat_id, reply_text)
            return True

        return False

    finally:
        if close_client:
            await client.aclose()


async def run_bot_listener(
    bot_token: str,
    authorized_chat_id: str,
    db_path: Optional[str] = None,
    poll_timeout: int = 30,
) -> None:
    """
    Long-polling loop receiving incoming updates from Telegram Bot API.
    Handles offsets safely to prevent duplicate update processing.
    """
    headers = {"User-Agent": "tech-news-mcp-chat/1.0"}
    async with httpx.AsyncClient(headers=headers, timeout=poll_timeout + 10.0) as client:
        # Check for active webhook conflicts
        active_webhook = await check_webhook_active(client, bot_token)
        if active_webhook:
            print(
                f"\n[ERROR] Telegram webhook is currently active: '{active_webhook}'.\n"
                "Telegram long-polling cannot start while a webhook is registered.\n"
                "To resolve, delete the webhook via Bot API or use webhook mode."
            )
            return

        print(f"🤖 Chat assistant started. Listening for messages from authorized chat ID: {authorized_chat_id}...")
        print("Press Ctrl+C to stop.\n")

        offset = 0
        get_updates_url = f"https://api.telegram.org/bot{bot_token}/getUpdates"

        while True:
            try:
                params = {"offset": offset, "timeout": poll_timeout}
                response = await client.get(get_updates_url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        updates = data.get("result", [])
                        for update in updates:
                            update_id = update.get("update_id")
                            if update_id is not None:
                                offset = max(offset, update_id + 1)

                            await process_telegram_update(
                                update=update,
                                bot_token=bot_token,
                                authorized_chat_id=authorized_chat_id,
                                client=client,
                                db_path=db_path,
                            )
                elif response.status_code == 409:
                    print("[WARNING] HTTP 409 Conflict: Another polling or webhook instance is active. Retrying in 5s...")
                    await asyncio.sleep(5.0)
                else:
                    await asyncio.sleep(2.0)

            except asyncio.CancelledError:
                print("\nShutting down chat listener cleanly...")
                break
            except Exception as e:
                # Never print token or credentials
                print(f"[Notice] Polling connection notice ({type(e).__name__}). Retrying in 3s...")
                await asyncio.sleep(3.0)


# ============================================================================
# Standalone Execution Entrypoint (DO NOT RUN DURING PHASE 5)
# ============================================================================

async def main():
    """Manual execution entrypoint for conversational assistant."""
    from dotenv import load_dotenv
    load_dotenv()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in your environment or .env.")
        sys.exit(1)

    try:
        await run_bot_listener(bot_token, chat_id)
    except KeyboardInterrupt:
        print("\nListener stopped by user.")


if __name__ == "__main__":
    asyncio.run(main())

