# tech-news-mcp 📰🤖

An automated, zero-cost AI news curator that gathers trending tech headlines from Hacker News, TechCrunch, and The Verge, summarizes them into a digestible format using Google Gemini (free-tier), deduplicates articles using SQLite, and delivers a daily HTML digest directly to Telegram.

---

## Architecture Overview

The system is organized into modular layers using modern agentic and AI tooling:

```
┌─────────────────────────────────────────────────────────────┐
│                      Scheduled Trigger                      │
│        (GitHub Actions Daily Cron / Local Execution)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Workflow Engine                 │
│                          (graph.py)                         │
│                                                             │
│   [fetch_news] ──► [filter_new] ──► [summarize] ──► [send]  │
│         │               │                                   │
│         │               └── (No new stories) ──► [END]      │
│         ▼                                                   │
│   (MCP Stdio Client)                                        │
└─────────┬───────────────────────────────────┬───────────────┘
          │                                   │
          ▼                                   ▼
┌──────────────────────────────┐    ┌─────────────────────────┐
│     Model Context Protocol   │    │  Google AI Studio API   │
│       Server (server.py)     │    │  (gemini-3.1-flash-lite)│
│                              │    │       (Free Tier)       │
│ • fetch_tech_news()          │    └─────────────────────────┘
│ • send_telegram_message()    │                   ▲
└─────────┬────────────────────┘                   │
          │                                        │
          ▼                                        │
┌─────────────────────────────────────────────────────────────┐
│                  External Integrations                      │
│ • Hacker News (Firebase REST API, async concurrent fetch)   │
│ • TechCrunch & The Verge (RSS Feeds via feedparser)         │
│ • SQLite URL State Store (sent.db deduplication)            │
│ • Telegram Bot API (HTML parse mode, automatic chunking)    │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts for College Viva
1. **Model Context Protocol (MCP)**:
   - An open standard developed to connect LLMs and AI applications with local or remote tools and resources.
   - In this project, `server.py` implements an MCP server using `FastMCP` running over standard I/O (`stdio`).
   - The LangGraph workflow consumes this server via `MultiServerMCPClient`, creating clean decoupling between tool implementation and workflow orchestration.
2. **LangGraph Workflow**:
   - A graph-based orchestration framework for complex agentic pipelines.
   - It represents execution as a state machine where nodes execute tasks and edges route state transitions based on conditional outcomes.
3. **Google Gemini (Free Tier)**:
   - Summarization is performed by Google Gemini (`gemini-3.1-flash-lite`) using Google AI Studio.
   - Strict budget constraints ensure the project operates at **₹0 cost**, preventing accidental billing or paid model fallbacks.
4. **SQLite URL Deduplication (`state_store.py`)**:
   - Prevents sending duplicate stories to Telegram across runs by storing previously delivered URLs in a local SQLite database (`sent.db`).

---

## Deterministic Story Selection Assumption
When `graph.py` selects articles for the digest:
- It uses a **deterministic, source-balanced round-robin** algorithm (`select_source_balanced`).
- Articles are grouped by source (Hacker News, TechCrunch, The Verge).
- The algorithm cycles through each source in alphabetical order, taking 1 top article from each source per round until a maximum of 10 articles are selected (e.g., HN #1, TC #1, Verge #1, HN #2, TC #2, Verge #2...).
- This ensures an equitable distribution across news outlets while strictly preserving each source's original ranking order.

---

## Prerequisites
- **Operating System**: macOS, Linux, or Windows (WSL recommended).
- **Python**: Version `3.11` or `3.12` installed (`python3 --version`).
- **Git**: Installed for repository management.

---

## Local Setup & Installation

### 1. Clone & Navigate to Project Root
```bash
cd /path/to/Telegram_miniproject
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*(On Windows PowerShell: `.venv\Scripts\Activate.ps1`)*

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration & Credentials Setup

> [!WARNING]
> **CRITICAL SECURITY RULE**: Never commit `.env` or paste actual API keys into version control. `.gitignore` is pre-configured to exclude `.env`, `.venv/`, and `*.db`.

Create your local `.env` file manually by copying the template:
```bash
cp .env.example .env
```

Open `.env` in your text editor and fill in the required values:
```env
GOOGLE_API_KEY=your_actual_google_ai_studio_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
TELEGRAM_BOT_TOKEN=your_actual_telegram_bot_token
TELEGRAM_CHAT_ID=your_actual_telegram_chat_id
```

### How to Get Your Telegram Bot Credentials
1. **Create a Bot via BotFather**:
   - Open Telegram and search for [@BotFather](https://t.me/BotFather).
   - Send `/newbot` and follow the prompts to choose a name and username.
   - Copy the generated API token (e.g., `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`) into `TELEGRAM_BOT_TOKEN`.
2. **Find Your Telegram Chat ID**:
   - Start a conversation with your newly created bot (press `/start`).
   - Open Telegram and search for [@userinfobot](https://t.me/userinfobot) or visit `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` in your browser.
   - Copy your numeric `Id` (e.g., `987654321`) into `TELEGRAM_CHAT_ID`.
   - *(If using a channel or group, add the bot as an administrator and use the channel ID, e.g. `-100...`)*.

### How to Get Your Google AI Studio API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **Get API key** -> **Create API key**.
4. Ensure your project is on the **Free tier**.
5. Paste the key into `GOOGLE_API_KEY`.
6. Leave `GEMINI_MODEL=gemini-3.1-flash-lite` (or another confirmed free-tier model).

---

## Running & Testing

### 1. Run Offline Unit Tests (Pytest)
Run the full test suite with all network, LLM, and Telegram calls mocked:
```bash
pytest -v
```
All 62 unit tests should pass in ~1.2 seconds.

### 2. Inspect the MCP Server Locally
You can test and inspect the MCP tools using the Model Context Protocol CLI inspector:
```bash
mcp dev server.py
```
*(Alternatively: `npx @modelcontextprotocol/inspector python server.py`)*

This opens an interactive web interface where you can test `fetch_tech_news` and inspect tool schemas directly.

### 3. Run the Workflow Locally (Manual Run)
Once you have created and populated your `.env` file with valid keys:
```bash
python graph.py
```

Expected workflow behavior:
1. Connects to `server.py` and fetches stories from all 3 sources.
2. Checks `sent.db` to filter out any previously sent URLs.
3. If new stories are found, selects up to 10 stories in balanced order and invokes Gemini for summarization.
4. Delivers the HTML digest to your Telegram chat.
5. Records the sent URLs in `sent.db` only after delivery confirmation.
6. If re-run immediately with no new stories published, prints `Workflow finished. Final status: No new stories` and safely exits without sending a duplicate message.

---

## How SQLite Deduplication Works (`state_store.py`)

- **Table Schema**:
  ```sql
  CREATE TABLE IF NOT EXISTS sent_articles (
      url TEXT PRIMARY KEY,
      sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```
- **Integrity**: `url` is defined as `PRIMARY KEY`. Any duplicate insertion attempt is safely ignored via `INSERT OR IGNORE`.
- **Delivery Safety Guarantee**: URLs are **never** recorded prior to or during LLM summarization. `state_store.record_sent_urls` is only invoked in `send_node` after the Telegram tool responds with a successful HTTP 200 delivery confirmation (`Successfully sent message...`). If network delivery fails, no URLs are stored, ensuring they are not lost for future runs.

---

## Automated Scheduling via GitHub Actions

The workflow is scheduled to run daily at **02:30 UTC (08:00 IST)** via `.github/workflows/daily.yml`.

### Setting Up GitHub Repository Secrets
To enable automated runs in your GitHub repository:
1. Push your repository to GitHub (ensure `.env` and `sent.db` are **not** committed).
2. On GitHub, navigate to: **Settings** -> **Secrets and variables** -> **Actions**.
3. Under **Repository secrets**, click **New repository secret** and add:
   - `GOOGLE_API_KEY`: Your Google AI Studio API key.
   - `GEMINI_MODEL`: `gemini-3.1-flash-lite`
   - `TELEGRAM_BOT_TOKEN`: Your Telegram Bot token.
   - `TELEGRAM_CHAT_ID`: Your Telegram numeric Chat ID.
4. **Trigger Manually**: Go to the **Actions** tab on GitHub, select **Daily Tech News Digest**, and click **Run workflow**.
5. **State Caching**: The GitHub Actions runner restores `sent.db` before execution and caches the updated `sent.db` after execution using `actions/cache`, maintaining historical deduplication across cloud runs.

---

---

## Personal Conversational AI Assistant (`chat_bot.py`)

In addition to scheduled daily digests, your bot can run as an interactive, conversational AI assistant.

### Architecture Comparison

| Feature | Daily Digest Workflow (`graph.py`) | Conversational Assistant (`chat_bot.py`) |
| :--- | :--- | :--- |
| **Execution Model** | Scheduled one-shot execution (cron / local run). | Continuous long-polling listener (`getUpdates`). |
| **Trigger** | GitHub Actions (08:00 IST) or `python graph.py`. | Any incoming text message from your Telegram chat. |
| **Orchestration** | `graph.py` StateGraph (`fetch` -> `filter` -> `summarize` -> `send`). | `chat_graph.py` StateGraph (`load_context` -> `news_intent` -> `generate` -> `save`). |
| **Storage Engine** | `sent.db` (stores delivered article URLs for deduplication). | `chat_memory.db` (stores persistent chat history per `chat_id`). |
| **Personality** | Concise HTML digest editor. | Friendly, approachable AI companion, tutor, and tech mentor. |

### How Conversational Memory Works (`chat_memory.py`)
- **Database**: Saved in a separate SQLite database named `chat_memory.db` (completely isolated from `sent.db`).
- **Thread Isolation**: Messages are indexed by `chat_id`. Multiple chats never leak context to each other.
- **Sliding-Window Optimization**: To respect Google Gemini's **Free-tier quotas**, only the most recent **10 messages** are injected into the prompt. This provides deep conversational continuity without burning excess tokens.
- **Persistence**: Chat history is preserved across bot restarts.
- **Resetting Memory**: Send `/reset` at any time in the Telegram chat to clear your conversation history and start fresh.

### Security & Privacy Protections
- **Authorized Access Only**: The bot strictly verifies that incoming updates match your configured `TELEGRAM_CHAT_ID`. Messages from unauthorized users are silently ignored.
- **Webhook Safety Check**: Before starting polling, `chat_bot.py` queries `getWebhookInfo`. If a remote webhook is registered, it halts with a clear explanation instead of silently deleting your configuration.

### How to Run the Conversational Assistant Locally
Ensure `.env` has your `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and `GOOGLE_API_KEY`:

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Start the interactive listener
python chat_bot.py
```

- Send any normal message in Telegram (e.g. *"Hey, can you explain what a closure is in Python?"* or *"Good morning! What are the latest tech headlines today?"*).
- If you ask about current tech news, the bot automatically queries verified news headlines via `server.py` rather than hallucinating developments.
- To stop the bot cleanly: Press **`Ctrl+C`** in your terminal.

---

## Troubleshooting Guide

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **`Telegram API error 401: Unauthorized`** | Invalid or misspelled `TELEGRAM_BOT_TOKEN`. | Verify your token by messaging `@BotFather` and check that `.env` has no surrounding quotes or whitespace. |
| **`Telegram API error 400: chat not found`** | Bot was never started by the user or wrong `TELEGRAM_CHAT_ID`. | Open Telegram, search for your bot username, press `/start`, and verify your chat ID with `@userinfobot`. |
| **`429 RESOURCE_EXHAUSTED`** | Google Gemini free-tier rate or quota limit reached. | The workflow catches this safely and halts. Wait for quota reset (per-minute or daily reset) or configure a different free model in `GEMINI_MODEL`. **Do not enable paid billing.** |
| **`RSS Feed Fetch Returns Empty`** | Outlet site temporarily down or blocking request. | `graph.py` catches individual source errors gracefully and continues fetching from remaining sources without aborting the workflow. |
| **`Telegram API error 409: Conflict`** | Another instance of `chat_bot.py` or another polling script is already running with this bot token. | Stop the other instance or terminate stray python processes (`ps aux \| grep chat_bot.py`). |
| **`Active webhook detected`** | An outgoing webhook URL is registered on Telegram. | Telegram prohibits long-polling while a webhook is set. Delete the webhook via `https://api.telegram.org/bot<TOKEN>/deleteWebhook` or keep webhook mode. |
| **`sent.db locked` or permission error** | Stray process holding SQLite file lock. | Ensure previous runs finished cleanly. Restart terminal session or inspect active Python processes. |

---

## Zero-Cost & Anti-Paid Safeguards
- This project is strictly engineered for educational and personal use under zero-cost tier limits.
- Never upgrade your Google Cloud or AI Studio project to a paid billing plan to bypass a rate limit.
- If a quota limit is encountered, the workflow exits safely and will resume automatically on the next scheduled run.

