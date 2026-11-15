# Nimbus Gear Support Assistant

An AI customer support chatbot for **Nimbus Gear**, a fictional online store selling tech
accessories. Customers ask questions in natural language about products, prices, shipping,
returns, warranty and store policies, and the assistant answers **only** from the store's
knowledge base using Retrieval-Augmented Generation (RAG).

- **Backend:** Python, FastAPI
- **LLM:** Google Gemini via Google AI Studio (`google-genai` SDK), free-tier model by default
- **Retrieval:** ChromaDB vector database + local `all-MiniLM-L6-v2` embeddings (ONNX, no API key)
- **Frontend:** Streamlit chat UI
- **Demo mode:** runs without any API key

## Features

- Answers questions about the company, products (specs, features, prices), shipping,
  delivery, returns, refunds, exchanges, warranty and FAQs.
- Semantic search: understands meaning, not keywords ("can I get my money back?" finds the
  refund policy).
- Conversation memory per session, including follow-ups such as "how much is it?".
- Grounded answers: the model is given only retrieved passages and is instructed to say so
  when the answer is not in them. Passages below a relevance threshold are discarded.
- Shows which knowledge-base sections each answer came from.
- Chat UI with welcome message, clickable suggested questions, session history, loading
  indicator, status badge and clear error messages.
- Input validation, structured error responses, request logging (`logs/app.log`).
- Graceful degradation: if Gemini fails (quota, network), the user still gets the most
  relevant knowledge-base passage instead of an error.

## Architecture

```
 Browser ──► Streamlit UI ──HTTP──► FastAPI backend
            (frontend/)             (app/main.py)
                                         │
                                 SessionStore (history)
                                         │
                                    ChatService  (app/rag/pipeline.py)
                                         │
              1. make follow-up standalone (Gemini rewrite / heuristic)
              2. embed query ─► ChromaDB similarity search ─► top-k passages
              3. drop passages below the relevance threshold
              4. Gemini: system rules + history + passages + question ─► answer
                 (demo mode: return the best passage verbatim)
```

At startup the backend chunks every Markdown file in `data/knowledge_base/` along its
headings (one chunk per product / policy section / FAQ entry), embeds the chunks and stores
them in ChromaDB under `data/chroma/`. The index is rebuilt automatically whenever the
documents change, so editing the knowledge base only needs a backend restart.

### Project structure

```
app/
  config.py            settings from environment / .env
  main.py              FastAPI app, routes, wiring
  schemas.py           request/response models and validation
  sessions.py          in-memory conversation history
  llm.py               LLM interface + Gemini client (the only provider-specific code)
  logging_config.py
  rag/
    chunker.py         Markdown -> chunks
    embeddings.py      ONNX MiniLM embedder + offline lexical fallback
    vector_store.py    ChromaDB wrapper
    prompts.py         system prompt and context formatting
    pipeline.py        retrieval + generation (ChatService)
    demo.py            no-LLM answers for demo mode / fallback
frontend/streamlit_app.py   chat UI
data/knowledge_base/        store documentation (Markdown)
scripts/ingest.py           optional: build the index ahead of time
tests/                      pytest suite
run.py                      starts backend + UI together
```

## Setup

Requires Python 3.10+ (developed on 3.12, Windows).

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

(On macOS/Linux activate with `source .venv/bin/activate`.)

### Add your Gemini API key

1. Create a free key at <https://aistudio.google.com/apikey>.
2. Copy `.env.example` to `.env`.
3. Put the key after `GEMINI_API_KEY=` in `.env`.

`.env` is git-ignored; the key is only ever read from the environment. The default model is
`gemini-2.5-flash`, which is available on the Gemini API free tier. To use another model
(for example `gemini-2.5-flash-lite`), set `GEMINI_MODEL` in `.env`. Free-tier requests are
rate-limited, and Google may use free-tier prompts to improve its products.

**No key?** Leave `GEMINI_API_KEY` empty and the app runs in **demo mode**: retrieval works
exactly the same, but instead of a written answer you get the most relevant knowledge-base
passage quoted verbatim. The UI shows a "Demo mode" badge.

## Running

```bash
python run.py
```

This starts the API on <http://127.0.0.1:8000> and opens the chat UI on
<http://localhost:8501>. The first start downloads the embedding model (about 80 MB) to
`~/.cache/chroma`; later starts are offline and fast. If the download is not possible the
app falls back to a simpler word-matching embedder and logs a warning.

To run the two parts separately:

```bash
uvicorn app.main:app --port 8000
```

```bash
streamlit run frontend/streamlit_app.py
```

### Things to ask

- "What is your return policy?"
- "Tell me about the Aero Pro headphones" → then "How much is it?" → "And the battery life?"
- "Do you ship to Canada, and how long does it take?"
- "Is the speaker waterproof?"
- "Which payment methods do you accept?"
- "What's the capital of France?" → the assistant says it doesn't have that information.

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Status, mode (`llm` / `demo`), model, number of indexed chunks |
| `POST` | `/chat` | `{"message": "...", "session_id": "optional"}` → answer, sources, session id |
| `DELETE` | `/sessions/{id}` | Clear a conversation |

Interactive docs: <http://127.0.0.1:8000/docs>

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"Do you ship internationally?\"}"
```

`mode` in the response is `llm`, `demo`, or `fallback` (Gemini is configured but the call
failed, so a quoted passage was returned).

## Configuration

All settings are environment variables (or `.env` entries); see `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | empty | Gemini key; empty = demo mode |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model |
| `EMBEDDING_BACKEND` | `auto` | `auto`, `onnx`, or `hash` (offline lexical fallback) |
| `RETRIEVAL_TOP_K` | `4` | Passages retrieved per question |
| `MIN_RELEVANCE` | per backend | Minimum cosine similarity for a passage to be used |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Backend address |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Where the UI finds the backend |

## Updating the knowledge base

Edit or add Markdown files in `data/knowledge_base/` and restart the backend. Use `#` for
the document title and `##` / `###` for sections; each section becomes one searchable chunk.

## Tests

```bash
pytest
```

The suite covers chunking, embeddings and vector search against the real knowledge base, the
RAG pipeline in demo and LLM mode (with a fake LLM: context injection, history, follow-up
rewriting, failure fallback), and the HTTP API (validation, sessions). Tests run offline:
they use the lexical `hash` embedder and never call Gemini.

## Limitations

- Conversation history is kept in server memory and is lost on restart.
- The assistant has no access to orders or accounts; it directs those questions to support.
- No authentication or rate limiting; add both before exposing the API publicly.
