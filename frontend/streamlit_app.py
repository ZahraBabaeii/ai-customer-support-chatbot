"""Streamlit chat interface for the Nimbus Gear support assistant."""

import os
import uuid

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_MESSAGE_CHARS = 1000

STORE_NAME = "Nimbus Gear"
SUPPORT_EMAIL = "support@nimbusgear.example"
SUPPORT_PHONE = "1-800-555-0142"

WELCOME = (
    "Hi, I'm **Nimbus**, your Nimbus Gear support assistant. I can help with products "
    "and prices, shipping and delivery, returns and refunds, warranty, and store policies. "
    "What would you like to know?"
)

SUGGESTIONS = [
    "What is your return policy?",
    "How much does shipping cost?",
    "Tell me about the Aero Pro headphones",
    "Do you ship internationally?",
    "What does the warranty cover?",
    "Which payment methods do you accept?",
]

CSS = """
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none;}
.block-container {max-width: 820px; padding-top: 3.5rem; padding-bottom: 6rem;}

.ng-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 22px; border-radius: 16px; margin-bottom: 18px;
    background: linear-gradient(135deg, #1e3a5f 0%, #2f6f8f 100%);
    color: #ffffff; box-shadow: 0 6px 20px rgba(30, 58, 95, 0.18);
}
.ng-logo {
    width: 46px; height: 46px; border-radius: 12px; flex-shrink: 0;
    background: rgba(255, 255, 255, 0.16);
    display: flex; align-items: center; justify-content: center; font-size: 24px;
}
.ng-title {font-size: 1.15rem; font-weight: 700; line-height: 1.2;}
.ng-subtitle {font-size: 0.85rem; opacity: 0.85;}
.ng-status {
    margin-left: auto; font-size: 0.78rem; font-weight: 600; white-space: nowrap;
    padding: 5px 11px; border-radius: 999px; background: rgba(255, 255, 255, 0.16);
}
.ng-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
}

[data-testid="stChatMessage"] {
    border-radius: 14px; padding: 0.85rem 1rem; margin-bottom: 0.4rem;
    border: 1px solid rgba(128, 128, 128, 0.18);
}
.ng-sources {font-size: 0.78rem; opacity: 0.7; margin-top: 0.35rem;}
.ng-suggest-label {font-size: 0.85rem; opacity: 0.7; margin: 0.6rem 0 0.3rem;}

div[data-testid="stButton"] > button {
    border-radius: 999px; font-size: 0.88rem; text-align: left;
    border: 1px solid rgba(47, 111, 143, 0.45);
}
div[data-testid="stButton"] > button:hover {
    border-color: #2f6f8f; color: #2f6f8f; background: rgba(47, 111, 143, 0.08);
}
</style>
"""


class BackendError(Exception):
    """A user-presentable description of what went wrong."""


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health() -> dict | None:
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def ask_backend(message: str, session_id: str) -> dict:
    try:
        response = httpx.post(
            f"{BACKEND_URL}/chat",
            json={"message": message, "session_id": session_id},
            timeout=httpx.Timeout(90, connect=5),
        )
    except httpx.ConnectError as exc:
        raise BackendError(
            "I can't reach the support service right now. Please make sure the backend "
            f"is running at {BACKEND_URL} and try again."
        ) from exc
    except httpx.TimeoutException as exc:
        raise BackendError("That took too long to answer. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise BackendError("A network error occurred. Please try again.") from exc

    if response.status_code == 422:
        raise BackendError(
            f"Please enter a message between 1 and {MAX_MESSAGE_CHARS} characters."
        )
    if response.status_code >= 400:
        raise BackendError("Something went wrong on our side. Please try again in a moment.")
    return response.json()


def reset_conversation() -> None:
    old_id = st.session_state.get("session_id")
    if old_id:
        try:
            httpx.delete(f"{BACKEND_URL}/sessions/{old_id}", timeout=5)
        except httpx.HTTPError:
            pass  # the server-side history simply expires on its own
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []


def render_header(health: dict | None) -> None:
    if health is None:
        color, label = "#f87171", "Offline"
    elif health["mode"] == "llm":
        color, label = "#4ade80", "Online"
    else:
        color, label = "#fbbf24", "Demo mode"
    st.markdown(
        f"""
        <div class="ng-header">
          <div class="ng-logo">☁️</div>
          <div>
            <div class="ng-title">{STORE_NAME} Support</div>
            <div class="ng-subtitle">Ask about products, shipping, returns and more</div>
          </div>
          <div class="ng-status">
            <span class="ng-dot" style="background:{color}"></span>{label}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(health: dict | None) -> None:
    with st.sidebar:
        st.markdown(f"### ☁️ {STORE_NAME}")
        st.caption("Tech accessories for work and travel.")

        if st.button("＋ New conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.divider()
        st.markdown("**Contact a person**")
        st.markdown(f"✉️ {SUPPORT_EMAIL}  \n📞 {SUPPORT_PHONE}")
        st.markdown(
            "**Support hours**  \nMon–Fri: 8 AM – 8 PM PT  \nSat: 9 AM – 5 PM PT  \nSun: closed"
        )

        st.divider()
        st.markdown("**Assistant status**")
        if health is None:
            st.error("Backend unreachable")
            st.caption(f"Expected at {BACKEND_URL}")
        elif health["mode"] == "llm":
            st.success(f"AI answers · {health['model']}")
            st.caption(f"{health['indexed_chunks']} knowledge-base passages indexed")
        else:
            st.warning("Demo mode")
            st.caption(
                "No API key is configured, so answers are passages quoted directly from "
                "the knowledge base. Add GEMINI_API_KEY to the .env file and restart the "
                "backend for full AI answers."
            )
            st.caption(f"{health['indexed_chunks']} knowledge-base passages indexed")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    labels = []
    for source in sources:
        label = (source["section"] or source["title"]).split(" > ")[-1]
        if label not in labels:
            labels.append(label)
    st.markdown(
        f'<div class="ng-sources">Sources: {" · ".join(labels)}</div>',
        unsafe_allow_html=True,
    )


def render_message(message: dict) -> None:
    avatar = "☁️" if message["role"] == "assistant" else "🙂"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("error"):
            st.error(message["content"], icon="⚠️")
        else:
            st.markdown(message["content"])
            render_sources(message.get("sources", []))


def handle_prompt(prompt: str) -> None:
    prompt = prompt.strip()
    if not prompt:
        return
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant", avatar="☁️"):
        with st.spinner("Looking that up…"):
            try:
                data = ask_backend(prompt, st.session_state.session_id)
                reply = {
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"],
                }
            except BackendError as exc:
                reply = {"role": "assistant", "content": str(exc), "error": True}
        if reply.get("error"):
            st.error(reply["content"], icon="⚠️")
        else:
            st.markdown(reply["content"])
            render_sources(reply["sources"])
    st.session_state.messages.append(reply)


def main() -> None:
    st.set_page_config(page_title=f"{STORE_NAME} Support", page_icon="☁️", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []

    health = fetch_health()
    render_header(health)
    render_sidebar(health)

    render_message({"role": "assistant", "content": WELCOME})
    for message in st.session_state.messages:
        render_message(message)

    # A clicked suggestion is stored and handled on the rerun, after the
    # suggestion buttons have disappeared.
    # The input box is pinned to the bottom wherever it is called; reading it
    # first lets the suggestions hide as soon as a question is sent.
    typed = st.chat_input("Type your question…", max_chars=MAX_MESSAGE_CHARS)
    prompt = typed or st.session_state.pop("pending_prompt", None)

    if not st.session_state.messages and prompt is None:
        st.markdown('<div class="ng-suggest-label">Popular questions</div>', unsafe_allow_html=True)
        columns = st.columns(2)
        for index, suggestion in enumerate(SUGGESTIONS):
            if columns[index % 2].button(suggestion, key=f"s{index}", use_container_width=True):
                st.session_state.pending_prompt = suggestion
                st.rerun()

    if prompt:
        handle_prompt(prompt)


main()
