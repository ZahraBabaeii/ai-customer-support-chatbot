"""Demo-mode responder: answers without an LLM.

It returns the most relevant knowledge-base passage verbatim, so answers are
always grounded, but it cannot rephrase, combine facts, or reason.
"""

import re

from app.config import ASSISTANT_NAME, STORE_NAME, SUPPORT_EMAIL, SUPPORT_PHONE
from app.rag.vector_store import SearchHit

_GREETING = re.compile(r"^(hi|hello|hey|hiya|good (morning|afternoon|evening))\b", re.I)
_THANKS = re.compile(r"^(thanks|thank you|thx|cheers|great,? thanks)\b", re.I)
_GOODBYE = re.compile(r"^(bye|goodbye|see you)\b", re.I)

NO_ANSWER = (
    f"I'm sorry, I don't have information about that in the {STORE_NAME} knowledge base. "
    f"Our support team can help: email {SUPPORT_EMAIL} or call {SUPPORT_PHONE}."
)


def small_talk_reply(message: str) -> str | None:
    """Reply to short greetings/thanks; None if the message is a real question."""
    text = message.strip()
    if len(text.split()) > 4:
        return None
    if _GREETING.match(text):
        return (
            f"Hello! I'm {ASSISTANT_NAME}, the {STORE_NAME} support assistant. Ask me about "
            "our products, shipping, returns, or store policies."
        )
    if _THANKS.match(text):
        return "You're welcome! Is there anything else I can help you with?"
    if _GOODBYE.match(text):
        return f"Goodbye, and thanks for visiting {STORE_NAME}!"
    return None


def select_passages(hits: list[SearchHit]) -> list[SearchHit]:
    """The top passage, plus the second one if it is nearly as relevant."""
    selected = hits[:1]
    if len(hits) > 1 and hits[1].score >= hits[0].score * 0.8:
        selected.append(hits[1])
    return selected


def extractive_answer(hits: list[SearchHit]) -> str:
    """Quote the selected passages as the answer."""
    selected = select_passages(hits)
    if not selected:
        return NO_ANSWER

    parts = []
    for hit in selected:
        heading = (hit.chunk.section or hit.chunk.title).split(" > ")[-1]
        parts.append(f"**{heading}**\n\n{hit.chunk.body}")
    return "Here's what I found:\n\n" + "\n\n---\n\n".join(parts)
