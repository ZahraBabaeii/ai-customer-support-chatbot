"""In-memory conversation history, keyed by session id."""

import threading
from collections import OrderedDict

from app.llm import Message


class SessionStore:
    """Keeps the last ``max_messages`` messages for up to ``max_sessions`` sessions.

    The least recently used session is evicted when the limit is reached.
    History lives in process memory, so it is lost when the server restarts.
    """

    def __init__(self, max_messages: int = 12, max_sessions: int = 1000) -> None:
        self._max_messages = max_messages
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[Message]] = OrderedDict()
        self._lock = threading.Lock()

    def history(self, session_id: str) -> list[Message]:
        with self._lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
            return list(self._sessions.get(session_id, []))

    def append_turn(self, session_id: str, user_message: str, answer: str) -> None:
        with self._lock:
            messages = self._sessions.setdefault(session_id, [])
            messages.extend([Message("user", user_message), Message("assistant", answer)])
            del messages[: -self._max_messages]
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
