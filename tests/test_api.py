from app.schemas import MAX_MESSAGE_CHARS
from app.sessions import SessionStore


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "demo"
    assert body["model"] is None
    assert body["indexed_chunks"] > 40


def test_chat_returns_grounded_answer_and_sources(client):
    response = client.post("/chat", json={"message": "Do you ship internationally?"})
    assert response.status_code == 200
    body = response.json()
    assert "Canada" in body["answer"]
    assert body["grounded"] is True
    assert body["sources"][0]["source"] == "shipping_policy.md"
    assert len(body["session_id"]) >= 8


def test_chat_keeps_context_within_a_session(client):
    first = client.post("/chat", json={"message": "Tell me about the Aero Pro headphones"})
    session_id = first.json()["session_id"]
    second = client.post("/chat", json={"message": "How much is it?", "session_id": session_id})
    assert second.json()["session_id"] == session_id
    assert "$249.00" in second.json()["answer"]


def test_sessions_are_isolated(client):
    client.post("/chat", json={"message": "Tell me about the Aero Pro headphones"})
    other = client.post("/chat", json={"message": "How much is it?"})
    assert "$249.00" not in other.json()["answer"]


def test_reset_session(client):
    session_id = client.post("/chat", json={"message": "Hello"}).json()["session_id"]
    assert client.delete(f"/sessions/{session_id}").json() == {"cleared": True}
    assert client.delete(f"/sessions/{session_id}").json() == {"cleared": False}


def test_rejects_invalid_input(client):
    assert client.post("/chat", json={"message": "   "}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422
    too_long = "a" * (MAX_MESSAGE_CHARS + 1)
    assert client.post("/chat", json={"message": too_long}).status_code == 422
    bad_session = {"message": "hi", "session_id": "../etc"}
    assert client.post("/chat", json=bad_session).status_code == 422


def test_session_store_trims_history_and_evicts_oldest():
    sessions = SessionStore(max_messages=4, max_sessions=2)
    for i in range(5):
        sessions.append_turn("a", f"q{i}", f"a{i}")
    assert [m.content for m in sessions.history("a")] == ["q3", "a3", "q4", "a4"]

    sessions.append_turn("b", "q", "a")
    sessions.append_turn("c", "q", "a")
    assert sessions.history("a") == []
    assert len(sessions.history("b")) == 2
