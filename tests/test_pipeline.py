from app.llm import Message
from app.rag import demo
from app.rag.pipeline import ChatService, is_follow_up
from tests.conftest import MIN_RELEVANCE, FakeLLM


def llm_service(store, llm):
    return ChatService(store, llm=llm, top_k=4, min_relevance=MIN_RELEVANCE)


# --- demo mode -------------------------------------------------------------

def test_demo_answer_is_grounded_in_knowledge_base(demo_service):
    result = demo_service.answer("What is your return window?")
    assert result.mode == "demo"
    assert result.grounded
    assert "30 days" in result.answer
    assert result.sources[0].source == "returns_and_refunds.md"


def test_demo_gives_product_price(demo_service):
    result = demo_service.answer("How much is the Nimbus Type 75 mechanical keyboard?")
    assert "$149.00" in result.answer


def test_demo_says_when_information_is_unavailable(demo_service):
    result = demo_service.answer("What is the capital of France?")
    assert result.answer == demo.NO_ANSWER
    assert not result.grounded
    assert result.sources == []


def test_demo_handles_greetings_without_retrieval(demo_service):
    result = demo_service.answer("Hello!")
    assert "Nimbus" in result.answer
    assert not result.grounded


def test_demo_follow_up_uses_conversation_context(demo_service):
    history = [
        Message("user", "Tell me about the Aero Pro headphones"),
        Message("assistant", "They are our flagship headphones."),
    ]
    result = demo_service.answer("How much is it?", history)
    assert "$249.00" in result.answer


# --- follow-up detection ---------------------------------------------------

def test_is_follow_up():
    assert is_follow_up("How much is it?")
    assert is_follow_up("What about international orders?")
    assert is_follow_up("And returns?")
    assert not is_follow_up("Which payment methods do you accept at checkout?")


def test_standalone_question_is_not_rewritten(store):
    llm = FakeLLM()
    service = llm_service(store, llm)
    history = [Message("user", "Hi"), Message("assistant", "Hello")]
    question = "Which payment methods do you accept at checkout?"
    assert service.search_query(question, history) == question
    assert llm.calls == []


# --- LLM mode --------------------------------------------------------------

def test_llm_receives_retrieved_context_and_history(store):
    llm = FakeLLM(reply="Overnight shipping is $24.95.")
    history = [Message("user", "Hi"), Message("assistant", "Hello!")]
    result = llm_service(store, llm).answer("How much is overnight shipping?", history)

    assert result.mode == "llm"
    assert result.answer == "Overnight shipping is $24.95."
    assert result.grounded

    system_prompt, messages = llm.calls[-1]
    assert "ONLY" in system_prompt
    assert messages[:2] == history
    final_turn = messages[-1].content
    assert "$24.95" in final_turn  # retrieved passage
    assert "How much is overnight shipping?" in final_turn


def test_llm_is_told_when_nothing_relevant_was_found(store):
    llm = FakeLLM(reply="I don't have that information.")
    result = llm_service(store, llm).answer("What is the capital of France?")
    assert not result.grounded
    assert "No relevant knowledge base entries" in llm.calls[-1][1][-1].content


def test_llm_rewrites_follow_up_into_search_query(store):
    llm = FakeLLM(reply="Aero Pro headphones price")
    history = [
        Message("user", "Tell me about the Aero Pro headphones"),
        Message("assistant", "They are our flagship headphones."),
    ]
    query = llm_service(store, llm).search_query("How much is it?", history)
    assert query == "Aero Pro headphones price"


def test_llm_failure_falls_back_to_extractive_answer(store):
    result = llm_service(store, FakeLLM(fail=True)).answer("What is your return window?")
    assert result.mode == "fallback"
    assert "30 days" in result.answer
