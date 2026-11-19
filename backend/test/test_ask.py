import asyncio
from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models.docModel import Document  # noqa: F401
from app.models.llm import LLMCall
from database.database import get_session
from main import app
from utils.llm import generate

# In-memory SQLite database for testing
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@pytest.fixture
def test_session() -> Generator[Session, None, None]:
    SQLModel.metadata.drop_all(test_engine)
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture
def client(
    test_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    def override_get_session():
        yield test_session

    app.dependency_overrides[get_session] = override_get_session
    monkeypatch.setattr("utils.llm.engine", test_engine)
    test_client = TestClient(app)
    yield test_client
    test_client.close()
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def mock_retriever_search():
    with patch("utils.llm.search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [
            {"text": "Python is a high-level language.", "filename": "python_guide.pdf", "page": 1, "score": 0.95},
            {"text": "Python supports multiple paradigms.", "filename": "python_guide.pdf", "page": 2, "score": 0.90},
        ]
        yield mock_search


class MockChunk:
    def __init__(self, content: str, input_tokens: int = 5, output_tokens: int = 5):
        self.content = content
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


class MockLLM:
    async def astream(self, question: str):
        yield MockChunk("Hello ", 10, 5)
        yield MockChunk("world!", 10, 10)


class MockRateLimitLLM:
    async def astream(self, question: str):
        request = httpx.Request("POST", "http://fake-provider")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("Rate limited", request=request, response=response)
        yield


class MockTimeoutLLM:
    async def astream(self, question: str):
        raise TimeoutError("Connection timed out")
        yield


# 1. Test successful streaming response and DB persistence
@patch("utils.llm.get_llm", return_value=MockLLM())
def test_ask_success(mock_get_llm, client: TestClient, test_session: Session):
    payload = {"question": "What is Python?", "provider": "Ollama"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    assert "event: token" in response.text
    assert 'data: {"text": "Hello "}' in response.text
    assert 'data: {"text": "world!"}' in response.text
    assert "event: saved" in response.text
    assert "event: pages" not in response.text
    assert "event: filename" not in response.text
    assert "event: Chunks" not in response.text
    assert "event: Response" not in response.text

    # Verify call saved into SQLite database
    saved_calls = test_session.exec(select(LLMCall)).all()
    assert len(saved_calls) == 1
    assert saved_calls[0].input == "What is Python?"
    assert saved_calls[0].output == "Hello world!"
    assert saved_calls[0].provider == "Ollama"


# 2. Test 429 rate limit error handling (exhausts retries and raises 503)
@patch("utils.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("utils.llm.get_llm", return_value=MockRateLimitLLM())
def test_ask_rate_limit_429(mock_get_llm, mock_sleep, client: TestClient):
    payload = {"question": "Test rate limit", "provider": "Groq"}
    with pytest.raises(Exception) as exc_info:
        try:
            response = client.post("/ask", json=payload)
            _ = response.text
        except Exception:  # noqa: TRY203
            raise           
        else:
            async def run():
                async for _ in generate(question=payload["question"], provider=payload["provider"]):
                    pass
            asyncio.run(run())

    cause = getattr(exc_info.value, "__cause__", exc_info.value)
    target = exc_info.value if isinstance(exc_info.value, HTTPException) else cause
    assert isinstance(target, HTTPException)
    assert target.status_code == 503
    assert "LLM provider is unavailable after all retry attempts" in target.detail


# 3. Test timeout handling (exhausts retries and raises 504 Gateway Timeout)
@patch("utils.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("utils.llm.get_llm", return_value=MockTimeoutLLM())
def test_ask_timeout(mock_get_llm, mock_sleep, client: TestClient):
    payload = {"question": "Test timeout", "provider": "Google"}
    with pytest.raises(Exception) as exc_info:
        try:
            response = client.post("/ask", json=payload)
            _ = response.text
        except Exception:  # noqa: TRY203
            raise
        else:
            async def run():
                async for _ in generate(question=payload["question"], provider=payload["provider"]):
                    pass
            asyncio.run(run())

    cause = getattr(exc_info.value, "__cause__", exc_info.value)
    target = exc_info.value if isinstance(exc_info.value, HTTPException) else cause
    assert isinstance(target, HTTPException)
    assert target.status_code == 504
    assert "LLM request timed out after all retry attempts" in target.detail


# 4. Test validation error: invalid provider
def test_ask_validation_invalid_provider(client: TestClient):
    payload = {"question": "Hello", "provider": "UnsupportedLLM"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 422


# 5. Test validation error: missing question field
def test_ask_validation_missing_question(client: TestClient):
    payload = {"provider": "Ollama"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 422


# 6. Test /page endpoint returns HTML
def test_page_endpoint(client: TestClient):
    response = client.get("/page")
    assert response.status_code == 200
    assert "doc-files" in response.text
    assert "doc-pages" in response.text


# 7. Test direct generate() yields token and saved events without raw pages/filename events
def test_generate_events(monkeypatch: pytest.MonkeyPatch, test_session: Session):
    monkeypatch.setattr("utils.llm.engine", test_engine)
    with patch("utils.llm.get_llm", return_value=MockLLM()):
        async def run():
            events = []
            async for ev in generate(question="What is Python?", provider="Ollama"):
                events.append(ev)
            return "".join(events)

        output = asyncio.run(run())
        assert "event: token" in output
        assert "event: saved" in output
        assert "event: pages" not in output
        assert "event: filename" not in output
        assert "event: Chunks" not in output
        assert "event: Response" not in output


# 8. Test RAG edge case: empty retrieval results
@patch("utils.llm.get_llm", return_value=MockLLM())
def test_ask_empty_retrieval_results(mock_get_llm, client: TestClient, mock_retriever_search):
    mock_retriever_search.return_value = []
    payload = {"question": "Question with no matching documents", "provider": "Ollama"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    assert "event: token" in response.text
    assert 'data: {"text": "Hello "}' in response.text
    assert 'data: {"text": "world!"}' in response.text
    assert "event: saved" in response.text
    assert "event: pages" not in response.text
    assert "event: filename" not in response.text


# 9. Test RAG edge case: missing chunk metadata (None page/filename)
@patch("utils.llm.get_llm", return_value=MockLLM())
def test_ask_missing_chunk_metadata(mock_get_llm, client: TestClient, mock_retriever_search):
    mock_retriever_search.return_value = [
        {"text": "Content with missing metadata", "filename": None, "page": None, "score": 0.85}
    ]
    payload = {"question": "Question with null chunk metadata", "provider": "Ollama"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: saved" in response.text
    assert "event: pages" not in response.text
    assert "event: filename" not in response.text


# 10. Test prompt formatting includes filename and page number for LLM context
def test_generate_prompt_contains_filename_and_pages(monkeypatch: pytest.MonkeyPatch, test_session: Session):
    monkeypatch.setattr("utils.llm.engine", test_engine)
    captured_messages = []

    class CapturingLLM:
        async def astream(self, messages):
            captured_messages.extend(messages)
            yield MockChunk("I don't know.", 5, 5)

    with patch("utils.llm.get_llm", return_value=CapturingLLM()):
        async def run():
            events = []
            async for ev in generate(question="What is Python?", provider="Ollama"):
                events.append(ev)
            return "".join(events)

        _ = asyncio.run(run())

    assert len(captured_messages) == 2
    system_msg, user_msg = captured_messages[0], captured_messages[1]

    # Verify system prompt instructs LLM on filename, page, and 'I don't know' rules
    system_prompt_text = system_msg.content
    assert "I don't know." in system_prompt_text
    assert "show the source filename and page number" in system_prompt_text
    assert "If and only if the response is successful" in system_prompt_text

    # Verify user prompt contains filenames and page numbers of retrieved chunks
    user_prompt_text = user_msg.content
    assert "Filename: python_guide.pdf" in user_prompt_text
    assert "Page 1" in user_prompt_text
    assert "Page 2" in user_prompt_text


# 10. Test RAG edge case: retrieval failure handling (HTTP 502)
@patch("utils.retry.asyncio.sleep", new_callable=AsyncMock)
def test_ask_retrieval_failure(mock_sleep, client: TestClient, mock_retriever_search):
    mock_retriever_search.side_effect = RuntimeError("Database connection failed during retrieval")
    payload = {"question": "What is Python?", "provider": "Ollama"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 502
    assert "LLM aprovider failed before streaming started" in response.json()["detail"]


# 11. Test direct generate() raises on retrieval failure
@patch("utils.retry.asyncio.sleep", new_callable=AsyncMock)
def test_generate_retrieval_failure_raises(mock_sleep, mock_retriever_search):
    mock_retriever_search.side_effect = RuntimeError("Vector store unreachable")

    async def run():
        async for _ in generate(question="What is Python?", provider="Ollama"):
            pass

    with pytest.raises((RuntimeError, AttributeError)):
        asyncio.run(run())
