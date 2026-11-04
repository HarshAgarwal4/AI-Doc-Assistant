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
def client(test_session: Session) -> Generator[TestClient, None, None]:
    def override_get_session():
        yield test_session

    app.dependency_overrides[get_session] = override_get_session
    test_client = TestClient(app)
    yield test_client
    test_client.close()
    app.dependency_overrides.clear()


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
    assert response.text == "Hello world!"

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
        client.post("/ask", json=payload)

    cause = getattr(exc_info.value, "__cause__", exc_info.value)
    assert isinstance(cause, HTTPException)
    assert cause.status_code == 503
    assert "LLM provider is unavailable after all retry attempts" in cause.detail


# 3. Test timeout handling (exhausts retries and raises 504 Gateway Timeout)
@patch("utils.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("utils.llm.get_llm", return_value=MockTimeoutLLM())
def test_ask_timeout(mock_get_llm, mock_sleep, client: TestClient):
    payload = {"question": "Test timeout", "provider": "Google"}
    with pytest.raises(Exception) as exc_info:
        client.post("/ask", json=payload)

    cause = getattr(exc_info.value, "__cause__", exc_info.value)
    assert isinstance(cause, HTTPException)
    assert cause.status_code == 504
    assert "LLM request timed out after all retry attempts" in cause.detail


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
