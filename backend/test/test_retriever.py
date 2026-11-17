from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlmodel import Session, SQLModel, create_engine

from app.models.chunkModel import Chunk, SearchResult
from app.models.docModel import Document
from main import app, search_endpoint
from utils.retriever import _search_sync, search

# In-memory SQLite engine for tests
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
)


@pytest.fixture
def session():
    SQLModel.metadata.drop_all(test_engine)
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as s:
        yield s
    SQLModel.metadata.drop_all(test_engine)


# Global autouse fixture ensuring real Ollama embedding is NEVER called in search tests
@pytest.fixture(autouse=True)
def prevent_real_embedding():
    """Ensure that search tests never invoke real Ollama embedding API."""
    mock_vector = [0.05] * 768
    with (
        patch("utils.retriever.generate_embedding", return_value=mock_vector) as mock_retriever_embed,
        patch("utils.embedding.generate_embedding", return_value=mock_vector),
        patch("langchain_ollama.OllamaEmbeddings.embed_query", return_value=mock_vector),
        patch("langchain_ollama.OllamaEmbeddings.embed_documents", return_value=[mock_vector]),
    ):
        yield mock_retriever_embed


# -------------------------------------------------------------
# 1. Validation Tests for search()
# -------------------------------------------------------------
def test_search_empty_query_raises_422():
    with pytest.raises(HTTPException) as exc_info:
        coro = search(q="", k=5)
        try:
            coro.send(None)
        finally:
            coro.close()
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Search query cannot be empty"


def test_search_whitespace_query_raises_422():
    with pytest.raises(HTTPException) as exc_info:
        coro = search(q="   ", k=5)
        try:
            coro.send(None)
        finally:
            coro.close()
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Search query cannot be empty"


def test_search_none_query_raises_422():
    with pytest.raises(HTTPException) as exc_info:
        coro = search(q=None, k=5)
        try:
            coro.send(None)
        finally:
            coro.close()
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Search query cannot be empty"


def test_search_k_less_than_one_raises_422():
    with pytest.raises(HTTPException) as exc_info:
        coro = search(q="valid query", k=0)
        try:
            coro.send(None)
        finally:
            coro.close()
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Number of chunks must be greater than zero"

    with pytest.raises(HTTPException) as exc_info_neg:
        coro = search(q="valid query", k=-1)
        try:
            coro.send(None)
        finally:
            coro.close()
    assert exc_info_neg.value.status_code == 422
    assert exc_info_neg.value.detail == "Number of chunks must be greater than zero"


# -------------------------------------------------------------
# 2. _search_sync() Logic Tests
# -------------------------------------------------------------
def test_search_sync_empty_db(session: Session):
    with patch("utils.retriever.engine", test_engine), \
         patch("utils.retriever.generate_embedding") as mock_embed:
        results = _search_sync("sample question", 5)
        assert results == []
        # When no ready document exists, embedding generation must NOT be called
        mock_embed.assert_not_called()


def test_search_sync_with_pending_document_only(session: Session):
    doc = Document(
        filename="pending.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        status="pending",
    )
    session.add(doc)
    session.commit()

    with patch("utils.retriever.engine", test_engine), \
         patch("utils.retriever.generate_embedding") as mock_embed:
        results = _search_sync("sample question", 5)
        assert results == []
        mock_embed.assert_not_called()


def test_search_sync_with_ready_document(session: Session):
    doc = Document(
        filename="sample.pdf",
        content_type="application/pdf",
        size_bytes=2048,
        status="ready",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    mock_chunk1 = MagicMock(spec=Chunk)
    mock_chunk1.text = "This is chunk 1 text about machine learning."
    mock_chunk1.page_number = 1

    mock_chunk2 = MagicMock(spec=Chunk)
    mock_chunk2.text = "This is chunk 2 text with more details."
    mock_chunk2.page_number = 2

    # Mock results returned from SQL query: (Chunk, Document.filename, distance)
    # distance = 0.2 -> score = 1 - 0.2 = 0.8
    # distance = 0.45 -> score = 1 - 0.45 = 0.55
    mock_db_results = [
        (mock_chunk1, "sample.pdf", 0.20),
        (mock_chunk2, "sample.pdf", 0.45),
    ]

    mock_query_vector = [0.05] * 768

    with patch("utils.retriever.engine", test_engine), \
         patch("utils.retriever.generate_embedding", return_value=mock_query_vector) as mock_embed, \
         patch("sqlmodel.Session.exec") as mock_exec:

        mock_exec.side_effect = [
            MagicMock(first=lambda: doc.id),
            MagicMock(all=lambda: mock_db_results),
        ]

        results = _search_sync("what is machine learning?", k=2)

        # Verify embedding was generated for query with same embedding model function
        mock_embed.assert_called_once_with("what is machine learning?")

        assert len(results) == 2
        assert results[0] == {
            "text": "This is chunk 1 text about machine learning.",
            "filename": "sample.pdf",
            "page": 1,
            "score": 0.8,
        }
        assert results[1] == {
            "text": "This is chunk 2 text with more details.",
            "filename": "sample.pdf",
            "page": 2,
            "score": 0.55,
        }


def test_search_never_uses_real_embedding(session: Session):
    doc = Document(
        filename="test.pdf",
        content_type="application/pdf",
        size_bytes=100,
        status="ready",
    )
    session.add(doc)
    session.commit()

    mock_chunk = MagicMock(spec=Chunk)
    mock_chunk.text = "Sample chunk"
    mock_chunk.page_number = 1
    mock_results = [(mock_chunk, "test.pdf", 0.1)]

    with patch("utils.retriever.engine", test_engine), \
         patch("utils.retriever.generate_embedding", return_value=[0.05] * 768) as mock_embed, \
         patch("sqlmodel.Session.exec") as mock_exec:
        mock_exec.side_effect = [
            MagicMock(first=lambda: doc.id),
            MagicMock(all=lambda: mock_results),
        ]
        results = _search_sync("query test", 1)

        # Confirm mock was used and real Ollama was never called
        mock_embed.assert_called_once_with("query test")
        assert len(results) == 1
        assert results[0]["score"] == 0.9


# -------------------------------------------------------------
# 3. FastAPI Endpoint Configuration & Handler Tests
# -------------------------------------------------------------
def test_search_route_registered_on_fastapi_app():
    route_paths = [r.path for r in app.routes]
    assert "/search" in route_paths

    route = next(r for r in app.routes if getattr(r, "path", None) == "/search")
    assert "GET" in route.methods
    assert route.response_model == list[SearchResult]


def test_search_endpoint_handler_delegation():
    mock_results = [
        {
            "text": "Found relevant text",
            "filename": "doc.pdf",
            "page": 1,
            "score": 0.92,
        }
    ]

    with patch("main.search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_results

        import inspect
        coro = search_endpoint(q="how does it work?", k=3)
        assert inspect.iscoroutine(coro)

        # Step the coroutine to completion
        try:
            coro.send(None)
        except StopIteration as stop:
            result = stop.value

        mock_search.assert_called_once_with(q="how does it work?", k=3)
        assert result == mock_results
