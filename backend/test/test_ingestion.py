import os
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models.chunkModel import Chunk
from app.models.docModel import Document
from utils.chunking import chunk_page_data
from utils.extraction import read_pdf_pages
from utils.process_doc import process_document

# In-memory SQLite engine for tests
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@pytest.fixture
def session():
    SQLModel.metadata.drop_all(test_engine)
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as s:
        yield s
    SQLModel.metadata.drop_all(test_engine)


# -------------------------------------------------------------
# 1. Chunking Tests
# -------------------------------------------------------------
def test_chunk_page_data_basic():
    pages = [
        {"page_number": 1, "text": "This is page one content for testing chunking."}
    ]
    chunks = chunk_page_data(pages, "sample.pdf")

    assert len(chunks) == 1
    assert chunks[0]["text"] == "This is page one content for testing chunking."
    assert chunks[0]["filename"] == "sample.pdf"
    assert chunks[0]["page_number"] == 1


def test_chunk_page_data_multiple_pages():
    pages = [
        {"page_number": 1, "text": "First page text."},
        {"page_number": 2, "text": "Second page text."},
    ]
    chunks = chunk_page_data(pages, "multi.pdf")

    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["filename"] == "multi.pdf"
    assert chunks[1]["page_number"] == 2
    assert chunks[1]["filename"] == "multi.pdf"


def test_chunk_page_data_empty_page_skipped():
    pages = [
        {"page_number": 1, "text": "   "},
        {"page_number": 2, "text": ""},
        {"page_number": 3, "text": "Valid text on page three."},
    ]
    chunks = chunk_page_data(pages, "test.pdf")

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 3
    assert chunks[0]["text"] == "Valid text on page three."


def test_chunk_size_and_overlap():
    long_text = ("Artificial intelligence is transforming technology. " * 30).strip()
    pages = [{"page_number": 1, "text": long_text}]

    chunks = chunk_page_data(pages, "long.pdf")

    # Long text should be split into multiple chunks
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 1000
        assert chunk["filename"] == "long.pdf"
        assert chunk["page_number"] == 1


# -------------------------------------------------------------
# 2. Process Document Tests (Status Changes & DB Insertion)
# -------------------------------------------------------------
def test_process_document_success(session: Session):
    # Setup document in pending status
    doc = Document(
        filename="test.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        status="pending",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    mock_pages = [
        {"page_number": 1, "text": "Page 1 test content for vector database."},
        {"page_number": 2, "text": "Page 2 test content with additional details."},
    ]
    mock_embedding = [0.05] * 768

    with (
        patch("utils.process_doc.engine", test_engine),
        patch("utils.process_doc.read_pdf_pages", return_value=mock_pages),
        patch("utils.process_doc.generate_embedding", return_value=mock_embedding),
    ):
        process_document("dummy_path.pdf", doc.id)

    # Verify status changed to ready
    session.refresh(doc)
    assert doc.status == "ready"

    # Verify Chunk table rows
    chunks = session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()
    assert len(chunks) == 2

    for chunk in chunks:
        assert chunk.filename == "test.pdf"
        assert chunk.file_name == "test.pdf"
        assert chunk.document_id == doc.id
        assert chunk.page_number in [1, 2]
        assert len(chunk.embedding) == 768
        assert chunk.embedding == mock_embedding


def test_process_document_broken_pdf(session: Session):
    # Setup document
    doc = Document(
        filename="broken.pdf",
        content_type="application/pdf",
        size_bytes=100,
        status="pending",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # Simulate broken PDF where extraction returns empty or raises an exception
    with (
        patch("utils.process_doc.engine", test_engine),
        patch("utils.process_doc.read_pdf_pages", side_effect=Exception("Corrupted PDF file")),
    ):
        process_document("broken_path.pdf", doc.id)

    # Verify document ends in 'failed' status and API did not crash
    session.refresh(doc)
    assert doc.status == "failed"


def test_process_document_empty_pdf(session: Session):
    doc = Document(
        filename="empty.pdf",
        content_type="application/pdf",
        size_bytes=100,
        status="pending",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # PDF with no text
    with (
        patch("utils.process_doc.engine", test_engine),
        patch("utils.process_doc.read_pdf_pages", return_value=[]),
    ):
        process_document("empty_path.pdf", doc.id)

    session.refresh(doc)
    assert doc.status == "failed"


def test_process_document_nonexistent_id(session: Session):
    # Calling with nonexistent UUID should not raise exception or crash
    random_id = uuid4()
    with patch("utils.process_doc.engine", test_engine):
        process_document("some_file.pdf", random_id)


# -------------------------------------------------------------
# 3. Extraction Test on Real PDF (if sample file exists)
# -------------------------------------------------------------
def test_read_pdf_pages_with_existing_file():
    sample_pdf = os.path.join("uploads", "Resume_Eng_Mgr_Chad_Norwood.2025-6-25.pdf")
    if os.path.exists(sample_pdf):
        pages = read_pdf_pages(sample_pdf)
        assert len(pages) > 0
        assert pages[0]["page_number"] == 1
        assert len(pages[0]["text"]) > 0
