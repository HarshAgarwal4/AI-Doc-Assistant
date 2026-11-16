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
    mock_embeddings = [mock_embedding, mock_embedding]

    with (
        patch("utils.process_doc.engine", test_engine),
        patch("utils.process_doc.read_document_pages", return_value=mock_pages),
        patch("utils.process_doc.generate_embeddings", return_value=mock_embeddings) as mock_embed,
    ):
        process_document("dummy_path.pdf", doc.id)
        mock_embed.assert_called_once_with([
            "Page 1 test content for vector database.",
            "Page 2 test content with additional details.",
        ])

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
        patch("utils.process_doc.read_document_pages", side_effect=Exception("Corrupted PDF file")),
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
        patch("utils.process_doc.read_document_pages", return_value=[]),
    ):
        process_document("empty_path.pdf", doc.id)

    session.refresh(doc)
    assert doc.status == "failed"


def test_process_document_nonexistent_id(session: Session):
    # Calling with nonexistent UUID should not raise exception or crash
    random_id = uuid4()
    with patch("utils.process_doc.engine", test_engine):
        process_document("some_file.pdf", random_id)


def test_process_document_reprocessing_clears_old_chunks(session: Session):
    doc = Document(
        filename="reprocess.pdf",
        content_type="application/pdf",
        size_bytes=500,
        status="ready",
        file_hash="old_hash_111",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # Add old chunks
    old_chunks = [
        Chunk(document_id=doc.id, text="Old chunk 1", filename="reprocess.pdf", page_number=1, embedding=[0.1] * 768),
        Chunk(document_id=doc.id, text="Old chunk 2", filename="reprocess.pdf", page_number=1, embedding=[0.2] * 768),
    ]
    session.add_all(old_chunks)
    session.commit()

    assert len(session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()) == 2

    # Reprocess with changed hash / new content
    new_pages = [{"page_number": 1, "text": "Brand new reprocessed content."}]
    new_embedding = [0.9] * 768

    with (
        patch("utils.process_doc.engine", test_engine),
        patch("utils.process_doc.compute_file_hash", return_value="new_hash_222"),
        patch("utils.process_doc.read_document_pages", return_value=new_pages),
        patch("utils.process_doc.generate_embeddings", return_value=[new_embedding]),
    ):
        process_document("reprocess.pdf", doc.id)

    session.refresh(doc)
    assert doc.status == "ready"
    assert doc.file_hash == "new_hash_222"

    chunks = session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()
    # Ensure old chunks were deleted and only the new 1 chunk exists (no duplicates accumulated!)
    assert len(chunks) == 1
    assert chunks[0].text == "Brand new reprocessed content."


def test_process_document_reprocessing_failure_purges_chunks(session: Session):
    doc = Document(
        filename="fail_reprocess.pdf",
        content_type="application/pdf",
        size_bytes=500,
        status="ready",
        file_hash="valid_hash_111",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # Add old chunks
    old_chunks = [
        Chunk(document_id=doc.id, text="Old chunk 1", filename="fail_reprocess.pdf", page_number=1, embedding=[0.1] * 768),
    ]
    session.add_all(old_chunks)
    session.commit()

    # Reprocess encounters an error during extraction
    with (
        patch("utils.process_doc.engine", test_engine),
        patch("utils.process_doc.compute_file_hash", return_value="diff_hash_333"),
        patch("utils.process_doc.read_document_pages", side_effect=RuntimeError("Extraction crashed")),
    ):
        process_document("fail_reprocess.pdf", doc.id)

    session.refresh(doc)
    assert doc.status == "failed"

    # Crucial assertion: Old chunks MUST be purged, not left behind in the database
    chunks = session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()
    assert len(chunks) == 0


def test_process_document_skip_duplicate_hash(session: Session):
    doc = Document(
        filename="duplicate.pdf",
        content_type="application/pdf",
        size_bytes=500,
        status="ready",
        file_hash="same_hash_444",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    chunk = Chunk(document_id=doc.id, text="Existing chunk", filename="duplicate.pdf", page_number=1, embedding=[0.5] * 768)
    session.add(chunk)
    session.commit()

    with (
        patch("utils.process_doc.engine", test_engine),
        patch("utils.process_doc.compute_file_hash", return_value="same_hash_444"),
        patch("utils.process_doc.read_document_pages") as mock_extract,
    ):
        process_document("duplicate.pdf", doc.id)
        # Should have skipped re-extraction since hash matches and document is already ready
        mock_extract.assert_not_called()

    session.refresh(doc)
    assert doc.status == "ready"
    chunks = session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()
    assert len(chunks) == 1


# -------------------------------------------------------------
# 3. Dynamic Extraction Tests (Zero static sample files needed)
# -------------------------------------------------------------
def test_read_pdf_pages_generated(tmp_path):
    import pymupdf

    pdf_file = tmp_path / "dynamic_test.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "First page sample text.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Second page sample text.")
    doc.save(str(pdf_file))
    doc.close()

    pages = read_pdf_pages(str(pdf_file))
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "First page sample text." in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Second page sample text." in pages[1]["text"]


def test_read_docx_pages_generated(tmp_path):
    import docx

    from utils.extraction import read_docx_with_docling

    docx_file = tmp_path / "dynamic_test.docx"
    doc = docx.Document()
    doc.add_paragraph("Dynamic DOCX page one text.")
    doc.add_page_break()
    doc.add_paragraph("Dynamic DOCX page two text.")
    doc.save(str(docx_file))

    pages = read_docx_with_docling(str(docx_file))
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Dynamic DOCX page one text." in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Dynamic DOCX page two text." in pages[1]["text"]


def test_read_document_pages_routing(tmp_path):
    from utils.extraction import read_document_pages

    pdf_path = str(tmp_path / "doc.pdf")
    docx_path = str(tmp_path / "doc.docx")

    with (
        patch("utils.extraction.read_pdf_pages", return_value=[{"page_number": 1, "text": "pdf_text"}]) as mock_pdf,
        patch("utils.extraction.read_docx_with_docling", return_value=[{"page_number": 1, "text": "docx_text"}]) as mock_docx,
    ):
        res_pdf = read_document_pages(pdf_path)
        assert res_pdf == [{"page_number": 1, "text": "pdf_text"}]
        mock_pdf.assert_called_once_with(pdf_path)

        res_docx = read_document_pages(docx_path)
        assert res_docx == [{"page_number": 1, "text": "docx_text"}]
        mock_docx.assert_called_once_with(docx_path)
