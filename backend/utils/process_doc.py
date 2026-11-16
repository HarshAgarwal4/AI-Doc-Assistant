import hashlib
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, delete, select

from app.models.chunkModel import Chunk
from app.models.docModel import Document
from database.database import engine
from utils.chunking import chunk_page_data
from utils.embedding import generate_embeddings
from utils.extraction import read_document_pages
from utils.logger import logger


def compute_file_hash(file_path: str) -> str | None:
    if not Path(file_path).exists():
        return None
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_document(file_path: str, document_id: UUID | str):
    """
    Process document in background:
    1. Compute file_hash and prevent duplicate processing if unchanged and ready
    2. Remove old chunks for this document to prevent duplicate accumulation
    3. Extract text (PyMuPDF for PDF, Docling for DOCX)
    4. Split pages into chunks
    5. Generate embeddings in batch using generate_embeddings
    6. Store all chunks at once in the database
    7. If processing fails, purge chunks so failed document has no orphan chunks
    """
    try:
        doc_uuid = UUID(str(document_id))

        with Session(engine) as session:
            document = session.get(Document, doc_uuid)
            if not document:
                logger.error(f"Document {document_id} not found.")
                return

            try:
                # Compute file hash
                current_hash = compute_file_hash(file_path) or document.file_hash

                # If already processed with same hash and ready with chunks, skip duplicate processing
                if document.file_hash == current_hash and document.status == "ready" and current_hash is not None:
                    existing_chunk = session.exec(select(Chunk).where(Chunk.document_id == document.id)).first()
                    if existing_chunk:
                        logger.info(f"Document {document.filename} already processed with hash {current_hash}. Skipping.")
                        return

                # Purge any old chunks to prevent duplicates during reprocessing
                session.exec(delete(Chunk).where(Chunk.document_id == document.id))

                document.file_hash = current_hash
                document.status = "processing"
                session.add(document)
                session.commit()

                # 1. Read document pages
                pages = read_document_pages(file_path)

                # 2. Split pages into chunks
                chunks_data = chunk_page_data(pages, document.filename)
                if not chunks_data:
                    raise ValueError("No text found in document.")

                # 3. Generate embeddings in batch
                texts = [c["text"] for c in chunks_data]
                embeddings = generate_embeddings(texts)

                # 4. Construct Chunk instances
                chunks = [
                    Chunk(
                        document_id=document.id,
                        text=c["text"],
                        filename=c["filename"],
                        page_number=c["page_number"],
                        embedding=emb,
                    )
                    for c, emb in zip(chunks_data, embeddings, strict=False)
                ]

                # 5. Store all chunks at once in the database
                session.add_all(chunks)

                # 6. Mark document as ready
                document.status = "ready"
                session.add(document)
                session.commit()

                logger.info(f"Saved {len(chunks)} chunks for {document.filename}")

            except Exception as e:  # noqa: BLE001
                logger.error(f"Processing failed for {document.filename}: {e}")
                session.rollback()
                # Purge chunks so failed document never keeps old/orphan chunks
                session.exec(delete(Chunk).where(Chunk.document_id == document.id))
                document.status = "failed"
                session.add(document)
                session.commit()

    except Exception as outer_err:  # noqa: BLE001
        logger.error(f"Unexpected error in process_document for {document_id}: {outer_err}")
