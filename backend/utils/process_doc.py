from uuid import UUID

from sqlmodel import Session

from app.models.chunkModel import Chunk
from app.models.docModel import Document
from database.database import engine
from utils.chunking import chunk_page_data
from utils.embedding import generate_embedding
from utils.extraction import read_pdf_pages
from utils.logger import logger


def process_document(file_path: str, document_id: UUID | str):
    doc_uuid = UUID(str(document_id))

    with Session(engine) as session:
        document = session.get(Document, doc_uuid)
        if not document:
            logger.error(f"Document {document_id} not found.")
            return

        try:
            document.status = "processing"
            session.add(document)
            session.commit()

            # 1. Read PDF page by page
            pages = read_pdf_pages(file_path)

            # 2. Split pages into chunks
            chunks_data = chunk_page_data(pages, document.filename)
            if not chunks_data:
                raise ValueError("No text found in PDF.")

            # 3. Generate embedding for each chunk individually
            chunks = []
            for c in chunks_data:
                embedding = generate_embedding(c["text"])
                chunk = Chunk(
                    document_id=document.id,
                    text=c["text"],
                    filename=c["filename"],
                    page_number=c["page_nu" \
                    "mber"],
                    embedding=embedding,
                )
                chunks.append(chunk)

            # 4. Store all chunks at once in the database
            session.add_all(chunks)

            # 5. Mark document as ready
            document.status = "ready"
            session.add(document)
            session.commit()

            logger.info(f"Saved {len(chunks)} chunks for {document.filename}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Processing failed for {document.filename}: {e}")
            session.rollback()
            document.status = "failed"
            session.add(document)
            session.commit()
