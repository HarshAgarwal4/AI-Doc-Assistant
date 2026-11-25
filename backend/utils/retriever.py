import asyncio

from fastapi import HTTPException, status
from langfuse import observe
from sqlmodel import Session, select

from app.models.chunkModel import Chunk
from app.models.docModel import Document
from database.database import engine
from utils.embedding import generate_embedding


def _search_sync(q: str, k: int):
    """
    Synchronous database + embedding operations.
    This entire function runs in a separate thread.
    """

    with Session(engine) as session:

        # 1. Check whether any document is ready
        ready_document = session.exec(
            select(Document.id)
            .where(Document.status == "ready")
            .limit(1)
        ).first()

        if ready_document is None:
            return []

        # 2. Generate query embedding
        query_embedding = generate_embedding(q)

        # 3. Calculate cosine distance
        distance = Chunk.embedding.cosine_distance(
            query_embedding
        ).label("distance")

        # 4. Retrieve top K matching chunks
        statement = (
            select(Chunk, Document.filename, distance)
            .join(
                Document,
                Chunk.document_id == Document.id
            )
            .where(Document.status == "ready")
            .order_by(distance.asc())
            .limit(k)
        )

        results = session.exec(statement).all()

        # 5. Format results
        return [
            {
                "text": chunk.text,
                "filename": filename,
                "page": chunk.page_number,
                "score": round(1 - float(dist), 4)
            }
            for chunk, filename, dist in results
        ]

@observe()
async def search(q: str, k: int = 5):

    # Validate query
    if q is None or not q.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Search query cannot be empty"
        )

    # Validate K
    if k < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Number of chunks must be greater than zero"
        )

    # Execute all synchronous work in a separate thread
    results = await asyncio.to_thread(
        _search_sync,
        q,
        k
    )

    return results
