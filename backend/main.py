from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from sqlmodel import Session, select

from app.models.docModel import Document
from database.database import create_db_and_tables, get_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(
    title="AI Doc Assistant",
    description="My first FastAPI backend with Postgres",
    lifespan=lifespan,
    version="1.0.0",
)


MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = [".pdf", ".txt", ".docx"]
CHUNK_SIZE = 1024 * 1024


@app.get("/", status_code=status.HTTP_200_OK)
def home():
    return {"message": "Welcome to AI Doc Assistant"}


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok"}


@app.post(
    "/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=Document,
)
async def post_document(
    file: Annotated[UploadFile, File()],
    session: Annotated[Session, Depends(get_session)],
):
    extension = Path(file.filename or "").suffix.lower()
    total_size = 0

    # Check file format
    if extension not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File format not allowed",
        )

    # Check file size if available
    if file.size is not None and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File size must not be greater than 10 MB",
        )

    # Read file chunk by chunk and enforce size limit
    while True:
        chunk = await file.read(CHUNK_SIZE)

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="File size must not exceed 10 MB",
            )

    # Reject empty files
    if total_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file not allowed",
        )

    try:
        document = Document(
            filename=Path(file.filename or "").name,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=total_size,
        )

        session.add(document)
        session.commit()
        session.refresh(document)

        return document

    except Exception:
        session.rollback()
        raise

    finally:
        await file.close()


@app.get(
    "/documents",
    response_model=list[Document],
    status_code=status.HTTP_200_OK,
)
def get_documents(
    session: Annotated[Session, Depends(get_session)],
):
    documents = session.exec(select(Document)).all()
    return documents
