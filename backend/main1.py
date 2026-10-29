from fastapi import FastAPI, status, HTTPException, File, UploadFile , Depends
from pathlib import Path
from datetime import datetime, timezone
from Database.database import create_db_and_tables , get_session
from sqlmodel import select , Session
from Database.DocModal import Document
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(
    title="AI Doc Assistant", description="My first FastAPI backend with Postgres", lifespan=lifespan , version="1.0.0"
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
    "/documents", status_code=status.HTTP_201_CREATED, response_model=Document
)
async def post_document(file: UploadFile = File(...) , session: Session = Depends(get_session)):
    extension = Path(file.filename or "").suffix.lower()
    total_size = 0
    # Checking correct file format or not
    if extension not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="file format not allowed",
        )

    if file.size is not None and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File size must not be greater than 10 MB",
        )

    # Reading File Content chunk by chunk because it is not necessary file size available always
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total_size = total_size + len(chunk)
        if total_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, detail="File size must not exceed 10 MB"
            )

    if total_size == 0:
        raise HTTPException(status_code=400, detail="Empty File Not Allowed")

    try:
        document = Document(
            filename = Path(file.filename or "").name,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=total_size
        )

        session.add(document)
        session.commit()
        session.refresh(document)

        return document
    except Exception as e:
        session.rollback()
        raise
    finally:
        await file.close()

@app.get(
    "/documents", response_model=list[Document], status_code=status.HTTP_200_OK
)
def get_documents(session : Session = Depends(get_session)):
    documents = session.exec(select(Document)).all()
    return documents