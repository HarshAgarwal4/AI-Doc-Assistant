from fastapi import FastAPI, status, HTTPException, File, UploadFile
from app.models.DocModel import DocumentModel
from pathlib import Path
from datetime import datetime, timezone

app = FastAPI(
    title="AI Doc Assistant", description="My first FastAPI backend", version="1.0.0"
)

DOCUMENTS: list[DocumentModel] = []
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = [".pdf", ".txt", ".docx"]
CHUNK_SIZE = 1024 * 1024


def generate_unique_id():
    pass


@app.get("/", status_code=status.HTTP_200_OK)
def home():
    return {"message": "Welcome to AI Doc Assistant"}


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok"}


@app.post(
    "/documents", status_code=status.HTTP_201_CREATED, response_model=DocumentModel
)
async def post_document(file: UploadFile = File(...)):
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
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty",
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

    document = DocumentModel(
        file_name=Path(file.filename).name,
        size_byte=total_size,
        updated_at=datetime.now(timezone.utc),
    )

    DOCUMENTS.append(document)
    return document


@app.get(
    "/documents", response_model=list[DocumentModel], status_code=status.HTTP_200_OK
)
def get_documents():
    return DOCUMENTS
