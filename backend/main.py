from fastapi import FastAPI , status , HTTPException , File , UploadFile
from app.models.DocModel import DocumentModel
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="AI Doc Assistant",
    description="My first FastAPI backend",
    version="1.0.0"
)

documents : list[DocumentModel] = []
MAX_FILE_SIZE = 10*1024*1024
Allowed_formats = ['.pdf' , '.txt' , '.docx']

@app.get("/" , status_code=status.HTTP_200_OK)
def home():
    return {"message": "Welcome to AI Doc Assistant"}


@app.get("/health" , status_code=status.HTTP_200_OK )
def health_check():
    return {"status": "ok"}

@app.post('/documents' , status_code=status.HTTP_201_CREATED , response_model=DocumentModel)
async def postDocument(file : UploadFile = File(...)):
    try:
        # extension = "."+str(str(file.filename).split('.')[1]).lower()     ---> My approach

        extension=Path(file.filename or "").suffix.lower()
        print(extension)
        # Checking correct file format or not
        if extension not in Allowed_formats:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail='file format not allowed',
            )

        # Reading File Content
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='File is empty',
            )

        if (len(content) > MAX_FILE_SIZE):
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail='File size must not greater than 10 MB',
            )

        id=len(documents)+1
        document = DocumentModel(
            id=id,
            filename=Path(file.filename).name,
            size_bytes=len(content),
            uploaded_At=datetime.now()
        )

        documents.append(document)
        return document
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='There is server error',
        )

@app.get('/documents' ,response_model=list[DocumentModel] , status_code=status.HTTP_200_OK)
def get_documents():
    return documents