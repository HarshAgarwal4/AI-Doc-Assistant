from pydantic import BaseModel
from datetime import datetime

class DocumentModel(BaseModel):
    id : int
    filename : str
    size_bytes : int
    uploaded_At : datetime