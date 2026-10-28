from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class DocumentModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    size_byte: int
    updated_at: datetime