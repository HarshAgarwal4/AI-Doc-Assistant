from sqlmodel import SQLModel, Field
from datetime import datetime, timezone

class Document(SQLModel , table=True):
    id:int | None = Field(default=None , primary_key=True)
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )