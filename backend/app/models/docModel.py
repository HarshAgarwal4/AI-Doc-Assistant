from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
