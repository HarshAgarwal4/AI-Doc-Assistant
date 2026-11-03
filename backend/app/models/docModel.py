
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, Column, DateTime, String, text
from sqlmodel import Field, SQLModel


class Document(SQLModel, table=True):
    __tablename__ = "document"

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_document_status",
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
    )

    filename: str
    content_type: str
    size_bytes: int

    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
    )

    status: str = Field(
        default="pending",
        sa_column=Column(
            String(20),
            nullable=False,
            server_default=text("'pending'"),
        ),
    )
