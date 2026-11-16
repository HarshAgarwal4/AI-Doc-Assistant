from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, ForeignKey
from sqlmodel import Field, SQLModel


class Chunk(SQLModel, table=True):
    __tablename__ = "chunk"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
    )

    document_id: UUID = Field(
        sa_column=Column(
            ForeignKey("document.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )

    text: str = Field(nullable=False)

    filename: str = Field(nullable=False)

    page_number: int = Field(nullable=False)

    embedding: list[float] = Field(
        sa_column=Column(Vector(768), nullable=False)
    )

    def __init__(self, **data):
        if "file_name" in data and "filename" not in data:
            data["filename"] = data.pop("file_name")
        super().__init__(**data)

    @property
    def file_name(self) -> str:
        return self.filename
