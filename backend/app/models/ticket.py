from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import Column, ForeignKey, String, text
from sqlmodel import Field, SQLModel


class TicketModel(SQLModel, table=True):

    __tablename__ = "tickets"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True
    )

    resp_id : UUID = Field(
        sa_column=Column(
            ForeignKey('llm_calls.id' , ondelete='CASCADE'),
            nullable=False,
            index=True
        )
    )

    query: str = Field(
        nullable=False
    )

    description: str = Field(
        nullable=False
    )

    status: Literal["pending", "resolved", "closed"] = Field(
        default="pending",
        sa_column=Column(
            String(20),
            nullable=False,
            server_default=text("'pending'"),
            index=True
        )
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False
    )
