from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy import Column, ForeignKey, String, text
from sqlmodel import Field, SQLModel


class Feedback(SQLModel , table=True):
    __tablename__ = 'llm_feedback'

    id : UUID = Field(
        primary_key=True,
        default_factory=uuid4
    )
    record_id : UUID = Field(
        sa_column=Column(
            ForeignKey('llm_calls.id' , ondelete="CASCADE"),
            nullable=False,
            index=True
        )
    )
    rating : Literal['Thumbs_up' , 'Thumbs_down'] = Field(
        sa_column=Column(
            String(20),
            nullable=False,
            index=True,
            server_default=text("'Thumbs_up'")
        )
    )

    comment : str | None = None

class FeedbackRequestModel(BaseModel):
    id : UUID
    comment : str | None = None
    rating : Literal['Thumbs_up' , 'Thumbs_down'] = Field(
        default='Thumbs_up'
    )

class FeedbackResponseModel(BaseModel):
    id: UUID
    msg : str