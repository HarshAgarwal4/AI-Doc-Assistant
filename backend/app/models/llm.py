from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlmodel import Field, SQLModel


class LLMCall(SQLModel , table=True):
    __tablename__ = "llm_calls"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True
    )
    provider: str = Field(index=True)
    model_name: str
    input : str
    output : str
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    time_taken: float
    cost_usd: float = Field(default=0.0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True
    )

class ModelDetails(BaseModel):
    provider : str
    model_name : str
    input_cost_per_million_tokens : float = 0
    output_cost_per_million_tokens : float = 0

class LLMRequestModel(BaseModel):
    question : str
    provider : Literal['Google' , "Groq" , "Mistral" , 'Ollama']