from uuid import UUID

from langchain.tools import tool
from sqlmodel import Session, select

import database.database as db
from app.models.ticket import TicketModel


def save_ticket(resp_id: str | UUID, query: str, description: str) -> TicketModel:
    if isinstance(resp_id, str):
        resp_id = UUID(resp_id)
    with Session(db.engine) as session:
        existing = session.exec(select(TicketModel).where(TicketModel.resp_id == resp_id)).first()
        if existing:
            return existing
        ticket = TicketModel(
            resp_id=resp_id,
            query=query,
            description=description,
        )
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return ticket


@tool
def create_support_ticket(resp_id: str, question: str, description: str) -> str:
    """Create a support ticket when the answer is not found in the documents. Returns the ticket ID."""
    ticket = save_ticket(resp_id=resp_id, query=question, description=description)
    return str(ticket.id)
