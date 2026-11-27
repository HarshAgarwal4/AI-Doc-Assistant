import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models.docModel import Document  # noqa: F401
from app.models.llm import LLMCall
from app.models.ticket import TicketModel
from database.database import get_session
from main import app
from tools.ticket import create_support_ticket, save_ticket
from utils.llm import generate

test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@pytest.fixture(autouse=True)
def setup_db(monkeypatch: pytest.MonkeyPatch):
    SQLModel.metadata.drop_all(test_engine)
    SQLModel.metadata.create_all(test_engine)
    monkeypatch.setattr("database.database.engine", test_engine)
    monkeypatch.setattr("utils.llm.db.engine", test_engine)
    monkeypatch.setattr("tools.ticket.db.engine", test_engine)
    monkeypatch.setattr("main.engine", test_engine)
    with Session(test_engine) as session:
        yield session
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture
def client(
    setup_db: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    def override_get_session():
        yield setup_db

    app.dependency_overrides[get_session] = override_get_session
    monkeypatch.setattr("main.engine", test_engine)
    test_client = TestClient(app)
    yield test_client
    test_client.close()
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def mock_search():
    with patch("utils.llm.search", new_callable=AsyncMock) as m:
        m.return_value = [
            {"text": "Python was created by Guido van Rossum.", "filename": "doc.pdf", "page": 1, "score": 0.9}
        ]
        yield m


@pytest.fixture(autouse=True)
def mock_classifier():
    with patch("utils.llm.detect_with_both", new_callable=AsyncMock) as m:
        m.return_value = {
            "query": "test",
            "svm": "benign",
            "naive_bayes": "benign",
            "final_prediction": 0,
            "final_label": "benign",
            "both_agree": True,
        }
        yield m


class MockChunk(AIMessageChunk):
    def __init__(self, content: str, input_tokens: int = 10, output_tokens: int = 5):
        super().__init__(content=content)
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}


class MockAgent:
    def __init__(self, chunks: list[MockChunk]):
        self.chunks = chunks

    async def astream(self, input_dict, stream_mode="messages"):
        for c in self.chunks:
            yield c, {}


def test_answer_found_workflow(setup_db: Session):
    session = setup_db
    mock_agent = MockAgent([MockChunk("Python was created by Guido van Rossum.")])
    with patch("utils.llm.create_agent", return_value=mock_agent):
        async def run():
            events = []
            async for ev in generate(question="Who created Python?", provider="Ollama"):
                events.append(ev)
            return "".join(events)

        output = asyncio.run(run())

    assert "event: token" in output
    assert "event: saved" in output

    calls = session.exec(select(LLMCall)).all()
    assert len(calls) == 1
    llm_call = calls[0]
    assert llm_call.input == "Who created Python?"
    assert llm_call.output == "Python was created by Guido van Rossum."
    assert llm_call.time_taken is not None

    # No ticket created when answer is found
    tickets = session.exec(select(TicketModel)).all()
    assert len(tickets) == 0


def test_answer_not_found_creates_ticket(setup_db: Session):
    session = setup_db
    mock_agent = MockAgent([MockChunk("I don't know.")])
    with patch("utils.llm.create_agent", return_value=mock_agent):
        async def run():
            events = []
            async for ev in generate(question="What is the quantum state of X?", provider="Ollama"):
                events.append(ev)
            return "".join(events)

        output = asyncio.run(run())

    assert "event: token" in output
    assert "event: saved" in output
    assert "ticket has been generated with id" in output

    calls = session.exec(select(LLMCall)).all()
    assert len(calls) == 1
    llm_call = calls[0]
    assert "ticket has been generated with id" in llm_call.output

    # Ticket created with same llm_call_id
    tickets = session.exec(select(TicketModel)).all()
    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.resp_id == llm_call.id
    assert "What is the quantum state of X?" in ticket.query


def test_tool_execution(setup_db: Session):
    session = setup_db
    # Create an initial LLMCall record
    doc = LLMCall(
        provider="Ollama",
        model_name="llama3.2:latest",
        input="Sample question",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # Invoke tool directly with resp_id
    result = asyncio.run(create_support_ticket.ainvoke({
        "resp_id": str(doc.id),
        "question": "Sample question",
        "description": "Need human support",
    }))
    assert isinstance(result, str)
    assert len(result) > 0

    tickets = session.exec(select(TicketModel)).all()
    assert len(tickets) == 1
    assert tickets[0].resp_id == doc.id
    assert tickets[0].query == "Sample question"
    assert tickets[0].description == "Need human support"
    assert tickets[0].status == "pending"


def test_save_ticket_idempotency(setup_db: Session):
    session = setup_db
    call_id = uuid4()
    t1 = save_ticket(call_id, "Query 1", "Desc 1")
    t2 = save_ticket(call_id, "Query 2", "Desc 2")

    assert t1.id == t2.id
    tickets = session.exec(select(TicketModel)).all()
    assert len(tickets) == 1


def test_jailbreak_creates_ticket(setup_db: Session, mock_classifier):
    session = setup_db
    mock_classifier.return_value = {
        "query": "malicious injection",
        "svm": "jailbreak",
        "naive_bayes": "jailbreak",
        "final_prediction": 1,
        "final_label": "jailbreak",
        "both_agree": True,
    }

    async def run():
        events = []
        async for ev in generate(question="drop tables", provider="Ollama"):
            events.append(ev)
        return "".join(events)

    output = asyncio.run(run())
    assert "event: token" in output
    assert "ticket has been generated with id" in output
    assert "event: saved" in output

    calls = session.exec(select(LLMCall)).all()
    assert len(calls) == 1
    assert "ticket has been generated with id" in calls[0].output

    tickets = session.exec(select(TicketModel)).all()
    assert len(tickets) == 1
    assert tickets[0].resp_id == calls[0].id


# ==========================================
# Tests for GET /ticket and GET /tickets
# ==========================================

def test_get_ticket_endpoint_empty(client: TestClient):
    response = client.get("/ticket")
    assert response.status_code == 200
    assert response.json() == []


def test_get_ticket_endpoint_populated(client: TestClient, setup_db: Session):
    session = setup_db
    call1 = LLMCall(provider="Ollama", model_name="llama3.2:latest", input="Query 1")
    call2 = LLMCall(provider="Groq", model_name="llama-3.3-70b-versatile", input="Query 2")
    session.add(call1)
    session.add(call2)
    session.commit()
    session.refresh(call1)
    session.refresh(call2)

    t1 = TicketModel(resp_id=call1.id, query="Query 1", description="Reason 1", status="pending")
    t2 = TicketModel(resp_id=call2.id, query="Query 2", description="Reason 2", status="resolved")
    session.add(t1)
    session.add(t2)
    session.commit()

    response = client.get("/ticket")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    queries = {item["query"] for item in data}
    assert queries == {"Query 1", "Query 2"}

    statuses = {item["status"] for item in data}
    assert statuses == {"pending", "resolved"}

    # Verify all expected schema fields are present
    first = data[0]
    assert "id" in first
    assert "resp_id" in first
    assert "query" in first
    assert "description" in first
    assert "status" in first
    assert "created_at" in first
    assert "updated_at" in first


def test_get_tickets_alias_endpoint(client: TestClient, setup_db: Session):
    session = setup_db
    call = LLMCall(provider="Google", model_name="gemini-3.5-flash-lite", input="Query Alias")
    session.add(call)
    session.commit()
    session.refresh(call)

    t = TicketModel(resp_id=call.id, query="Query Alias", description="Alias check", status="pending")
    session.add(t)
    session.commit()

    res_single = client.get("/ticket")
    res_plural = client.get("/tickets")

    assert res_single.status_code == 200
    assert res_plural.status_code == 200
    assert res_single.json() == res_plural.json()
    assert len(res_plural.json()) == 1


def test_ticket_status_types(client: TestClient, setup_db: Session):
    session = setup_db
    call = LLMCall(provider="Ollama", model_name="llama3.2", input="Status test")
    session.add(call)
    session.commit()
    session.refresh(call)

    for st in ["pending", "resolved", "closed"]:
        session.add(TicketModel(resp_id=call.id, query=f"Query {st}", description=f"Desc {st}", status=st))
    session.commit()

    response = client.get("/ticket")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    found_statuses = {item["status"] for item in data}
    assert found_statuses == {"pending", "resolved", "closed"}
