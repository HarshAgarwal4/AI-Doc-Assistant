from collections.abc import Generator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models.feedback import Feedback
from app.models.llm import LLMCall
from database.database import get_session
from main import app

# In-memory SQLite database for testing (no external API calls or real database)
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@pytest.fixture
def test_session() -> Generator[Session, None, None]:
    SQLModel.metadata.drop_all(test_engine)
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture
def client(
    test_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    def override_get_session():
        yield test_session

    app.dependency_overrides[get_session] = override_get_session
    monkeypatch.setattr("main.engine", test_engine)
    test_client = TestClient(app)
    yield test_client
    test_client.close()
    app.dependency_overrides.clear()


def create_sample_llm_call(session: Session) -> LLMCall:
    call = LLMCall(
        id=uuid4(),
        provider="Groq",
        model_name="llama-3.3-70b-versatile",
        input="What are the key findings?",
        output="The key findings are revenue grew 27.5%.",
        input_tokens=100,
        output_tokens=50,
        time_taken=1.23,
        cost_usd=0.0015,
    )
    session.add(call)
    session.commit()
    session.refresh(call)
    return call


# 1. Test POST /feedback with Thumbs_up and comment
def test_post_feedback_thumbs_up_success(client: TestClient, test_session: Session):
    llm_call = create_sample_llm_call(test_session)

    payload = {
        "id": str(llm_call.id),
        "rating": "Thumbs_up",
        "comment": "Accurate and well cited.",
    }

    response = client.post("/feedback", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["msg"] == "Feedback saved successfully"

    # Verify saved directly in database
    saved_feedback = test_session.exec(select(Feedback)).first()
    assert saved_feedback is not None
    assert saved_feedback.record_id == llm_call.id
    assert saved_feedback.rating == "Thumbs_up"
    assert saved_feedback.comment == "Accurate and well cited."


# 2. Test POST /feedback with Thumbs_down
def test_post_feedback_thumbs_down_success(client: TestClient, test_session: Session):
    llm_call = create_sample_llm_call(test_session)

    payload = {
        "id": str(llm_call.id),
        "rating": "Thumbs_down",
        "comment": "Missed page citation.",
    }

    response = client.post("/feedback", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["msg"] == "Feedback saved successfully"

    saved = test_session.exec(select(Feedback)).first()
    assert saved is not None
    assert saved.record_id == llm_call.id
    assert saved.rating == "Thumbs_down"
    assert saved.comment == "Missed page citation."


# 3. Test POST /feedback without comment (optional field)
def test_post_feedback_without_comment(client: TestClient, test_session: Session):
    llm_call = create_sample_llm_call(test_session)

    payload = {
        "id": str(llm_call.id),
        "rating": "Thumbs_up",
    }

    response = client.post("/feedback", json=payload)

    assert response.status_code == 201
    saved = test_session.exec(select(Feedback)).first()
    assert saved is not None
    assert saved.comment is None
    assert saved.rating == "Thumbs_up"


# 4. Test POST /feedback with default rating when omitted
def test_post_feedback_default_rating(client: TestClient, test_session: Session):
    llm_call = create_sample_llm_call(test_session)

    payload = {
        "id": str(llm_call.id),
    }

    response = client.post("/feedback", json=payload)

    assert response.status_code == 201
    saved = test_session.exec(select(Feedback)).first()
    assert saved is not None
    assert saved.rating == "Thumbs_up"


# 5. Test POST /feedback with invalid rating literal (validation failure)
def test_post_feedback_invalid_rating(client: TestClient, test_session: Session):
    llm_call = create_sample_llm_call(test_session)

    payload = {
        "id": str(llm_call.id),
        "rating": "neutral",
    }

    response = client.post("/feedback", json=payload)
    assert response.status_code == 422


# 6. Test POST /feedback with missing required id
def test_post_feedback_missing_id(client: TestClient):
    payload = {
        "rating": "Thumbs_up",
        "comment": "No ID provided",
    }

    response = client.post("/feedback", json=payload)
    assert response.status_code == 422


# 7. Test POST /feedback with malformed UUID
def test_post_feedback_malformed_uuid(client: TestClient):
    payload = {
        "id": "not-a-valid-uuid",
        "rating": "Thumbs_up",
    }

    response = client.post("/feedback", json=payload)
    assert response.status_code == 422


# 8. Test GET /feedback when database is empty
def test_get_feedback_empty(client: TestClient):
    response = client.get("/feedback")

    assert response.status_code == 200
    assert response.json() == []


# 9. Test GET /feedback and /feedbacks with populated feedback
def test_get_feedback_populated(client: TestClient, test_session: Session):
    call1 = create_sample_llm_call(test_session)
    call2 = create_sample_llm_call(test_session)

    fb1 = Feedback(record_id=call1.id, rating="Thumbs_up", comment="Very helpful")
    fb2 = Feedback(record_id=call2.id, rating="Thumbs_down", comment="Incorrect number")
    test_session.add(fb1)
    test_session.add(fb2)
    test_session.commit()

    # Test /feedback
    response1 = client.get("/feedback")
    assert response1.status_code == 200
    feedbacks = response1.json()
    assert len(feedbacks) == 2
    assert {f["rating"] for f in feedbacks} == {"Thumbs_up", "Thumbs_down"}
    assert {f["comment"] for f in feedbacks} == {"Very helpful", "Incorrect number"}

    # Test alias /feedbacks
    response2 = client.get("/feedbacks")
    assert response2.status_code == 200
    assert len(response2.json()) == 2


# 10. Test multiple feedback entries for different LLM calls
def test_multiple_feedbacks_independent(client: TestClient, test_session: Session):
    call1 = create_sample_llm_call(test_session)
    call2 = create_sample_llm_call(test_session)

    res1 = client.post("/feedback", json={"id": str(call1.id), "rating": "Thumbs_up", "comment": "Good"})
    res2 = client.post("/feedback", json={"id": str(call2.id), "rating": "Thumbs_down", "comment": "Bad"})

    assert res1.status_code == 201
    assert res2.status_code == 201

    all_fb = client.get("/feedback").json()
    assert len(all_fb) == 2
    ratings = {fb["rating"] for fb in all_fb}
    assert ratings == {"Thumbs_up", "Thumbs_down"}
