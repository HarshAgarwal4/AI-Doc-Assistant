
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from main import app
from database.database import get_session
from app.models.docModel import Document


# In-memory SQLite database for testing
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


# Fixture: Create a fresh database for every test
@pytest.fixture
def test_session() -> Generator[Session, None, None]:
    SQLModel.metadata.drop_all(test_engine)
    SQLModel.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        yield session

    SQLModel.metadata.drop_all(test_engine)


# Fixture: Override PostgreSQL session with SQLite
@pytest.fixture
def client(test_session: Session) -> Generator[TestClient, None, None]:

    def override_get_session():
        yield test_session

    app.dependency_overrides[get_session] = override_get_session

    # Avoid running PostgreSQL lifespan during tests
    test_client = TestClient(app)

    yield test_client

    test_client.close()
    app.dependency_overrides.clear()


# 1. Test home endpoint
def test_home(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to AI Doc Assistant"
    }


# 2. Test health endpoint
def test_health(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# 3. Test valid document upload
def test_document_upload(client: TestClient):
    response = client.post(
        "/documents",
        files={
            "file": (
                "hello.txt",
                b"Hello world",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["id"] is not None
    assert data["filename"] == "hello.txt"
    assert data["content_type"] == "text/plain"
    assert data["size_bytes"] == 11
    assert data["uploaded_at"] is not None

    document = Document.model_validate(data)
    assert document.filename == "hello.txt"


# 4. Test unsupported file format
def test_wrong_file_format(client: TestClient):
    response = client.post(
        "/documents",
        files={
            "file": (
                "image.png",
                b"image data",
                "image/png",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "file format not allowed"


# 5. Test file greater than 10 MB
def test_greater_file_size(client: TestClient):
    response = client.post(
        "/documents",
        files={
            "file": (
                "large.txt",
                b"x" * (10 * 1024 * 1024 + 1),
                "text/plain",
            )
        },
    )

    assert response.status_code == 413


# 6. Test empty file upload
def test_empty_file(client: TestClient):
    response = client.post(
        "/documents",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


# 7. Test fetching documents from empty database
def test_list_documents_empty(client: TestClient):
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


# 8. Test uploading multiple documents
def test_multiple_documents(client: TestClient):

    documents = [
        ("first.txt", b"First document"),
        ("second.txt", b"Second document"),
    ]

    for filename, content in documents:
        response = client.post(
            "/documents",
            files={
                "file": (
                    filename,
                    content,
                    "text/plain",
                )
            },
        )

        assert response.status_code == 201

    response = client.get("/documents")

    assert response.status_code == 200

    result = response.json()

    assert len(result) == 2

    assert {doc["filename"] for doc in result} == {
        "first.txt",
        "second.txt",
    }
