
from collections.abc import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine
from main1 import app
from Database.database import get_session


# In-memory SQLite database used only for tests
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

@pytest.fixture
def test_session() -> Generator[Session, None, None]:
    # Start every test with a clean database
    SQLModel.metadata.drop_all(test_engine)
    SQLModel.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        yield session

    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture
def client(test_session: Session) -> Generator[TestClient, None, None]:
    # Replace the PostgreSQL session with the SQLite session
    def override_get_session():
        yield test_session

    app.dependency_overrides[get_session] = override_get_session

    # Do not enter TestClient as a context manager here,
    # because that would run the app's lifespan startup.
    test_client = TestClient(app)

    yield test_client

    test_client.close()
    app.dependency_overrides.clear()


def test_health(client: TestClient):
    """Test the health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home(client: TestClient):
    """Test the home endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to AI Doc Assistant"
    }


def test_document_upload(client: TestClient):
    """Test uploading a valid text document."""
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
    assert data["size_bytes"] == len(b"Hello world")


def test_wrong_file_format(client: TestClient):
    """Test that unsupported file formats are rejected."""
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


def test_greater_file_size(client: TestClient):
    """Test that files larger than 10 MB are rejected."""
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


def test_empty_file(client: TestClient):
    """Test that empty files are rejected."""
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


def test_list_documents(client: TestClient):
    """Test listing documents from an empty database."""
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_multiple_documents(client: TestClient):
    """Test uploading and listing multiple documents."""
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
