from fastapi.testclient import TestClient
from app.models.DocModel import DocumentModel
from main import app, DOCUMENTS
import pytest

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_documents():
    """Clear all documents"""
    DOCUMENTS.clear()


def test_health():
    """Testing health of api status"""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_Doc_upload():
    """Testing document upload"""

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
    assert DocumentModel.model_validate(response.json())


def test_Wrong_file_format():
    """Testing it handle proper file formats or not"""

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


def test_Greater_file_size():
    """Testing what happen when we upload file greater than 10 mb"""

    response = client.post(
        "/documents",
        files={
            "file": (
                "hello.txt",
                b"x" * (10 * 1024 * 1024 + 1),
                "text/plain",
            )
        },
    )

    assert response.status_code == 413


def test_Empty_File():
    """Testing what happen when we upload a completely empty file"""

    response = client.post(
        "/documents",
        files={
            "file": (
                "hello.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_list_docs():
    """Testing fetching docs list"""
    response = client.get("/documents")
    assert response.status_code == 200
