from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from app.models.resumeModel import ResumeData
from main import app


def make_pdf_bytes(text: str = "") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_docx_bytes(paragraphs: list[str]) -> bytes:
    doc = DocxDocument()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def make_mock_llm(side_effect=None, return_value=None):
    mock_structured = MagicMock()
    if side_effect is not None:
        mock_structured.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        mock_structured.ainvoke = AsyncMock(return_value=return_value)

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm, mock_structured


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_resume_data() -> ResumeData:
    return ResumeData(
        name="Aarav Mehta",
        email="aarav.mehta@example.com",
        phone="91 9876543210",
        skills=["Python", "FastAPI", "PostgreSQL"],
    )


def test_extract_pdf_success(client: TestClient, sample_resume_data: ResumeData):
    pdf_bytes = make_pdf_bytes(
        "Aarav Mehta\naarav.mehta@example.com\n91 9876543210\nSkills: Python, FastAPI, PostgreSQL"
    )
    mock_llm, mock_structured = make_mock_llm(return_value=sample_resume_data)

    with patch("utils.resume.llm", mock_llm):
        response = client.post(
            "/extract",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "name": "Aarav Mehta",
        "email": "aarav.mehta@example.com",
        "phone": "91 9876543210",
        "skills": ["Python", "FastAPI", "PostgreSQL"],
    }
    mock_llm.with_structured_output.assert_called_once_with(ResumeData)
    mock_structured.ainvoke.assert_awaited_once()
    sent_messages = mock_structured.ainvoke.call_args.args[0]
    assert "Aarav Mehta" in sent_messages[1].content


def test_extract_docx_success(client: TestClient, sample_resume_data: ResumeData):
    docx_bytes = make_docx_bytes(
        [
            "Aarav Mehta",
            "Email: aarav.mehta@example.com",
            "Phone: 91 9876543210",
            "Skills: Python, FastAPI, PostgreSQL",
        ]
    )
    mock_llm, mock_structured = make_mock_llm(return_value=sample_resume_data)

    with patch("utils.resume.llm", mock_llm):
        response = client.post(
            "/extract",
            files={
                "file": (
                    "resume.docx",
                    docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    assert response.status_code == 200
    assert response.json()["name"] == "Aarav Mehta"
    assert response.json()["skills"] == ["Python", "FastAPI", "PostgreSQL"]
    mock_structured.ainvoke.assert_awaited_once()
    sent_messages = mock_structured.ainvoke.call_args.args[0]
    assert "aarav.mehta@example.com" in sent_messages[1].content


def test_extract_txt_success(client: TestClient, sample_resume_data: ResumeData):
    txt_bytes = b"Name: Aarav Mehta\nEmail: aarav.mehta@example.com\nSkills: Python, FastAPI"
    mock_llm, mock_structured = make_mock_llm(return_value=sample_resume_data)

    with patch("utils.resume.llm", mock_llm):
        response = client.post(
            "/extract",
            files={"file": ("resume.txt", txt_bytes, "text/plain")},
        )

    assert response.status_code == 200
    assert response.json() == sample_resume_data.model_dump()
    mock_structured.ainvoke.assert_awaited_once()


def test_extract_nullable_fields_and_empty_skills(client: TestClient):
    pdf_bytes = make_pdf_bytes("Minimal resume with no contact details")
    minimal_data = ResumeData(name=None, email=None, phone=None, skills=[])
    mock_llm, _ = make_mock_llm(return_value=minimal_data)

    with patch("utils.resume.llm", mock_llm):
        response = client.post(
            "/extract",
            files={"file": ("minimal.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
    }


def test_extract_retries_on_validation_error_and_succeeds(
    client: TestClient,
    sample_resume_data: ResumeData,
):
    pdf_bytes = make_pdf_bytes("Aarav Mehta - Python Developer")
    invalid_dict = {
        "name": "Aarav Mehta",
        "email": "not-a-valid-email",
        "phone": "91 9876543210",
        "skills": ["Python"],
    }
    mock_llm, mock_structured = make_mock_llm(
        side_effect=[invalid_dict, sample_resume_data]
    )

    with patch("utils.resume.llm", mock_llm):
        response = client.post(
            "/extract",
            files={"file": ("retry.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == sample_resume_data.model_dump()
    assert mock_structured.ainvoke.await_count == 2


def test_extract_fails_after_exceeding_validation_retries(client: TestClient):
    pdf_bytes = make_pdf_bytes("Aarav Mehta - Invalid output")
    invalid_dict = {
        "name": "Aarav Mehta",
        "email": "still-invalid-email",
        "phone": None,
        "skills": [],
    }
    mock_llm, mock_structured = make_mock_llm(
        side_effect=[invalid_dict, invalid_dict]
    )

    with patch("utils.resume.llm", mock_llm):
        response = client.post(
            "/extract",
            files={"file": ("bad_output.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 500
    assert "Error in Parsing Resume" in response.json()["detail"]
    assert mock_structured.ainvoke.await_count == 2


def test_extract_llm_exception_returns_500(client: TestClient):
    pdf_bytes = make_pdf_bytes("Aarav Mehta")
    mock_llm, _ = make_mock_llm(
        side_effect=RuntimeError("Mock LLM failure")
    )

    with patch("utils.resume.llm", mock_llm):
        response = client.post(
            "/extract",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 500
    assert "Mock LLM failure" in response.json()["detail"]


def test_extract_unsupported_file_format():
    non_raising_client = TestClient(app, raise_server_exceptions=False)
    mock_llm, mock_structured = make_mock_llm()

    with patch("utils.resume.llm", mock_llm):
        response = non_raising_client.post(
            "/extract",
            files={"file": ("resume.png", b"fake-image-bytes", "image/png")},
        )

    assert response.status_code == 500
    mock_structured.ainvoke.assert_not_awaited()


def test_extract_empty_pdf_does_not_call_llm():
    empty_pdf = make_pdf_bytes("")
    non_raising_client = TestClient(app, raise_server_exceptions=False)
    mock_llm, mock_structured = make_mock_llm()

    with patch("utils.resume.llm", mock_llm):
        response = non_raising_client.post(
            "/extract",
            files={"file": ("empty.pdf", empty_pdf, "application/pdf")},
        )

    assert response.status_code in (400, 500)
    mock_structured.ainvoke.assert_not_awaited()


def test_extract_missing_file_returns_422(client: TestClient):
    response = client.post("/extract")
    assert response.status_code == 422
