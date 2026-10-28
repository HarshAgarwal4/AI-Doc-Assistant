# 📄 AI Doc Assistant

An intelligent document processing and management backend built with **FastAPI**, **Pydantic v2**, and managed using **uv** for ultra-fast Python package resolution and virtual environment workflows.

---

## 🚀 Features

- **High-Performance API**: Built on [FastAPI](https://fastapi.tiangolo.com/) and asynchronous ASGI with [Uvicorn](https://www.uvicorn.org/).
- **Strict Data Validation**: Document schemas powered by [Pydantic v2](https://docs.pydantic.dev/).
- **Document Ingestion & Validation**:
  - Supported file extensions: `.pdf`, `.txt`, `.docx`.
  - Format verification with `415 Unsupported Media Type` rejection.
  - File size restriction up to **10 MB** (`413 Content Too Large`).
  - Empty file detection (`400 Bad Request`).
  - Metadata tracking: Document ID, filename, size in bytes, and upload timestamp.
- **Document Management**: Endpoints for listing all uploaded documents and retrieving metadata.
- **Interactive Documentation**: Auto-generated interactive API docs via Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **Automated Testing Suite**: Full integration test coverage using `pytest` and `httpx` / `TestClient`.
- **Modern uv Tooling**: Lightning-fast virtual environment, dependency management, and reproducible lockfiles with `uv`.

---

## 📁 Project Structure

```text
AI Doc Assistant/
├── .gitignore               # Workspace-level Git ignore rules
├── README.md                # Project documentation (this file)
└── backend/                 # Backend application directory
    ├── .gitignore           # Backend Git ignore rules
    ├── .python-version      # Target Python version (3.12)
    ├── pyproject.toml       # Project metadata, dependencies, and build config
    ├── uv.lock              # Pinned lockfile for reproducible environments
    ├── main.py              # FastAPI application entrypoint & API routes
    ├── app/
    │   ├── controllers/     # Application logic & controllers
    │   ├── models/          # Data schemas
    │   │   └── DocModel.py  # Pydantic DocumentModel definition
    │   └── routes/          # API route definitions
    ├── Tests/               # Automated test suite
    │   └── tests.py         # Pytest test cases for validation & endpoints
    └── src/                 # Package source tree
        └── backend/
            └── __init__.py
```

---

## 🛠️ Prerequisites

- **Python**: `>= 3.12`
- **uv**: Astral's high-performance Python package and project manager.

### Installing `uv`

If you don't have `uv` installed, install it using the recommended method for your operating system:

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
*Or via winget:*
```powershell
winget install --id=astral-sh.uv -e
```

#### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Via pip (Cross-platform)
```bash
pip install uv
```

Verify the installation:
```bash
uv --version
```

---

## ⚡ Setup & Installation with `uv`

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "AI Doc Assistant"
   ```

2. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

3. **Install dependencies and create virtual environment**:
   `uv` reads `pyproject.toml` and `uv.lock`, automatically provisions the correct Python version if needed, creates a `.venv` virtual environment, and installs all required dependencies:

   ```bash
   # Sync standard runtime dependencies
   uv sync

   # Or sync with development dependencies (e.g., pytest, httpx)
   uv sync --all-groups
   ```

---

## 🏃 Running the Application

Start the FastAPI development server with hot-reload enabled using `uv run`:

```bash
# From within the backend/ folder:
uv run uvicorn main:app --reload --port 8000
```

Once started:
- **API Base URL**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🧪 Running Tests

Run the automated test suite with `pytest` using `uv run`:

```bash
# From within the backend/ folder:
uv run python -m pytest Tests/tests.py -v
```

### Test Coverage Highlights:
- ✅ **API Health Check**: Verifies `/health` returns status `ok`.
- ✅ **Document Upload**: Validates successful upload of supported documents.
- ✅ **Format Enforcement**: Confirms unsupported extensions (e.g., `.png`, `.exe`) return `415`.
- ✅ **File Size Enforcement**: Validates files exceeding 10 MB trigger `413`.
- ✅ **Empty File Rejection**: Confirms 0-byte uploads return `400`.
- ✅ **Document Listing**: Verifies retrieval of all stored documents.

---

## 📡 API Reference

### 1. Root & Health

| Method | Endpoint | Description | Status Code | Response Example |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Root welcome message | `200 OK` | `{"message": "Welcome to AI Doc Assistant"}` |
| `GET` | `/health` | Health check endpoint | `200 OK` | `{"status": "ok"}` |

### 2. Document Endpoints

#### Upload Document
- **URL**: `/documents`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Field**: `file` (Binary file)
- **Allowed Extensions**: `.pdf`, `.txt`, `.docx`
- **Max File Size**: 10 MB

**Response (`201 Created`)**:
```json
{
  "id": 1,
  "filename": "annual_report.pdf",
  "size_bytes": 1048576,
  "uploaded_At": "2026-10-28T14:30:00.000000"
}
```

**Error Responses**:
- `400 Bad Request`: Empty file (`{"detail": "File is empty"}`)
- `413 Content Too Large`: File exceeds 10 MB (`{"detail": "File size must not greater than 10 MB"}`)
- `415 Unsupported Media Type`: File format not allowed (`{"detail": "file format not allowed"}`)

---

#### List Uploaded Documents
- **URL**: `/documents`
- **Method**: `GET`
- **Response (`200 OK`)**:
```json
[
  {
    "id": 1,
    "filename": "annual_report.pdf",
    "size_bytes": 1048576,
    "uploaded_At": "2026-10-28T14:30:00.000000"
  }
]
```

---

## 💻 Example API Calls

### Upload a Document (`curl`)
```bash
curl -X POST "http://localhost:8000/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.txt"
```

### Upload a Document (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/documents" `
  -Method Post `
  -Form @{ file = Get-Item ".\sample.txt" }
```

### Fetch Documents List (`curl`)
```bash
curl -X GET "http://localhost:8000/documents" -H "accept: application/json"
```

---

## 📦 Managing Dependencies with `uv`

Common `uv` commands for project maintenance:

| Action | Command |
| :--- | :--- |
| **Install / Sync all dependencies** | `uv sync --all-groups` |
| **Add a production dependency** | `uv add <package_name>` |
| **Add a development dependency** | `uv add --dev <package_name>` |
| **Remove a dependency** | `uv remove <package_name>` |
| **Update lockfile** | `uv lock --upgrade` |
| **Run commands inside virtualenv** | `uv run <command>` (e.g. `uv run python script.py`) |

---

## 🗺️ Roadmap

- [ ] Persistent database storage (PostgreSQL / SQLite via SQLAlchemy or Tortoise ORM).
- [ ] Text extraction pipeline for PDF, TXT, and DOCX documents.
- [ ] Document chunking and vector embeddings generation.
- [ ] Vector database integration (ChromaDB / Qdrant / Pinecone).
- [ ] Retrieval-Augmented Generation (RAG) Q&A endpoint powered by LLMs.
