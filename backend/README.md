# AI Doc Assistant

A high-performance document management API built with **FastAPI, Pydantic v2, SQLModel, PostgreSQL, and Docker**. The application supports document validation, metadata persistence, and automated testing, with a reproducible development environment powered by `uv`.

## 🚀 Features

- **High-Performance API:** Built with FastAPI and Uvicorn.
- **Strict Data Validation:** Request and response validation using Pydantic v2.
- **Document Ingestion & Validation:**
  - Supported extensions: `.pdf`, `.txt`, `.docx`.
  - Unsupported formats return `415 Unsupported Media Type`.
  - Maximum file size: 10 MB (`413 Content Too Large`).
  - Empty files return `400 Bad Request`.
- **Persistent Metadata Storage:** SQLModel and PostgreSQL store document metadata.
- **Document Metadata:** Tracks ID, filename, content type, file size in bytes, and upload timestamp.
- **Document Management:** Endpoints for uploading documents, listing all documents, and retrieving metadata.
- **Interactive API Documentation:** Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **Docker Support:** Run the application and PostgreSQL using a single Docker Compose command.
- **Persistent Database Volume:** Document metadata survives container recreation and application restarts.
- **Environment-Based Configuration:** Database credentials and connection settings are supplied through `.env`.
- **Automated Testing:** Pytest and HTTPX/TestClient, with SQLite available for database-independent tests.
- **Modern Python Tooling:** Dependency management and reproducible lockfiles using `uv`.

## 📁 Project Structure

```text
AI Doc Assistant/
├── README.md
└── backend/
    ├── .gitignore
    ├── .python-version
    ├── .env
    ├── .env.example
    ├── .dockerignore
    ├── Dockerfile
    ├── docker-compose.yml
    ├── pyproject.toml
    ├── uv.lock
    ├── main.py
    ├── app/
    │   ├── controllers/
    │   ├── models/
    │   │   ├── __init__.py
    │   │   └── document.py
    │   └── routes/
    ├── Database/
    │   ├── __init__.py
    │   └── database.py
    ├── Tests/
    │   └── tests.py
    └── src/
        └── backend/
            └── __init__.py
```

The structure above represents the intended layout. Keep your existing filenames if they differ, and adjust the imports and Dockerfile paths accordingly.

## 🛠️ Prerequisites

### For local development

- Python 3.12 or later, compatible with the project configuration.
- `uv`.
- PostgreSQL if you intend to run the application locally against a local PostgreSQL server.

### For Docker

- Docker Desktop on Windows or macOS, or Docker Engine with the Docker Compose plugin on Linux.
- No local PostgreSQL installation is required when using the PostgreSQL Docker container.

### Installing uv

**Windows (PowerShell)**

```powershell
winget install --id=astral-sh.uv -e
```

Alternatively:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the installation:

```bash
uv --version
```

## ⚡ Setup & Installation

Clone the repository:

```bash
git clone <repository-url>
cd "AI Doc Assistant/backend"
```

Install the project dependencies:

```bash
uv sync --all-groups
```

This creates or updates the virtual environment and installs the dependencies defined in `pyproject.toml` and `uv.lock`.

## 🔐 Environment Configuration

Create a `.env` file inside the `backend/` directory.

Example development configuration:

```dotenv
POSTGRES_USER=postgres
POSTGRES_PASSWORD=replace_with_a_secure_password
POSTGRES_DB=ai_doc_assistant

DATABASE_URL=postgresql+psycopg://postgres:replace_with_a_secure_password@localhost:5432/ai_doc_assistant
```

Replace the example password with your own local development password.

### Environment variables

| Variable | Description |
|---|---|
| `POSTGRES_USER` | PostgreSQL user created during database initialization |
| `POSTGRES_PASSWORD` | Password used to initialize the PostgreSQL user |
| `POSTGRES_DB` | Database created during initialization |
| `DATABASE_URL` | SQLAlchemy-compatible connection URL used by the application |

### `.env.example`

Commit a `.env.example` file containing placeholders, not real credentials:

```dotenv
POSTGRES_USER=postgres
POSTGRES_PASSWORD=replace_me
POSTGRES_DB=ai_doc_assistant

DATABASE_URL=postgresql+psycopg://postgres:replace_me@db:5432/ai_doc_assistant
```

**Important:** The `.env` file is for local development and must not be committed to Git. Add `.env` to `.gitignore`. Docker Compose reads `.env` for variable substitution, while the application service receives its environment configuration through Compose.

## 🐳 Running with Docker

Docker Compose runs two services:

- **app:** The FastAPI application.
- **db:** PostgreSQL 16.

The database uses a named Docker volume so its data persists when containers are stopped and recreated.

### 1. Verify Docker

Start Docker Desktop and wait for the Docker Engine to become available.

```bash
docker --version
docker compose version
docker info
```

### 2. Validate the Compose configuration

From the `backend/` directory:

```bash
docker compose config
```

Check that the resolved PostgreSQL variables are populated and the application's database URL uses the hostname `db`.

### 3. Build and start the services

```bash
docker compose up --build
```

Docker downloads the PostgreSQL image if needed, builds the application image, creates the network and persistent volume, and starts the services.

Once the application is running:

| Resource | URL |
|---|---|
| API base URL | http://localhost:8000 |
| Health check | http://localhost:8000/health |
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| List documents | http://localhost:8000/documents |

The PostgreSQL service is accessible to the application through the internal hostname `db`. You do not need to expose PostgreSQL's port on your host unless you want to connect to it from a local database client.

### 4. Run in the background

```bash
docker compose up --build -d
```

View running containers:

```bash
docker compose ps
```

View application and database logs:

```bash
docker compose logs -f app
docker compose logs -f db
```

Stop the services:

```bash
docker compose down
```

The named database volume is preserved by this command.

### 5. Verify persistent storage

To verify that document metadata survives container recreation:

1. Start the application using `docker compose up --build`.
2. Open Swagger UI at `/docs`.
3. Upload a valid document using `POST /documents`.
4. Call `GET /documents` and verify that the document appears.
5. Stop the services using `docker compose down`.
6. Start them again using `docker compose up --build`.
7. Call `GET /documents` again.

Previously stored records should still be present because the PostgreSQL database files are stored in the named volume.

**Warning:** The command below removes the Compose-managed volumes and permanently deletes the database data stored in them:

```bash
docker compose down -v
```

Use it only when you intentionally want to reset the development database.

## 🖥️ Running Locally Without Docker

For local development, the application can run directly through `uv`. A PostgreSQL server must be running and accessible using the local connection URL.

Configure `DATABASE_URL` in `.env` to use `localhost`:

```dotenv
DATABASE_URL=postgresql+psycopg://postgres:your_local_password@localhost:5432/ai_doc_assistant
```

Start the API:

```bash
uv run uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs.

**Connection URL distinction:**

- Local application: `localhost:5432`.
- Application inside Docker Compose: `db:5432`.

These hostnames refer to different network environments. The Compose configuration should override the application's `DATABASE_URL` so the container connects to the `db` service.

## 🧪 Running Tests

Run the test suite from the `backend/` directory:

```bash
uv run python -m pytest Tests/ -v
```

To run a particular test file:

```bash
uv run python -m pytest Tests/tests.py -v
```

Tests should be able to use SQLite where appropriate, without requiring a PostgreSQL server to be running.

### Test coverage

- API health check.
- Successful document uploads.
- Unsupported file format rejection.
- Maximum file size enforcement.
- Empty file rejection.
- Document listing.
- Document metadata persistence.
- Database model and CRUD behavior.

For database-dependent tests, configure a separate SQLite engine or override the database dependency so tests do not modify your development PostgreSQL database.

## 📡 API Reference

### 1. Root and Health

| Method | Endpoint | Description | Success |
|---|---|---|---|
| GET | `/` | Root welcome message, if implemented | 200 OK |
| GET | `/health` | Health check | 200 OK |

Health response:

```json
{
  "status": "ok"
}
```

### 2. Upload a Document

| Property | Value |
|---|---|
| Method | `POST` |
| Endpoint | `/documents` |
| Content type | `multipart/form-data` |
| Form field | `file` |
| Supported extensions | `.pdf`, `.txt`, `.docx` |
| Maximum size | 10 MB |

Example success response:

```json
{
  "id": 1,
  "filename": "annual_report.pdf",
  "content_type": "application/pdf",
  "size_bytes": 1048576,
  "uploaded_at": "2026-10-28T14:30:00Z"
}
```

Possible error responses:

| Status | Meaning |
|---|---|
| 400 Bad Request | Empty file or invalid upload |
| 413 Content Too Large | File exceeds 10 MB |
| 415 Unsupported Media Type | Unsupported document format |
| 500 Internal Server Error | Database operation failed |

The exact response body and validation behavior depend on the application's implemented routes and exception handlers.

### 3. List Documents

| Property | Value |
|---|---|
| Method | `GET` |
| Endpoint | `/documents` |
| Success status | `200 OK` |

Example response:

```json
[
  {
    "id": 1,
    "filename": "annual_report.pdf",
    "content_type": "application/pdf",
    "size_bytes": 1048576,
    "uploaded_at": "2026-10-28T14:30:00Z"
  }
]
```

The endpoint retrieves document metadata from PostgreSQL through SQLModel.

## 💻 Example API Calls

### Upload a document using curl

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "accept: application/json" \
  -F "file=@sample.txt"
```

### Upload a document using PowerShell

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/documents" `
  -Method Post `
  -Form @{ file = Get-Item ".\sample.txt" }
```

### List uploaded documents

```bash
curl -X GET "http://localhost:8000/documents" \
  -H "accept: application/json"
```

## 📦 Managing Dependencies with uv

| Action | Command |
|---|---|
| Install dependencies | `uv sync` |
| Install all dependency groups | `uv sync --all-groups` |
| Add a production dependency | `uv add <package_name>` |
| Add a development dependency | `uv add --dev <package_name>` |
| Remove a dependency | `uv remove <package_name>` |
| Update the lockfile | `uv lock` |
| Upgrade dependencies | `uv lock --upgrade` |
| Run a Python command | `uv run python <file.py>` |
| Run tests | `uv run python -m pytest Tests/ -v` |

## 🔄 Git Workflow & Pull Request

Before submitting Assignment 4:

1. Implement SQLModel and PostgreSQL integration.
2. Add the Dockerfile and `docker-compose.yml`.
3. Add `.env.example` and update `.gitignore`.
4. Verify that `docker compose up --build` starts the application and database.
5. Verify that document records persist after `docker compose down` and `docker compose up --build`.
6. Run the existing test suite and fix any regressions.
7. Push your changes to the appropriate feature branch.
8. Open a pull request.
9. Ask your partner to review the changes and address feedback.
10. Merge the pull request after approval, following the team's branch policy.

Never commit actual database passwords or other secrets.

## 🗺️ Roadmap

- [x] PostgreSQL integration with SQLModel.
- [x] Document metadata persistence.
- [x] Dockerfile and Docker Compose configuration.
- [x] Persistent PostgreSQL storage using a named volume.
- [ ] PDF, TXT, and DOCX text extraction.
- [ ] Document chunking and vector embedding generation.
- [ ] Vector database integration with ChromaDB, Qdrant, or Pinecone.
- [ ] Retrieval-Augmented Generation (RAG) question-answering endpoint.
- [ ] Authentication and authorization.
- [ ] Production deployment and database migrations.

## 📄 License

Add your project's chosen license here if applicable.
