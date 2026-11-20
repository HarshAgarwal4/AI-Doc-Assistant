# AI Doc Assistant

A high-performance document management API built with **FastAPI, Pydantic v2, SQLModel, PostgreSQL, Alembic, and Docker**. The application supports document validation, metadata persistence, and automated testing, with a reproducible development environment powered by `uv`.

## 🚀 Features

- **High-Performance API:** Built with FastAPI and Uvicorn.
- **Strict Data Validation:** Request and response validation using Pydantic v2.
- **Document Ingestion & Validation:**
  - Supported extensions: `.pdf`, `.txt`, `.docx`.
  - Unsupported formats return `415 Unsupported Media Type`.
  - Maximum file size: 10 MB (`413 Content Too Large`).
  - Empty files return `400 Bad Request`.
- **Persistent Metadata Storage:** SQLModel and PostgreSQL store document metadata.
- **Database Migrations:** Schema version control and automated migrations powered by Alembic.
- **Document Metadata:** Tracks ID, filename, content type, file size in bytes, and upload timestamp.
- **Document Management:** Endpoints for uploading documents, listing all documents, and retrieving metadata.
- **Interactive API Documentation:** Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **Docker Support:** Run the application and PostgreSQL using a single Docker Compose command.
- **Persistent Database Volume:** Document metadata survives container recreation and application restarts.
- **Environment-Based Configuration:** Database credentials and connection settings are supplied through `.env`.
- **Automated Testing:** Pytest and HTTPX/TestClient, with SQLite available for database-independent tests.
- **Modern Python Tooling:** Dependency management and reproducible lockfiles using `uv`.
- **Structured Resume Extraction:** `POST /extract` parses `.pdf`, `.docx`, and `.txt` resumes into a validated `ResumeData` JSON schema using LangChain's `with_structured_output` parser with validation-error retries.
- **Automated Extraction Evaluation:** `scripts/eval_extract.py` benchmarks `POST /extract` against `resumes/` and `expected_outputs.json` and logs field-level accuracy.
- **Background Ingestion & pgvector Storage (Assignment 9):**
  - **Non-blocking Uploads:** `POST /documents` returns `201 Created` with `status: "pending"` immediately; parsing and vectorization run asynchronously via FastAPI `BackgroundTasks`.
  - **Page-by-Page Extraction:** PyMuPDF (`fitz`) for fast `.pdf` extraction and Docling for `.docx` files.
  - **Chunking with Metadata:** Splits text into ~800-character chunks with 100-character overlap, preserving `text`, `filename`, and 1-indexed `page_number`.
  - **Local Vector Embeddings:** 768-dimensional embeddings powered by Ollama (`nomic-embed-text`).
  - **pgvector Vector Database:** Stores vectors in a dedicated `Chunk` table using PostgreSQL's `vector(768)` type.
  - **Batch Ingestion:** Chunks are saved in bulk via `session.add_all()`.
  - **Fail-Safe Lifecycle:** Tracks document status (`pending` -> `processing` -> `ready`). Broken or empty documents transition to `failed` and log errors without crashing the API.
- **Vector Similarity Search (Assignment 10 / Retriever):**
  - **Semantic Retrieval:** `GET /search?q=...&k=5` embeds queries using the same `nomic-embed-text` model and searches indexed chunks via PostgreSQL `pgvector` cosine distance (`<=>`).
  - **Ranked Results:** Returns top-`k` matching chunks with `text`, `filename`, 1-indexed `page`, and similarity `score` (`1 - distance`).
  - **Ready Filter:** Queries only chunks from fully processed (`ready`) documents.
- **Streaming Retrieval-Augmented Generation (RAG - `POST /ask`):**
  - **Server-Sent Events (SSE) Streaming:** Real-time token streaming (`event: token`) followed by persistence confirmation (`event: saved`).
  - **Multi-Provider Support:** Google Gemini (`gemini-3.5-flash-lite`), Groq (`gpt-oss-120b`), Mistral AI (`ministral-3b-2512`), and local Ollama (`llama3.2:1b`).
  - **Resilience & Rate Limiting:** Decorator-driven retries with exponential backoff on HTTP 429 rate limits and provider timeouts.
  - **Cost & Audit Tracking:** Every completion is saved to the `LLMCall` PostgreSQL table tracking input/output tokens, execution latency, and exact cost in USD.
  - **Accurate Document Citations:** Synthesizes multi-page information, handles calculations, and outputs clean source citations (`Sources: - filename.pdf, Page X`).
- **Multi-Layered Security & Jailbreak Defense:**
  - **ML-Based Classification (`backend/ML/`):** Dual-model ensemble (Linear SVM + Multinomial Naive Bayes with TF-IDF) trained on `rag_security_training.csv` to classify prompts before retrieval. Adversarial queries immediately yield `"I don't know."` at zero token cost.
  - **Rule-Based Sanitizer (`utils/prompt_parser.py`):** Filters out malicious HTML/XML script tags, zero-width characters, LLM control tokens, instruction resets, roleplay overrides (`DAN`, `developer mode`), and JSON chat continuation attacks.
- **Interactive Streaming Playground (`GET /page`):**
  - Browser-based playground for testing document Q&A, switching LLM providers on the fly, aborting in-flight streams, and viewing live performance metrics.

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
    ├── alembic.ini
    ├── pyproject.toml
    ├── uv.lock
    ├── main.py
    ├── linear_svm_model.pkl
    ├── naive_bayes_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── rag_security_training.csv
    ├── ML/
    │   ├── cleaning.py
    │   ├── model.py
    │   └── predict.py
    ├── app/
    │   └── models/
    │       ├── docModel.py
    │       ├── chunkModel.py
    │       ├── resumeModel.py
    │       └── llm.py
    ├── database/
    │   └── database.py
    ├── utils/
    │   ├── chunking.py
    │   ├── embedding.py
    │   ├── extraction.py
    │   ├── process_doc.py
    │   ├── retriever.py
    │   ├── prompt_parser.py
    │   ├── retry.py
    │   ├── logger.py
    │   ├── llm.py
    │   └── resume.py
    ├── migrations/
    │   ├── versions/
    │   ├── env.py
    │   ├── script.py.mako
    │   └── README
    ├── test/
    │   ├── test_db.py
    │   ├── test_ingestion.py
    │   ├── test_retriever.py
    │   ├── test_ask.py
    │   ├── test_prompt_parser.py
    │   └── test_resume.py
    └── uploads/
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

Apply database migrations:

After running `uv sync`, apply database migrations using Alembic:

```bash
uv run alembic upgrade head
```

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

## 🗄️ Database Migrations with Alembic

Database schema migrations are managed using **Alembic**. This ensures reproducible, version-controlled schema evolution across local environments and Docker deployments.

### Common Alembic Commands

- **Apply all migrations:**
  ```bash
  uv run alembic upgrade head
  ```

- **Generate a new migration (after modifying models):**
  ```bash
  uv run alembic revision --autogenerate -m "describe_changes"
  ```

- **Roll back the last migration:**
  ```bash
  uv run alembic downgrade -1
  ```

- **Check current migration version:**
  ```bash
  uv run alembic current
  ```

- **View migration history:**
  ```bash
  uv run alembic history --verbose
  ```

## 🐳 Running with Docker

Docker Compose runs three services:

- **db:** PostgreSQL 16 database.
- **migrate:** Runs `alembic upgrade head` after PostgreSQL becomes healthy, ensuring all schema migrations are applied before the application starts.
- **app:** The FastAPI application (starts automatically once migrations complete successfully).

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

1. Configure `DATABASE_URL` in `.env` to use `localhost`:

```dotenv
DATABASE_URL=postgresql+psycopg://postgres:your_local_password@localhost:5432/ai_doc_assistant
```

2. Run database migrations:

```bash
uv run alembic upgrade head
```

3. Start the API:

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

### 4. Extract Resume Data

| Property | Value |
|---|---|
| Method | `POST` |
| Endpoint | `/extract` |
| Content type | `multipart/form-data` |
| Form field | `file` |
| Supported extensions | `.pdf`, `.docx`, `.txt` |
| Success status | `200 OK` |

How it works (`utils/resume.py` & `app/models/resumeModel.py`):
- `extract_text` reads the uploaded file and extracts plain text via PyMuPDF (`.pdf`), `python-docx` (`.docx`), or raw bytes (`.txt`).
- `extract_resume_data` uses `llm.with_structured_output(ResumeData)` to parse the resume into a strict Pydantic v2 schema (`name`, `email`, `phone`, `skills`) with `extra="forbid"`.
- If `ResumeData.model_validate` raises a `ValidationError`, the error is appended to the conversation and retried with the structured LLM before returning `422 Unprocessable Content`.

Example response:

```json
{
  "name": "Aarav Mehta",
  "email": "aarav.mehta@example.com",
  "phone": "91 9876543210",
  "skills": ["Python", "FastAPI", "PostgreSQL"]
}
```

### 5. Ask Questions with Streaming RAG (`POST /ask`)

Streams real-time answers grounded strictly in retrieved document context via Server-Sent Events (SSE).

| Property | Value |
|---|---|
| Method | `POST` |
| Endpoint | `/ask` |
| Content type | `application/json` |
| Response type | `text/event-stream` |

**Request Body:**

```json
{
  "question": "What are the key findings of the annual financial report?",
  "provider": "Groq"
}
```

*Supported providers:* `Google`, `Groq`, `Mistral`, `Ollama`.

**Event Stream Workflow:**
1. **Security Classification:** The raw question is classified by `detect_with_both` (SVM + Naive Bayes). If flagged as an adversarial jailbreak or prompt injection, it immediately streams `"I don't know."` and exits without invoking the LLM.
2. **Prompt Normalization:** Benign queries are parsed via `parse_prompt()` to strip dangerous script tags and delimiters.
3. **Context Retrieval:** Top chunks are retrieved from PostgreSQL via vector similarity search.
4. **Token Streaming:** Emits `event: token` containing delta text:
   ```text
   event: token
   data: {"text": "The company reported..."}
   ```
5. **Database Audit & Persistence:** Emits `event: saved` after committing the call to the `LLMCall` table:
   ```text
   event: saved
   data: {
     "message": "Response generated and saved successfully",
     "record_id": "c1f7b02e-...",
     "time_taken": 1.45,
     "input_tokens": 1250,
     "output_tokens": 85,
     "cost_usd": 0.000238
   }
   ```

### 6. Interactive Streaming Playground (`GET /page`)

Returns a responsive, browser-ready UI (`http://localhost:5000/page`) to test streaming queries, switch LLM providers, abort active requests in real-time, and inspect request latency and database IDs.

### 7. Get Document Chunks (`GET /chunks/{id}`)

Retrieves all extracted chunks and metadata for a specific document ID.

```bash
curl -X GET "http://localhost:5000/chunks/<document_uuid>"
```

## 📊 Resume Extraction Evaluation (`scripts/eval_extract.py`)

Run the evaluation script from the `backend/` directory while the API server is running at `http://localhost:5000`.

### 1. Groq Evaluation Results

**Model:** Groq (LLM)

| Metric | Result |
|---|---:|
| Total Resumes | 20 |
| Valid Responses | 18 |
| Failed Responses | 2 |
| Correct Fields | 72 |
| Total Fields | 72 |
| **Field Accuracy** | **100.00%** |
| Total API Time | 72.73 seconds |
| Average API Time | 4.04 seconds |
| Total Execution Time | 138.02 seconds |

**Observation:**
- Achieved 100% field accuracy across all successfully evaluated fields.
- 2 responses failed during extraction.
- Faster API response time compared to the local Ollama model.

### 2. Ollama Evaluation Results

**Model:** Llama 3.2 (3B) – Local Model

| Metric | Result |
|---|---:|
| Total Resumes | 20 |
| Valid Responses | 20 |
| Failed Responses | 0 |
| Correct Fields | 71 |
| Total Fields | 80 |
| **Field Accuracy** | **88.75%** |
| Total API Time | 140.73 seconds |
| Average API Time | 9.35 seconds |
| Total Execution Time | 288.67 seconds |

**Observation:**
- Successfully processed all 20 resumes without any failed responses.
- Achieved 88.75% field accuracy.
- The smaller local model struggled with some complex resume extraction cases.
- Slower response time compared to Groq.

### 3. Model Comparison

| Metric | Groq | Ollama (Llama 3.2 3B) |
|---|---:|---:|
| Total Resumes | 20 | 20 |
| Valid Responses | 18 | 20 |
| Failed Responses | 2 | 0 |
| Correct Fields | 72 | 71 |
| Total Fields | 72 | 80 |
| Field Accuracy | **100.00%** | 88.75% |
| Average API Time | **4.04 sec** | 9.35 sec |
| Total Execution Time | **138.02 sec** | 288.67 sec |

### 4. Conclusion

- **Groq:** Delivered higher field accuracy and faster API responses, but encountered 2 failed responses.
- **Ollama:** Successfully processed all resumes without failures but had lower extraction accuracy and slower execution.

Groq performed better in terms of extraction accuracy and speed, whereas Ollama demonstrated better response reliability in this evaluation.

**Final Verdict:** Groq is the better choice for accurate and fast resume extraction based on these results. However, Ollama remains a useful option for offline processing and privacy-sensitive applications.

> **Note:** These results are based on a dataset of 20 resumes. Further evaluation with larger and more diverse datasets is recommended before drawing definitive conclusions.

```bash
uv run python scripts/eval_extract.py
```

- Loads test cases from `expected_outputs.json` and sends each file in `resumes/` to `POST /extract`.
- Compares extracted `name`, `email`, `phone`, and `skills` against expected outputs and logs mismatches, errors, and final field accuracy using Python `logging`.

## ⚡ Claude Prompt Caching (Quick Notes)

- **What it does:** Stores reusable prompt prefixes (system instructions, tool definitions, reference docs) in memory so Claude does not reprocess them on every request.
- **Faster & cheaper:** Cache hits cut latency (time-to-first-token) and give around a 90% input token discount (`0.1x` base input price or lower).
- **Write vs. read pricing:** Cache writes cost `1.25x` (5-minute default TTL) or `2.0x` (1-hour TTL) of base input price, so reuse from the 2nd request onward saves money.
- **How to enable:** Add `"cache_control": {"type": "ephemeral"}` breakpoints to static blocks in `system`, `tools`, or `messages` (requires minimum token length, typically 512+ tokens).
- **Order matters:** Caching matches exact prefixes from top to bottom—keep static instructions/tools first and dynamic user input at the very end.
- **Verify cache hits:** Check `response.usage` for `cache_read_input_tokens > 0` (hit) vs. `cache_creation_input_tokens > 0` (write).

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

## 🚀 Assignment 9: Background Ingestion – PDF to pgvector

When a PDF or DOCX file is uploaded, it is automatically converted into searchable vector embeddings without blocking the client.

### Ingestion Workflow
1. **Save File**: The file is streamed to the `uploads/` directory (mounted as a Docker volume and ignored by git).
2. **Immediate 201 Response**: `POST /documents` returns HTTP `201 Created` with `status: "pending"` right away.
3. **Background Task**: FastAPI `BackgroundTasks` calls `process_document`:
   - Updates document status to `processing`.
   - Extracts text page by page (PyMuPDF for `.pdf`, Docling for `.docx`).
   - Splits text into chunks of ~800 characters with 100-character overlap, preserving `text`, `filename`, and `page_number`.
   - Generates 768-dimensional embeddings via local Ollama (`nomic-embed-text`).
   - Bulk inserts all chunks into PostgreSQL `chunk` table with `pgvector` (`session.add_all()`).
   - Sets document status to `ready`.
4. **Error Handling**: Corrupted or empty files update status to `failed` and log errors without crashing the API.

### Test Endpoints with curl

**Upload a PDF (returns 201 immediately):**
```bash
curl -X POST "http://localhost:5000/documents" \
  -F "file=@uploads/sample.pdf"
```

**List documents and check processing status:**
```bash
curl -X GET "http://localhost:5000/documents"
```

## 🔍 Semantic Search & Retrieval (`GET /search`)

Search over ingested documents using vector embeddings and cosine distance.

### Endpoint Details
- **Method:** `GET /search`
- **Query Parameters:**
  - `q` (string, required): Search query or question.
  - `k` (integer, optional, default: `5`): Maximum number of top matching chunks to retrieve.

### Example Request
```bash
curl -X GET "http://localhost:5000/search?q=What+is+photosynthesis&k=3" \
  -H "accept: application/json"
```

### Example Response
```json
[
  {
    "text": "Photosynthesis is the process used by plants to convert light energy into chemical energy...",
    "filename": "biology_notes.pdf",
    "page": 2,
    "score": 0.8842
  }
]
```

### Running Tests

```bash
# Run unit tests for vector retriever & search endpoint
uv run pytest test/test_retriever.py -v

# Run unit tests for ingestion, chunking, and error handling
uv run pytest test/test_ingestion.py -v

# Run database tests
uv run pytest test/test_db.py -v
```

## 📦 Managing Dependencies with uv

| Action | Command |
|---|---|
| Install dependencies | `uv sync` |
| Install all dependency groups | `uv sync --all-groups` |
| Apply database migrations | `uv run alembic upgrade head` |
| Create database migration | `uv run alembic revision --autogenerate -m "<msg>"` |
| Add a production dependency | `uv add <package_name>` |
| Add a development dependency | `uv add --dev <package_name>` |
| Remove a dependency | `uv remove <package_name>` |
| Update the lockfile | `uv lock` |
| Upgrade dependencies | `uv lock --upgrade` |
| Run a Python command | `uv run python <file.py>` |
| Run ingestion tests | `uv run pytest test/test_ingestion.py -v` |
| Run all tests | `uv run pytest -v` |

## 🔄 Git Workflow & Pull Request

Before submitting:

1. Implement SQLModel and PostgreSQL pgvector integration.
2. Verify Dockerfile, `docker-compose.yml`, and `uploads/` volume mounting.
3. Verify that `docker compose up --build` starts the application, database, and Ollama model downloads.
4. Verify migrations apply cleanly with `uv run alembic upgrade head`.
5. Run the test suite: `uv run pytest test/test_ingestion.py -v`.
6. Push changes to the feature branch and open a pull request.

Never commit actual database passwords or other secrets.

## 🗺️ Roadmap

- [x] PostgreSQL integration with SQLModel.
- [x] Document metadata persistence.
- [x] Dockerfile and Docker Compose configuration.
- [x] Persistent PostgreSQL storage using a named volume.
- [x] Database migrations with Alembic (`uv run alembic upgrade head`).
- [x] PDF and DOCX text extraction (PyMuPDF & Docling).
- [x] Document chunking preserving page numbers and filenames.
- [x] Vector embedding generation with Ollama (`nomic-embed-text`).
- [x] Vector database integration with PostgreSQL `pgvector` (`Chunk` table).
- [x] Asynchronous background ingestion with FastAPI `BackgroundTasks`.
- [x] Vector similarity search and retriever endpoint (`GET /search`).
- [x] Streaming Retrieval-Augmented Generation (RAG) endpoint (`POST /ask`).
- [x] Interactive Browser Playground (`GET /page`).
- [x] Multi-layered security defense (ML jailbreak classifier + prompt parser).
- [x] LLM call logging, latency measurement, and dollar cost tracking.
- [ ] Authentication and authorization.
- [ ] Production deployment.

## 📄 License

Add your project's chosen license here if applicable.
