import asyncio
import hashlib
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlmodel import Session, select

from app.models.chunkModel import Chunk, SearchResult
from app.models.docModel import Document
from app.models.feedback import Feedback, FeedbackRequestModel, FeedbackResponseModel
from app.models.llm import LLMRequestModel
from app.models.resumeModel import ResumeData
from app.models.ticket import TicketModel
from database.database import engine, get_session
from utils.llm import generate
from utils.logger import logger
from utils.process_doc import process_document
from utils.prompt_parser import parse_prompt
from utils.resume import extract_resume_data, extract_text
from utils.retriever import search

app = FastAPI(
    title="AI Doc Assistant",
    description="My first FastAPI backend with Postgres",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = [".pdf", ".txt", ".docx"]
CHUNK_SIZE = 1024 * 1024
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/", status_code=status.HTTP_200_OK)
def home():
    return {"message": "Welcome to AI Doc Assistant"}


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok"}


@app.post(
    "/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=Document,
)
async def post_document(
    file: Annotated[UploadFile, File()],
    session: Annotated[Session, Depends(get_session)],
    background_tasks : BackgroundTasks
):
    extension = Path(file.filename or "").suffix.lower()
    total_size = 0

    # Check file format
    if extension not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File format not allowed",
        )

    # Check file size if available
    if file.size is not None and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File size must not be greater than 10 MB",
        )

    safe_filename = Path(file.filename or "upload.pdf").name
    file_path = UPLOAD_DIR / safe_filename

    try:
        hasher = hashlib.sha256()
        with open(file_path, "wb") as f:  # noqa: ASYNC230
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    f.close()
                    if file_path.exists():
                        file_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="File size must not exceed 10 MB",
                    )
                hasher.update(chunk)
                f.write(chunk)

        # Reject empty files
        if total_size == 0:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file not allowed",
            )

        file_hash = hasher.hexdigest()

        # Check if file with exact same hash already exists
        existing_document = session.exec(
            select(Document).where(Document.file_hash == file_hash)
        ).first()

        if existing_document:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document with the exact same content already exists",
            )

        document = Document(
            filename=safe_filename,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=total_size,
            file_hash=file_hash,
            status="pending",
        )

        session.add(document)
        session.commit()
        session.refresh(document)

        if extension in [".pdf", ".docx"]:
            background_tasks.add_task(
                process_document,
                str(file_path),
                document.id,
            )

        return document

    except HTTPException:
        raise
    except Exception:
        session.rollback()
        if file_path.exists():
            file_path.unlink()
        raise
    finally:
        await file.close()


@app.get(
    "/documents",
    response_model=list[Document],
    status_code=status.HTTP_200_OK,
)
def get_documents(
    session: Annotated[Session, Depends(get_session)],
):
    documents = session.exec(select(Document)).all()
    return documents

@app.delete("/documents/{id}", status_code=status.HTTP_200_OK)
def delete_document(
    id: UUID,
    session: Annotated[Session, Depends(get_session)],
):
    document = session.get(Document, id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    file_path = UPLOAD_DIR / document.filename
    if file_path.exists():
        file_path.unlink()
    session.delete(document)
    session.commit()
    return {"message": "Document deleted successfully"}

@app.get("/chunks/{id}", response_model=list[Chunk], status_code=status.HTTP_200_OK)
def get_document_chunks(
    id: UUID,
    session: Annotated[Session, Depends(get_session)],
):
    document = session.get(Document, id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    chunks = session.exec(select(Chunk).where(Chunk.document_id == id)).all()
    return [
        chunk.model_dump(exclude={"embedding"})
        for chunk in chunks
    ]


@app.get(
    "/search",
    response_model=list[SearchResult],
    status_code=status.HTTP_200_OK,
)
async def search_endpoint(
    q: Annotated[str, Query(description="Search query string")],
    k: Annotated[int, Query(description="Number of chunks to retrieve" , ge=1 , le=50)] = 5,
):
    return await search(q=q, k=k)

@app.get("/page", response_class=HTMLResponse)
def get_page():
    return HTMLResponse(r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Streaming Playground</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 35px auto;
            padding: 20px;
            background: #f3f4f6;
            color: #111827;
        }

        h2 {
            text-align: center;
        }

        .form {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px #00000010;
        }

        input, select, button {
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 15px;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }

        .request-card {
            background: white;
            padding: 18px;
            margin: 15px 0;
            border-radius: 10px;
            box-shadow: 0 2px 8px #00000010;
        }

        .request-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .request-header h3 {
            margin: 0;
        }

        .status {
            font-size: 13px;
            padding: 5px 10px;
            border-radius: 20px;
            background: #fef3c7;
            color: #92400e;
        }

        .status.completed {
            background: #dcfce7;
            color: #166534;
        }

        .status.error {
            background: #fee2e2;
            color: #991b1b;
        }

        .status.cancelled {
            background: #e5e7eb;
            color: #374151;
        }

        .output {
            background: #111827;
            color: #4ade80;
            padding: 18px;
            min-height: 100px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            line-height: 1.6;
            border-radius: 8px;
            margin-top: 15px;
        }

        .actions {
            display: flex;
            gap: 10px;
        }

        .actions button {
            width: auto;
            padding: 8px 15px;
        }

        .cancel-btn {
            background: #dc2626;
        }

        .cancel-btn:hover {
            background: #b91c1c;
        }

        .meta {
            color: #6b7280;
            font-size: 13px;
            margin-top: 10px;
        }

        #activeCount {
            text-align: center;
            color: #4b5563;
        }

        .empty {
            text-align: center;
            color: #6b7280;
            padding: 30px;
        }

        .feedback-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid #e5e7eb;
        }

        .feedback-label {
            font-size: 14px;
            font-weight: 500;
            color: #374151;
        }

        .feedback-btn {
            width: auto !important;
            padding: 6px 14px !important;
            font-size: 14px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            margin: 0 !important;
        }

        .thumb-up-btn {
            background: #10b981;
        }

        .thumb-up-btn:hover {
            background: #059669;
        }

        .thumb-down-btn {
            background: #ef4444;
        }

        .thumb-down-btn:hover {
            background: #dc2626;
        }

        .feedback-success {
            font-size: 14px;
            color: #059669;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
    </style>
</head>

<body>

    <h2>AI Streaming Playground</h2>

    <p id="activeCount">Active requests: 0</p>

    <div class="form">

        <input
            type="text"
            id="question"
            placeholder="Enter your question..."
        >

        <select id="provider">
            <option value="Google">Google</option>
            <option value="Groq">Groq</option>
            <option value="Mistral">Mistral</option>
            <option value="Ollama">Ollama</option>
        </select>

        <button id="askBtn" onclick="askQuestion()">
            Send Request
        </button>

    </div>

    <h3>Request History</h3>

    <div id="requests">
        <div class="empty">No requests yet.</div>
    </div>

<script>

let requestCounter = 0;
const activeRequests = new Map();

function updateActiveCount() {
    document.getElementById("activeCount").textContent =
        `Active requests: ${activeRequests.size}`;
}

function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[char]);
}

function setStatus(element, message, className = "") {
    element.textContent = message;
    element.className = "status";

    if (className) {
        element.classList.add(className);
    }
}

function parseSSEEvent(rawEvent) {
    let eventName = "message";
    const dataLines = [];

    for (const line of rawEvent.split(/\r?\n/)) {
        if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
        }
    }

    if (!dataLines.length) return null;

    return {
        event: eventName,
        data: JSON.parse(dataLines.join("\n"))
    };
}

async function askQuestion() {
    const question = document.getElementById("question").value.trim();
    const provider = document.getElementById("provider").value;

    if (!question) {
        alert("Please enter a question.");
        return;
    }

    requestCounter++;

    const requestId = requestCounter;
    const controller = new AbortController();

    const container = document.getElementById("requests");

    if (requestCounter === 1) {
        container.innerHTML = "";
    }

    const card = document.createElement("div");
    card.className = "request-card";

    card.innerHTML = `
        <div class="request-header">
            <h3>Request #${requestId}</h3>
            <span class="status">Generating...</span>
        </div>

        <p><strong>Question:</strong> ${escapeHTML(question)}</p>
        <p class="meta">Provider: ${escapeHTML(provider)}</p>
        <p class="doc-files" style="font-size: 13px; color: #4b5563; margin: 4px 0;"></p>
        <p class="doc-pages" style="font-size: 13px; color: #4b5563; margin: 4px 0;"></p>

        <div class="output"></div>

        <div class="actions">
            <button class="cancel-btn">Cancel</button>
        </div>

        <div class="feedback-container" style="display: none;">
            <span class="feedback-label">Was this response helpful?</span>
            <button class="feedback-btn thumb-up-btn" type="button">👍 Thumbs Up</button>
            <button class="feedback-btn thumb-down-btn" type="button">👎 Thumbs Down</button>
            <span class="feedback-success" style="display: none;"></span>
        </div>

        <p class="meta"></p>
    `;

    container.prepend(card);

    const status = card.querySelector(".status");
    const output = card.querySelector(".output");
    const cancelButton = card.querySelector(".cancel-btn");
    const meta = card.querySelectorAll(".meta")[1];
    const docFiles = card.querySelector(".doc-files");
    const docPages = card.querySelector(".doc-pages");
    const feedbackContainer = card.querySelector(".feedback-container");
    const thumbUpBtn = card.querySelector(".thumb-up-btn");
    const thumbDownBtn = card.querySelector(".thumb-down-btn");
    const feedbackSuccess = card.querySelector(".feedback-success");
    const feedbackLabel = card.querySelector(".feedback-label");

    let reader = null;
    let saved = false;
    let receivedToken = false;

    const startTime = performance.now();

    cancelButton.onclick = () => controller.abort();

    activeRequests.set(requestId, controller);
    updateActiveCount();

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },
            body: JSON.stringify({
                question,
                provider
            }),
            signal: controller.signal
        });

        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}: ${await response.text()}`
            );
        }

        if (!response.body) {
            throw new Error("Streaming response body unavailable.");
        }

        reader = response.body.getReader();

        const decoder = new TextDecoder();
        let buffer = "";

        function handleEvent(rawEvent) {
            const parsed = parseSSEEvent(rawEvent);

            if (!parsed) return;

            const { event, data } = parsed;

            console.log("SSE EVENT RECEIVED:", event, data);

            if (event === "token") {
                if (!receivedToken) {
                    output.textContent = "";
                    receivedToken = true;
                }

                output.textContent += data.text || "";
                output.scrollTop = output.scrollHeight;
            }

            else if (event === "saved") {
                saved = true;

                setStatus(status, "Completed", "completed");

                meta.textContent =
                    `Database ID: ${data.record_id} | ` +
                    `Generation time: ${data.time_taken}s`;

                if (data.record_id) {
                    feedbackContainer.style.display = "flex";

                    const sendFeedback = async (rating) => {
                        thumbUpBtn.disabled = true;
                        thumbDownBtn.disabled = true;
                        try {
                            const res = await fetch("/feedback", {
                                method: "POST",
                                headers: {
                                    "Content-Type": "application/json"
                                },
                                body: JSON.stringify({
                                    id: data.record_id,
                                    rating: rating
                                })
                            });
                            if (!res.ok) {
                                throw new Error("Failed to save feedback");
                            }
                            const resJson = await res.json();
                            thumbUpBtn.style.display = "none";
                            thumbDownBtn.style.display = "none";
                            feedbackLabel.style.display = "none";
                            feedbackSuccess.textContent = "✓ " + (resJson.msg || "Feedback saved successfully");
                            feedbackSuccess.style.display = "inline";
                        } catch (err) {
                            console.error("Feedback error:", err);
                            thumbUpBtn.disabled = false;
                            thumbDownBtn.disabled = false;
                            alert("Failed to save feedback. Please try again.");
                        }
                    };

                    thumbUpBtn.onclick = () => sendFeedback("Thumbs_up");
                    thumbDownBtn.onclick = () => sendFeedback("Thumbs_down");
                }
            }

            else if (event === "error") {
                throw new Error(
                    data.message || "Server generation failed."
                );
            }
        }

        function processBuffer() {
            while (true) {
                const separator = /\r?\n\r?\n/.exec(buffer);

                if (!separator) break;

                const rawEvent = buffer.slice(0, separator.index);

                buffer = buffer.slice(
                    separator.index + separator[0].length
                );

                handleEvent(rawEvent);
            }
        }

        while (true) {
            const { value, done } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            processBuffer();
        }

        // Flush remaining decoder data.
        buffer += decoder.decode();

        // Handle a final event without a trailing blank line.
        if (buffer.trim()) {
            handleEvent(buffer);
            buffer = "";
        }

        if (!saved) {
            throw new Error(
                "Stream closed without receiving the saved event."
            );
        }

        const elapsed = (
            (performance.now() - startTime) / 1000
        ).toFixed(2);

        meta.textContent += ` | Total time: ${elapsed}s`;

    } catch (error) {
        if (error.name === "AbortError") {
            setStatus(status, "Cancelled", "cancelled");
            output.textContent += "\n\nRequest cancelled.";
        } else {
            setStatus(status, "Failed", "error");
            output.textContent += "\n\nError: " + error.message;
            console.error("Request failed:", error);
        }

    } finally {
        activeRequests.delete(requestId);
        updateActiveCount();

        cancelButton.disabled = true;
        cancelButton.textContent = "Finished";

        if (reader) {
            try {
                reader.releaseLock();
            } catch (error) {
                console.warn("Reader release:", error);
            }
        }
    }
}

document.getElementById("question").addEventListener(
    "keydown",
    event => {
        if (event.key === "Enter") {
            askQuestion();
        }
    }
);
</script>

</body>
</html>
""")

async def stream_response(first_chunk, generator):

    yield first_chunk

    async for chunk in generator:
        yield chunk

@app.post('/ask')
async def ask_llm(req: LLMRequestModel):

    clean_question = parse_prompt(req.question)

    generator = generate(
        question=clean_question,
        provider=req.provider,
    )

    try:
        first_chunk = await anext(generator)

    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="LLM request timed out",
        )

    except HTTPException:
        raise

    except Exception:  # noqa: BLE001
        logger.exception("LLM INITIALIZATION FAILED")

        raise HTTPException(
            status_code=502,
            detail="LLM aprovider failed before streaming started",
        )

    return StreamingResponse(
        stream_response(first_chunk, generator),
        media_type="text/event-stream",
        status_code=200,
    )

@app.post('/extract' , status_code=status.HTTP_200_OK , response_model=ResumeData)
async def extract(file : Annotated[UploadFile , File()]):
    content = await extract_text(file)
    if content is None or content.strip() == '':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='file is empty'
        )
    response = await extract_resume_data(content)
    return response


def save_feedback(f: FeedbackRequestModel) -> Feedback:
    with Session(engine) as session:
        feedback_obj = Feedback(
            record_id=f.id,
            rating=f.rating,
            comment=f.comment,
        )
        session.add(feedback_obj)
        session.commit()
        session.refresh(feedback_obj)
        return feedback_obj


@app.post("/feedback", status_code=status.HTTP_201_CREATED, response_model=FeedbackResponseModel)
async def feedback(f: FeedbackRequestModel):
    feedback_record = await asyncio.to_thread(save_feedback, f)
    return FeedbackResponseModel(
        id=feedback_record.id,
        msg="Feedback saved successfully",
    )


def fetch_all_feedbacks() -> list[Feedback]:
    with Session(engine) as session:
        return list(session.exec(select(Feedback)).all())


@app.get("/feedback", response_model=list[Feedback], status_code=status.HTTP_200_OK)
async def get_feedbacks():
    return await asyncio.to_thread(fetch_all_feedbacks)


@app.get("/feedbacks", response_model=list[Feedback], status_code=status.HTTP_200_OK, include_in_schema=False)
async def get_feedbacks_alias():
    return await asyncio.to_thread(fetch_all_feedbacks)


@app.get(
    "/ticket",
    response_model=list[TicketModel],
    status_code=status.HTTP_200_OK,
)
def get_tickets(
    session: Annotated[Session, Depends(get_session)],
):
    return session.exec(select(TicketModel)).all()


@app.get(
    "/tickets",
    response_model=list[TicketModel],
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def get_tickets_alias(
    session: Annotated[Session, Depends(get_session)],
):
    return session.exec(select(TicketModel)).all()


