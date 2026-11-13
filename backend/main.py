from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlmodel import Session, select

from app.models.docModel import Document
from app.models.llm import LLMRequestModel
from app.models.resumeModel import ResumeData
from database.database import get_session
from utils.llm import generate
from utils.logger import logger
from utils.resume import extract_resume_data, extract_text

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

    # Read file chunk by chunk and enforce size limit
    while True:
        chunk = await file.read(CHUNK_SIZE)

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="File size must not exceed 10 MB",
            )

    # Reject empty files
    if total_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file not allowed",
        )

    try:
        document = Document(
            filename=Path(file.filename or "").name,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=total_size,
        )

        session.add(document)
        session.commit()
        session.refresh(document)

        return document

    except Exception:
        session.rollback()
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

        <div class="output"></div>

        <div class="actions">
            <button class="cancel-btn">Cancel</button>
        </div>

        <p class="meta"></p>
    `;

    container.prepend(card);

    const status = card.querySelector(".status");
    const output = card.querySelector(".output");
    const cancelButton = card.querySelector(".cancel-btn");
    const meta = card.querySelectorAll(".meta")[1];

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

    generator = generate(
        question=req.question,
        provider=req.provider,
    )

    try:
        first_chunk = await anext(generator)

    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="LLM request timed out",
        )

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

