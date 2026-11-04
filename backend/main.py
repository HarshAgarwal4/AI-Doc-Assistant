from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlmodel import Session, select

from app.models.docModel import Document
from app.models.llm import LLMRequestModel
from database.database import get_session
from utils.llm import generate

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

@app.get('/page')
def get_page():
    return HTMLResponse(
        """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LLM Streaming Tester</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background: #f4f4f4;
        }

        h2 {
            text-align: center;
        }

        input, select, button {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            box-sizing: border-box;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 5px;
        }

        button:disabled {
            background: gray;
            cursor: not-allowed;
        }

        #output {
            background: #111827;
            color: #4ade80;
            padding: 20px;
            min-height: 200px;
            border-radius: 8px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            line-height: 1.6;
        }
    </style>
</head>

<body>

    <h2>LLM Streaming Tester</h2>

    <input
        type="text"
        id="question"
        placeholder="Enter your question..."
    />

    <select id="provider">
        <option value="Google">Google</option>
        <option value="Groq">Groq</option>
        <option value="Mistral">Mistral</option>
        <option value="Ollama">Ollama</option>
    </select>

    <button id="askBtn" onclick="askQuestion()">
        Ask AI
    </button>

    <h3>Live Response:</h3>

    <div id="output">Your response will appear here...</div>

    <script>
        async function askQuestion() {
            const question = document.getElementById("question").value;
            const provider = document.getElementById("provider").value;
            const output = document.getElementById("output");
            const button = document.getElementById("askBtn");

            if (!question.trim()) {
                alert("Please enter a question");
                return;
            }

            output.textContent = "";
            button.disabled = true;
            button.textContent = "Generating...";

            try {
                const response = await fetch("http://127.0.0.1:5000/ask", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        question: question,
                        provider: provider
                    })
                });

                if (!response.ok) {
                    throw new Error(await response.text() || `HTTP ${response.status}`);
                }

                if (!response.body) {
                    throw new Error("Streaming is not supported by this response.");
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { value, done } = await reader.read();

                    if (done) break;

                    output.textContent += decoder.decode(value, {
                        stream: true
                    });
                }

                output.textContent += decoder.decode();

            } catch (error) {
                output.textContent += "\\n\\nError: " + error.message;
            } finally {
                button.disabled = false;
                button.textContent = "Ask AI";
            }
        }
    </script>

</body>
</html>
"""
    )

@app.post('/ask' , status_code=status.HTTP_200_OK)
def ask_llm(req : LLMRequestModel , session : Annotated[Session , Depends(get_session)]):
    return StreamingResponse(
        generate(question=req.question ,session=session, provider=req.provider),
        media_type="text/plain"
    )
    