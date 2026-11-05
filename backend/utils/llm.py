import asyncio
import json
import os
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from sqlmodel import Session

from app.models.llm import LLMCall, ModelDetails
from database.database import engine
from utils.logger import logger
from utils.retry import retry

load_dotenv()

MODELS = {
    "Google": ModelDetails(
        provider="Google",
        model_name="gemini-3.5-flash-lite",
        input_cost_per_million_tokens=0.30,
        output_cost_per_million_tokens=2.50,
    ),
    "Groq": ModelDetails(
        provider="Groq",
        model_name="openai/gpt-oss-120b",
        input_cost_per_million_tokens=0.15,
        output_cost_per_million_tokens=0.60,
    ),
    "Mistral": ModelDetails(
        provider="Mistral",
        model_name="ministral-3b-2512",
        input_cost_per_million_tokens=0.1,
        output_cost_per_million_tokens=0.1,
    ),
    "Ollama": ModelDetails(
        provider="Ollama",
        model_name=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        input_cost_per_million_tokens=0.0,
        output_cost_per_million_tokens=0.0,
    ),
}


def get_llm(provider: str):

    if provider not in MODELS:
        raise ValueError(f"Unsupported provider: {provider}")

    model = MODELS[provider].model_name

    if provider == "Google":
        return ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model=model)

    elif provider == "Groq":
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model,
            temperature=0,
        )

    elif provider == "Mistral":
        return ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model=model,
            temperature=0,
        )

    elif provider == "Ollama":
        return ChatOllama(
            model=model,
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
        )


def save_llm_call(data: dict):
    logger.info("DB SESSION STARTED")
    with Session(engine) as session:
        doc = LLMCall(**data)

        logger.info("DB OBJECT CREATED")
        session.add(doc)

        logger.info("DB COMMIT STARTED")
        session.commit()

        logger.info("DB COMMIT COMPLETED")
        session.refresh(doc)

        logger.info("DB REFRESH COMPLETED")

    return doc.id


def make_event(event: str, data: dict) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data)}\n\n"
    )

@retry(max_retry=3)
async def generate(
    question: str,
    provider: str = "Ollama",
):
    logger.info(
        "GENERATOR STARTED | Provider: %s",
        provider,
    )

    input_tokens = 0
    output_tokens = 0
    output_text = ""

    start_time = time.perf_counter()
    # 1. Initialize LLM
    llm = get_llm(provider)
    logger.info("LLM INITIALIZED | Provider: %s", provider)
    # 2. Stream LLM response
    async for chunk in llm.astream(question):
        usage = chunk.usage_metadata
        if usage:
            input_tokens = max(
                input_tokens,
                usage.get("input_tokens") or 0,
            )
            output_tokens = max(
                output_tokens,
                usage.get("output_tokens") or 0,
            )
        # Extract text from chunk
        content = chunk.content
        chunk_text = ""
        if isinstance(content, str):
            chunk_text = content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    chunk_text += item
                elif isinstance(item, dict):
                    text_content = item.get("text")
                    if isinstance(text_content, str):
                        chunk_text += text_content
        # Accumulate complete response
        output_text += chunk_text
        # Send token to frontend
        if chunk_text:
            yield make_event(
                "token",
                {"text": chunk_text},
            )
    logger.info("LLM GENERATION COMPLETED")
    # 3. Calculate generation time
    total_time = time.perf_counter() - start_time
    # 4. Calculate API cost
    model_details = MODELS[provider]
    cost = (
        model_details.input_cost_per_million_tokens * input_tokens
        + model_details.output_cost_per_million_tokens * output_tokens
    ) / 1_000_000
    # 5. Prepare database data
    data = {
        "model_name": model_details.model_name,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "time_taken": total_time,
        "input": question,
        "output": output_text,
        "cost_usd": float(cost),
    }
    logger.info(
        "RESPONSE PREPARED | Input tokens: %s | Output tokens: %s",
        input_tokens,
        output_tokens,
    )
    # 6. Save database record in worker thread
    logger.info("DATABASE SAVING STARTED")
    doc_id = await asyncio.to_thread(
        save_llm_call,
        data,
    )
    logger.info(
        "DATABASE SAVING COMPLETED | Record ID: %s",
        doc_id,
    )
    # 7. Send saved event after successful commit
    yield make_event(
        "saved",
        {
            "message": "Response generated and saved successfully",
            "record_id": str(doc_id),
            "time_taken": round(total_time, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(float(cost), 8),
        },
    )
    logger.info("SAVED EVENT YIELDED")





