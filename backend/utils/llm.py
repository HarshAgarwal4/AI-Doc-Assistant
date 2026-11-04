import os
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from sqlmodel import Session

from app.models.llm import LLMCall, ModelDetails
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
        model_name='ministral-3b-2512',
        input_cost_per_million_tokens=0.1,
        output_cost_per_million_tokens=0.1,
    ),

    "Ollama": ModelDetails(
        provider="Ollama",
        model_name="llama3.2:latest",
        input_cost_per_million_tokens=0.0,
        output_cost_per_million_tokens=0.0,
    ),
}

def get_llm(provider: str):

    if provider not in MODELS:
        raise ValueError(f"Unsupported provider: {provider}")

    model = MODELS[provider].model_name

    if provider == "Google":
        return ChatGoogleGenerativeAI(
            api_key=os.getenv('GOOGLE_API_KEY'),
            model=model
        )

    elif provider == "Groq":
        return ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model=model,
            temperature=0,
        )

    elif provider == "Mistral":
        return ChatMistralAI(
            api_key=os.getenv('MISTRAL_API_KEY'),
            model=model,
            temperature=0,
        )

    elif provider == "Ollama":
        return ChatOllama(
            model=model,
            temperature=0,
        )

@retry(max_retry=3)
async def generate(question: str, session : Session ,provider: str = "Ollama"):
    llm = get_llm(provider)

    input_tokens = 0
    output_tokens = 0

    input_text = question
    output_text = ""
    start_time = time.perf_counter()
    async for chunk in llm.astream(question):

        # Capture token usage
        usage = chunk.usage_metadata
        if usage:
            input_tokens = max(
                input_tokens,
                usage.get("input_tokens") or 0
            )

            output_tokens = max(
                output_tokens,
                usage.get("output_tokens") or 0
            )

        # Extract generated text
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

        # Save complete output
        output_text += chunk_text

        # Stream response
        if chunk_text:
            yield chunk_text
    total_time = time.perf_counter() - start_time
    Cost = float((MODELS[provider].input_cost_per_million_tokens * input_tokens + MODELS[provider].output_cost_per_million_tokens * output_tokens)/1000000)
    logger.info("Cost: %s", Cost)
    # Final information
    logger.info(
        "\n========== FINAL LLM DETAILS ==========\n"
        "Provider: %s\n"
        "Model: %s\n\n"
        "INPUT TEXT:\n%s\n\n"
        "OUTPUT TEXT:\n%s\n\n"
        "TOKEN USAGE:\n"
        "Input Tokens: %s\n"
        "Output Tokens: %s\n"
        "Total Tokens: %s\n"
        "Time Taken: %s\n"
        "=======================================",
        provider,
        MODELS[provider].model_name,
        input_text,
        output_text,
        input_tokens,
        output_tokens,
        input_tokens + output_tokens,
        total_time,
    )

    try :
        doc = LLMCall(
            model_name=MODELS[provider].model_name,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_taken=total_time,
            input=input_text,
            output=output_text,
            cost_usd=Cost
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
    except Exception as e:  # noqa: BLE001
        logger.error("Database saving failed: %s", e)
