import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=ollama_host,
)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return embedding_model.embed_documents(texts)


def generate_embedding(text: str) -> list[float]:
    return embedding_model.embed_query(text)
