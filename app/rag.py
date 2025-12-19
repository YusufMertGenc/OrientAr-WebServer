import json
import requests
from pathlib import Path
from typing import Dict

from chromadb import PersistentClient
from .config import settings

# Paths
_base_dir = Path(__file__).resolve().parent.parent
_data_path = _base_dir / "data" / "campus_kb.json"

# Chroma (embeddings already exist)
_chroma_client = PersistentClient(path=settings.chroma_dir)
_collection = _chroma_client.get_or_create_collection("campus_kb")


def _embed_question(question: str) -> list[float]:
    """
    Get embedding from local Ollama (via ngrok)
    """
    resp = requests.post(
        f"{settings.embedding_base_url}/api/embeddings",
        json={
            "model": settings.embedding_model,
            "prompt": question
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def rag_query_with_scores(question: str, top_k: int = 5) -> Dict:
    query_embedding = _embed_question(question)

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }
