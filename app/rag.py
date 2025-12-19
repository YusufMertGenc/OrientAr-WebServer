import json
from pathlib import Path
from typing import Dict, List

import requests
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from .config import settings


# ---- Ollama embedding wrapper ----
class OllamaEmbeddingFunction:
    def name(self) -> str:
        return "ollama-nomic-embed-text"

    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{settings.embedding_base_url}/api/embeddings",
                json={
                    "model": settings.embedding_model,
                    "prompt": text
                },
                timeout=60
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings




# ---- Paths ----
_base_dir = Path(__file__).resolve().parent.parent
_data_path = _base_dir / "data" / "campus_kb.json"


# ---- Chroma ----
_chroma_client = PersistentClient(path=settings.chroma_dir)

_embedding_fn = OllamaEmbeddingFunction()

_collection = _chroma_client.get_or_create_collection(
    name="campus_kb",
    embedding_function=_embedding_fn
)


def load_kb_to_chroma():
    if not _data_path.exists():
        raise FileNotFoundError("campus_kb.json not found")

    with open(_data_path, "r", encoding="utf-8") as f:
        kb_items = json.load(f)

    ids = [item["id"] for item in kb_items]
    docs = [item["text"] for item in kb_items]
    metas = [
        {
            "entity": item["entity"],
            "type": item["type"]
        }
        for item in kb_items
    ]

    _collection.delete(where={})

    _collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas
    )


def rag_query_with_scores(question: str, top_k: int = 5) -> Dict:
    results = _collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }
