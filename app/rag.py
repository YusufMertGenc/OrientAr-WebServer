import json
from pathlib import Path
from typing import Dict, List

import requests
from chromadb import PersistentClient

from .config import settings


# ---- Paths ----
_base_dir = Path(__file__).resolve().parent.parent
_data_path = _base_dir / "data" / "campus_kb.json"


# ---- Chroma ----
_chroma_client = PersistentClient(path=settings.chroma_path)
_collection = _chroma_client.get_or_create_collection("campus_kb")


def ollama_embed(texts: List[str]) -> List[List[float]]:
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




def load_kb_to_chroma():
    if not _data_path.exists():
        raise FileNotFoundError("campus_kb.json not found")

    with open(_data_path, "r", encoding="utf-8") as f:
        kb_items = json.load(f)

    ids = [item["id"] for item in kb_items]
    docs = [item["text"] for item in kb_items]
    metas = [
        {"entity": item["entity"], "type": item["type"]}
        for item in kb_items
    ]

    # 🔥 ÖNEMLİ: tamamen temizle
    _collection.delete(where={})

    embeddings = ollama_embed(docs)

    _collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings
    )
    print("CHROMA COUNT AFTER LOAD:", _collection.count())


def rag_query_with_scores(question: str, top_k: int = 5) -> Dict:
    query_embedding = ollama_embed([question])[0]

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

