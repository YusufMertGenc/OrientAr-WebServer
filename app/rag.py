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

    # ---- CORE DATA ----
    ids = [item["id"] for item in kb_items]
    docs = [item["content"] for item in kb_items]

    # ---- METADATA (OPTIONAL, SAFE) ----
    # Metadata retrieval için kullanılmıyor.
    # Sadece debug / trace amaçlı.
    metas = [{"id": item["id"]} for item in kb_items]

    # ---- CLEAN EXISTING COLLECTION ----
    try:
        existing = _collection.get(include=[])
        if existing["ids"]:
            _collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # ---- EMBEDDINGS ----
    embeddings = ollama_embed(docs)

    _collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings
    )

    print("CHROMA COUNT AFTER LOAD:", _collection.count())


def rag_query(question: str, top_k: int = 5, max_distance: float = 1.0) -> Dict:
    
    query_embedding = ollama_embed([question])[0]

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )

    docs = results["documents"][0] if results.get("documents") else []
    dists = results["distances"][0] if results.get("distances") else []

    if not docs:
        return {
            "documents": [],
            "reason": "no_retrieval"
        }

    # 🔥 Distance filter (asıl RAG farkı)
    filtered_docs = [
        doc for doc, dist in zip(docs, dists)
        if dist <= max_distance
    ]

    if not filtered_docs:
        return {
            "documents": [],
            "reason": "low_relevance"
        }

    return {
        "documents": filtered_docs,
        "reason": "ok"
    }
