import json
from pathlib import Path
from typing import List, Dict

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from .config import settings

# Paths
_base_dir = Path(__file__).resolve().parent.parent
_data_path = _base_dir / "data" / "campus_kb.json"

# Chroma
_chroma_client = PersistentClient(path=settings.chroma_dir)
_collection = _chroma_client.get_or_create_collection("campus_kb")

# Embedder
_embedder = SentenceTransformer(settings.embedding_model)


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

    try:
        _collection.delete(where={})
    except:
        pass

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
