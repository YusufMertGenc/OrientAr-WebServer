# rag.py  ✅ KB-driven (ID-as-topic) RAG + topic routing + rerank (no keyword lists)

"""
RAG Module (KB-driven)

Implements a Retrieval-Augmented Generation pipeline that learns topics
directly from the Knowledge Base (KB) using document IDs as topics.

Flow:
- load_kb_to_chroma(): loads KB, builds topic and document embeddings (startup only)
- ollama_embed(): central embedding function with in-memory caching
- top_topics(): routes a question to the most relevant KB topics
- rerank_documents(): selects the most answer-relevant documents
- rag_query(): end-to-end RAG pipeline used by the API

No hard-coded keywords or entities are used.
"""


import json
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from chromadb import PersistentClient

from .config import settings


# ---- Paths ----
_base_dir = Path(__file__).resolve().parent.parent
_data_path = _base_dir / "data" / "campus_kb.json"


# ---- Chroma ----
_chroma_client = PersistentClient(path=settings.chroma_path)
_collection = _chroma_client.get_or_create_collection("campus_kb")


# ---- Caches / In-memory indexes ----
_EMBED_CACHE: Dict[str, List[float]] = {}

_KB_BY_ID: Dict[str, str] = {}          # doc_id -> content
_TOPIC_IDS: List[str] = []              # doc_id list
_TOPIC_TEXTS: List[str] = []            # "library working hours" style
_TOPIC_EMBS: List[List[float]] = []     # embeddings for topics


# ---------------- Embedding ----------------
def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
    # Hashing is used to create a deterministic cache key for each text.
    # This prevents recomputing embeddings for the same input.


def ollama_embed(texts: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []

    for text in texts:
        key = _hash_text(text)
        if key in _EMBED_CACHE:
            embeddings.append(_EMBED_CACHE[key])
            continue

        resp = requests.post(
            f"{settings.embedding_base_url}/api/embeddings",
            json={
                "model": settings.embedding_model,
                "prompt": text          # 🔥 DOĞRU ALAN BU
            },
            timeout=60
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Embedding request failed: {resp.text}")

        data = resp.json()
        if "embedding" not in data or not data["embedding"]:
            raise RuntimeError(f"Empty embedding returned from Ollama for model {settings.embedding_model}")

        emb = data["embedding"]
        _EMBED_CACHE[key] = emb
        embeddings.append(emb)

    return embeddings






# ---------------- Similarity ----------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def keyword_overlap_score(query: str, doc: str) -> float:
    q_words = set(query.lower().split())
    d_words = set(doc.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & d_words) / len(q_words)


# ---------------- Topic text from KB id ----------------
def id_to_topic_text(doc_id: str) -> str:
    # "library_working_hours" -> "library working hours"
    # also handles "metu_ncc_overview" -> "metu ncc overview"
    return doc_id.replace("_", " ").strip()


# ---------------- Topic routing ----------------
def top_topics(
    question: str,
    q_emb: List[float],
    top_n: int = 7
) -> List[str]:
    # If topic index is not initialized, routing cannot be performed
    if not _TOPIC_IDS or not _TOPIC_EMBS:
        return []

    scored: List[Tuple[float, str]] = []

    # Compare the already-computed question embedding with topic embeddings
    for topic_id, t_emb in zip(_TOPIC_IDS, _TOPIC_EMBS):
        scored.append((cosine_similarity(q_emb, t_emb), topic_id))

    # Sort topics by similarity score in descending order
    scored.sort(reverse=True, key=lambda x: x[0])

    # Return the IDs of the most relevant topics
    return [topic_id for _, topic_id in scored[:top_n]]


# ---------------- Reranker ----------------
def rerank_documents(
    query: str,
    query_embedding: List[float],
    docs: List[str],
    top_n: int = 3
) -> List[str]:
    if not docs:
        return []

    scored: List[Tuple[float, str]] = []

    # Precompute keyword set once
    q_words = set(query.lower().split())

    for doc in docs:
        # Lexical signal
        d_words = set(doc.lower().split())
        overlap = len(q_words & d_words) / max(len(q_words), 1)

        # Lightweight semantic proxy (length-normalized)
        semantic_hint = min(len(d_words) / 100.0, 1.0)

        # Hybrid score
        score = (0.6 * overlap) + (0.4 * semantic_hint)
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_n]]




# ---------------- Load KB ----------------
def load_kb_to_chroma():
    global _KB_BY_ID, _TOPIC_IDS, _TOPIC_TEXTS, _TOPIC_EMBS

    try:
        if _collection.count() > 0:
            print("Chroma already populated, skipping embedding.")
            return
    except Exception:
        pass


    if not _data_path.exists():
        raise FileNotFoundError("campus_kb.json not found")

    with open(_data_path, "r", encoding="utf-8") as f:
        kb_items = json.load(f)

    ids = [item["id"] for item in kb_items]
    docs = [item["content"] for item in kb_items]

    _KB_BY_ID = {i: d for i, d in zip(ids, docs)}

    # Build topic index from ids (KB-driven, no hardcoded keywords)
    _TOPIC_IDS = ids
    _TOPIC_TEXTS = [id_to_topic_text(i) for i in ids]
    _TOPIC_EMBS = ollama_embed(_TOPIC_TEXTS)

    # (Optional) keep metadata for trace/debug
    metas = [{"id": item_id, "topic": id_to_topic_text(item_id)} for item_id in ids]

    # Clean existing collection
    try:
        existing = _collection.get(include=[])
        if existing.get("ids"):
            _collection.delete(ids=existing["ids"])
    except Exception:
        pass
    doc_embeddings = ollama_embed(docs)
    # Embed documents once and add
    if not doc_embeddings or len(doc_embeddings) != len(docs):
        raise RuntimeError("Document embeddings could not be generated correctly")

    doc_embeddings = ollama_embed(docs)
    _collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=doc_embeddings
    )

    print("CHROMA COUNT AFTER LOAD:", _collection.count())


# ---------------- Main RAG ----------------
def rag_query(question: str, top_k: int = 8) -> Dict:
    """
    Enterprise-style flow (KB-driven):
    1) Topic routing using embeddings of KB IDs (self-describing topics)
    2) Candidate pool = topic-hit docs + a small global dense fallback
    3) Fast rerank -> top 3 docs to LLM
    """

    #  QUESTION EMBEDDING – ONLY ONCE
    q_emb = ollama_embed([question])[0]

    # 1) Topic routing (reuse q_emb)
    routed_ids = top_topics(question, q_emb, top_n=5)

    candidate_docs: List[str] = []
    seen = set()

    # 2a) Add docs from routed topics
    for doc_id in routed_ids:
        doc = _KB_BY_ID.get(doc_id)
        if doc and doc not in seen:
            seen.add(doc)
            candidate_docs.append(doc)

    # 2b) Small global dense fallback (reuse same q_emb)
    try:
        results = _collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents"]
        )
        dense_docs = results.get("documents", [[]])[0]
        for d in dense_docs:
            if d and d not in seen:
                seen.add(d)
                candidate_docs.append(d)
    except Exception:
        pass

    if not candidate_docs:
        return {"documents": []}

    # 3) Rerank and return top docs
    best_docs = rerank_documents(
        query=question,
        query_embedding=q_emb,
        docs=candidate_docs,
        top_n=3
    )
    if not best_docs:
        best_docs = candidate_docs[:3]

    return {"documents": best_docs}
