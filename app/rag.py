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
    q_emb: List[float],
    top_n: int = 7
) -> List[Tuple[str, float]]:
    # If topic index is not initialized, routing cannot be performed
    if not _TOPIC_IDS or not _TOPIC_EMBS:
        return []

    scored: List[Tuple[str, float]] = []

    # Compare question embedding with topic embeddings
    for topic_id, t_emb in zip(_TOPIC_IDS, _TOPIC_EMBS):
        sim = cosine_similarity(q_emb, t_emb)
        scored.append((topic_id, sim))

    # Sort by similarity score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return (topic_id, similarity_score)
    return scored[:top_n]



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

    if not _data_path.exists():
        raise FileNotFoundError("campus_kb.json not found")

    # 1) JSON'u her zaman oku (restart'ta da)
    with open(_data_path, "r", encoding="utf-8") as f:
        kb_items = json.load(f)

    ids = [item["id"] for item in kb_items]
    docs = [item["content"] for item in kb_items]

    _KB_BY_ID = {i: d for i, d in zip(ids, docs)}
    _TOPIC_IDS = ids
    # 2) Topic text'i güçlendir (sadece id değil, content'in ilk kısmı da)
    _TOPIC_TEXTS = [
        f"{id_to_topic_text(i)}. {(_KB_BY_ID[i][:200]).strip()}"
        for i in ids
    ]
    _TOPIC_EMBS = ollama_embed(_TOPIC_TEXTS)

    # 3) Chroma doluysa sadece index yükledik, çık
    try:
        if _collection.count() > 0:
            print("Chroma already populated -> skipping doc embedding build, using existing store.")
            return
    except Exception:
        pass

    # 4) İlk kez dolduruluyorsa embeddings üret ve add et
    metas = [{"id": item_id, "topic": id_to_topic_text(item_id)} for item_id in ids]
    doc_embeddings = ollama_embed(docs)

    _collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=doc_embeddings)
    print("CHROMA COUNT AFTER LOAD:", _collection.count())



# ---------------- Main RAG ----------------
def rag_query(question: str, top_k: int = 10) -> Dict:
    """
    Improved KB-driven RAG:
    1) Topic routing (semantic, ID-as-topic)
    2) Candidate pool = routed docs + global dense retrieval
    3) Hybrid rerank using Chroma distance + lexical overlap
    """

    # 1) QUESTION EMBEDDING (once)
    q_emb = ollama_embed([question])[0]

    # 2) Topic routing
    routed_topics = top_topics(q_emb, top_n=5)
    max_topic_score = max((score for _, score in routed_topics), default=0.0)

    # Domain relevance guard (same as before)
    if max_topic_score < 0.35:
        return {
            "documents": [],
            "domain_score": max_topic_score
        }

    routed_ids = [topic_id for topic_id, _ in routed_topics]

    candidate_docs: List[str] = []
    seen = set()

    # 3a) Routed-topic docs (high precision)
    for doc_id in routed_ids:
        doc = _KB_BY_ID.get(doc_id)
        if doc and doc not in seen:
            seen.add(doc)
            candidate_docs.append(doc)

    # 3b) Global dense fallback (with distance)
    dense_docs = []
    try:
        results = _collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "distances"]
        )
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, dist in zip(docs, dists):
            if doc and doc not in seen:
                dense_docs.append((doc, dist))
                seen.add(doc)
    except Exception:
        pass

    # Nothing found at all
    if not candidate_docs and not dense_docs:
        return {
            "documents": [],
            "domain_score": max_topic_score
        }

    # 4) Hybrid rerank (semantic distance + keyword overlap)
    scored: List[Tuple[float, str]] = []

    q_words = set(question.lower().split())

    # Routed docs: boost slightly (they passed topic routing)
    for doc in candidate_docs:
        d_words = set(doc.lower().split())
        overlap = len(q_words & d_words) / max(len(q_words), 1)
        score = 0.6 + (0.4 * overlap)   # base boost
        scored.append((score, doc))

    # Dense docs: use distance → similarity
    for doc, dist in dense_docs:
        d_words = set(doc.lower().split())
        overlap = len(q_words & d_words) / max(len(q_words), 1)
        semantic_sim = 1.0 / (1.0 + dist)   # distance → similarity
        score = (0.7 * semantic_sim) + (0.3 * overlap)
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])

    best_docs = [doc for _, doc in scored[:3]]

    return {
        "documents": best_docs,
        "domain_score": max_topic_score
    }
