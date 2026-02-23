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
from .firebase_kb import init_firebase, fetch_kb_items, fetch_kb_version
import requests
from chromadb import PersistentClient

from .config import settings

META_DOC_ID = "__kb_meta__"

def get_indexed_kb_version() -> str | None:
    try:
        res = _collection.get(ids=[META_DOC_ID], include=["metadatas"])
        metas = res.get("metadatas", [])
        if metas and metas[0]:
            return metas[0].get("kbVersion")
    except Exception:
        pass
    return None

def set_indexed_kb_version(kb_version: str):
    # small meta doc so we can compare versions on restart
    _collection.upsert(
        ids=[META_DOC_ID],
        documents=["kb meta"],
        metadatas=[{"kbVersion": kb_version}],
        embeddings=ollama_embed(["kb meta"])
    )

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
def load_kb_to_chroma(service_account_path: str | None = None):
    global _KB_BY_ID, _TOPIC_IDS, _TOPIC_TEXTS, _TOPIC_EMBS

    # 0) Firebase init (once)
    init_firebase(service_account_path)

    # 1) Firestore'dan ham KB çek
    kb_version = fetch_kb_version()
    kb_items = fetch_kb_items()

    if not kb_items:
        raise RuntimeError("Firestore KB is empty (chatbot_kb_items).")

    ids = [item["id"] for item in kb_items]
    docs = [item["content"] for item in kb_items]

    # 2) In-memory indexler (restart bug'ı burada biter)
    _KB_BY_ID = {i: d for i, d in zip(ids, docs)}
    _TOPIC_IDS = ids

    # Topic text: id + snippet (routing güçlensin)
    _TOPIC_TEXTS = [
        f"{id_to_topic_text(i)}. {(_KB_BY_ID[i][:200]).strip()}"
        for i in ids
    ]
    _TOPIC_EMBS = ollama_embed(_TOPIC_TEXTS)

    # 3) Chroma version check
    indexed_version = get_indexed_kb_version()
    try:
        chroma_has_docs = _collection.count() > 0
    except Exception:
        chroma_has_docs = False

    if chroma_has_docs and indexed_version == kb_version:
        print(f"Chroma up-to-date. kbVersion={kb_version}. Skipping doc embedding build.")
        return

    # 4) Rebuild Chroma (simple + safe)
    print(f"Rebuilding Chroma. Firestore kbVersion={kb_version}, indexed={indexed_version}, chroma_has_docs={chroma_has_docs}")

    try:
        existing = _collection.get(include=[])
        if existing.get("ids"):
            _collection.delete(ids=existing["ids"])
    except Exception:
        pass

    metas = [{"id": item_id, "topic": id_to_topic_text(item_id)} for item_id in ids]
    doc_embeddings = ollama_embed(docs)

    _collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=doc_embeddings
    )

    set_indexed_kb_version(kb_version)
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

# rag.py içine ekle

import threading
import time
from .firebase_kb import fetch_kb_fingerprint

_last_fp = None
_watcher_started = False

def start_kb_watcher(interval_sec: int = 600):
    global _watcher_started
    if _watcher_started:
        return
    _watcher_started = True

    def _loop():
        global _last_fp
        while True:
            try:
                fp = fetch_kb_fingerprint()  # (count, max_update_time_iso)

                # ilk kez
                if _last_fp is None:
                    _last_fp = fp
                else:
                    if fp != _last_fp:
                        print(f"[KB WATCHER] Change detected. old={_last_fp}, new={fp}. Rebuilding...")
                        load_kb_to_chroma()   # zaten init_firebase içeride
                        _last_fp = fp
                    else:
                        print(f"[KB WATCHER] No change. fp={fp}")

            except Exception as e:
                print(f"[KB WATCHER] Error: {e}")

            time.sleep(interval_sec)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()