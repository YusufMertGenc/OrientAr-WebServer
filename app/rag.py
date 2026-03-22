from __future__ import annotations

import asyncio
import json
import logging
import hashlib
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import httpx
import requests
from cachetools import TTLCache
from chromadb import PersistentClient

from .config import settings
from .firebase_kb import (
    init_firebase,
    fetch_kb_items,
    fetch_kb_version,
    fetch_kb_fingerprint,
)

logger = logging.getLogger("orientar")

# -------------------- HTTP / caches --------------------

_embed_http_client: Optional[httpx.AsyncClient] = None


# -------------------- Constants / Tunables --------------------

EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "2500"))
EMBED_RETRY_CHARS = int(os.getenv("EMBED_RETRY_CHARS", "1200"))
TOPIC_SNIPPET_CHARS = int(os.getenv("TOPIC_SNIPPET_CHARS", "250"))

TOPIC_TOP_N = int(os.getenv("TOPIC_TOP_N", "4"))
TOPIC_DOMAIN_GUARD = float(os.getenv("TOPIC_DOMAIN_GUARD", "0.38"))

DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "5"))
FINAL_DOCS = int(os.getenv("FINAL_DOCS", "3"))

EMBED_TIMEOUT = int(os.getenv("EMBED_TIMEOUT", "20"))

QUERY_DOC_MAX_CHARS = int(os.getenv("QUERY_DOC_MAX_CHARS", "1400"))

COLLECTION_NAME = "campus_kb"

# -------------------- Chroma --------------------

_chroma_client = PersistentClient(path=settings.chroma_path)
_collection = None

_INDEX_STATE_PATH = Path(settings.chroma_path) / "kb_index_state.json"


def get_collection():
    global _collection
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def recreate_collection():
    global _collection
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
        print("[CHROMA] old collection deleted", flush=True)
    except Exception as e:
        print(f"[CHROMA] delete_collection warning: {e}", flush=True)

    _collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)
    print("[CHROMA] fresh collection ready", flush=True)
    return _collection


def get_local_indexed_kb_version() -> Optional[str]:
    try:
        if not _INDEX_STATE_PATH.exists():
            return None
        data = json.loads(_INDEX_STATE_PATH.read_text(encoding="utf-8"))
        return data.get("kbVersion")
    except Exception as e:
        print(f"[RAG] local state read failed: {e}", flush=True)
        return None


def set_local_indexed_kb_version(kb_version: str):
    try:
        _INDEX_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _INDEX_STATE_PATH.write_text(
            json.dumps({"kbVersion": kb_version}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[RAG] local index state updated: {kb_version}", flush=True)
    except Exception as e:
        print(f"[RAG] local state write failed: {e}", flush=True)


# -------------------- In-memory caches/indexes --------------------

_EMBED_CACHE: Dict[str, List[float]] = {}
_KB_BY_ID: Dict[str, Dict] = {}

_TOPIC_IDS: List[str] = []
_TOPIC_TEXTS: List[str] = []
_TOPIC_EMBS: List[List[float]] = []

_REBUILDING = False

# -------------------- Utils --------------------


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _norm_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _clamp_for_embed(text: str, max_chars: int = EMBED_MAX_CHARS) -> str:
    t = _norm_ws(text)
    if len(t) <= max_chars:
        return t
    return t[:max_chars]


def _clamp_for_query(text: str, max_chars: int = QUERY_DOC_MAX_CHARS) -> str:
    t = _norm_ws(text)
    if len(t) <= max_chars:
        return t
    return t[:max_chars]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _url_path_hint(url: str) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        p = urlparse(url).path.strip("/")
        p = p.replace("-", " ").replace("/", " / ")
        return _norm_ws(p)
    except Exception:
        return ""


def _sanitize_meta(meta: Dict) -> Dict:
    clean: Dict = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def _make_response_cache_key(question: str, top_k: int) -> str:
    return hashlib.sha256(f"{question.strip().lower()}|{top_k}".encode()).hexdigest()

# -------------------- Async embedding client --------------------


async def get_embed_http_client() -> httpx.AsyncClient:
    global _embed_http_client
    if _embed_http_client is None:
        _embed_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=10.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
    return _embed_http_client


async def _ollama_embed_one_async(prompt: str) -> List[float]:
    client = await get_embed_http_client()

    resp = await client.post(
        f"{settings.embedding_base_url}/api/embeddings",
        json={"model": settings.embedding_model, "prompt": prompt},
    )
    if resp.status_code == 200:
        data = resp.json()
        emb = data.get("embedding")
        if emb:
            return emb
        raise RuntimeError(f"Empty embedding from Ollama ({settings.embedding_model})")

    prompt2 = prompt[:EMBED_RETRY_CHARS]
    resp2 = await client.post(
        f"{settings.embedding_base_url}/api/embeddings",
        json={"model": settings.embedding_model, "prompt": prompt2},
    )
    if resp2.status_code == 200:
        data2 = resp2.json()
        emb2 = data2.get("embedding")
        if emb2:
            return emb2
        raise RuntimeError(f"Empty embedding (retry) from Ollama ({settings.embedding_model})")

    raise RuntimeError(f"Embedding request failed: {resp.text}")


async def ollama_embed_async(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []

    for raw in texts:
        t = _clamp_for_embed(raw)
        if not t:
            t = "empty"

        key = _md5(t)
        if key in _EMBED_CACHE:
            out.append(_EMBED_CACHE[key])
            continue

        emb = await _ollama_embed_one_async(t)
        _EMBED_CACHE[key] = emb
        out.append(emb)

    return out

# -------------------- Sync embedding --------------------


def _ollama_embed_one(prompt: str) -> List[float]:
    resp = requests.post(
        f"{settings.embedding_base_url}/api/embeddings",
        json={"model": settings.embedding_model, "prompt": prompt},
        timeout=EMBED_TIMEOUT,
    )
    if resp.status_code == 200:
        data = resp.json()
        emb = data.get("embedding")
        if emb:
            return emb
        raise RuntimeError(f"Empty embedding from Ollama ({settings.embedding_model})")

    prompt2 = prompt[:EMBED_RETRY_CHARS]
    resp2 = requests.post(
        f"{settings.embedding_base_url}/api/embeddings",
        json={"model": settings.embedding_model, "prompt": prompt2},
        timeout=EMBED_TIMEOUT,
    )
    if resp2.status_code == 200:
        data2 = resp2.json()
        emb2 = data2.get("embedding")
        if emb2:
            return emb2
        raise RuntimeError(f"Empty embedding (retry) from Ollama ({settings.embedding_model})")

    raise RuntimeError(f"Embedding request failed: {resp.text}")


def ollama_embed(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    total = len(texts)

    for i, raw in enumerate(texts, start=1):
        t = _clamp_for_embed(raw)

        if not t:
            t = "empty"

        key = _md5(t)
        if key in _EMBED_CACHE:
            out.append(_EMBED_CACHE[key])
            if i % 10 == 0 or i == total:
                print(f"[EMBED] cache progress {i}/{total}", flush=True)
            continue

        emb = _ollama_embed_one(t)
        _EMBED_CACHE[key] = emb
        out.append(emb)

        if i % 10 == 0 or i == total:
            print(f"[EMBED] progress {i}/{total}", flush=True)

    return out

# -------------------- Topic text builder --------------------


def _topic_text_from_item(item: Dict) -> str:
    doc_id = _norm_ws(item.get("id", ""))
    title = _norm_ws(item.get("title", ""))
    section = _norm_ws(item.get("section", ""))
    lang = _norm_ws(item.get("lang", ""))
    url = _norm_ws(item.get("url", ""))
    source = _norm_ws(item.get("source", ""))

    content = item.get("content", "") or ""
    snippet = _norm_ws(content)[:TOPIC_SNIPPET_CHARS]

    url_hint = _url_path_hint(url)
    id_hint = _norm_ws(doc_id.replace("_", " ").replace("__", " "))

    parts: List[str] = []
    if title:
        parts.append(title)
    if section:
        parts.append(section)
    if url_hint:
        parts.append(url_hint)
    if lang:
        parts.append(f"lang:{lang}")
    if source:
        parts.append(f"src:{source}")
    if not parts and id_hint:
        parts.append(id_hint)
    if not parts:
        parts.append("kb item")

    base = ". ".join(parts)
    topic = f"{base}. {snippet}"
    return _clamp_for_embed(topic, max_chars=EMBED_MAX_CHARS)

# -------------------- Load KB & Build Indexes --------------------


def load_kb_to_chroma(service_account_path: str | None = None):
    global _KB_BY_ID, _TOPIC_IDS, _TOPIC_TEXTS, _TOPIC_EMBS, _REBUILDING

    print("STEP 1: load_kb_to_chroma started", flush=True)
    _REBUILDING = True
    try:
        print("STEP 2: init_firebase starting", flush=True)
        init_firebase(service_account_path)
        print("STEP 3: init_firebase done", flush=True)

        print("STEP 4: fetch_kb_version starting", flush=True)
        kb_version = fetch_kb_version()
        print("STEP 5: fetch_kb_version done", flush=True)

        print("STEP 6: fetch_kb_fingerprint starting", flush=True)
        fp_count, fp_max_ut = fetch_kb_fingerprint()
        print("STEP 7: fetch_kb_fingerprint done", flush=True)

        kb_version_effective = f"{kb_version}|count={fp_count}|ut={fp_max_ut}"

        print("STEP 8: fetch_kb_items starting", flush=True)
        kb_items = fetch_kb_items()
        print(f"STEP 9: fetch_kb_items done, item_count={len(kb_items) if kb_items else 0}", flush=True)

        if not kb_items:
            raise RuntimeError("Firestore KB is empty (chatbot_kb_items).")

        print("STEP 10: normalizing items", flush=True)
        normalized: List[Dict] = []
        for it in kb_items:
            if it.get("isDeleted") is True:
                continue
            doc_id = it.get("id")
            content = it.get("content", "")
            if not doc_id or not (content and str(content).strip()):
                continue
            normalized.append(it)

        print(f"STEP 11: normalization done, normalized_count={len(normalized)}", flush=True)

        if not normalized:
            raise RuntimeError("Firestore KB has no active (non-deleted) items.")

        _KB_BY_ID = {it["id"]: it for it in normalized}

        _TOPIC_IDS = [it["id"] for it in normalized]
        _TOPIC_TEXTS = [_topic_text_from_item(it) for it in normalized]

        print(f"STEP 12: topic texts prepared, topic_count={len(_TOPIC_TEXTS)}", flush=True)
        print("STEP 13: topic embedding start", flush=True)
        _TOPIC_EMBS = ollama_embed(_TOPIC_TEXTS)
        print("STEP 14: topic embedding done", flush=True)

        print("STEP 15: checking local indexed version", flush=True)
        indexed_version = get_local_indexed_kb_version()
        print(
            f"STEP 16: local version check done, indexed_version={indexed_version}, kb_version_effective={kb_version_effective}",
            flush=True,
        )

        if indexed_version == kb_version_effective:
            print(f"[RAG] Local state says index is up-to-date. kbVersion={kb_version_effective}", flush=True)
            print("STEP 17: early exit because local state is up-to-date", flush=True)
            return

        print(
            f"[RAG] Rebuilding Chroma. Firestore kbVersion={kb_version_effective}, indexed={indexed_version}",
            flush=True,
        )

        print("STEP 18: recreating collection", flush=True)
        collection = recreate_collection()
        print("STEP 19: fresh collection ready", flush=True)

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict] = []

        print("STEP 20: preparing docs/metas for Chroma", flush=True)
        for it in normalized:
            doc_id = str(it.get("id", "")).strip()
            if not doc_id:
                continue

            content = str(it.get("content", "") or "")
            doc_text = _clamp_for_embed(content, max_chars=EMBED_MAX_CHARS)
            if not doc_text:
                continue

            raw_meta = {
                "id": doc_id,
                "title": str(it.get("title", "") or ""),
                "section": str(it.get("section", "") or ""),
                "url": str(it.get("url", "") or ""),
                "lang": str(it.get("lang", "") or ""),
                "source": str(it.get("source", "") or ""),
                "pageId": str(it.get("pageId", "") or ""),
                "chunkIndex": int(it.get("chunkIndex", -1) if it.get("chunkIndex", None) is not None else -1),
                "chunkCount": int(it.get("chunkCount", -1) if it.get("chunkCount", None) is not None else -1),
            }

            ids.append(doc_id)
            docs.append(doc_text)
            metas.append(_sanitize_meta(raw_meta))

        print(f"STEP 21: docs prepared, doc_count={len(docs)}", flush=True)

        if not ids:
            raise RuntimeError("[RAG] No valid docs to index after normalization.")

        print("STEP 22: doc embedding start", flush=True)
        doc_embeddings = ollama_embed(docs)
        print("STEP 23: doc embedding done", flush=True)

        if not (len(ids) == len(docs) == len(metas) == len(doc_embeddings)):
            raise RuntimeError(
                f"[RAG] Alignment mismatch ids={len(ids)} docs={len(docs)} metas={len(metas)} embs={len(doc_embeddings)}"
            )

        print("STEP 24: adding docs to Chroma", flush=True)
        collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=doc_embeddings,
        )
        print("STEP 25: Chroma add done", flush=True)

        set_local_indexed_kb_version(kb_version_effective)

        print("[RAG] CHROMA COUNT AFTER LOAD:", collection.count(), flush=True)
        print("STEP 26: load_kb_to_chroma finished successfully", flush=True)

    except Exception as e:
        print(f"STEP ERROR: load_kb_to_chroma failed with error: {e}", flush=True)
        raise

    finally:
        _REBUILDING = False
        print("STEP FINAL: _REBUILDING set to False", flush=True)

# -------------------- Topic routing --------------------


def top_topics(q_emb: List[float], top_n: int = TOPIC_TOP_N) -> List[Tuple[str, float]]:
    if not _TOPIC_IDS or not _TOPIC_EMBS:
        return []

    scored: List[Tuple[str, float]] = []
    for topic_id, t_emb in zip(_TOPIC_IDS, _TOPIC_EMBS):
        scored.append((topic_id, cosine_similarity(q_emb, t_emb)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# -------------------- Reranking --------------------


def _lexical_overlap(query: str, doc: str) -> float:
    q_words = set(_norm_ws(query).lower().split())
    d_words = set(_norm_ws(doc).lower().split())
    if not q_words:
        return 0.0
    return len(q_words & d_words) / max(len(q_words), 1)


def rerank_documents(query: str, candidates: List[Tuple[str, float]]) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for doc_text, sem in candidates:
        lex = _lexical_overlap(query, doc_text)
        score = (0.75 * sem) + (0.25 * lex)
        scored.append((score, doc_text))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored[:FINAL_DOCS]]

# -------------------- Main RAG --------------------


async def rag_query_async(question: str, top_k: int = DENSE_TOP_K) -> Dict:
    

    if _REBUILDING:
        return {"documents": [], "domain_score": 0.0, "in_domain": False}

    t0 = time.perf_counter()
    q = question or ""

    t_embed0 = time.perf_counter()
    q_emb = (await ollama_embed_async([q]))[0]
    t_embed1 = time.perf_counter()

    t_topic0 = time.perf_counter()
    routed = await asyncio.to_thread(top_topics, q_emb, TOPIC_TOP_N)
    max_topic_score = max((s for _, s in routed), default=0.0)
    t_topic1 = time.perf_counter()

    if max_topic_score < TOPIC_DOMAIN_GUARD:
        result = {"documents": [], "domain_score": max_topic_score, "in_domain": False}
        return result

    routed_ids = [tid for tid, _ in routed]
    candidates: List[Tuple[str, float]] = []
    seen = set()

    for doc_id in routed_ids[:2]:
        it = _KB_BY_ID.get(doc_id)
        if not it:
            continue
        doc = it.get("content", "") or ""
        doc = _clamp_for_query(doc, max_chars=QUERY_DOC_MAX_CHARS)
        if doc and doc not in seen:
            seen.add(doc)
            candidates.append((doc, 0.85))

    t_dense0 = time.perf_counter()
    try:
        collection = get_collection()
        res = await asyncio.to_thread(
            collection.query,
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, dist in zip(docs, dists):
            if not doc or doc in seen:
                continue
            doc2 = _clamp_for_query(doc, max_chars=QUERY_DOC_MAX_CHARS)
            if not doc2:
                continue
            seen.add(doc2)
            sem_sim = 1.0 / (1.0 + float(dist))
            candidates.append((doc2, sem_sim))
    except Exception as e:
        logger.warning(f"[RAG] Dense query failed: {e}")
    t_dense1 = time.perf_counter()

    if not candidates:
        result = {"documents": [], "domain_score": max_topic_score, "in_domain": True}
        
        return result

    t_rerank0 = time.perf_counter()
    best_docs = await asyncio.to_thread(rerank_documents, q, candidates)
    t_rerank1 = time.perf_counter()

    total = time.perf_counter() - t0

    logger.info(
        "[RAG_TIMING] "
        f"embed={t_embed1 - t_embed0:.2f}s "
        f"topic={t_topic1 - t_topic0:.2f}s "
        f"dense={t_dense1 - t_dense0:.2f}s "
        f"rerank={t_rerank1 - t_rerank0:.2f}s "
        f"total={total:.2f}s"
    )

    result = {
        "documents": best_docs,
        "domain_score": max_topic_score,
        "in_domain": True,
    }
    return result


def rag_query(question: str, top_k: int = DENSE_TOP_K) -> Dict:
    if _REBUILDING:
        return {"documents": [], "domain_score": 0.0, "in_domain": False}

    q = question or ""
    q_emb = ollama_embed([q])[0]

    routed = top_topics(q_emb, top_n=TOPIC_TOP_N)
    max_topic_score = max((s for _, s in routed), default=0.0)

    if max_topic_score < TOPIC_DOMAIN_GUARD:
        return {"documents": [], "domain_score": max_topic_score, "in_domain": False}

    routed_ids = [tid for tid, _ in routed]
    candidates: List[Tuple[str, float]] = []
    seen = set()

    for doc_id in routed_ids[:2]:
        it = _KB_BY_ID.get(doc_id)
        if not it:
            continue
        doc = it.get("content", "") or ""
        doc = _clamp_for_query(doc, max_chars=QUERY_DOC_MAX_CHARS)
        if doc and doc not in seen:
            seen.add(doc)
            candidates.append((doc, 0.85))

    try:
        collection = get_collection()
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, dist in zip(docs, dists):
            if not doc or doc in seen:
                continue
            doc2 = _clamp_for_query(doc, max_chars=QUERY_DOC_MAX_CHARS)
            if not doc2:
                continue
            seen.add(doc2)
            sem_sim = 1.0 / (1.0 + float(dist))
            candidates.append((doc2, sem_sim))
    except Exception as e:
        print(f"[RAG] Dense query failed: {e}", flush=True)

    if not candidates:
        return {"documents": [], "domain_score": max_topic_score, "in_domain": True}

    best_docs = rerank_documents(q, candidates)
    return {
        "documents": best_docs,
        "domain_score": max_topic_score,
        "in_domain": True,
    }

# -------------------- KB Watcher --------------------

_last_fp = None
_watcher_started = False
_reload_lock = None


def start_kb_watcher(interval_sec: int = 600):
    global _watcher_started, _last_fp, _reload_lock
    if _watcher_started:
        return
    _watcher_started = True

    import threading
    _reload_lock = threading.Lock()

    def _loop():
        global _last_fp
        while True:
            try:
                init_firebase(None)
                fp = fetch_kb_fingerprint()

                if _last_fp is None:
                    _last_fp = fp
                    print(f"[KB WATCHER] init fp={fp}", flush=True)
                else:
                    if fp != _last_fp:
                        if _reload_lock.acquire(blocking=False):
                            try:
                                print(f"[KB WATCHER] Change detected old={_last_fp}, new={fp}. Rebuilding...", flush=True)
                                load_kb_to_chroma()
                                _last_fp = fp
                            finally:
                                _reload_lock.release()
                        else:
                            print("[KB WATCHER] rebuild already running, skip", flush=True)
                    else:
                        print(f"[KB WATCHER] No change fp={fp}", flush=True)

            except Exception as e:
                print(f"[KB WATCHER] Error: {e}", flush=True)

            time.sleep(interval_sec)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


def is_in_domain(question: str) -> Tuple[bool, float]:
    if _REBUILDING:
        return False, 0.0

    q = (question or "").strip()
    if not q:
        return False, 0.0

    q_emb = ollama_embed([q])[0]
    routed = top_topics(q_emb, top_n=TOPIC_TOP_N)
    max_topic_score = max((s for _, s in routed), default=0.0)

    return max_topic_score >= TOPIC_DOMAIN_GUARD, max_topic_score