import json
import re
import time
import uuid
import traceback
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse
from .rag import rag_query_async, load_kb_to_chroma, start_kb_watcher
from .llm_client import (
    generate_intent_response,
    match_predefined_response,
    OUT_OF_DOMAIN_MESSAGE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orientar")

active_requests = 0
active_requests_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] Loading KB to Chroma...")
    load_kb_to_chroma()
    start_kb_watcher(interval_sec=600)
    logger.info("[STARTUP] App is ready.")
    yield
    logger.info("[SHUTDOWN] App shutting down...")


app = FastAPI(
    title="OrientAR Chatbot API",
    version="0.7.3",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global active_requests
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    async with active_requests_lock:
        active_requests += 1
        current_active = active_requests

    logger.info(
        f"[REQ_START] id={request_id} path={request.url.path} "
        f"method={request.method} active={current_active}"
    )

    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.perf_counter() - start

        async with active_requests_lock:
            active_requests -= 1
            current_active = active_requests

        logger.info(
            f"[REQ_END] id={request_id} path={request.url.path} "
            f"duration={duration:.2f}s active={current_active}"
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("[UNHANDLED ERROR] %r", exc)
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "active_requests": active_requests}


@app.post("/chatbot/query", response_model=ChatResponse)
async def chatbot_query(req: ChatRequest):
    started = time.perf_counter()
    question = req.question.strip()

    if not question:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ChatResponse(
            answer="Please enter a campus-related question.",
            confidence=0.0,
            context_used=[],
            latency_ms=latency_ms,
            domain_score=0.0,
            in_domain=False,
        )

    predefined = match_predefined_response(question)
    if predefined:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ChatResponse(
            answer=predefined["message"],
            confidence=float(predefined["confidence"]),
            context_used=[],
            latency_ms=latency_ms,
            domain_score=1.0,
            in_domain=True,
        )

    rag_started = time.perf_counter()
    rag_result = await rag_query_async(question, top_k=4)
    rag_duration = time.perf_counter() - rag_started

    documents = rag_result.get("documents", [])
    domain_score = float(rag_result.get("domain_score", 0.0))
    in_domain = bool(rag_result.get("in_domain", False))

    logger.info(
        f"[RAG_RESULT] in_domain={in_domain} domain_score={domain_score:.3f} "
        f"doc_count={len(documents)} rag_time={rag_duration:.2f}s"
    )

    if not in_domain:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ChatResponse(
            answer=OUT_OF_DOMAIN_MESSAGE,
            confidence=0.95,
            context_used=[],
            latency_ms=latency_ms,
            domain_score=domain_score,
            in_domain=False,
        )

    if not documents:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ChatResponse(
            answer="I’m not sure based on the available campus information.",
            confidence=0.25,
            context_used=[],
            latency_ms=latency_ms,
            domain_score=domain_score,
            in_domain=True,
        )

    llm_started = time.perf_counter()
    llm_json = await generate_intent_response(question, documents)
    llm_duration = time.perf_counter() - llm_started

    latency_ms = int((time.perf_counter() - started) * 1000)

    answer = extract_answer_from_llm_result(llm_json)
    confidence = safe_float(
        llm_json.get("confidence", 0.5) if isinstance(llm_json, dict) else 0.5,
        default=0.5
    )
    if not answer:
        answer = "I’m not sure based on the available campus information."
        confidence = 0.25

    logger.info(
        f"[CHAT_DONE] total={latency_ms}ms llm_time={llm_duration:.2f}s "
        f"rag_time={rag_duration:.2f}s confidence={confidence:.2f}"
    )

    return ChatResponse(
        answer=answer,
        confidence=confidence,
        context_used=documents,
        latency_ms=latency_ms,
        domain_score=domain_score,
        in_domain=True,
    )


def safe_float(value, default: float = 0.5) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_float(value, default: float = 0.5) -> float:
    try:
        return float(value)
    except Exception:
        return default


def extract_answer_from_llm_result(llm_result) -> str:
    """
    generate_intent_response bazen şöyle dönebiliyor:

    1) {"message": "normal answer", "confidence": 0.8}
    2) {"message": "{\"message\": \"actual answer\", \"confidence\": 0.3}"}
    3) "{\"message\": \"actual answer\", \"confidence\": 0.3}"
    4) { \"message\": \"actual answer\", ... } gibi bozuk/escaped JSON string

    Bu fonksiyon hepsinden sadece gerçek answer metnini çıkarır.
    """
    if llm_result is None:
        return ""

    if isinstance(llm_result, dict):
        for key in ["message", "answer", "response", "text"]:
            value = llm_result.get(key)
            if value:
                cleaned = final_answer_cleanup(value)
                if cleaned:
                    return cleaned
        return ""

    return final_answer_cleanup(llm_result)


def final_answer_cleanup(raw_answer) -> str:
    if raw_answer is None:
        return ""

    s = str(raw_answer).strip()
    if not s:
        return ""

    # Markdown code block temizliği
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s).strip()

    # Dış tırnak temizliği
    s = s.strip()

    # 1) Normal JSON parse
    parsed = try_parse_json_answer(s)
    if parsed:
        return parsed

    # 2) Eğer JSON string olarak encode edilmişse birkaç kez çözmeyi dene
    current = s
    for _ in range(3):
        try:
            decoded = json.loads(current)
            if isinstance(decoded, str):
                current = decoded.strip()
                parsed = try_parse_json_answer(current)
                if parsed:
                    return parsed
            elif isinstance(decoded, dict):
                parsed = extract_answer_from_llm_result(decoded)
                if parsed:
                    return parsed
                break
            else:
                break
        except Exception:
            break

    # 3) Escaped quote temizliği: { \"message\": \"...\" }
    unescaped = (
        s.replace('\\"', '"')
         .replace("\\n", "\n")
         .replace("\\r", "\r")
         .replace("\\t", "\t")
         .strip()
    )

    parsed = try_parse_json_answer(unescaped)
    if parsed:
        return parsed

    # 4) JSON bozuk/truncated ise regex ile message içeriğini çıkar
    regex_answer = extract_answer_by_regex(unescaped)
    if regex_answer:
        return regex_answer

    regex_answer = extract_answer_by_regex(s)
    if regex_answer:
        return regex_answer

    # 5) Son fallback: baştaki JSON field adını elle kırp
    fallback = strip_leading_json_message_wrapper(unescaped)
    fallback = fallback.strip()
    fallback = fallback.strip(' "\'')
    fallback = re.sub(r"\s+", " ", fallback).strip()

    return fallback


def try_parse_json_answer(s: str) -> str:
    try:
        obj = json.loads(s)
    except Exception:
        return ""

    if isinstance(obj, dict):
        for key in ["message", "answer", "response", "text"]:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return final_answer_cleanup(value)

    if isinstance(obj, str) and obj.strip():
        return final_answer_cleanup(obj)

    return ""


def extract_answer_by_regex(s: str) -> str:
    """
    Geçerli JSON değilse bile şunları yakalar:
    { "message": "....", "confidence": 0.3 }
    { \"message\": \"....\", \"confidence\": 0.3 }
    Truncated durumda confidence yoksa bile message sonrası metni alır.
    """
    if not s:
        return ""

    patterns = [
        r'"(?:message|answer|response|text)"\s*:\s*"(.+?)"\s*,\s*"(?:confidence|context_used|latency_ms|domain_score|in_domain)"',
        r'"(?:message|answer|response|text)"\s*:\s*"(.+?)"\s*\}',
        r'"(?:message|answer|response|text)"\s*:\s*"(.+)$',
    ]

    for pattern in patterns:
        m = re.search(pattern, s, flags=re.DOTALL)
        if m:
            value = m.group(1)
            value = value.replace('\\"', '"')
            value = value.replace("\\n", " ")
            value = value.replace("\\r", " ")
            value = value.replace("\\t", " ")
            value = re.sub(r'"\s*,\s*"confidence"\s*:\s*[\d.]+.*$', "", value, flags=re.DOTALL)
            value = re.sub(r"\s+", " ", value).strip()
            value = value.strip(' "\'{}')
            return value

    return ""


def strip_leading_json_message_wrapper(s: str) -> str:
    """
    Son çare:
    { "message": "bla bla
    gibi kalan wrapper'ı kırpar.
    """
    s = s.strip()

    s = re.sub(
        r'^\{\s*"?(?:message|answer|response|text)"?\s*:\s*"?',
        "",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )

    s = re.sub(
        r'"\s*,\s*"?confidence"?\s*:\s*[\d.]+.*$',
        "",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return s