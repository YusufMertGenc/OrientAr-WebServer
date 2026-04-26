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

    answer = final_answer_cleanup(str(llm_json.get("message", "")))
    confidence = float(llm_json.get("confidence", 0.5))

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


def final_answer_cleanup(text: str) -> str:
    s = (text or "").strip()
    s = s.strip(' "\'')

    if '\\"message\\"' in s or s.startswith('{\\'):
        try:
            s2 = s.replace('\\"', '"')
            obj = json.loads(s2)
            if isinstance(obj, dict) and "message" in obj:
                s = str(obj["message"]).strip()
        except Exception:
            pass

    if s.startswith("{") and '"message"' in s:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "message" in obj:
                s = str(obj["message"]).strip()
        except Exception:
            m = re.search(r'"message"\s*:\s*"(.+?)"', s)
            if m:
                s = m.group(1).strip()

    s = re.sub(r"\s+", " ", s).strip()
    s = s.lstrip('"').strip()
    return s