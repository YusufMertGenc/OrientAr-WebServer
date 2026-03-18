import time
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse
from .rag import rag_query, load_kb_to_chroma, start_kb_watcher
from .llm_client import (
    generate_intent_response,
    match_predefined_response,
    OUT_OF_DOMAIN_MESSAGE,
)

app = FastAPI(title="OrientAR Chatbot API", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print("[UNHANDLED ERROR]", repr(exc))
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.on_event("startup")
def startup():
    load_kb_to_chroma()
    start_kb_watcher(interval_sec=600)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chatbot/query", response_model=ChatResponse)
def chatbot_query(req: ChatRequest):
    started = time.time()
    question = req.question.strip()

    predefined = match_predefined_response(question)
    if predefined:
        latency_ms = int((time.time() - started) * 1000)
        return ChatResponse(
            answer=predefined["message"],
            confidence=float(predefined["confidence"]),
            context_used=[],
            latency_ms=latency_ms,
            domain_score=1.0,
            in_domain=True,
        )

    rag_result = rag_query(question, top_k=4)
    documents = rag_result.get("documents", [])
    domain_score = float(rag_result.get("domain_score", 0.0))
    in_domain = bool(rag_result.get("in_domain", False))

    if not in_domain:
        latency_ms = int((time.time() - started) * 1000)
        return ChatResponse(
            answer=OUT_OF_DOMAIN_MESSAGE,
            confidence=0.95,
            context_used=[],
            latency_ms=latency_ms,
            domain_score=domain_score,
            in_domain=False,
        )

    if not documents:
        latency_ms = int((time.time() - started) * 1000)
        return ChatResponse(
            answer="I’m not sure based on the available campus information.",
            confidence=0.25,
            context_used=[],
            latency_ms=latency_ms,
            domain_score=domain_score,
            in_domain=True,
        )

    llm_json = generate_intent_response(question, documents)
    latency_ms = int((time.time() - started) * 1000)

    answer = str(llm_json.get("message", "")).strip()
    confidence = float(llm_json.get("confidence", 0.5))

    if not answer:
        answer = "I’m not sure based on the available campus information."
        confidence = 0.25

    return ChatResponse(
        answer=answer,
        confidence=confidence,
        context_used=documents,
        latency_ms=latency_ms,
        domain_score=domain_score,
        in_domain=True,
    )