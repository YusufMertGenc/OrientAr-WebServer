# app/main.py ✅ FINAL

import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse
from .rag import rag_query, load_kb_to_chroma, start_kb_watcher
from .llm_client import generate_intent_response


app = FastAPI(title="OrientAR Chatbot API", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Global exception handler (logda traceback gör + client'a kontrollü dön)
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
    # ilk yükleme
    load_kb_to_chroma()
    # watcher
    start_kb_watcher(interval_sec=600)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chatbot/query", response_model=ChatResponse)
def chatbot_query(req: ChatRequest):
    rag_result = rag_query(req.question, top_k=5)
    documents = rag_result.get("documents", [])

    if not documents:
        return ChatResponse(
            message="I’m not sure based on the available information.",
            confidence=0.2,
            context_used=[],
        )

    llm_json = generate_intent_response(req.question, documents)

    # Eğer llm client fallback ile "error" dönmüşse bile message var → 200 OK dön
    return ChatResponse(
        message=str(llm_json.get("message", "")).strip(),
        confidence=float(llm_json.get("confidence", 0.5)),
        context_used=documents,
    )