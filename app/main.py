from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse
from .rag import load_kb_to_chroma, rag_query_with_scores
from .llm_client import generate_intent_response


app = FastAPI(title="OrientAR Chatbot API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_kb_to_chroma()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chatbot/query", response_model=ChatResponse)
def chatbot_query(req: ChatRequest):
    try:
        rag = rag_query_with_scores(req.question, top_k=5)
        documents = rag["documents"]

        llm_json = generate_intent_response(req.question, documents)

        return ChatResponse(
            message=llm_json["message"],
            confidence=llm_json["confidence"],
            context_used=documents
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

