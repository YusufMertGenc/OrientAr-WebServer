from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse
from .rag import rag_query, load_kb_to_chroma
from .llm_client import generate_intent_response


app = FastAPI(title="OrientAR Chatbot API", version="0.4.1")

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
        rag_result = rag_query(req.question, top_k=5)

        if rag_result["reason"] != "ok":
            return ChatResponse(
                message="I’m not sure based on the available information.",
                confidence=0.2,
                context_used=[]
            )

        documents = rag_result["documents"]
        llm_json = generate_intent_response(req.question, documents)

        return ChatResponse(
            message=llm_json["message"],
            confidence=llm_json["confidence"],
            context_used=documents
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
