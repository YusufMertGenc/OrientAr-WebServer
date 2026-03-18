from pydantic import BaseModel, Field
from typing import List


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    context_used: List[str]
    latency_ms: int
    domain_score: float = 0.0
    in_domain: bool = True