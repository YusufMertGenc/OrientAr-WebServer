from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    message: str
    confidence: float
    context_used: List[str]
