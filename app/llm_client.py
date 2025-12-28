"""
LLM Client

Responsible for generating the final answer using the selected RAG context.
Builds a constrained prompt, sends it to the LLM, and parses a JSON-only response.

Key points:
- Uses a strict system prompt to avoid hallucination
- Limits context size for performance
- Expects a structured JSON output (message + confidence)
"""

import json
import requests
from typing import List

from .config import settings


INTENT_SYSTEM_PROMPT = """
You are OrientAR, an assistant for a campus application.
Assume all questions are about METU NCC unless stated otherwise.

You may combine and reason over multiple pieces of the provided context
to produce a clear and helpful answer.
You MUST NOT use any knowledge that is not present in the context.

If the question is related to "How to" or "Help me",
provide step-by-step instructions ONLY if the context EXPLICITLY describes the steps.
If the context does not contain procedural steps, say you do not know.

Rules:
- Use ONLY the provided context.
- If the answer cannot be derived from the context, say you do not know.
- Do NOT make assumptions or add external information.
- Respond ONLY with valid JSON.

JSON FORMAT:
{
  "message": "<answer>",
  "confidence": <number between 0 and 1>
}

"""


def build_intent_prompt(question: str, context_passages: List[str]) -> str:
    # hard limit context size for speed
    trimmed = []
    total_chars = 0

    for p in context_passages:
        if total_chars > 1500:
            break
        trimmed.append(p)
        total_chars += len(p)

    context_text = "\n\n".join(trimmed) if trimmed else "No relevant campus info found."

    return f"""
CONTEXT:
{context_text}

QUESTION:
{question}
""".strip()


def generate_intent_response(question: str, context_passages: List[str]) -> dict:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_intent_prompt(question, context_passages)}
        ],
        "temperature": 0.2,
        "options": {
            "num_ctx": 2048,
            "num_predict": 128
        },
        "stream": False
    }

    resp = requests.post(
        f"{settings.llm_base_url}/api/chat",
        json=payload,
        timeout=120
    )
    resp.raise_for_status()

    raw = resp.json()["message"]["content"]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "message": raw,
            "confidence": 0.5
        }
