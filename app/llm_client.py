import json
import requests
from typing import List

from .config import settings


INTENT_SYSTEM_PROMPT = """
You are OrientAR, an assistant for a campus application.

Your task:
- Answer the user's question using the provided context.
- Respond ONLY with valid JSON.
- Do NOT include explanations or markdown.

Rules:
- If relevant information is found in the context, use it to answer.
- If no relevant information is found, respond with a polite message saying you do not know.
- If the user message is casual or informal and no context is relevant, respond naturally.
- Do NOT make up information that is not in the context.

JSON FORMAT:
{
  "message": "<user-facing response>",
  "confidence": <number between 0 and 1>
}
"""


def build_intent_prompt(question: str, context_passages: List[str]) -> str:
    context_text = "\n\n".join(context_passages) if context_passages else "No relevant campus info found."

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
        "temperature": 0.2
    }

    resp = requests.post(settings.llm_base_url, json=payload, timeout=30)
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    return json.loads(raw)
