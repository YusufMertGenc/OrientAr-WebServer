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
- The value of "message" MUST be plain text.
- DO NOT put JSON, lists, or objects inside the "message" field.

JSON FORMAT:
{
  "message": "<answer>",
  "confidence": <number between 0 and 1>
}

"""

def _clean_json_string(text: str) -> str:
    """Markdown taglerini temizler ve sadece { } arasındaki JSON'ı alır."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    # İlk { ve son } bularak aradaki temiz JSON'ı al
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
    return text.strip()

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


import json

import re  # <--- Bunu en tepeye ekle

# generate_intent_response fonksiyonunu komple bununla değiştir:
def generate_intent_response(question: str, context_passages: List[str]) -> dict:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_intent_prompt(question, context_passages)}
        ],
        "temperature": 0.2,
        "options": {
            "num_ctx": 1024,
            "num_predict": 64 
        },
        "stream": False
    }

    try:
        resp = requests.post(
            f"{settings.llm_base_url}/api/chat",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]

        # Önce temizlemeyi dene
        cleaned = _clean_json_string(raw)
        return json.loads(cleaned)

    except (json.JSONDecodeError, ValueError):
        # 🔥 BURASI ÖNEMLİ: JSON patladıysa ham metni basma!
        # Regex ile sadece "message"ın karşısındaki yazıyı al.
        
        # 1. Regex ile çekmeye çalış (en temiz yöntem)
        match = re.search(r'(?:"?message"?\s*:\s*"?)(.*?)(?:["}]|$)', raw, re.DOTALL)
        
        if match:
            text = match.group(1).strip()
            # Eğer kesildiği için sonda tırnak (") kaldıysa sil
            if text.endswith('"'): text = text[:-1]
            return {"message": text, "confidence": 0.5}

        # 2. Regex de bulamazsa manuel temizlik (Brute force)
        text = raw.replace('{"message":', '').replace('message:', '').replace('{', '').replace('}', '').strip()
        if text.startswith('"'): text = text[1:]
        if text.endswith('"'): text = text[:-1]
        
        return {
            "message": text,
            "confidence": 0.5
        }