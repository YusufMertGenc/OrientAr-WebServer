# app/llm_client.py  ✅ FINAL

import json
import re
import time
from typing import List, Dict

import requests
from .config import settings


INTENT_SYSTEM_PROMPT = """
You are OrientAR, an assistant for a campus application.
Assume all questions are about METU NCC unless stated otherwise.

You MUST use ONLY the provided context.
You MUST NOT use any external knowledge.

CRITICAL RULES:
- Always finish sentences. Never stop mid-sentence.
- If context contains a list (e.g. dormitories), list ALL items mentioned.
- Do not hallucinate new items.

Output:
- Respond ONLY with valid JSON (no markdown, no extra text).
- "message" must be plain text.
- JSON format:
  {"message": "...", "confidence": 0.xx}
""".strip()


def _clean_json_string(text: str) -> str:
    text = (text or "").strip()

    # remove fenced blocks if any
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def build_intent_prompt(question: str, context_passages: List[str]) -> str:
    # Passage-based trimming (daha stabil)
    trimmed: List[str] = []
    total_chars = 0
    LIMIT = 3500  # context limit

    for p in context_passages or []:
        if not p:
            continue
        pp = p.strip()
        # each passage clamp
        if len(pp) > 1800:
            pp = pp[:1800]
        if total_chars + len(pp) > LIMIT:
            break
        trimmed.append(pp)
        total_chars += len(pp)

    context_text = "\n\n".join([f"[DOC {i+1}]\n{t}" for i, t in enumerate(trimmed)]) if trimmed else "No relevant campus info found."

    return f"""
CONTEXT:
{context_text}

QUESTION:
{question}

Remember:
- Use only context
- Finish sentences
- Output ONLY JSON
""".strip()


def _safe_fallback(raw: str) -> Dict:
    raw = (raw or "").strip()

    # try to pull message if model produced something close to JSON
    m = re.search(r'"message"\s*:\s*"([^"]+)"', raw, re.DOTALL)
    if m:
        msg = m.group(1).strip()
        return {"message": msg, "confidence": 0.45}

    # else: just return cleaned text
    msg = re.sub(r"\s+", " ", raw)
    if len(msg) > 600:
        msg = msg[:600] + "..."
    if not msg:
        msg = "I’m not sure based on the available information."
    return {"message": msg, "confidence": 0.35}


def generate_intent_response(question: str, context_passages: List[str]) -> Dict:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_intent_prompt(question, context_passages)},
        ],
        "temperature": 0.2,
        "options": {
            "num_ctx": 2048,      # 1024 küçük kalabiliyor
            "num_predict": 96,   # 64 çok kısa → yarım cümle yapar
        },
        "stream": False,
    }

    last_err = None

    for attempt in range(3):  # retry
        try:
            resp = requests.post(
                f"{settings.llm_base_url}/api/chat",
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()

            raw = resp.json()["message"]["content"]
            cleaned = _clean_json_string(raw)

            try:
                obj = json.loads(cleaned)
                # guarantee keys
                msg = str(obj.get("message", "")).strip()
                conf = obj.get("confidence", 0.5)
                try:
                    conf = float(conf)
                except Exception:
                    conf = 0.5
                if not msg:
                    return {"message": "I’m not sure based on the available information.", "confidence": 0.25}
                return {"message": msg, "confidence": max(0.0, min(1.0, conf))}
            except Exception:
                # JSON parse patladı
                return _safe_fallback(raw)

        except Exception as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))

    # all failed
    return {"message": "Temporary error while answering. Please retry.", "confidence": 0.2, "error": str(last_err)}