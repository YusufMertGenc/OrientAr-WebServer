# app/llm_client.py  ✅ FINAL (robust JSON + double-JSON fix + retries)

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
- "message" must be plain text (NOT JSON string).
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
    trimmed: List[str] = []
    total_chars = 0
    LIMIT = 3500

    for p in context_passages or []:
        if not p:
            continue
        pp = p.strip()
        if len(pp) > 1800:
            pp = pp[:1800]
        if total_chars + len(pp) > LIMIT:
            break
        trimmed.append(pp)
        total_chars += len(pp)

    context_text = (
        "\n\n".join([f"[DOC {i+1}]\n{t}" for i, t in enumerate(trimmed)])
        if trimmed
        else "No relevant campus info found."
    )

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

    # try to pull message from near-JSON
    m = re.search(r'"message"\s*:\s*"([^"]+)"', raw, re.DOTALL)
    if m:
        msg = m.group(1).strip()
        return {"message": msg, "confidence": 0.45}

    msg = re.sub(r"\s+", " ", raw)
    if len(msg) > 600:
        msg = msg[:600] + "..."
    if not msg:
        msg = "I’m not sure based on the available information."
    return {"message": msg, "confidence": 0.35}


def _normalize_llm_obj(obj) -> Dict:
    if not isinstance(obj, dict):
        return {"message": str(obj), "confidence": 0.5}

    msg = obj.get("message", "")
    conf = obj.get("confidence", 0.5)

    # confidence normalize
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    if not isinstance(msg, str):
        msg = str(msg)

    s = msg.strip()

    # ✅ 1) Eğer message içi JSON string'i gibi görünüyorsa (tam olmasa bile) JSON bloğunu ayıkla
    # ör: "{\"message\": \"...\", \"confidence\": 0.6"  -> içte { ... } parçasını çek
    inner_candidate = _clean_json_string(s)

    # ✅ 2) inner JSON parse dene
    if inner_candidate and inner_candidate.startswith("{"):
        try:
            inner = json.loads(inner_candidate)
            if isinstance(inner, dict) and "message" in inner:
                return _normalize_llm_obj(inner)  # recursion
        except Exception:
            pass

    # ✅ 3) Parse olmadıysa regex ile iç message'ı çek (kırpılmış olsa bile)
    m = re.search(r'"message"\s*:\s*"(.+?)"\s*(?:,|})', s, re.DOTALL)
    if m:
        extracted = m.group(1).strip()
        # escaped quotes temizle
        extracted = extracted.replace('\\"', '"')
        return {"message": extracted, "confidence": conf}

    # ✅ 4) En son: düz metin gibi döndür
    return {"message": s, "confidence": conf}

def generate_intent_response(question: str, context_passages: List[str]) -> Dict:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_intent_prompt(question, context_passages)},
        ],
        "temperature": 0.2,
        "options": {
            "num_ctx": 2048,
            "num_predict": 96,
        },
        "stream": False,
    }

    last_err = None

    for attempt in range(3):
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
                return _normalize_llm_obj(obj)
            except Exception:
                # JSON parse failed
                return _safe_fallback(raw)

        except Exception as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))

    return {"message": "Temporary error while answering. Please retry.", "confidence": 0.2, "error": str(last_err)}