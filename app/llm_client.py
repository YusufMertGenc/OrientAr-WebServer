import json
import re
import time
from typing import List, Dict, Any

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


# -------------------- helpers --------------------

# message: / "message": / Message= ... prefix siler (tekrar tekrar olsa bile)
_RE_LEADING_MESSAGE_PREFIX = re.compile(
    r'^\s*(?:["\']?\s*message\s*["\']?\s*[:=]\s*)+',
    re.IGNORECASE
)

_RE_INNER_MESSAGE_UNESCAPED = re.compile(
    r'"message"\s*:\s*"(?P<msg>.*?)(?:"\s*,|"\s*}|"$|$)',
    re.DOTALL
)

# iç içe JSON string içinden "message":"..." yakalar (escape'li de olsa)
_RE_INNER_JSON_MESSAGE = re.compile(
    r'"message"\s*:\s*"(?P<msg>(?:\\.|[^"\\])*)"',
    re.DOTALL
)
_RE_ESCAPED_MESSAGE = re.compile(
    r'\\"message\\"\s*:\s*\\"(?P<msg>.*?)(?:\\"(?:\s*,|\s*})|$)',
    re.DOTALL
)

# near-json içinden message yakalama (kırpılmış JSON durumları için)
_RE_NEAR_JSON_MESSAGE = re.compile(
    r'"message"\s*:\s*"(?P<msg>.+?)"\s*(?:,|}|$)',
    re.DOTALL
)


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
        return text[start: end + 1]
    return text


def build_intent_prompt(question: str, context_passages: List[str]) -> str:
    trimmed: List[str] = []
    total_chars = 0
    LIMIT = 1800

    for p in context_passages or []:
        if not p:
            continue
        pp = p.strip()
        if len(pp) > 700:
            pp = pp[:700]
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


def _safe_fallback(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()

    # "message: ...." varsa baştan kırp
    raw2 = _RE_LEADING_MESSAGE_PREFIX.sub("", raw).strip()

    # near-json içinden message çek
    m = _RE_NEAR_JSON_MESSAGE.search(raw2)
    if m:
        msg = m.group("msg").strip()
        msg = msg.replace('\\"', '"').replace("\\n", "\n")
        msg = _RE_LEADING_MESSAGE_PREFIX.sub("", msg).strip()
        return {"message": msg, "confidence": 0.35}

    # plain text dön
    msg = re.sub(r"\s+", " ", raw2)
    if len(msg) > 600:
        msg = msg[:600] + "..."
    if not msg:
        msg = "I’m not sure based on the available information."
    return {"message": msg, "confidence": 0.30}


def _normalize_llm_obj(obj) -> Dict[str, Any]:
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

    if msg is None:
        msg = ""
    if not isinstance(msg, str):
        msg = str(msg)

    s = msg.strip()
    s = _RE_LEADING_MESSAGE_PREFIX.sub("", s).strip()

    #  CASE A: message field itself looks like JSON object string (UNESCAPED)
    # e.g. {"message":"...", "confidence":0.3}
    if s.startswith("{") and '"message"' in s:
        # 1) try parse full JSON
        try:
            inner = json.loads(s)
            if isinstance(inner, dict) and "message" in inner:
                return _normalize_llm_obj(inner)
        except Exception:
            pass

        # 2) if it's truncated JSON, regex extract message value
        m = _RE_INNER_MESSAGE_UNESCAPED.search(s)
        if m:
            extracted = m.group("msg").strip()
            extracted = extracted.replace("\\n", "\n").replace("\\t", "\t")
            extracted = _RE_LEADING_MESSAGE_PREFIX.sub("", extracted).strip()
            return {"message": extracted, "confidence": conf}

    #  CASE B: message field is ESCAPED json string (rare)
    # e.g. {\"message\": \"...\" ...}
    if '\\"message\\"' in s or s.startswith('{\\\"'):
        m = _RE_ESCAPED_MESSAGE.search(s)
        if m:
            extracted = m.group("msg").strip()
            extracted = extracted.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
            extracted = _RE_LEADING_MESSAGE_PREFIX.sub("", extracted).strip()
            return {"message": extracted, "confidence": conf}

        # brute unescape then retry as unescaped json
        s2 = s.replace('\\"', '"')
        if s2.startswith("{") and '"message"' in s2:
            try:
                inner2 = json.loads(s2)
                if isinstance(inner2, dict) and "message" in inner2:
                    return _normalize_llm_obj(inner2)
            except Exception:
                pass
            m2 = _RE_INNER_MESSAGE_UNESCAPED.search(s2)
            if m2:
                extracted = m2.group("msg").strip()
                extracted = extracted.replace("\\n", "\n").replace("\\t", "\t")
                extracted = _RE_LEADING_MESSAGE_PREFIX.sub("", extracted).strip()
                return {"message": extracted, "confidence": conf}

    #  Normal: plain text already
    return {"message": s, "confidence": conf}


def generate_intent_response(question: str, context_passages: List[str]) -> Dict[str, Any]:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_intent_prompt(question, context_passages)},
        ],
        "temperature": 0.2,
        "options": {
            "num_ctx": 1536,
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
                timeout=90,
            )
            resp.raise_for_status()

            raw = resp.json()["message"]["content"]
            cleaned = _clean_json_string(raw)

            try:
                obj = json.loads(cleaned)
                return _normalize_llm_obj(obj)
            except Exception:
                # JSON parse failed -> fallback
                return _safe_fallback(raw)

        except Exception as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))

    return {"message": "Temporary error while answering. Please retry.", "confidence": 0.2, "error": str(last_err)}