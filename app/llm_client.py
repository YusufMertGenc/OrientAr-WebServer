import json
import re
import time
import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional

import httpx
from .config import settings

logger = logging.getLogger("orientar")

INTENT_SYSTEM_PROMPT = """
You are OrientAR, a campus orientation assistant for METU Northern Cyprus Campus (METU NCC).

Your job is to help students with:
- campus orientation
- campus facilities
- student life
- clubs and societies
- navigation-related campus questions
- general METU NCC information

Rules:
- Use ONLY the provided context.
- Do NOT invent facts that are not in the context.
- Keep answers concise, helpful, and focused.
- Answer in 2-3 complete sentences unless a short list is necessary.
- Do not leave sentences unfinished.
- If the context is insufficient, say so briefly.
- Never output markdown.
- Never output code fences.
- Return ONLY valid JSON.
- If the question is broad, include multiple relevant aspects from the context.

JSON format:
{"message": "...", "confidence": 0.xx}
""".strip()

OUT_OF_DOMAIN_MESSAGE = (
    "I mainly help with METU NCC campus-related questions such as orientation, "
    "facilities, student life, clubs, and navigation."
)

PREDEFINED_RESPONSES = {
    "what_is_orientar": {
        "message": (
            "OrientAR is a campus orientation assistant designed to help students "
            "get used to METU NCC by providing campus-related information and guidance."
        ),
        "confidence": 0.95,
    },
    "how_can_you_help": {
        "message": (
            "I can help you with METU NCC campus-related questions such as orientation, "
            "facilities, student life, clubs, and general campus information."
        ),
        "confidence": 0.95,
    },
    "new_student_help": {
        "message": (
            "I can help new students get used to METU NCC by answering campus-related "
            "questions, explaining facilities, sharing student life information, and "
            "guiding them on orientation-related topics."
        ),
        "confidence": 0.95,
    },
}

_RE_LEADING_MESSAGE_PREFIX = re.compile(
    r'^\s*(?:["\']?\s*message\s*["\']?\s*[:=]\s*)+',
    re.IGNORECASE
)

_RE_INNER_MESSAGE_UNESCAPED = re.compile(
    r'"message"\s*:\s*"(?P<msg>.*?)(?:"\s*,|"\s*}|"$|$)',
    re.DOTALL
)

_RE_ESCAPED_MESSAGE = re.compile(
    r'\\"message\\"\s*:\s*\\"(?P<msg>.*?)(?:\\"(?:\s*,|\s*})|$)',
    re.DOTALL
)

_RE_NEAR_JSON_MESSAGE = re.compile(
    r'"message"\s*:\s*"(?P<msg>.+?)"\s*(?:,|}|$)',
    re.DOTALL
)

_http_client: Optional[httpx.AsyncClient] = None
_llm_semaphore = asyncio.Semaphore(settings.llm_max_concurrency)


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=50,
            ),
        )
    return _http_client


def _make_llm_cache_key(question: str, context_passages: List[str]) -> str:
    base = question.strip().lower() + "||" + "||".join((context_passages or []))
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def match_predefined_response(question: str) -> Optional[Dict[str, Any]]:
    q = (question or "").strip().lower()

    if "what is orientar" in q or "what does orientar do" in q:
        return PREDEFINED_RESPONSES["what_is_orientar"]

    if "how can you help" in q or "what can you do" in q:
        return PREDEFINED_RESPONSES["how_can_you_help"]

    if "i am a new student" in q or "i'm a new student" in q:
        if "help" in q or "get used to the campus" in q:
            return PREDEFINED_RESPONSES["new_student_help"]

    return None


def _clean_json_string(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def build_intent_prompt(question: str, context_passages: List[str]) -> str:
    trimmed: List[str] = []
    total_chars = 0
    LIMIT = 1400

    for p in context_passages or []:
        if not p:
            continue
        pp = p.strip()
        if len(pp) > 500:
            pp = pp[:500]
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
- Use only the context
- Keep the answer concise
- Output ONLY valid JSON
""".strip()


def _safe_fallback(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    raw2 = _RE_LEADING_MESSAGE_PREFIX.sub("", raw).strip()

    m = _RE_NEAR_JSON_MESSAGE.search(raw2)
    if m:
        msg = m.group("msg").strip()
        msg = msg.replace('\\"', '"').replace("\\n", "\n")
        msg = _RE_LEADING_MESSAGE_PREFIX.sub("", msg).strip()
        return {"message": msg, "confidence": 0.35}

    msg = re.sub(r"\s+", " ", raw2)
    if len(msg) > 500:
        msg = msg[:500].rstrip() + "..."
    if not msg:
        msg = "I’m not sure based on the available information."
    return {"message": msg, "confidence": 0.30}


def _normalize_llm_obj(obj) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"message": str(obj), "confidence": 0.5}

    msg = obj.get("message", "")
    conf = obj.get("confidence", 0.5)

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

    if s.startswith("{") and '"message"' in s:
        try:
            inner = json.loads(s)
            if isinstance(inner, dict) and "message" in inner:
                return _normalize_llm_obj(inner)
        except Exception:
            pass

        m = _RE_INNER_MESSAGE_UNESCAPED.search(s)
        if m:
            extracted = m.group("msg").strip()
            extracted = extracted.replace("\\n", "\n").replace("\\t", "\t")
            extracted = _RE_LEADING_MESSAGE_PREFIX.sub("", extracted).strip()
            return {"message": extracted, "confidence": conf}

    if '\\"message\\"' in s or s.startswith('{\\\"'):
        m = _RE_ESCAPED_MESSAGE.search(s)
        if m:
            extracted = m.group("msg").strip()
            extracted = extracted.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
            extracted = _RE_LEADING_MESSAGE_PREFIX.sub("", extracted).strip()
            return {"message": extracted, "confidence": conf}

        s2 = s.replace('\\"', '"')
        if s2.startswith("{") and '"message"' in s2:
            try:
                inner2 = json.loads(s2)
                if isinstance(inner2, dict) and "message" in inner2:
                    return _normalize_llm_obj(inner2)
            except Exception:
                pass

    return {"message": s, "confidence": conf}


async def generate_intent_response(question: str, context_passages: List[str]) -> Dict[str, Any]:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_intent_prompt(question, context_passages)},
        ],
        "temperature": 0.15,
        "options": {
            "num_ctx": 1280,
            "num_predict": 120,
        },
        "stream": False,
    }

    last_err = None
    client = await get_http_client()

    for attempt in range(3):
        try:
            queue_wait_started = time.perf_counter()
            async with _llm_semaphore:
                queue_wait = time.perf_counter() - queue_wait_started
                started = time.perf_counter()

                resp = await client.post(
                    f"{settings.llm_base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()

                raw = resp.json()["message"]["content"]
                cleaned = _clean_json_string(raw)

                duration = time.perf_counter() - started
                logger.info(
                    f"[LLM] wait={queue_wait:.2f}s duration={duration:.2f}s "
                    f"attempt={attempt+1}"
                )

                try:
                    obj = json.loads(cleaned)
                    result = _normalize_llm_obj(obj)
                except Exception:
                    result = _safe_fallback(raw)

                return result

        except Exception as e:
            last_err = e
            logger.warning(f"[LLM] attempt={attempt+1} error={repr(e)}")
            await asyncio.sleep(0.5 * (attempt + 1))

    return {
        "message": "Temporary error while answering. Please retry.",
        "confidence": 0.2,
        "error": str(last_err),
    }