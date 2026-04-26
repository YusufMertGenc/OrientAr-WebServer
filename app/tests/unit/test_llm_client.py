from __future__ import annotations


def test_match_predefined_response_what_is_orientar():
    from app.llm_client import match_predefined_response

    result = match_predefined_response("What is OrientAR?")
    assert result is not None
    assert "OrientAR" in result["message"]
    assert result["confidence"] == 0.95


def test_match_predefined_response_unknown_returns_none():
    from app.llm_client import match_predefined_response

    result = match_predefined_response("What is the weather in London?")
    assert result is None


def test_clean_json_string_removes_code_fence():
    from app.llm_client import _clean_json_string

    raw = """```json
    {"message": "Hello", "confidence": 0.9}
    ```"""
    cleaned = _clean_json_string(raw)
    assert cleaned == '{"message": "Hello", "confidence": 0.9}'


def test_clean_json_string_extracts_json_from_surrounding_text():
    from app.llm_client import _clean_json_string

    raw = 'random text before {"message":"Hi","confidence":0.8} random text after'
    cleaned = _clean_json_string(raw)
    assert cleaned == '{"message":"Hi","confidence":0.8}'


def test_normalize_llm_obj_clamps_confidence_and_keeps_message():
    from app.llm_client import _normalize_llm_obj

    obj = {"message": "Library info", "confidence": 3}
    result = _normalize_llm_obj(obj)

    assert result["message"] == "Library info"
    assert result["confidence"] == 1.0


def test_normalize_llm_obj_handles_nested_message_json():
    from app.llm_client import _normalize_llm_obj

    obj = {"message": '{"message":"Campus bus is free.","confidence":0.7}', "confidence": 0.4}
    result = _normalize_llm_obj(obj)

    assert result["message"] == "Campus bus is free."
    assert 0.0 <= result["confidence"] <= 1.0


def test_normalize_llm_obj_non_dict():
    from app.llm_client import _normalize_llm_obj

    result = _normalize_llm_obj("plain output")
    assert result["message"] == "plain output"
    assert result["confidence"] == 0.5


def test_safe_fallback_extracts_message_from_near_json():
    from app.llm_client import _safe_fallback

    raw = 'message: {"message":"The library is open.","confidence":0.8}'
    result = _safe_fallback(raw)

    assert "library" in result["message"].lower()
    assert result["confidence"] == 0.35


def test_safe_fallback_plain_text():
    from app.llm_client import _safe_fallback

    raw = "some malformed output without json"
    result = _safe_fallback(raw)

    assert result["message"] == "some malformed output without json"
    assert result["confidence"] == 0.30