from __future__ import annotations

import pytest


def test_final_answer_cleanup_plain_text():
    from app.main import final_answer_cleanup

    raw = '   "Hello   world"   '
    assert final_answer_cleanup(raw) == "Hello world"


def test_final_answer_cleanup_valid_json():
    from app.main import final_answer_cleanup

    raw = '{"message": "Library is near the cafeteria."}'
    assert final_answer_cleanup(raw) == "Library is near the cafeteria."


def test_final_answer_cleanup_escaped_json():
    from app.main import final_answer_cleanup

    raw = '{\\"message\\": \\"Gym is open until 10 PM.\\"}'
    assert final_answer_cleanup(raw) == "Gym is open until 10 PM."


def test_final_answer_cleanup_extracts_message_with_regex_fallback():
    from app.main import final_answer_cleanup

    raw = '{"message":"Student Affairs is in the admin building","confidence":0.8'
    cleaned = final_answer_cleanup(raw)
    assert "Student Affairs" in cleaned


@pytest.mark.asyncio
async def test_health_check_returns_status_and_active_requests(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "active_requests", 3)
    result = await main.health_check()

    assert result == {"status": "ok", "active_requests": 3}

def test_final_answer_cleanup_empty_string():
    from app.main import final_answer_cleanup

    assert final_answer_cleanup("") == ""


def test_final_answer_cleanup_none():
    from app.main import final_answer_cleanup

    assert final_answer_cleanup(None) == ""