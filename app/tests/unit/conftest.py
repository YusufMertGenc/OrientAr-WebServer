from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """
    Creates a TestClient while disabling startup side effects such as
    KB loading and watcher startup.
    """
    import app.main as main

    monkeypatch.setattr(main, "load_kb_to_chroma", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "start_kb_watcher", lambda *args, **kwargs: None)

    with TestClient(main.app) as test_client:
        yield test_client