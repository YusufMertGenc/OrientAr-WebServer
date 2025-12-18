from typing import Dict, Any

# Basit in-memory session (şimdilik)
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def get_session(user_id: str) -> Dict[str, Any]:
    return SESSION_STORE.get(user_id, {})


def set_session(user_id: str, data: Dict[str, Any]):
    SESSION_STORE[user_id] = data


def clear_session(user_id: str):
    if user_id in SESSION_STORE:
        del SESSION_STORE[user_id]
