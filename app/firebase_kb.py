import os
import base64
import tempfile
import firebase_admin
from firebase_admin import credentials, firestore
from typing import List, Dict, Tuple, Optional
from .config import settings
COL_ITEMS = "chatbot_kb_items"
COL_META = "chatbot_kb_meta"
DOC_META = "current"

_app_inited = False


def init_firebase(service_account_path: str | None = None):
    """
    Firebase Admin init logic (safe for local + production).

    Priority order:
    1) FIREBASE_SA_B64 (base64 encoded service account JSON)
    2) GOOGLE_APPLICATION_CREDENTIALS (file path)
    3) Explicit service_account_path
    """
    global _app_inited

    if _app_inited:
        return

    if firebase_admin._apps:
        _app_inited = True
        return

    sa_b64 = settings.firebase_sa_b64
    if sa_b64:
        decoded = base64.b64decode(sa_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name

        cred = credentials.Certificate(tmp_path)
        firebase_admin.initialize_app(cred)
        print("Firebase initialized from FIREBASE_SA_B64")
        _app_inited = True
        return

    gac_path = settings.google_application_credentials
    if gac_path and os.path.exists(gac_path):
        cred = credentials.Certificate(gac_path)
        firebase_admin.initialize_app(cred)
        print("Firebase initialized from GOOGLE_APPLICATION_CREDENTIALS")
        _app_inited = True
        return

    if service_account_path and os.path.exists(service_account_path):
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        print("Firebase initialized from explicit path")
        _app_inited = True
        return

    raise RuntimeError(
        "Firebase initialization failed. "
        "Provide FIREBASE_SA_B64 or GOOGLE_APPLICATION_CREDENTIALS "
        "or a valid service_account_path."
    )


def fetch_kb_version() -> str:
    db = firestore.client()
    doc = db.collection(COL_META).document(DOC_META).get()
    if not doc.exists:
        return "unknown"
    data = doc.to_dict() or {}
    return data.get("kbVersion", "unknown")


def fetch_kb_items() -> List[Dict]:
    """
    Returns:
    [{"id": "<doc_id>", "content": "<text>", "updatedAt": ..., "isDeleted": ...}, ...]
    """
    db = firestore.client()
    items: List[Dict] = []

    for doc in db.collection(COL_ITEMS).stream():
        d = doc.to_dict() or {}

        if d.get("isDeleted") is True:
            continue

        content = d.get("content", "")
        if not content:
            continue

        items.append({
            "id": doc.id,
            "content": content,
            "updatedAt": d.get("updatedAt"),
            "isDeleted": d.get("isDeleted", False),
        })

    return items


def fetch_kb_fingerprint() -> Tuple[int, Optional[str]]:
    """
    Robust fingerprint that doesn't rely on user fields (updatedAt/meta).
    Uses Firestore system update_time.

    Returns: (count, max_update_time_iso)
    """
    db = firestore.client()
    count = 0
    max_ut = None  # datetime

    for doc in db.collection(COL_ITEMS).stream():
        d = doc.to_dict() or {}
        if d.get("isDeleted") is True:
            continue
        content = d.get("content", "")
        if not content:
            continue

        count += 1
        ut = getattr(doc, "update_time", None)  # Firestore system timestamp
        if ut and (max_ut is None or ut > max_ut):
            max_ut = ut

    return count, (max_ut.isoformat() if max_ut else None)