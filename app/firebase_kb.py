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


def get_field_case_insensitive(d: Dict, field_name: str, default=None):
    """
    Firestore document fieldlarını büyük/küçük harf fark etmeden okur.
    Örn: content, Content, CONTENT hepsini yakalar.
    """
    if field_name in d:
        return d[field_name]

    target = field_name.lower()
    for key, value in d.items():
        if str(key).lower() == target:
            return value

    return default


def init_firebase(service_account_path: str | None = None):
    """
    Firebase Admin init logic.

    Priority order:
    1) FIREBASE_SA_B64
    2) GOOGLE_APPLICATION_CREDENTIALS
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
    return get_field_case_insensitive(data, "kbVersion", "unknown")


def build_searchable_content(title: str, category: str, content: str) -> str:
    """
    Title/category/content alanlarını tek metin haline getirir.
    Böylece RAG title ve category'den de faydalanabilir.
    """
    parts = []

    if title:
        parts.append(f"Title: {title}")

    if category:
        parts.append(f"Category: {category}")

    if content:
        parts.append(f"Content: {content}")

    return "\n".join(parts).strip()


def fetch_kb_items() -> List[Dict]:
    """
    Returns:
    [
        {
            "id": "<doc_id>",
            "content": "<searchable_text>",
            "updatedAt": ...,
            "isDeleted": ...
        },
        ...
    ]

    Büyük/küçük harf uyumludur:
    content / Content
    title / Title
    category / Category
    isDeleted / IsDeleted
    updatedAt / UpdatedAt
    """
    db = firestore.client()
    items: List[Dict] = []

    for doc in db.collection(COL_ITEMS).stream():
        d = doc.to_dict() or {}

        is_deleted = get_field_case_insensitive(d, "isDeleted", False)
        if is_deleted is True:
            continue

        content = get_field_case_insensitive(d, "content", "")
        title = get_field_case_insensitive(d, "title", "")
        category = get_field_case_insensitive(d, "category", "")
        updated_at = get_field_case_insensitive(d, "updatedAt", None)

        if content is None:
            content = ""
        if title is None:
            title = ""
        if category is None:
            category = ""

        content = str(content).strip()
        title = str(title).strip()
        category = str(category).strip()

        if not content:
            continue

        searchable_content = build_searchable_content(
            title=title,
            category=category,
            content=content,
        )

        items.append({
            "id": doc.id,
            "content": searchable_content,
            "updatedAt": updated_at,
            "isDeleted": is_deleted,
        })

    return items


def fetch_kb_fingerprint() -> Tuple[int, Optional[str]]:
    """
    Robust fingerprint.
    Firestore system update_time kullanır.

    Returns:
    (count, max_update_time_iso)
    """
    db = firestore.client()
    count = 0
    max_ut = None

    for doc in db.collection(COL_ITEMS).stream():
        d = doc.to_dict() or {}

        is_deleted = get_field_case_insensitive(d, "isDeleted", False)
        if is_deleted is True:
            continue

        content = get_field_case_insensitive(d, "content", "")
        if content is None:
            content = ""

        content = str(content).strip()

        if not content:
            continue

        count += 1

        ut = getattr(doc, "update_time", None)
        if ut and (max_ut is None or ut > max_ut):
            max_ut = ut

    return count, (max_ut.isoformat() if max_ut else None)