import json
import time
from pathlib import Path
from collections import Counter

import firebase_admin
from firebase_admin import credentials, firestore


COL_ITEMS = "chatbot_kb_items"
COL_META = "chatbot_kb_meta"
DOC_META = "current"


def get_field_case_insensitive(d: dict, field_name: str, default=None):
    """
    Dict içinden field'ı büyük/küçük harf fark etmeden okur.
    Örn: content, Content, CONTENT hepsini yakalar.
    """
    if field_name in d:
        return d[field_name]

    target = field_name.lower()
    for key, value in d.items():
        if str(key).lower() == target:
            return value

    return default


def init_firebase_once(service_account_path: str):
    """
    Import scripti birden fazla çalıştırılırsa Firebase already initialized hatası vermesin.
    """
    if firebase_admin._apps:
        return

    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)


def import_kb(service_account_path: str, kb_json_path: str):
    init_firebase_once(service_account_path)
    db = firestore.client()

    with open(kb_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("KB JSON must be a list of items.")

    print(f"Loaded JSON items: {len(items)}")

    ids = [get_field_case_insensitive(item, "id") for item in items]
    id_counts = Counter(ids)

    duplicate_ids = [doc_id for doc_id, count in id_counts.items() if count > 1]
    missing_ids = [index for index, doc_id in enumerate(ids) if not doc_id]

    if missing_ids:
        raise ValueError(f"Some items have missing id. Item indexes: {missing_ids}")

    if duplicate_ids:
        print("WARNING: Duplicate ids found. These will overwrite each other in Firestore:")
        for doc_id in duplicate_ids:
            print(f"- {doc_id}")

    now = int(time.time())

    BATCH_LIMIT = 450
    batch = db.batch()
    batch_count = 0
    total = 0
    skipped_empty_content = 0

    for item in items:
        doc_id = get_field_case_insensitive(item, "id")
        content = get_field_case_insensitive(item, "content", "")
        title = get_field_case_insensitive(item, "title", "")
        category = get_field_case_insensitive(item, "category", "")

        if content is None:
            content = ""

        content = str(content).strip()

        if not content:
            skipped_empty_content += 1
            print(f"Skipping item with empty content: {doc_id}")
            continue

        if title is None:
            title = ""
        if category is None:
            category = ""

        title = str(title).strip()
        category = str(category).strip()

        ref = db.collection(COL_ITEMS).document(str(doc_id))

        # Firestore'a standart küçük harfli fieldlar yazıyoruz.
        # Böylece backend tarafında content/title/category sorunsuz okunur.
        batch.set(ref, {
            "content": content,
            "title": title,
            "category": category,
            "updatedAt": now,
            "isDeleted": False,
        }, merge=True)

        batch_count += 1
        total += 1

        if batch_count >= BATCH_LIMIT:
            batch.commit()
            batch = db.batch()
            batch_count = 0
            print(f"{total} documents committed...")

    if batch_count > 0:
        batch.commit()

    db.collection(COL_META).document(DOC_META).set({
        "kbVersion": f"v{now}",
        "lastUpdatedAt": now,
        "itemCount": total,
    }, merge=True)

    print("Import finished.")
    print(f"JSON item count: {len(items)}")
    print(f"Unique id count: {len(set(ids))}")
    print(f"Imported documents: {total}")
    print(f"Skipped empty content: {skipped_empty_content}")
    print(f"Meta kbVersion: v{now}")


ROOT_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    import_kb(
        service_account_path=str(ROOT_DIR / "app" / "orientar-4ae58-firebase-adminsdk-fbsvc-c66789f440.json"),
        kb_json_path=str(ROOT_DIR / "data" / "campus_kb.json"),
    )