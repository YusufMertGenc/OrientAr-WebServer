import json
import time
import firebase_admin
from firebase_admin import credentials, firestore

COL_ITEMS = "chatbot_kb_items"
COL_META = "chatbot_kb_meta"
DOC_META = "current"

def import_kb(service_account_path: str, kb_json_path: str):
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    with open(kb_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    now = int(time.time())

    BATCH_LIMIT = 450
    batch = db.batch()
    count = 0
    total = 0

    for item in items:
        doc_id = item["id"]
        content = item["content"]

        ref = db.collection(COL_ITEMS).document(doc_id)

        batch.set(ref, {
            "content": content,
            "updatedAt": now,
            "isDeleted": False
        }, merge=True)

        count += 1
        total += 1

        if count >= BATCH_LIMIT:
            batch.commit()
            batch = db.batch()
            count = 0
            print(f"{total} documents committed...")

    if count > 0:
        batch.commit()

    db.collection(COL_META).document(DOC_META).set({
        "kbVersion": f"v{now}",
        "lastUpdatedAt": now
    }, merge=True)

    print(f"Done. Total documents: {total}")

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent  # project root

if __name__ == "__main__":
    import_kb(
        service_account_path=str(ROOT_DIR / "app" / "orientar-4ae58-firebase-adminsdk-fbsvc-c66789f440.json"),
        kb_json_path=str(ROOT_DIR / "data" / "campus_kb.json")
    )