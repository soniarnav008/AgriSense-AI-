import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "agrisense.db"

def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            record_type TEXT NOT NULL,
            crop TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """)

def save_record(record_type, crop, payload):
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO assessments (created_at, record_type, crop, payload) VALUES (?, ?, ?, ?)",
            (created_at, record_type, crop, json.dumps(payload, ensure_ascii=False))
        )

def get_records(limit=200):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT created_at, record_type, crop, payload FROM assessments ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    records = []
    for row in rows:
        records.append({
            "created_at": row["created_at"],
            "record_type": row["record_type"],
            "crop": row["crop"],
            "payload": json.loads(row["payload"]),
        })
    return records

def clear_records():
    with _connect() as conn:
        conn.execute("DELETE FROM assessments")
