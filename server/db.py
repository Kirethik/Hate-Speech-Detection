
import sqlite3, json, uuid
from datetime import datetime
from pathlib import Path

def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id TEXT PRIMARY KEY,
                text TEXT,
                language TEXT,
                verdict TEXT,
                hate_prob REAL,
                created_at TEXT,
                result_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                analysis_id TEXT,
                correct_label INTEGER,
                bad_suggestion TEXT,
                note TEXT,
                created_at TEXT
            )
        """)

def save_analysis(db_path: str, result: dict) -> str:
    analysis_id = str(uuid.uuid4())
    verdict = "hate" if result["detection"]["is_hate"] else "clean"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO analysis_history VALUES (?,?,?,?,?,?,?)",
            (analysis_id, result["input"]["text"][:500],
             result["input"]["language"], verdict,
             result["detection"]["hate_prob"],
             datetime.utcnow().isoformat(),
             json.dumps(result))
        )
    return analysis_id

def get_history(db_path: str, limit: int) -> list[dict]:
    return []

def save_feedback(db_path: str, feedback: dict) -> str:
    return "fid-123"
