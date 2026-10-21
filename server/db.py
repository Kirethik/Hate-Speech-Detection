"""
Civitas AI — SQLite Persistence Layer

Manages persistent storage for:
- Analysis queries and full canonical JSON results
- Human moderator / user feedback
- Analytics aggregation metrics
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any


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
                rating TEXT,
                accepted_rewrite TEXT,
                created_at TEXT
            )
        """)


def save_analysis(db_path: str, result: dict) -> str:
    analysis_id = str(uuid.uuid4())
    verdict = "hate" if result.get("detection", {}).get("is_hate", False) else "clean"
    now_str = datetime.now(timezone.utc).isoformat()
    
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO analysis_history VALUES (?,?,?,?,?,?,?)",
            (
                analysis_id,
                result.get("input", {}).get("text", "")[:500],
                result.get("input", {}).get("language", "en"),
                verdict,
                float(result.get("detection", {}).get("hate_prob", 0.0)),
                now_str,
                json.dumps(result)
            )
        )
    return analysis_id


def get_history(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT id, text, language, verdict, hate_prob, created_at, result_json FROM analysis_history ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        history = []
        for r in rows:
            history.append({
                "id": r["id"],
                "text": r["text"],
                "language": r["language"],
                "verdict": r["verdict"],
                "hate_prob": r["hate_prob"],
                "created_at": r["created_at"],
                "result": json.loads(r["result_json"]) if r["result_json"] else {}
            })
        return history


def save_feedback(db_path: str, feedback: dict) -> str:
    init_db(db_path)
    feedback_id = str(uuid.uuid4())
    now_str = datetime.now(timezone.utc).isoformat()
    
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?)",
            (
                feedback_id,
                feedback.get("analysis_id", ""),
                feedback.get("correct_label"),
                feedback.get("bad_suggestion", ""),
                feedback.get("note", ""),
                feedback.get("rating", "helpful"),
                feedback.get("accepted_rewrite", ""),
                now_str
            )
        )
    return feedback_id


def get_analytics_summary(db_path: str) -> Dict[str, Any]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        total_queries = cursor.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
        hate_count = cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE verdict='hate'").fetchone()[0]
        clean_count = total_queries - hate_count

        lang_counts = {}
        for lang, count in cursor.execute("SELECT language, COUNT(*) FROM analysis_history GROUP BY language").fetchall():
            lang_counts[lang or "en"] = count

        total_feedback = cursor.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

        return {
            "total_analyses": total_queries,
            "hateful_detected": hate_count,
            "clean_detected": clean_count,
            "hate_ratio": round(hate_count / total_queries, 4) if total_queries > 0 else 0.0,
            "language_distribution": lang_counts,
            "total_feedback_submissions": total_feedback,
        }
