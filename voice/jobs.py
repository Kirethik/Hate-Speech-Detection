"""
Job queue for audio processing.

- Jobs persist in SQLite (survive restarts; interrupted jobs are marked failed).
- One worker processes jobs sequentially in a background thread, so models stay
  loaded once and never fight over the GPU/CPU.
- Progress is pushed to any number of subscribers (Server-Sent Events).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .pipeline import JobOptions

log = logging.getLogger("civitas.jobs")

SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_jobs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status TEXT NOT NULL,          -- queued | running | done | failed
    stage TEXT, progress REAL DEFAULT 0, message TEXT,
    filename TEXT, options TEXT, result TEXT, error TEXT
)"""


class JobStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def create(self, filename: str, opts: dict) -> str:
        jid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO voice_jobs (id, created_at, updated_at, status, stage, progress, message,"
                      " filename, options) VALUES (?,?,?,?,?,?,?,?,?)",
                      (jid, now, now, "queued", "queued", 0.0, "Waiting in queue", filename, json.dumps(opts)))
        return jid

    def update(self, jid: str, **kw):
        if not kw:
            return
        kw["updated_at"] = time.time()
        for k in ("result", "options"):
            if k in kw and not isinstance(kw[k], str):
                kw[k] = json.dumps(kw[k], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in kw)
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE voice_jobs SET {cols} WHERE id=?", (*kw.values(), jid))

    def get(self, jid: str, with_result: bool = False) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM voice_jobs WHERE id=?", (jid,)).fetchone()
        return self._row(r, with_result) if r else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM voice_jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r, False) for r in rows]

    def by_status(self, *statuses: str) -> list[str]:
        q = ",".join("?" * len(statuses))
        with self._conn() as c:
            return [r["id"] for r in c.execute(
                f"SELECT id FROM voice_jobs WHERE status IN ({q}) ORDER BY created_at", statuses)]

    @staticmethod
    def _row(r: sqlite3.Row, with_result: bool) -> dict:
        d = dict(r)
        d["options"] = json.loads(d["options"] or "{}")
        res = json.loads(d.pop("result") or "null")
        if with_result:
            d["result"] = res
        elif res:
            ae = res.get("audio_edit", {})
            d["summary"] = {"language": ae.get("language"), "duration_s": ae.get("duration_s"),
                            "edits": len(ae.get("edits", [])), "is_hate": res.get("detection", {}).get("is_hate")}
        return d


class JobManager:
    def __init__(self, store: JobStore, jobs_dir: Path, run: Callable[[str, Path, Path, JobOptions, Callable], dict]):
        """run(job_id, src_path, out_dir, opts, emit) -> result dict (blocking)."""
        self.store, self.jobs_dir, self.run = store, jobs_dir, run
        self.queue: asyncio.Queue[str] | None = None
        self.subs: dict[str, list[asyncio.Queue]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="civitas-worker")
        self._task: asyncio.Task | None = None

    # ---- lifecycle --------------------------------------------------------
    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()
        for jid in self.store.by_status("running"):
            self.store.update(jid, status="failed", stage="failed", error="server restarted during processing")
        for jid in self.store.by_status("queued"):
            self.queue.put_nowait(jid)
        self._task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._task:
            self._task.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ---- API --------------------------------------------------------------
    def job_dir(self, jid: str) -> Path:
        return self.jobs_dir / jid

    def submit(self, filename: str, data: bytes, opts: JobOptions) -> str:
        jid = self.store.create(filename, opts.__dict__)
        d = self.job_dir(jid)
        d.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix.lower() or ".bin"
        (d / f"input{suffix}").write_bytes(data)
        self.queue.put_nowait(jid)
        return jid

    def subscribe(self, jid: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subs.setdefault(jid, []).append(q)
        return q

    def unsubscribe(self, jid: str, q: asyncio.Queue):
        lst = self.subs.get(jid, [])
        if q in lst:
            lst.remove(q)
        if not lst:
            self.subs.pop(jid, None)

    def position(self, jid: str) -> int:
        ids = self.store.by_status("queued")
        return ids.index(jid) + 1 if jid in ids else 0

    # ---- internals --------------------------------------------------------
    def _publish(self, jid: str, event: dict):
        for q in list(self.subs.get(jid, [])):
            q.put_nowait(event)

    def _emit_threadsafe(self, jid: str):
        last = {"t": 0.0}

        def emit(stage: str, progress: float, message: str = ""):
            now = time.time()
            if stage not in ("done",) and now - last["t"] < 0.2:
                return  # throttle DB writes
            last["t"] = now
            self.store.update(jid, stage=stage, progress=round(progress, 3), message=message)
            ev = {"id": jid, "status": "running", "stage": stage, "progress": round(progress, 3),
                  "message": message}
            self.loop.call_soon_threadsafe(self._publish, jid, ev)

        return emit

    async def _worker(self):
        while True:
            jid = await self.queue.get()
            job = self.store.get(jid)
            if not job or job["status"] != "queued":
                continue
            d = self.job_dir(jid)
            src = next(iter(sorted(d.glob("input.*"))), None)
            opts = JobOptions(**{k: v for k, v in job["options"].items() if k in JobOptions.__dataclass_fields__})
            self.store.update(jid, status="running", stage="starting", progress=0.01, message="Starting")
            self._publish(jid, {"id": jid, "status": "running", "stage": "starting", "progress": 0.01})
            try:
                if src is None:
                    raise FileNotFoundError("uploaded file missing")
                result = await self.loop.run_in_executor(
                    self.pool, self.run, jid, src, d, opts, self._emit_threadsafe(jid))
                self.store.update(jid, status="done", stage="done", progress=1.0,
                                  message=f"{len(result.get('audio_edit', {}).get('edits', []))} sentence(s) replaced",
                                  result=result)
                self._publish(jid, {"id": jid, "status": "done", "stage": "done", "progress": 1.0})
            except Exception as e:
                log.exception("job %s failed", jid)
                self.store.update(jid, status="failed", stage="failed", error=f"{type(e).__name__}: {e}")
                self._publish(jid, {"id": jid, "status": "failed", "stage": "failed", "error": str(e)})
