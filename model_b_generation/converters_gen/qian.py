"""
Qian et al. (2019) "A Benchmark Dataset for Learning to Intervene in Online
Hate Speech": Reddit and Gab conversations with human-written interventions.
English. Task = respond.

EXPECTED (verify against your download; see docs/SOURCES.md):
    data/raw/qian_counter/reddit.csv, data/raw/qian_counter/gab.csv
    text            numbered posts, one per line: "1. first post\n2. reply ..."
    hate_speech_idx list of post numbers that are hateful, e.g. "[1, 3]" (or "n/a")
    response        list of intervention strings, e.g. "['reply 1', 'reply 2']"

Each (hateful post, intervention) becomes one pair.
"""

import ast
import logging
import re
from pathlib import Path

import pandas as pd

from ._common import empty_df, finalize, group_id, make_row

logger = logging.getLogger(__name__)

_POST = re.compile(r"^\s*(\d+)\.\s*(.*)$")


def _posts(text: str) -> dict[int, str]:
    out = {}
    for line in str(text).splitlines():
        m = _POST.match(line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


def _as_list(v) -> list:
    if isinstance(v, list):
        return v
    try:
        parsed = ast.literal_eval(str(v))
        return parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, SyntaxError):
        return []


def _pairs(df: pd.DataFrame, source: str) -> list:
    need = {"text", "hate_speech_idx", "response"}
    if not need <= set(df.columns):
        logger.warning("qian %s: expected columns %s, got %s", source, need, list(df.columns))
        return []
    rows = []
    for text, idx, resp in zip(df["text"], df["hate_speech_idx"], df["response"]):
        posts = _posts(text)
        responses = [r for r in _as_list(resp) if isinstance(r, str) and r.strip()]
        for i in _as_list(idx):
            post = posts.get(int(i)) if str(i).isdigit() else None
            if not post:
                continue
            gid = group_id(post)
            rows += [make_row("respond", post, r, "en", source, gid=gid) for r in responses]
    return rows


def convert(raw_dir: str = "data/raw") -> pd.DataFrame:
    root = Path(raw_dir) / "qian_counter"
    rows = []
    for name in ("reddit", "gab"):
        path = root / f"{name}.csv"
        if path.exists():
            rows += _pairs(pd.read_csv(path), f"qian_{name}")
    if not rows:
        logger.warning("Qian counter-speech not found under %s", root)
        return empty_df()
    return finalize(rows)
