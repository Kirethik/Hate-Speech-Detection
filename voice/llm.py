"""
LLM judge + rewriter, via any OpenAI-compatible endpoint (Ollama by default).

One call returns: verdict, target, hateful spans, a rewrite, a counter-narrative
and `tts_text` (the text to synthesise for the requested mode, in the script
the TTS engine needs).
"""

from __future__ import annotations

import json
import logging
import re

from label_maps import TARGET_CLASSES

from .types import Verdict

log = logging.getLogger("civitas.llm")

SEVERITIES = ["none", "offensive", "hate"]

SCHEMA = {
    "type": "object",
    "properties": {
        "is_hate": {"type": "boolean"},
        "severity": {"type": "string", "enum": SEVERITIES},
        "target": {"type": "string", "enum": TARGET_CLASSES},
        "spans": {"type": "array", "items": {"type": "string"}},
        "rewrite": {"type": "string"},
        "counter": {"type": "string"},
        "tts_text": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["is_hate", "severity", "target", "spans", "rewrite", "counter", "tts_text", "reason"],
}

LANG_NAMES = {"en": "English", "hi": "Hindi", "ur": "Urdu", "ta": "Tamil", "te": "Telugu",
              "ml": "Malayalam", "kn": "Kannada", "bn": "Bengali", "mr": "Marathi"}

SYSTEM = f"""You are the moderation component of a voice platform. Spoken audio was transcribed;
you get the whole transcript for context and ONE target sentence from it.

1. Classify the target sentence:
   - "hate": attacks, dehumanises or calls for harm against people because of a protected
     characteristic (religion, caste, ethnicity, gender, sexuality, disability, nationality...).
   - "offensive": insults, slurs or profanity without that kind of group target.
   - "none": everything else. Quoting or criticising hate is NOT hate. Use the context.
   target must be one of: {", ".join(TARGET_CLASSES)} ("none" if not hate).
2. If severity is "offensive" or "hate", write in the SAME language and SAME script as the target
   sentence, spoken style, first person, roughly the same number of words (+/-20%):
   - rewrite: what the speaker could have said instead: keep any legitimate point or concern,
     remove the hate/insult completely.
   - counter: a first-person sentence that calmly challenges the hateful idea.
   - tts_text: copy of rewrite if mode=rewrite, copy of counter if mode=counter. Exception: if the
     language is Urdu, write tts_text in Devanagari script (it is read by a Hindi voice).
   - spans: exact substrings of the target sentence that are hateful or offensive.
3. If severity is "none": is_hate=false, target="none", spans=[], rewrite="", counter="", tts_text="".
4. reason: one short English sentence explaining the decision.
Return ONLY a JSON object with keys: is_hate, severity, target, spans, rewrite, counter, tts_text, reason."""


TONES = {
    "calm": "calm and de-escalating",
    "factual": "factual and objective, focused on verifiable points",
    "firm": "firm and direct about not accepting harm, but respectful",
    "empathetic": "empathetic, acknowledging the underlying feelings",
    "professional": "formal and professional, suitable for a workplace",
}

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"rewrites": {"type": "array", "items": {"type": "string"}},
                   "responses": {"type": "array", "items": {"type": "string"}}},
    "required": ["rewrites", "responses"],
}

REWRITE_SYSTEM = """You help people say things without hate or insults.
Given a TEXT, write in the SAME language and SAME script as the TEXT, in the requested tone:
- rewrites: {n} different versions of what the author could say instead. Keep any legitimate
  point, concern or criticism; remove all hate, slurs, insults and generalisations about groups.
  First person, natural, about the same length.
- responses: {n} short replies a bystander or moderator could give that calmly counter any hateful
  idea in the TEXT (if the TEXT is not hateful, a constructive reply).
Return ONLY JSON: {{"rewrites": [...], "responses": [...]}}"""


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(cand)
    except (json.JSONDecodeError, ValueError):
        return None


def to_verdict(d: dict, mode: str) -> Verdict | None:
    """Validate/normalise a raw LLM dict. Returns None if unusable."""
    if not isinstance(d, dict):
        return None
    sev = str(d.get("severity", "none")).lower().strip()
    if sev not in SEVERITIES:
        sev = "hate" if d.get("is_hate") else "none"
    tgt = str(d.get("target", "none")).lower().strip()
    if tgt not in TARGET_CLASSES:
        tgt = "other" if sev == "hate" else "none"
    rewrite = str(d.get("rewrite") or "").strip()
    counter = str(d.get("counter") or "").strip()
    tts = str(d.get("tts_text") or "").strip() or (counter if mode == "counter" else rewrite)
    spans = [str(s) for s in (d.get("spans") or []) if str(s).strip()]
    if sev == "none":
        return Verdict(False, "none", "none", [], "", "", "", str(d.get("reason", "")))
    return Verdict(sev == "hate", sev, tgt, spans, rewrite, counter, tts, str(d.get("reason", "")))


class Llm:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 180.0,
                 temperature: float = 0.2, client=None, reasoning_effort: str = ""):
        if client is None:
            import httpx
            from openai import OpenAI

            local = any(h in base_url for h in ("127.0.0.1", "localhost", "://ollama", "://0.0.0.0"))
            client = OpenAI(base_url=base_url, api_key=api_key or "local", timeout=timeout,
                            http_client=httpx.Client(timeout=timeout, trust_env=not local))
        self.c, self.model, self.temperature = client, model, temperature
        self.reasoning_effort = (reasoning_effort or "").strip()
        self.name = model

    def _create(self, **kw):
        """chat.completions.create with reasoning_effort; dropped for good if the server rejects it."""
        if self.reasoning_effort:
            try:
                return self.c.chat.completions.create(reasoning_effort=self.reasoning_effort, **kw)
            except Exception as e:
                if "reasoning" not in str(e).lower():
                    raise
                log.warning("server rejected reasoning_effort=%s (%s); sending without it",
                            self.reasoning_effort, e)
                self.reasoning_effort = ""
        return self.c.chat.completions.create(**kw)

    def _ask(self, messages: list[dict], structured: bool) -> str:
        kw = dict(model=self.model, temperature=self.temperature, messages=messages)
        if structured:
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "verdict", "schema": SCHEMA}}
        r = self._create(**kw)
        return r.choices[0].message.content or ""

    def judge(self, context: str, sentence: str, lang: str, mode: str = "rewrite",
              feedback: str = "", tone: str = "calm") -> Verdict | None:
        user = (f"language: {lang} ({LANG_NAMES.get(lang, lang)})\nmode: {mode}\n"
                f"tone for rewrite/counter: {TONES.get(tone, TONES['calm'])}\n\n"
                f"TRANSCRIPT (context):\n{context[:6000]}\n\nTARGET SENTENCE:\n{sentence}")
        if feedback:
            user += f"\n\nNOTE: {feedback}"
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        for structured in (True, True, False):
            try:
                raw = self._ask(msgs, structured)
            except Exception as e:
                log.warning("LLM call failed (%s)", e)
                if structured:
                    continue  # server may not support response_format; retry plain
                return None
            v = to_verdict(_extract_json(raw), mode)
            if v is not None:
                return v
        return None

    def rewrite(self, text: str, lang: str, tone: str = "calm", n: int = 2) -> dict | None:
        """Rewrite Studio: always produce alternatives, even for non-hateful input."""
        user = (f"language: {lang} ({LANG_NAMES.get(lang, lang)})\n"
                f"tone: {TONES.get(tone, TONES['calm'])}\n\nTEXT:\n{text}")
        msgs = [{"role": "system", "content": REWRITE_SYSTEM.format(n=n)}, {"role": "user", "content": user}]
        for structured in (True, True, False):
            try:
                kw = dict(model=self.model, temperature=max(self.temperature, 0.5), messages=msgs)
                if structured:
                    kw["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "rewrites", "schema": REWRITE_SCHEMA}}
                raw = self._create(**kw).choices[0].message.content or ""
            except Exception as e:
                log.warning("LLM rewrite failed (%s)", e)
                if structured:
                    continue
                return None
            d = _extract_json(raw)
            if isinstance(d, dict):
                rw = [str(x).strip() for x in d.get("rewrites") or [] if str(x).strip()][:n]
                rs = [str(x).strip() for x in d.get("responses") or [] if str(x).strip()][:n]
                if rw or rs:
                    return {"rewrites": rw, "responses": rs}
        return None

    def ping(self) -> bool:
        try:
            self.c.models.list()
            return True
        except Exception:
            return False
