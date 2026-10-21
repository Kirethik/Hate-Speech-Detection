import json
from types import SimpleNamespace

from voice.llm import Llm, _extract_json, to_verdict

GOOD = {"is_hate": True, "severity": "hate", "target": "religion", "spans": ["vermin"],
        "rewrite": "I disagree with them.", "counter": "Nobody is vermin.", "tts_text": "", "reason": "x"}


def test_extract_json_variants():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert _extract_json('<think>hmm {no}</think> {"a": 3}') == {"a": 3}
    assert _extract_json("no json here") is None


def test_to_verdict_normalises():
    v = to_verdict(dict(GOOD), "rewrite")
    assert v.is_hate and v.target == "religion" and v.tts_text == "I disagree with them."
    v = to_verdict(dict(GOOD), "counter")
    assert v.tts_text == "Nobody is vermin."
    v = to_verdict({**GOOD, "target": "aliens", "severity": "HATE"}, "rewrite")
    assert v.target == "other" and v.severity == "hate"
    v = to_verdict({"severity": "none", "rewrite": "ignored"}, "rewrite")
    assert not v.is_hate and v.rewrite == "" and v.target == "none"
    assert to_verdict(None, "rewrite") is None


class FakeClient:
    """First call (structured) raises, second returns fenced JSON."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.kwargs = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.kwargs.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])


def test_judge_retries_and_parses():
    c = FakeClient([RuntimeError("response_format unsupported"), "```json\n" + json.dumps(GOOD) + "\n```"])
    v = Llm("", "m", client=c).judge("ctx", "they are vermin", "en")
    assert v and v.spans == ["vermin"]
    assert "response_format" in c.kwargs[0]


def test_judge_gives_up_cleanly():
    c = FakeClient([RuntimeError("down")] * 3)
    assert Llm("", "m", client=c).judge("ctx", "s", "en") is None


def test_feedback_is_sent():
    c = FakeClient([json.dumps(GOOD)])
    Llm("", "m", client=c).judge("ctx", "s", "en", feedback="still toxic")
    assert "still toxic" in c.kwargs[0]["messages"][1]["content"]


def test_reasoning_effort_is_sent():
    c = FakeClient([json.dumps(GOOD)])
    Llm("", "m", client=c, reasoning_effort="none").judge("ctx", "s", "en")
    assert c.kwargs[0]["reasoning_effort"] == "none"


def test_reasoning_effort_dropped_when_server_rejects_it():
    c = FakeClient([RuntimeError("unknown field reasoning_effort"), json.dumps(GOOD)])
    llm = Llm("", "m", client=c, reasoning_effort="none")
    assert llm.judge("ctx", "s", "en") is not None
    assert "reasoning_effort" not in c.kwargs[1] and llm.reasoning_effort == ""


def test_reasoning_effort_kept_on_unrelated_errors():
    c = FakeClient([RuntimeError("timeout"), json.dumps(GOOD)])
    llm = Llm("", "m", client=c, reasoning_effort="none")
    assert llm.judge("ctx", "s", "en") is not None
    assert c.kwargs[1]["reasoning_effort"] == "none"
