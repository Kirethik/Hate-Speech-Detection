"""Model B data: converters on fixture files, leak-free splits, silver rows, review CLI."""

import json

import pandas as pd
import pytest

from model_b_generation.converters_gen import PAIR_COLUMNS, conan, indic_conan, multitarget_conan, paradetox, qian
from model_b_generation.converters_gen._common import group_id, hash_split, make_row
from model_b_generation.prepare_splits_gen import apply_review, prepare, strip_pii


@pytest.fixture
def raw(tmp_path):
    (tmp_path / "multitarget_conan").mkdir()
    pd.DataFrame({
        "INDEX": [1, 2], "HATE_SPEECH": ["h one", "h two"],
        "COUNTER_NARRATIVE": ["c one", "c two"], "TARGET": ["MUSLIMS", "DISABLED"],
        "VERSION": ["V1", "V2"],
    }).to_csv(tmp_path / "multitarget_conan" / "Multitarget-CONAN.csv", index=False)
    (tmp_path / "conan").mkdir()
    pd.DataFrame({"cn_id": ["EN001", "FR001"], "hateSpeech": ["en hate", "fr hate"],
                  "counterSpeech": ["en cn", "fr cn"]}).to_csv(tmp_path / "conan" / "CONAN.csv", index=False)
    (tmp_path / "qian_counter").mkdir()
    pd.DataFrame({"id": [1], "text": ["1. fine post\n2. hateful post\n3. another hateful"],
                  "hate_speech_idx": ["[2, 3]"], "response": ["['reply a', 'reply b']"]}
                 ).to_csv(tmp_path / "qian_counter" / "reddit.csv", index=False)
    (tmp_path / "paradetox").mkdir()
    pd.DataFrame({"en_toxic_comment": ["toxic x", ""], "en_neutral_comment": ["neutral x", "n"]}
                 ).to_csv(tmp_path / "paradetox" / "pd.csv", index=False)
    (tmp_path / "indic_conan").mkdir()
    pd.DataFrame({"hate_speech": ["hi hate"], "counter_narrative": ["hi cn"], "language": ["Hindi"]}
                 ).to_csv(tmp_path / "indic_conan" / "ic.csv", index=False)
    return str(tmp_path)


@pytest.mark.parametrize("module,n,task", [
    (multitarget_conan, 2, "respond"), (conan, 1, "respond"), (qian, 4, "respond"),
    (paradetox, 1, "rewrite"), (indic_conan, 1, "respond"),
])
def test_converters_emit_pair_schema(raw, module, n, task):
    df = module.convert(raw)
    assert list(df.columns) == PAIR_COLUMNS
    assert len(df) == n
    assert set(df["task"]) == {task}
    assert set(df["split"]) <= {"train", "val"}
    assert df[["source_text", "target_text"]].map(lambda s: bool(s.strip())).all().all()


def test_conan_targets_and_language(raw):
    mt = multitarget_conan.convert(raw)
    assert list(mt["target"]) == ["religion", "disability"]
    assert conan.convert(raw)["source_text"].tolist() == ["en hate"]
    assert indic_conan.convert(raw)["language"].tolist() == ["hi"]


def test_qian_pairs_every_hateful_post_with_every_reply(raw):
    df = qian.convert(raw)
    assert sorted(set(df["source_text"])) == ["another hateful", "hateful post"]
    assert df.groupby("source_text").size().tolist() == [2, 2]


def test_missing_sources_return_empty(tmp_path):
    for m in (conan, qian, indic_conan, multitarget_conan):
        if m is multitarget_conan:
            continue  # falls back to the HF hub
        assert m.convert(str(tmp_path)).empty


def _pairs(n=400):
    rows = []
    for i in range(n):
        rows.append(make_row("rewrite" if i % 2 else "respond", f"source sentence {i}",
                             f"target sentence {i}", "en", "fixture"))
    return pd.DataFrame(rows)


def test_prepare_is_leak_free_and_makes_test():
    pairs = _pairs()
    dup = pairs.iloc[[0]].copy()
    dup["group_id"] = "ffffffffffffffff"  # same text, different group: must not leak
    dup["split"] = "val" if pairs.iloc[0]["split"] == "train" else "train"
    splits = prepare(pd.concat([pairs, dup], ignore_index=True))
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["val"]) and len(splits["test"])
    keys = {k: set(v["source_text"].str.casefold()) for k, v in splits.items()}
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"])
    assert not (set(splits["val"]["group_id"]) & set(splits["test"]["group_id"]))


def test_silver_follows_its_origin_group():
    from model_b_generation.silver import make_silver_rows
    pairs = _pairs(200)
    en = pairs.iloc[:40]
    silver = make_silver_rows(en, "ta", [f"ta src {i}" for i in range(40)],
                              [f"ta tgt {i}" for i in range(40)])
    splits = prepare(pairs, silver)
    for name, df in splits.items():
        ta = df[df["language"] == "ta"]
        origin = pairs.set_index("group_id").loc[ta["group_id"], "split"]
        want = {"train"} if name == "train" else {"val"}
        assert set(origin) <= want


def test_review_drop_and_fix():
    from model_b_generation.silver import make_silver_rows
    en = _pairs(4)
    silver = make_silver_rows(en, "hi", ["a", "b", "c", "d"], ["w", "x", "y", "z"])
    sid = silver["silver_id"].tolist()
    out = apply_review(silver, {sid[0]: {"decision": "drop"},
                                sid[1]: {"decision": "fix", "target_text": "fixed"}})
    assert len(out) == 3
    assert out.set_index("silver_id").loc[sid[1], "target_text"] == "fixed"


def test_romanized_urdu_silver_is_latin():
    from model_b_generation.silver import make_silver_rows
    pytest.importorskip("indic_transliteration")
    df = make_silver_rows(_pairs(2), "ur_roman", ["सब लोग", "नमस्ते"], ["अच्छा", "दोस्त"])
    assert all(ch.isascii() for ch in "".join(df["source_text"]))


def test_strip_pii():
    s = strip_pii("mail a@b.com or https://x.y/z call +91 98765 43210 @someone")
    assert "a@b.com" not in s and "https" not in s and "98765" not in s and "@someone" not in s


def test_hash_split_is_stable():
    g = group_id("hello")
    assert hash_split(g) == hash_split(g)


def test_review_cli_records_decisions(tmp_path):
    from model_b_generation import review_silver
    from model_b_generation.silver import make_silver_rows
    silver = make_silver_rows(_pairs(3), "te", ["a", "b", "c"], ["x", "y", "z"])
    sp = tmp_path / "s.parquet"
    silver.to_parquet(sp)
    answers = iter(["k", "f", "", "better", "d"])
    rv = tmp_path / "r.jsonl"
    review_silver.main(["--silver", str(sp), "--review", str(rv), "--language", "te"],
                       input_fn=lambda _: next(answers))
    recs = [json.loads(l) for l in rv.read_text(encoding="utf-8").splitlines()]
    assert [r["decision"] for r in recs] == ["keep", "fix", "drop"]
    assert recs[1] == {**recs[1], "target_text": "better"} and "source_text" not in recs[1]
    # already-reviewed rows are not shown again
    review_silver.main(["--silver", str(sp), "--review", str(rv)], input_fn=lambda _: "q")
