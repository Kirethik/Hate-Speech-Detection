"""
Smoke tests for train.py on CPU with a tiny randomly initialised XLM-R
(real tokenizer, 2 layers x 32 hidden), so the full training loop, resume
and --eval_only paths run in seconds without downloading xlm-roberta-base.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory):
    try:
        tok = transformers.XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    except Exception as e:  # offline
        pytest.skip(f"xlm-roberta-base tokenizer unavailable: {e}")
    cfg = transformers.XLMRobertaConfig(
        vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=160,
    )
    d = tmp_path_factory.mktemp("tiny_xlmr")
    transformers.XLMRobertaModel(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


def _write_splits(d: Path):
    rng = np.random.default_rng(0)
    langs = ["en", "hi", "ta", "te", "ml", "ur_roman"]
    rows = []
    for i in range(240):
        hate = int(i % 2)
        lang = langs[(i // 2) % len(langs)]
        text = (f"these people are vermin number{i}" if hate else f"have a lovely day friend {i}")
        spans = json.dumps([[18, 24]]) if (hate and lang == "en") else "[]"
        rows.append({
            "text": text, "language": lang, "hate_label": hate,
            "target_label": 1 if hate else 0, "severity_label": 2 if hate else 0,
            "rationale_spans": spans, "source": "fixture",
        })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0)
    df.iloc[:160].to_csv(d / "train.csv", index=False)
    df.iloc[160:200].to_csv(d / "val.csv", index=False)
    df.iloc[200:].to_parquet(d / "test.parquet", index=False)  # parquet path is supported too


def _args(data, out, enc, *extra):
    return ["--data_dir", str(data), "--output_dir", str(out), "--encoder_name", enc,
            "--epochs", "1", "--batch_size", "8", "--evals_per_epoch", "2",
            "--num_workers", "0", "--no_amp", "--lang_weights", "", *extra]


def test_train_resume_and_eval_only(tmp_path, tiny_encoder):
    import train
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    _write_splits(data)

    assert train.main(_args(data, out, tiny_encoder, "--save_every", "5")) == 0
    for name in ("best_model.pt", "last.pt", "best_metrics.json", "metrics_history.csv", "train.log"):
        assert (out / name).exists(), name
    ckpt = torch.load(out / "best_model.pt", weights_only=False)
    assert ckpt["normalization_version"] == train.NORMALIZATION_VERSION
    assert 0.0 < ckpt["threshold"] < 1.0
    assert not (out / "test_report.json").exists()   # training never scores test

    # a second run with --resume and more epochs continues instead of restarting
    last = torch.load(out / "last.pt", weights_only=False)
    assert last["state"]["epoch"] == 1 and last["state"]["batch"] == 0
    assert train.main(_args(data, out, tiny_encoder, "--resume") + ["--epochs", "2"]) == 0
    log_text = (out / "train.log").read_text(encoding="utf-8")
    assert "resumed from" in log_text
    assert torch.load(out / "last.pt", weights_only=False)["state"]["epoch"] == 2

    assert train.main(["--eval_only", "--split", "test", "--data_dir", str(data),
                       "--output_dir", str(out), "--batch_size", "8", "--no_amp"]) == 0
    report = json.loads((out / "test_report.json").read_text())
    assert "hate_macro_f1" in report and "hate_by_script" in report


def test_rationale_survives_normalisation(tiny_encoder):
    """Offsets are mapped through normalize_code_mixed instead of dropping the row."""
    import train
    tok = transformers.XLMRobertaTokenizerFast.from_pretrained(tiny_encoder)
    text = "these  people are v e r m i n"          # normalisation changes this text
    start = text.index("v e r")
    df = pd.DataFrame([{"text": text, "language": "en", "hate_label": 1, "target_label": -1,
                        "severity_label": -1, "rationale_spans": json.dumps([[start, len(text)]]),
                        "source": "x", "script": "latin"}])
    ds = train.PreTokenizedDataset(df, tok, 32)
    assert ds.has_rationale[0]
    labels, ids = ds.rationale[0], ds.input_ids[0]
    flagged = tok.decode([int(t) for t, l in zip(ids, labels) if l == 1]).strip()
    assert flagged == "vermin"


def test_refuses_leaked_splits(tmp_path, tiny_encoder):
    import train
    data = tmp_path / "data"
    data.mkdir()
    _write_splits(data)
    val = pd.read_csv(data / "val.csv")
    train_df = pd.read_csv(data / "train.csv")
    pd.concat([train_df, val.iloc[:1]]).to_csv(data / "train.csv", index=False)
    assert train.main(_args(data, tmp_path / "out", tiny_encoder)) == 1


def test_amp_choice():
    import train
    assert train.pick_amp(torch.device("cpu"), False) == (None, False)


# --------------------------------------------------------------------------- #
# augmenters
# --------------------------------------------------------------------------- #
def test_spelling_augment():
    import random
    from spelling_augment import expand_with_spelling_augmentation, perturb_word
    rng = random.Random(0)
    assert perturb_word("women", rng, "space") in {"w omen", "wo men", "wom en"}
    assert len(perturb_word("women", rng, "delete")) == 4
    assert sorted(perturb_word("women", rng, "swap")) == sorted("women")
    assert perturb_word("cat", rng) == "cat"  # too short
    df = pd.DataFrame({"text": [f"people like this are terrible {i}" for i in range(40)],
                       "hate_label": [i % 2 for i in range(40)],
                       "rationale_spans": ["[]"] * 39 + ["[[0, 6]]"], "source": "s"})
    out = expand_with_spelling_augmentation(df, frac=0.5, seed=1)
    added = out.iloc[len(df):]
    assert len(added) > 0 and (added["source"] == "s_typo").all()
    assert set(added["hate_label"]) == {0, 1}          # both classes perturbed
    assert not added["text"].isin(df["text"]).any()


def test_script_augment_expansion():
    pytest.importorskip("indic_transliteration")
    from script_augment import expand_with_script_augmentation
    df = pd.DataFrame({
        "text": ["नमस्ते दोस्तों", "hello friend", "நீ ஒரு நல்ல மனிதன்", "tum acche ho"],
        "language": ["hi", "en", "ta", "hi"], "hate_label": [0, 0, 0, 0],
        "rationale_spans": ["[]"] * 4, "source": "s", "script": ["native", "latin", "native", "latin"],
    })
    out = expand_with_script_augmentation(df, frac=1.0, seed=0)
    added = out.iloc[len(df):]
    assert len(added) == 3                               # English rows are never transliterated
    assert (added["source"] == "s_translit").all()
    assert set(added["script"]) == {"latin", "native"}


def test_infer_loads_training_checkpoint(tmp_path, tiny_encoder):
    """infer.load_model/score_batch/predict work on what train.py writes."""
    import infer
    import train
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    _write_splits(data)
    assert train.main(_args(data, out, tiny_encoder)) == 0

    m = infer.load_model(str(out / "best_model.pt"), "cpu")
    model, tok, thr = m  # legacy tuple unpacking still works
    assert thr == m.threshold
    text = "these p e o p l e are vermin"
    r = infer.score_batch(m, [text, "have a nice day"])
    assert len(r) == 2
    assert set(r[0]["severity"]["probs"]) == set(m.severity_classes)
    assert abs(sum(r[0]["target"]["probs"].values()) - 1.0) < 1e-4
    assert r[0]["normalized_text"] == "these people are vermin"
    for span in r[0]["rationale"]:
        assert text[span["start_char"]:span["end_char"]] == span["text"]

    legacy = infer.predict(model, tok, thr, text, torch.device("cpu"))
    assert legacy["label"] in ("ABUSIVE", "clean")
    assert "vermin" in " ".join(legacy["reasons"]).lower()  # dehumanisation lexicon fires


def test_rationale_spans_widen_to_words_and_merge():
    import infer
    text = "you are vermin trash"
    # normalized == original here; offsets of subword pieces "ver" + "min" and "trash"
    offsets = [(0, 0), (8, 11), (11, 14), (15, 20), (0, 0)]
    scores = [0.0, 0.9, 0.6, 0.7, 0.0]
    spans = infer._rationale_spans(text, offsets, scores, list(range(len(text))))
    assert spans == [{"start_char": 8, "end_char": 20, "text": "vermin trash", "score": 0.9}]
