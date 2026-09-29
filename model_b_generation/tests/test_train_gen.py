"""
train_gen / evaluate_gen / inference smoke test on CPU with a tiny random mT5
(real mt5 tokenizer, 2 layers x 32 hidden), so the loop, resume, results/
files and adapter loading run in seconds without downloading mt0-base.
"""

import json

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")


@pytest.fixture(scope="session")
def tiny_mt5(tmp_path_factory):
    try:
        tok = transformers.AutoTokenizer.from_pretrained("google/mt5-small")
    except Exception as e:  # offline and not cached
        pytest.skip(f"mt5 tokenizer unavailable: {e}")
    cfg = transformers.MT5Config(vocab_size=len(tok), d_model=32, d_kv=8, d_ff=64, num_layers=2,
                                 num_decoder_layers=2, num_heads=2,
                                 decoder_start_token_id=tok.pad_token_id)
    d = tmp_path_factory.mktemp("tiny_mt5")
    transformers.MT5ForConditionalGeneration(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


def _data(d):
    from model_b_generation.converters_gen._common import make_row
    rows = [make_row("rewrite" if i % 2 else "respond", f"input number {i}", f"output number {i}",
                     ["en", "hi", "ta"][i % 3], "fixture") for i in range(60)]
    df = pd.DataFrame(rows)
    df.iloc[:40].to_parquet(d / "train.parquet")
    df.iloc[40:52].to_parquet(d / "val.parquet")
    df.iloc[52:].to_parquet(d / "test.parquet")
    (d / "manifest.json").write_text("{}")


def _args(data, out, base, *extra):
    return ["--data_dir", str(data), "--output_dir", str(out), "--base", base, "--epochs", "1",
            "--batch_size", "8", "--grad_accum", "1", "--evals_per_epoch", "2",
            "--eval_gen_rows", "6", "--num_workers", "0", "--lora_r", "2",
            "--max_output_length", "8", "--save_every", "1", *extra]


def test_train_resume_results_and_eval(tmp_path, tiny_mt5):
    from model_b_generation import evaluate_gen, train_gen
    data, out = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    _data(data)

    state = train_gen.main(_args(data, out, tiny_mt5))
    assert state["done"] and state["global_step"] == 5
    res = out / "results"
    for f in ("metrics.json", "metrics_history.jsonl", "config.json", "env.json", "samples.jsonl"):
        assert (res / f).exists(), f
    hist = [json.loads(l) for l in (res / "metrics_history.jsonl").read_text().splitlines()]
    assert len(hist) == 3 and {"chrf", "copy_rate", "val_loss", "by_group"} <= set(hist[0])
    assert (out / "best" / "adapter_config.json").exists()
    assert json.loads((res / "config.json").read_text())["args"]["lora_r"] == 2

    # a finished run is a no-op on --resume
    again = train_gen.main(_args(data, out, tiny_mt5, "--resume"))
    assert again["global_step"] == 5

    # resume from the middle of an epoch
    ck = torch.load(out / "last" / train_gen.STATE_FILE, weights_only=False)
    ck["state"].update(done=False, epoch=0, batch=2, global_step=2, bad_evals=0)
    torch.save(ck, out / "last" / train_gen.STATE_FILE)
    resumed = train_gen.main(_args(data, out, tiny_mt5, "--resume"))
    assert resumed["global_step"] == 5 and resumed["done"]

    rep = evaluate_gen.main(["--data_dir", str(data), "--run_dir", str(out), "--base", tiny_mt5,
                             "--max_output_length", "8"])
    assert rep["n"] == 8 and (res / "test_report.json").exists()

    import results_io
    z = results_io.zip_results(out, tmp_path / "civitas_results_model_b.zip")
    import zipfile
    names = zipfile.ZipFile(z).namelist()
    assert "results/metrics.json" in names and not any("adapter" in n for n in names)


def test_inference_loads_adapter_and_generates(tmp_path, tiny_mt5):
    from model_b_generation import train_gen
    from model_b_generation.infer_gen import generate_candidates
    from model_b_generation.model_gen import load_for_inference
    data, out = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    _data(data)
    train_gen.main(_args(data, out, tiny_mt5))

    mb = load_for_inference(str(out / "best"), base=tiny_mt5, device="cpu")
    assert mb.adapter_dir is not None
    cands = generate_candidates(mb, ["rewrite | English | unknown: x", "respond | Tamil | religion: y"],
                                k=3, max_new_tokens=5)
    assert len(cands) == 2 and all(len(c) == 3 for c in cands)

    bare = load_for_inference(str(tmp_path / "nope"), base=tiny_mt5, device="cpu")
    assert bare.adapter_dir is None
