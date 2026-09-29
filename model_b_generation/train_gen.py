"""
Train Model B (mt0-base + LoRA) for Rewrite + Respond. Built for Colab:
resumable after a disconnect, writes everything to --output_dir (Drive).

    python -m model_b_generation.train_gen --data_dir /content/gen_data \
        --output_dir /content/drive/MyDrive/civitas/model_b_v1 --resume

output_dir/
    last/            adapter + trainer_state.pt   (resume point, every --save_every steps)
    best/            adapter + tokenizer          (-> artifacts/model_b/adapter on your PC)
    results/         metrics.json, metrics_history.jsonl, config.json, env.json, samples.jsonl

Checkpoint selection uses val only:
    score = chrF x (1 - copy_rate) x safety_rate   (safety_rate = 1 without Model A)
Test is scored separately, once, by evaluate_gen.py.

Precision: fp32 on T4 (mT5 overflows in fp16), bf16 autocast on L4/A100.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import results_io
from model_b_generation import gen_metrics
from model_b_generation.dataset_gen import GenDataset, make_collate, read_pairs, sampling_weights
from model_b_generation.model_gen import DEFAULT_BASE, load_for_training

STATE_FILE = "trainer_state.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Model B (mt0 + LoRA)")
    p.add_argument("--data_dir", required=True, help="holds train/val .parquet (or .csv)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_input_length", type=int, default=128)
    p.add_argument("--max_output_length", type=int, default=64)
    p.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    p.add_argument("--evals_per_epoch", type=int, default=2)
    p.add_argument("--eval_gen_rows", type=int, default=600,
                   help="val rows used for generation metrics each eval (stratified by task|language)")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--save_every", type=int, default=300, help="optimizer steps between last/ saves")
    p.add_argument("--sample_power", type=float, default=0.5,
                   help="task|language bucket sampling: 1 = natural mix, 0 = uniform")
    p.add_argument("--model_a_ckpt", default="", help="Model A best_model.pt for the safety metric")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_train_rows", type=int, default=0, help="debug: subsample train")
    return p.parse_args(argv)


def pick_dtype(precision: str, device: torch.device):
    if precision == "bf16" or (precision == "auto" and device.type == "cuda"
                               and torch.cuda.is_bf16_supported()
                               and torch.cuda.get_device_capability(0)[0] >= 8):
        return torch.bfloat16
    return None  # fp32


def stratified_rows(df, n: int, seed: int):
    if n <= 0 or len(df) <= n:
        return df.reset_index(drop=True)
    keys = df["task"] + "|" + df["language"]
    per = max(1, n // keys.nunique())
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby(keys)]
    import pandas as pd
    return pd.concat(parts).reset_index(drop=True)


@torch.no_grad()
def generate(model, tok, prompts, device, max_new_tokens, batch_size=32, amp_dtype=None):
    model.eval()
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        outs += tok.batch_decode(ids, skip_special_tokens=True)
    return [o.strip() for o in outs]


@torch.no_grad()
def val_loss(model, loader, device, amp_dtype):
    model.eval()
    total, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            total += model(**b).loss.float().item()
        n += 1
    return total / max(n, 1)


def model_a_scorer(ckpt: str, device):
    if not ckpt or not Path(ckpt).exists():
        return None, 0.5
    from infer import load_model, score_batch
    ma = load_model(ckpt, device)
    return (lambda texts: [r["hate_prob"] for r in score_batch(ma, texts)]), ma.threshold


def selection_score(m: dict) -> float:
    safety = m["safety_rate"] if m.get("safety_rate") is not None else 1.0
    return m["chrf"] * (1.0 - m["copy_rate"]) * safety


def epoch_order(n_rows: int, weights, epoch: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    return torch.multinomial(weights, n_rows, replacement=True, generator=g).tolist()


def main(argv=None):
    a = parse_args(argv)
    started = time.time()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = pick_dtype(a.precision, device)
    print(f"device={device} precision={'bf16' if amp_dtype else 'fp32'}")

    data = Path(a.data_dir)
    find = lambda s: next(p for p in (data / f"{s}.parquet", data / f"{s}.csv") if p.exists())
    train_df, val_df = read_pairs(find("train")), read_pairs(find("val"))
    if a.max_train_rows:
        train_df = train_df.sample(min(a.max_train_rows, len(train_df)), random_state=a.seed)
    gen_df = stratified_rows(val_df, a.eval_gen_rows, a.seed)
    print(f"train={len(train_df)} val={len(val_df)} val_gen={len(gen_df)}")

    last_dir, best_dir = out / "last", out / "best"
    resuming = a.resume and (last_dir / STATE_FILE).exists()
    model, tok = load_for_training(a.base, a.lora_r, a.lora_alpha, a.lora_dropout,
                                   adapter_dir=str(last_dir) if resuming else None)
    model.to(device)

    train_ds = GenDataset(train_df, tok, a.max_input_length, a.max_output_length)
    val_ds = GenDataset(val_df, tok, a.max_input_length, a.max_output_length)
    collate = make_collate(tok.pad_token_id)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size * 2, collate_fn=collate,
                            num_workers=a.num_workers)
    weights = sampling_weights(train_df, a.sample_power)

    from transformers.optimization import Adafactor, get_linear_schedule_with_warmup
    params = [p for p in model.parameters() if p.requires_grad]
    opt = Adafactor(params, lr=a.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    batches_per_epoch = math.ceil(len(train_ds) / a.batch_size)
    total_steps = max(1, batches_per_epoch * a.epochs // a.grad_accum)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup_ratio * total_steps), total_steps)

    state = {"epoch": 0, "batch": 0, "global_step": 0, "best_score": -1.0, "bad_evals": 0,
             "done": False}
    if resuming:
        ck = torch.load(last_dir / STATE_FILE, map_location="cpu", weights_only=False)
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        state.update(ck["state"])
        torch.set_rng_state(ck["torch_rng"]); random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"])
        print(f"resumed from {last_dir}: epoch {state['epoch']} batch {state['batch']} "
              f"step {state['global_step']} best {state['best_score']:.2f}")
    elif not a.resume or not (out / "results" / "config.json").exists():
        results_io.write_config(out, vars(a), data / "manifest.json")
    if state["done"]:
        print("run already finished (patience or epochs reached); nothing to do")
        return state

    score_fn, ma_threshold = model_a_scorer(a.model_a_ckpt, device)
    eval_every = max(1, batches_per_epoch // max(1, a.evals_per_epoch))

    def save_last():
        model.save_pretrained(last_dir)
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "state": state,
                    "torch_rng": torch.get_rng_state(), "py_rng": random.getstate(),
                    "np_rng": np.random.get_state()}, last_dir / STATE_FILE)

    def evaluate():
        vloss = val_loss(model, val_loader, device, amp_dtype)
        prompts = [GenDataset(gen_df, tok).prompt(i) for i in range(len(gen_df))]
        outs = generate(model, tok, prompts, device, a.max_output_length, amp_dtype=amp_dtype)
        groups = (gen_df["task"] + "|" + gen_df["language"]).tolist()
        m = gen_metrics.report(outs, gen_df["target_text"].tolist(), gen_df["source_text"].tolist(),
                               groups, score_fn, ma_threshold)
        m.update(val_loss=round(vloss, 4), epoch=state["epoch"], batch=state["batch"],
                 global_step=state["global_step"])
        m["selection_score"] = round(selection_score(m), 3)
        results_io.append_jsonl(out, "metrics_history.jsonl", m)
        results_io.write_env(out, started)
        print(f"[eval step {state['global_step']}] val_loss={vloss:.4f} chrF={m['chrf']:.1f} "
              f"copy={m['copy_rate']:.3f} safety={m['safety_rate']} score={m['selection_score']:.2f}")
        if m["selection_score"] > state["best_score"]:
            state["best_score"], state["bad_evals"] = m["selection_score"], 0
            model.save_pretrained(best_dir); tok.save_pretrained(best_dir)
            results_io.write_json(out, "metrics.json", {k: v for k, v in m.items()})
            samples = []
            for g in sorted(set(groups)):
                for i in [j for j, x in enumerate(groups) if x == g][:10]:
                    samples.append({"group": g, "prompt": prompts[i], "output": outs[i],
                                    "reference": gen_df["target_text"].iloc[i]})
            results_io.write_jsonl(out, "samples.jsonl", samples)
            print(f"  new best -> {best_dir}")
        else:
            state["bad_evals"] += 1
        model.train()
        return state["bad_evals"] >= a.patience

    model.train()
    stop = False
    while state["epoch"] < a.epochs and not stop:
        order = epoch_order(len(train_ds), weights, state["epoch"], a.seed)
        start = state["batch"] * a.batch_size
        loader = DataLoader(train_ds, batch_size=a.batch_size, sampler=order[start:],
                            collate_fn=collate, num_workers=a.num_workers)
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**b).loss / a.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {state['global_step']} "
                                   "(if you forced bf16 on a T4, use --precision fp32)")
            loss.backward()
            state["batch"] += 1
            if state["batch"] % a.grad_accum == 0 or state["batch"] == batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                state["global_step"] += 1
                if state["global_step"] % 50 == 0:
                    print(f"epoch {state['epoch']} step {state['global_step']}/{total_steps} "
                          f"loss {loss.item() * a.grad_accum:.4f}")
                if state["global_step"] % a.save_every == 0:
                    save_last()
            if state["batch"] % eval_every == 0 or state["batch"] == batches_per_epoch:
                stop = evaluate()
                save_last()
                if stop:
                    print(f"early stop: {a.patience} evals without improvement")
                    break
        if not stop:
            state["epoch"] += 1
            state["batch"] = 0
            save_last()

    state["done"] = True
    save_last()
    results_io.write_env(out, started)
    print(f"done. best selection score {state['best_score']:.2f}; adapter in {best_dir}")
    return state


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
