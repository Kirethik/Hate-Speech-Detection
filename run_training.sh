#!/usr/bin/env bash
# Launches the corrected Model A training run in the background.
#
# batch_size 16 x grad_accum 2 = effective batch 32. Deliberately not a raw
# batch of 32: AdamW's fp32 moments for xlm-roberta-base already cost ~4.4 GB of
# the 6 GB card, so a peak-length batch of 32 risks an OOM hours into the run.
# 16x2 gets the same effective batch with headroom to spare.
#
# Progress goes to train.log (tail -f it). Crash output lands in train.stderr.log.

set -euo pipefail
cd "$(dirname "$0")"

PY=/home/kirethik/python-envs/general/bin/python

if pgrep -f "train\.py" >/dev/null; then
    echo "A train.py process is still running — stop it first, then re-run this script." >&2
    exit 1
fi

$PY prepare_splits.py

nohup $PY train.py \
    --batch_size 16 \
    --grad_accum 2 \
    --epochs 4 \
    --evals_per_epoch 3 \
    --patience 4 \
    --identity_aug 1 \
    > train.stderr.log 2>&1 &

echo "launched training as PID $! — watch it with:  tail -f train.log"
