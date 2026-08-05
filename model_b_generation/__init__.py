# Model B — Counter-Narrative Generator (QLoRA mT5-small)
#
# This package is self-contained. It imports nothing from Model A's
# training internals — the only coupling point is gating.py, which
# consumes Model A's *output* (a dict of predictions from infer.py).
