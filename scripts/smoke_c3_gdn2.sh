#!/usr/bin/env bash
# scripts/smoke_c3_gdn2.sh — the C3 Gated DeltaNet-2 arm's smoke leg (laptop, seconds).
#
# ############################################################################
# ⛔⛔ THIS IS NEVER A CLAIM VENUE. ⛔⛔
#
# d_model 64, 2 layers, 6 optimisation steps, 0.6 MB of enwik8. Those shapes
# exist to make every SEAM execute, not to make any number mean anything. A bpc
# from this script is not evidence for or against GDN-2, the CLU or anything
# else, IN EITHER DIRECTION. Charter §2: claims live at 26-47 M with >=3 seeds.
# ############################################################################
#
# WHAT IT PROVES (task `c3-rival-arms` §1.6, in order):
#   load -> train -> checkpoint -> RESUME (bit-identical) -> eval -> slices
#   -> byte ledger (published + shrink-to-match rows, reproducing RIVAL_SPECS)
#
#   leg 1  a fresh run, 3 steps, banked
#   leg 2  --resume to 6 steps: the banked step is lifted and training continues
#   leg 3  the uninterrupted 6-step reference, for the bit-identity comparison
#
# ⛔ ZERO ladder arms are trained. The ladder rows are arithmetic on a config.
#
# USAGE:  bash scripts/smoke_c3_gdn2.sh [OUT_DIR]
# ⚠ enwik8 must already be staged (~36 MB, once):
#       python -c "from chlu.data.corpora import stage_corpus; stage_corpus('enwik8')"

set -euo pipefail

OUT="${1:-.claude/outputs/c3-rival-gdn2/smoke}"
PY="${PY:-python}"
EXP="chlu.experiments.exp_c3_rival_gdn2"

echo "=== leg 1: fresh, 3 steps, banked ==============================="
"$PY" -m "$EXP" --out "$OUT/interrupted" --steps 3 --no-download

echo "=== leg 2: --resume to 6 steps + slices ========================="
"$PY" -m "$EXP" --out "$OUT/interrupted" --steps 6 --resume --slices --no-download

echo "=== leg 3: uninterrupted 6 steps (the reference) ================"
"$PY" -m "$EXP" --out "$OUT/uninterrupted" --steps 6 --slices --no-download

echo "=== resume bit-identity ========================================="
"$PY" - "$OUT" <<'EOF'
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
from chlu.utils.checkpoints import load_model

out = sys.argv[1]
a = load_model(f"{out}/uninterrupted/ckpt/gdn2_model.pkl")
b = load_model(f"{out}/interrupted/ckpt/gdn2_model.pkl")
la = jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_inexact_array))
lb = jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_inexact_array))
d = sum(1 for x, y in zip(la, lb, strict=True) if not bool(jnp.all(x == y)))
print(f"leaves {len(la)} | differing {d}")
assert d == 0, "⛔ resume is NOT bit-identical"
print("✅ resume is bit-identical")
EOF
