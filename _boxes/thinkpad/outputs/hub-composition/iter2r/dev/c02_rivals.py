"""C02 — every arm and every §C control, on the SAME DEV corners, on THIS box.

    python c02_rivals.py <tag> <seed> <corner spec>
    corner spec: "sel[:n]" | "val[:n]" | comma-separated indices

⛔ SELECTION ONLY. `dev.dev_setup` refuses any corner that touches eval_a / eval_b.
Resumable at (tag, corner, arm) granularity: a kill costs ONE ARM, never the run.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp  # noqa: E402
import ctx  # noqa: E402
import dev  # noqa: E402
import rivals  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "ref"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SPEC = sys.argv[3] if len(sys.argv) > 3 else "sel:4"
ARMS = sys.argv[4].split(",") if len(sys.argv) > 4 else None


def parse(spec):
    if spec.startswith("sel"):
        idx = comp.SELECTION_IDX
    elif spec.startswith("val"):
        idx = comp.VALIDATION_IDX
    else:
        return [int(x) for x in spec.split(",")]
    n = spec.split(":")
    return idx[: int(n[1])] if len(n) > 1 else idx


T0 = time.time()
IDX = parse(SPEC)
ctx.log(f"C02 tag={TAG} seed={SEED} spec={SPEC} -> {len(IDX)} corners: {IDX}")
c = ctx.build_context(SEED)
ctx.log(f"context ready in {time.time() - T0:.1f}s")

ok, refused = 0, []
for ci in IDX:
    try:
        rivals.run_corner(c, ci, tag=TAG, seed=SEED, arms=ARMS)
        ok += 1
    except dev.HeldOutOverlap as e:
        ctx.log(f"[c{ci}] ⛔ REFUSED (touches a held-out set): {e}")
        refused.append(ci)
    except AssertionError as e:
        # ⭐ a corner whose identity does not hold is REFUSED and NAMED, never trimmed
        ctx.log(f"[c{ci}] ⛔ IDENTITY/ASSERT FAILED — corner dropped and named: {e}")
        refused.append(ci)

ctx.log(f"C02 done: {ok} corners scored, {len(refused)} refused {refused}, "
        f"{time.time() - T0:.1f}s total")
