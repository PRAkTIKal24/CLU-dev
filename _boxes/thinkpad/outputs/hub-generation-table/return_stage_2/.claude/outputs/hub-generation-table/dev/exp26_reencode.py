"""DEV-26 — ⛔⛔ THE BLOCKER: does the frozen encoder reproduce its own cache on THIS box?

``exp_gen_store._assert_cache_reencodes`` is the pinned harness's gate at the point where
``gather_features`` extends cache coverage.  Seeds 1 and 2 were written at
``cache_scope="used"`` (9,288 rows) while the cell's store pool is 32,768, so **every run of
the scored cell on seeds 1 and 2 must pass this gate**.

On the first attempt (seed 1) it FAILED: ``max |delta| = 0.00195312 = 81 float16 ULP``
against a bound of 1 ULP, 640/131,072 elements differing (0.488 %, under the 1 % bound).

This measures the gate on **all three seeds**, at full strength, and quantifies the
discrepancy, so the failure is reported as a number rather than a symptom.
⛔ Nothing here is waved through: COMMON §0's override covers exactly one named test and
explicitly does not cover any other red check.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx  # noqa: E402

OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    "exp26_reencode.json")
res = {}
for SEED in (0, 1, 2):
    from chlu.core.gen_bridge import load_feature_cache, param_hash
    from chlu.experiments.exp_gen_store import _assert_cache_reencodes, bridge_dir
    from chlu.utils.checkpoints import load_model

    cfg = ctx.load_config(SEED)
    bdir = bridge_dir(cfg, "shapes3d", "recon", SEED)
    encoder = load_model(bdir / "encoder.pkl")
    crows, feats, cman = load_feature_cache(bdir, verify=False)
    ehash = param_hash(encoder)
    row = {
        "bridge_dir": str(bdir),
        "cache_scope": cman["scope"],
        "cache_rows": int(cman["n_rows"]),
        "cache_dtype": cman["dtype"],
        "cache_sha256": cman["features_sha256"],
        "encoder_param_hash": ehash,
        "encoder_hash_matches_cache_manifest": bool(ehash == cman["encoder_param_hash"]),
    }
    cfg.gen_store.encode_chunk = 2048
    t = time.time()
    try:
        out = _assert_cache_reencodes("shapes3d", cfg, encoder, crows, feats, cman,
                                      n=64, log=ctx.log, will_extend=True,
                                      reason="DEV-26 diagnostic, full strength")
        row["gate"] = "PASS"
        row["detail"] = {k: v for k, v in out.items() if k != "note"} if isinstance(out, dict) else str(out)
    except AssertionError as exc:
        row["gate"] = "FAIL"
        row["detail"] = str(exc)
    row["seconds"] = round(time.time() - t, 1)
    res[f"seed_{SEED}"] = row
    ctx.log(f"seed {SEED}: scope={row['cache_scope']} rows={row['cache_rows']} "
            f"encoder_hash_ok={row['encoder_hash_matches_cache_manifest']} "
            f"GATE={row['gate']} [{row['seconds']}s]")
    if row["gate"] == "FAIL":
        ctx.log("    " + row["detail"][:400])
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

ctx.log("=== VERDICT ===")
for k, v in res.items():
    ctx.log(f"  {k}: scope={v['cache_scope']} gate={v['gate']} "
            f"(a 'used'-scope seed MUST pass this gate for the scored cell to run at all)")
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"wrote {OUTP}")
