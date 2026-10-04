"""⭐ The three behaviours, as they actually print. Stubs the two data sources so the
whole thing runs on a laptop; the gate arithmetic is the shipped arithmetic."""
import numpy as np
from chlu.config import get_default_config
from chlu.experiments import exp_gen_store as X

rng = np.random.default_rng(0)
feats = rng.uniform(4.0, 7.5, size=(8, 64)).astype(np.float32).astype(np.float16).astype(np.float32)
rows = np.arange(8, dtype=np.int64)
cman = {"dtype": "float16", "D": 64, "feat_shape": (1, 1, 64), "n_rows": 8,
        "encoder_param_hash": "deadbeef"}
bad = (feats + np.float32(8 * 0.00390625)).astype(np.float32)   # exactly 8 float16 ULP

X.stream_images = lambda _d, r, _rt, _b: np.zeros((np.asarray(r).size, 1, 1, 64), np.float32)
X.encode_images = lambda _e, imgs, _b: bad[: int(np.asarray(imgs).shape[0])]
cfg = get_default_config()

print("=" * 78)
print("(A) ENFORCED — a run that WILL extend the cache, encoder disagrees")
print("=" * 78)
try:
    X._assert_cache_reencodes("shapes3d", cfg, object(), rows, feats, cman, log=print,
                              will_extend=True, reason="this run will extend the cache")
    print("  !!! NO ABORT — THE GATE LOST ITS TEETH")
except AssertionError as e:
    print(f"  ⛔ ABORTED: {str(e)[:160]}…")

print()
print("=" * 78)
print("(B) NOT ENFORCED — the SAME disagreement, on a fully covered run")
print("=" * 78)
st = X._assert_cache_reencodes(
    "shapes3d", cfg, object(), rows, feats, cman, log=print, will_extend=False,
    reason="cache-coverage gate: 33864/33864 rows cached, 0 shortfall (seed 0, "
           "scope='full') ⇒ this run encodes nothing new and cannot extend the cache")
print("  returned:", {k: st[k] for k in (
    "max_abs_delta", "max_ulp_delta", "n_differing", "frac_differing", "n_checked",
    "ulp_bound", "frac_bound", "enforced", "would_have_failed", "would_have_failed_why")})

print()
print("=" * 78)
print("(C) THE EXTENSION PATH — (B) passed, then the run asks for an uncached row")
print("=" * 78)
print("  _REENCODE_VERIFIED after the unenforced pass:", X._REENCODE_VERIFIED)
try:
    X.gather_features("shapes3d", cfg, object(), rows, feats, cman,
                      np.array([0, 99], np.int64), log=print)
    print("  !!! EXTENDED ANYWAY — THE HOLE IS OPEN")
except AssertionError as e:
    print(f"  ⛔ ABORTED AT THE POINT OF EXTENSION: {str(e)[:160]}…")
