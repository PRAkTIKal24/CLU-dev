"""CPU bit-identity + cost probe for `encode_images`.

Run the SAME script against two trees via PYTHONPATH (main vs the branch) and
compare the sha256 of the raw feature bytes.  ⛔ Any difference in the digest
means the precision change moved the laptop's numbers, which is the laundering
control failing.
"""
import hashlib
import json
import sys
import time

import jax
import numpy as np

from chlu.core.gen_bridge import GenEncoder, encode_images, param_hash

TAG = sys.argv[1]
OUT = sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5

# The shipped default bridge encoder: 3x64x64 -> (128, 4, 4), D = 2048.
enc = GenEncoder((3, 64, 64), [32, 64, 128, 128], 4, 2, 8, jax.random.PRNGKey(0))
rng = np.random.default_rng(1234)
imgs = rng.random((64, 3, 64, 64), dtype=np.float32)

t0 = time.perf_counter()
feats = encode_images(enc, imgs, 32)          # includes trace + compile
t_first = time.perf_counter() - t0

reps = []
for _ in range(REPS):
    t = time.perf_counter()
    f = encode_images(enc, imgs, 32)
    reps.append(time.perf_counter() - t)
    assert f.tobytes() == feats.tobytes(), "encode_images is not deterministic in-process"

raw = np.ascontiguousarray(feats, dtype=np.float32).tobytes()
rec = {
    "tag": TAG,
    "jax": jax.__version__,
    "backend": jax.devices()[0].platform,
    "device_kind": jax.devices()[0].device_kind,
    "encoder_param_hash": param_hash(enc),
    "feat_shape": list(feats.shape),
    "dtype": str(feats.dtype),
    "n_elements": int(feats.size),
    "sha256": hashlib.sha256(raw).hexdigest(),
    "sum_f64": float(np.asarray(feats, np.float64).sum()),
    "max": float(feats.max()),
    "t_first_call_s": t_first,
    "t_repeat_s": reps,
    "t_repeat_median_s": float(np.median(reps)),
}
np.save(f"{OUT}/{TAG}_features.npy", feats)
with open(f"{OUT}/{TAG}_probe.json", "w") as fh:
    json.dump(rec, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(json.dumps(rec, indent=2, sort_keys=True))
