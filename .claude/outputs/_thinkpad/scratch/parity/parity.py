#!/usr/bin/env python
"""Cross-machine parity probe for the frozen encoder's float32 conv path.

Runs identically on any machine that has a ready bundle. Needs NO dataset, NO GPU,
NO scored run. Deterministic input, so the only variable is the platform.

    VERIFY_PYTHON parity.py          (from inside a workspace, after `. ./env_hub.sh`)
"""
import hashlib, os, platform, sys
import numpy as np

root = os.environ.get("CHLU_CLAUDE_ROOT") or os.path.join(
    os.environ.get("CHLU_ROOT", os.getcwd()), ".claude")
sys.path.insert(0, os.environ.get("CHLU_ROOT", os.getcwd()))

from chlu.core.gen_bridge import encode_images, ENCODE_PRECISION  # noqa: E402
from chlu.utils.checkpoints import load_model                              # noqa: E402

enc_path = os.path.join(root, "outputs", "gen-feature-bridge", "runs",
                        "shapes3d_recon", "seed_0", "encoder.pkl")
encoder = load_model(enc_path)

# deterministic input — ⛔ never the dataset, so this probe needs no h5
rng = np.random.default_rng(20260916)
imgs = rng.random((64, 3, 64, 64), dtype=np.float32)

feats = np.asarray(encode_images(encoder, imgs, 32), dtype=np.float32)
raw = feats.tobytes()

print(f"platform        : {platform.system()} {platform.machine()} py{platform.python_version()}")
print(f"jax/numpy       : {__import__('jax').__version__} / {np.__version__}")
print(f"ENCODE_PRECISION: {ENCODE_PRECISION}")
print(f"feats           : shape={feats.shape} dtype={feats.dtype}")
print(f"sum(float64)    : {feats.astype(np.float64).sum():.10f}")
print(f"max             : {feats.max():.10f}")
print(f"SHA256          : {hashlib.sha256(raw).hexdigest()}")
