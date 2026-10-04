"""w23 spoke/seed5-diagnosis -- QUESTION 1.  ZERO-FIT trace of the closed-form CLU read.

    python scripts/w23_seed5_trace.py <out_dir>

Seed 5, BOTH, TEST, base design (A1: latent 128, K 2, 6 atoms, ridge 1.0, [bias|main|pair]
318).  Loads the PERSISTED codec and clu.pkl (file + param sha256 asserted against the
.sha256 file AND MANIFEST.json before anything is read), rebuilds the eval rows exactly as the
runner did (deterministic banks + hold-out + cap_split, nothing fitted), reproduces the CLU's
per-tuple n-way counts, and then traces the read for every held-out n-way tuple:

  1. the design row (318 columns; the non-zero ones labelled),
  2. its projection on the fitted head  c_k = phi W_k  (6 coefficients per well, K = 2),
  3. the addressed wells/atoms and the read  z_hat = sum_k c_k G_k,
  4. the emitted four facts vs the truth,
  5. the locus tests:
       CONTENT  -- can the deployed atoms hold the right value?  oracle c*_k = argmin
                   ||c G_k - z_true[ch_k]||, decode(sum c*_k G_k) == truth ?
       CODEC    -- decode(encode(truth)) == truth ?  (frozen codec, outside the CLU)
       ADDRESS  -- ||c - c*||, and whether the head's c decodes right,
       DESIGN   -- estimability of phi(t) in the row space of the train design (full, and
                   restricted to the train rows that share this tuple's pair support), the
                   rank of that restricted design, per-column rank drop, ridge leverage;
                   in-sample CLU accuracy on the train rows carrying each of the tuple's
                   pair cells, and the train outcome mix of each pair cell,
       ISOLATION -- block-swap counterfactual between the failing tuple and a same-family
                   tuple the CLU gets right (the difference IS the finding).

⛔ Nothing is fitted, trained, or re-solved.  The head, store, A and Mx are read from the
pickle; ``A`` is asserted equal to Phi_train^T Phi_train rebuilt from the train words.
"""
from __future__ import annotations
import hashlib
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
K = 4
SEED = 5
from chlu.paths import run_dir
PD = str(run_dir("cmaze_t", "symmetric", "test", "persist", "BOTH_seed5", "test"))
def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_file(pdir, name, man):
    """.sha256 side file and MANIFEST must agree, and the file must hash to them."""
    side = open(os.path.join(pdir, name + ".sha256")).read().split()
    side_param, side_file = side[1], side[3]
    fh = fsha(os.path.join(pdir, name))
    rec = {"file_sha256_computed": fh, "file_sha256_sidecar": side_file,
           "file_sha256_manifest": man[name]["file_sha256"],
           "param_sha256_sidecar": side_param,
           "param_sha256_manifest": man[name]["param_sha256"]}
    ok = fh == side_file == man[name]["file_sha256"] and side_param == man[name]["param_sha256"]
    rec["file_match"] = ok
    assert ok, (name, rec)
    return rec


