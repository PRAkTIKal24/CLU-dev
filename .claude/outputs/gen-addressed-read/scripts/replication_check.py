"""⭐ PREREG §4 assertion A6 — is this arm's `x0` the SAME object the diagnostic scored?

`gen-read-cell-arm` persisted, per seed and per rung, the launch its chain started from
(`settled_states_s{S}_{rung}_B_full.npz::x0`) and measured it as a WIRING DIAGNOSTIC.
This spoke promotes that launch to a scored arm. ⛔ If the two arrays are not identical
the wiring differs and ⛔ nothing here may be compared with that diagnostic.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path("/Users/user/Desktop/CHLU")
MINE = REPO / ".claude/outputs/gen-addressed-read/runs/artefacts/shapes3d_recon"
THEIRS = REPO / ".claude/outputs/gen-read-cell-arm/runs/artefacts/shapes3d_recon"

out = {"what": __doc__.strip().splitlines()[0], "per_seed_per_rung": {}}
allok = True
for seed in (0, 1, 2):
    for rung in ("A", "B"):
        a = MINE / f"seed_{seed}" / f"x0_states_s{seed}_{rung}_B_full.npz"
        b = THEIRS / f"seed_{seed}" / f"settled_states_s{seed}_{rung}_B_full.npz"
        k = f"seed_{seed}/{rung}"
        if not a.exists() or not b.exists():
            out["per_seed_per_rung"][k] = {"NOT_RUN": f"missing {a if not a.exists() else b}"}
            allok = False
            continue
        x, y = np.load(a)["x0"], np.load(b)["x0"]
        same = bool(x.shape == y.shape and np.array_equal(x, y))
        out["per_seed_per_rung"][k] = {
            "bit_identical": same,
            "shape_mine": list(x.shape), "shape_theirs": list(y.shape),
            "max_abs_delta": float(np.max(np.abs(x - y))) if x.shape == y.shape else None,
        }
        allok &= same
        print(f"seed {seed} rung {rung}: bit-identical = {same}  shape {x.shape}")
out["A6_ALL_BIT_IDENTICAL"] = bool(allok)
out["verdict"] = (
    "⭐ REPLICATED BIT-FOR-BIT: this arm's `x0` IS the array `gen-read-cell-arm` "
    "measured as a diagnostic — the promotion changed the SCORING (ledger, floors, "
    "controls), ⛔ not the read." if allok else
    "⛔⛔ NOT BIT-IDENTICAL — the wiring differs from the diagnostic's and ⛔ no number "
    "here may be compared with it (PREREG §6 F5 FIRED).")
print("\n" + out["verdict"])
(REPO / ".claude/outputs/gen-addressed-read/runs/replication_check.json").write_text(
    json.dumps(out, indent=1))
