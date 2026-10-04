"""Assemble the PINNED per-request refusal-score file + the by-kind tables.

⭐ Incremental by design: it writes whatever seeds exist and marks the file
``"status": "complete"`` (written LAST) only when every expected seed is present.
⛔ A seed that did not run is listed under ``not_run``, never silently dropped.
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-arms-a7-wiring")
RUNS = ROOT / "runs"
PUB = Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-baselines/runs/shapes3d_recon")
LEGS = OrderedDict([("historical_688128", 688_128), ("deployed_524288", 524_288)])
SEEDS = (0, 1, 2)


def auroc(neg, pos):
    s = np.concatenate([np.asarray(neg, float), np.asarray(pos, float)])
    y = np.concatenate([np.zeros(len(neg), bool), np.ones(len(pos), bool)])
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    sr = s[o]
    i = 0
    ranks = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = ranks[i:j + 1].mean()
        i = j + 1
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    payload = OrderedDict()
    payload["schema"] = "gen-arms-a7-wiring/refusal-scores/1"
    payload["what"] = (
        "⭐ PER-REQUEST refusal scores for every arm variant, every seed, both byte legs. "
        "A future re-score against a new control is a JOIN on `ids`, ⛔ not a re-run — the "
        "absence of exactly this table is what blocked `gen-baselines-consolidate`."
    )
    payload["legs"] = {k: {"byte_target_bytes": v} for k, v in LEGS.items()}
    payload["controls"] = {
        "duplicate_factor": "a7_redundant (A7)", "out_of_range": "base (instantiated by B)",
        "A7D": "⭐ DECLARED SENSITIVITY — A7 with the redundant term removed; ⛔ not a control",
    }
    data, not_run, checks = OrderedDict(), [], []
    for leg in LEGS:
        data[leg] = OrderedDict()
        for s in SEEDS:
            d = RUNS / leg / f"seed_{s}"
            sc, man = d / "refusal_scores.json", d / "baselines_manifest.json"
            if not (sc.exists() and man.exists()):
                not_run.append(f"{leg}/seed_{s}")
                continue
            payload_s = json.loads(sc.read_text())
            m = json.loads(man.read_text())
            bt = m["provenance"]["byte_target"]
            assert int(bt["total_bytes"]) == LEGS[leg], (leg, bt["total_bytes"])
            data[leg][f"seed_{s}"] = {
                "byte_target": bt,
                "control_map": payload_s["control_map"],
                "scores": payload_s["scores"],
            }
            checks.append({"leg": leg, "seed": s, "realised_bytes": int(bt["total_bytes"]),
                           "expected": bt.get("expected_total_bytes")})
    payload["seeds_present"] = {k: sorted(v) for k, v in data.items()}
    payload["not_run"] = not_run
    payload["byte_target_checks"] = checks
    payload["scores"] = data
    complete = not not_run
    payload["status"] = "complete" if complete else "partial"   # ⭐ written LAST
    out = ROOT / "refusal_scores.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
        fh.write("\n")
    print(f"{out}  status={payload['status']}  not_run={not_run}  "
          f"{out.stat().st_size / 1e6:.1f} MB")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
