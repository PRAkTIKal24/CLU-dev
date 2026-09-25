"""Build every w19 panel set into ONE directory (no timestamps, no absolute build paths in
any written file, so two builds are byte-comparable).

    python -m w19_success_failure_panels.build <out_dir>
    bash w19_success_failure_panels/build_twice.sh      # build twice, shasum, install
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from . import cmazet, t7


def _names(obj):
    if isinstance(obj, dict):
        return {k: _names(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_names(v) for v in obj]
    return obj


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    T = t7.build(out_dir)
    C = cmazet.build(out_dir)
    fams = cmazet.family_counts()
    scan_sha = hashlib.sha256(Path(T["scan"]).read_bytes()).hexdigest()
    figs = {("t7_" + k): [Path(p).name for p in v] for k, v in T["figures"].items()}
    figs.update({("cmazet_" + k): [Path(p).name for p in v] for k, v in C["figures"].items()})
    declared = {
        "t7_rival_succeeds_clu_fails": [s for s in T["selection"]["rival_succeeds_clu_fails"]
                                        if "DECLARED" in s],
        "t7_per_arm": [s for s in T["selection"]["per_arm"] if "DECLARED" in s],
        "cmazet_arm_columns": (
            "DECLARED NOT BUILDABLE: every CMAZE-T arm column (CLU, Transformer, MLP, LSTM; "
            "and in the deletion panel CLU downdate, rival as trained, rival retrained without). "
            "w16 maze-wm persisted no arm object and no per-transition prediction: the frozen "
            "codec, frozen state encoder, CLU store, controls and rival weights are trained "
            "in-process by scripts/w16_cmaze_t.py::full and never written; "
            "w16-maze-wm/t1/A1_test_seed*.json holds aggregate scores only. Re-training here "
            "would be a new TEST-leg run with a new codec (forbidden: zero scored runs, never "
            "substitute a codec). The panels drawn are the ENGINE's s_t, action and true "
            "s_t+1 only. See NEEDS-ARTEFACTS.json."),
        "cmazet_one_success_one_failure_per_arm": (
            "DECLARED NOT BUILDABLE for the same reason: choosing a transition where an arm "
            "succeeds or fails needs that arm's per-transition output."),
        "cmazet_rule_a_family_seed0": C["transitions"].get("families_fallback"),
    }
    req = {"schema": "w19-success-failure-panels/requests/1",
           "t7": {"K_star": 32,
                  "selection_rules": t7.__doc__, "scan_file": "outputs/w19-success-failure-panels/scan/verdicts.json",
                  "scan_sha256": scan_sha, "counts_per_seed": T["counts"],
                  "selected": T["selection"]},
           "cmazet": {"selection_rules": cmazet.__doc__, "transitions": C["transitions"],
                      "selection_index": C["selection"],
                      "re_derivation_check": C["re_derivation_check"],
                      "family_counts_per_seed": fams},
           "declared": declared, "figures": figs}
    from w17_rendered_compositions import selection as S
    req["t7"]["arms_order"] = [{"key": a, "label": l, "colour": c} for a, l, c in S.ARMS]
    (out_dir / "REQUESTS-w19.json").write_text(json.dumps(req, indent=1, default=str), encoding="utf-8")
    (out_dir / "SAME-DECODER-w19.json").write_text(json.dumps(
        {"t7_per_seed": T["same_decoder"],
         "cmazet": "no decoder/codec is used: no arm column is rendered (declared); the "
                   "engine panels are drawn from integer engine state"}, indent=1), encoding="utf-8")
    (out_dir / "STATUS-w19.json").write_text(json.dumps(
        {"STATUS": "RENDERED (T7) / ENGINE-ONLY with arm columns DECLARED (CMAZE-T)",
         "figures": figs, "declared": declared}, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
