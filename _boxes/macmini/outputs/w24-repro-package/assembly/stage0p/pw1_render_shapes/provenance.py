"""PROVENANCE-render.json for one build dir: every source file with its sha256, and every
figure file's sha256.  Paths are written ~-relative so two builds are byte-comparable.

    python -m pw1_render_shapes.provenance <build_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import common as C

HOME = "/Users/batmac/clu/"


def _rel(p) -> str:
    return str(p).replace(HOME, "~/clu/")


def main(bdir: Path) -> None:
    req = json.loads((bdir / "REQUESTS-pw1.json").read_text(encoding="utf-8"))
    carried = req["carried_fit_seed0"]
    src = {}

    def add(role, p):
        p = Path(p)
        src[role] = {"path": _rel(p), "sha256": C.sha256(p)}

    add("scan_seed0 (selection input)", C.OUT / "scan" / "verdicts-seed0.json")
    for s in (1, 2):
        p = C.OUT / "scan" / f"verdicts-seed{s}.json"
        if p.exists():
            add(f"scan_seed{s} (counts only)", p)
    add("n-CLU object (K*=32, 131,072 rows; = w20 store_K32_n131072)", C.CLU_PKL[0])
    ck = C.W20_STAGE / "rivals" / "test" / "ckpt" / "seed_0"
    for f in C.FAMILIES:
        for ext in ("eqx", "json"):
            add(f"{f} ({carried[f]}) .rival.{ext}", ck / f"{f}__{carried[f]}.rival.{ext}")
    add("lookup table (floor)", C.STAGE / "outputs/spoke-a-rivals/rivals/seed_0/table.rival.pkl")
    add("w20 TEST object manifest", C.W20_MANIFEST)
    dl = C.STAGE / "outputs/gen-decoder-lift/winner"
    add("frozen decoder seed 0", dl / "decoder_v1_s0.pkl")
    add("decoder winner_hashes (stage)", dl / "winner_hashes.json")
    add("decoder winner_hashes (evidence, asserted)", C.WINNER_HASHES)
    cl = C.STAGE / "outputs/gen-external-classifier/seed_0"
    add("judge classifier seed 0", cl / "external_classifier.pkl")
    add("judge classifier manifest", cl / "classifier_manifest.json")
    vh = C.STAGE / "outputs/data/gen/shapes3d/value_holdout"
    add("split factor_index.npz", vh / "factor_index.npz")
    add("split value_cells.json", vh / "value_cells.json")
    add("feature cache manifest seed 0",
        C.STAGE / "outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/seed_0/feature_cache.json")
    add("SHAPES-MATCHED.json (carried fit)", C.SHAPES_MATCHED)
    for col in C.ORDER:
        add(f"w20 score file seed0 {col} (reproduction target)", C.score_file(0, col, carried))
    code = {}
    for p in sorted((C.REPO / "pw1_render_shapes").glob("*.py")) + [
            C.REPO / "w17_rendered_compositions" / "same_decoder.py",
            C.REPO / "w17_rendered_compositions" / "figure.py"]:
        code[str(p.relative_to(C.REPO))] = C.sha256(p)
    figs = {p.name: C.sha256(p) for p in sorted(bdir.glob("rc_pw1_*"))}
    figs["REQUESTS-pw1.json"] = C.sha256(bdir / "REQUESTS-pw1.json")
    prov = {"schema": "pw1-render-shapes/provenance/1",
            "figures": {"rc_pw1_shapes_per_model": "PW-1 item 9 (a): per-model success/failure grid",
                        "rc_pw1_shapes_baseline_wins": "PW-1 item 9 (b): baseline succeeds, n-CLU fails"},
            "seed": 0, "select_seed": req["select_seed"], "sources": src, "code": code,
            "outputs_sha256": figs,
            "gates": {k: (v if k != "G4_positive_control" else "see REQUESTS-pw1.json gates")
                      for k, v in req["gates"].items()},
            "fits_run": 0}
    (bdir / "PROVENANCE-render.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
