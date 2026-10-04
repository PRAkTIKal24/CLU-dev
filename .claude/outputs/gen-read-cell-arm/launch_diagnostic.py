"""⭐ WIRING DIAGNOSTIC, ⛔ NOT A TABLE ROW.

The scored rows say the dynamical read lands at ~0.16 (rung A) / ~0.07 (rung B) on
composition-B while the codebook read it replaces sits at 0.5949.  Before that is
reported as a property of the ARM it has to be separated from a property of THIS
WIRING.  The decisive split is the LAUNCH: ``x0`` is a pure function of (request,
address book, atoms) and never touches the chain.

- If the launch scores well and the settled state does not, the settling is what
  destroys the read — a fact about the arm.
- If the launch scores as badly as the settled state, something upstream of the
  chain is wrong and no verdict may be written.

⛔ This is `gen-dynamical-read`'s ``zero_step`` control used as an instrument, and it
is reported as a diagnostic. ⛔ It is NOT added to the generation table.
"""
import json, sys
from pathlib import Path

import numpy as np

from chlu.config import load_config
from chlu.experiments.exp_gen_cell import (
    _cell_root, load_external_classifier, render, score_images, distinct_counts,
)
from chlu.experiments.exp_gen_compose import resolve_decoder
from chlu.experiments.exp_gen_store import bridge_dir

cfg = load_config(Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-val-split-fix/cell_rescore.yaml"))
ART = Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-read-cell-arm/runs/artefacts/shapes3d_recon")
out = {}
for seed in (0, 1, 2):
    bdir = bridge_dir(cfg, "shapes3d", "recon", seed)
    with open(bdir / "feature_cache.json") as fh:
        feat_shape = tuple(int(x) for x in json.load(fh)["feat_shape"])
    cfg.gen_compose.claude_root = str(_cell_root(cfg))
    cfg.gen_compose.decoder_source = str(cfg.gen_cell.decoder_source)
    cfg.gen_compose.winner_root = cfg.gen_cell.winner_root
    cfg.gen_compose.winner_decoder_pattern = str(cfg.gen_cell.winner_decoder_pattern)
    cfg.gen_compose.expected_decoder_recipe = str(cfg.gen_cell.expected_decoder_recipe)
    decoder, dinfo, _ = resolve_decoder(cfg, bdir, seed, log=lambda *a: None)
    ext, _ = load_external_classifier(cfg, seed, log=lambda *a: None)
    d = ART / f"seed_{seed}"
    dec = np.load(d / f"decoded_s{seed}_dynread_B.npz", allow_pickle=True)
    labels, names = dec["labels"], [str(x) for x in dec["factor_names"]]
    for budget in ("A", "B"):
        z = np.load(d / f"settled_states_s{seed}_{budget}_B_full.npz")
        for what in ("x0", "x_settled"):
            X = np.asarray(z[what], np.float32)
            imgs = render(decoder, X, feat_shape, 256)
            sc = score_images(ext, imgs, labels, names, 256)
            dc = distinct_counts(ext, X, imgs, 256)
            out.setdefault(f"{budget}", {}).setdefault(what, []).append(
                (float(sc["aggregate"]), int(dc["factor_tuples"])))
            print(f"seed {seed} rung {budget} {what:10s} composition-B={sc['aggregate']:.4f} "
                  f"tuples={dc['factor_tuples']}/72")
print()
for budget, v in out.items():
    for what, rows in v.items():
        a = [r[0] for r in rows]
        t = [r[1] for r in rows]
        print(f"rung {budget} {what:10s} {np.mean(a):.4f} ± {np.std(a):.4f}  "
              f"per-seed {[round(x,4) for x in a]}  tuples {t}")
Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-read-cell-arm/runs/launch_diagnostic.json"
     ).write_text(json.dumps(out, indent=1))
