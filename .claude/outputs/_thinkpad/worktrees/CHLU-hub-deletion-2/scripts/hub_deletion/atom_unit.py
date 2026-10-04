#!/usr/bin/env python
"""PREREG §4d — `delete_atom(j)`: the SCOPED operation, with its residue measured.

⛔⛔ THIS IS NOT AN EXACT DELETE AND IS NEVER REPORTED AS ONE.
Zeroing mode `j`'s row in a well's codebook does not give a store that never held
`j`: the other 63 modes of that well were **fitted with `j` present** — they took
their positions in a 1200-step optimisation that was competing with it, and the
address book's counts were computed over codes that included it.  A canonical
rebuild would move all of them.  So this is a *scoped* edit to a deployed object,
its residue is measured, and it sits beside the exact operator as a contrast:

    exact   delete(v) := rebuild(live \\ rows(v))      O(n), byte-identical
    scoped  delete_atom(j) := zero the column          O(1), NOT canonical

⭐ The interesting number is not that the scoped edit is cheap — it obviously is —
but **how much of `j`'s content survives it**: the requests that used to address `j`
are re-routed to whichever mode is now nearest, and the residue is whether the probe
can still read what `j` used to render.

⛔ Distinct from `gen_store.delete_store(mode=zero|mean|scramble)`, which empties the
WHOLE store and is the laundering control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import jax  # noqa: E402

from chlu.config import CHLUConfig  # noqa: E402
from chlu.core.gen_bridge import (  # noqa: E402
    assert_decoder_hash, dataset_module, load_feature_cache,
)
from chlu.core.gen_compose import compose_codes, read_compose, request_posterior  # noqa: E402
from chlu.core.gen_dream import AddressBook  # noqa: E402
from chlu.core.gen_store import GenStore  # noqa: E402
from chlu.eval.gen_probe import decode_features, factor_labels, predict_factors  # noqa: E402
from chlu.experiments.exp_gen_store import bridge_dir  # noqa: E402
from chlu.utils.checkpoints import load_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_unit import eval_row_sets  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="det_B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--well", type=int, default=0)
    ap.add_argument("--n-atoms", type=int, default=4, help="how many modes to scope-delete")
    ap.add_argument("--n-dev", type=int, default=4096)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    d = Path(args.out) / args.tag
    bman = json.loads((d / "build.json").read_text())
    config = CHLUConfig()
    bdir = bridge_dir(config, "shapes3d", "recon", args.seed)
    crows, feats, cman = load_feature_cache(bdir, verify=False)
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    bridge_man = json.loads((bdir / "bridge_manifest.json").read_text())
    decoder, probe = load_model(bdir / "decoder.pkl"), load_model(bdir / "probe.pkl")
    assert_decoder_hash(decoder, bridge_man["decoder_param_hash"])

    split, _ = dataset_module("shapes3d").build_split(config.gen_data.data_root)
    grid = split.grid
    names = list(grid.names)

    z = np.load(d / "modes.npz")
    nw = len([k for k in z.files if k.startswith("well_")])
    nf = len([k for k in z.files if k.startswith("cond_")])
    wc = [z[f"chan_{j}"] for j in range(nw)]
    modes0 = [np.array(z[f"well_{j}"], np.float32) for j in range(nw)]
    book = AddressBook(z["log_prior"], [z[f"cond_{f}"] for f in range(nf)])
    codes_live = np.asarray(z["codes"], np.int64)

    rows = eval_row_sets(split, config, args.seed, args.n_dev)["dev"]
    labs = np.asarray(factor_labels(grid, rows), np.int32)
    R = labs.astype(np.int64)
    P = request_posterior(book, R)
    codes = np.asarray(compose_codes(P)["code"], np.int64)

    def score(modes):
        st = GenStore(wc, modes[0].shape[0], feat_shape, jax.random.PRNGKey(0), init=modes)
        x = np.asarray(read_compose(st, codes), np.float32)
        dec = decode_features(decoder, x, 256)
        pr = np.asarray(predict_factors(probe, dec, 256), np.int32)
        ok = (pr == labs)
        return {"per_factor": {names[j]: float(ok[:, j].mean()) for j in range(len(names))},
                "aggregate": float(np.mean([ok[:, j].mean() for j in range(len(names))]))}, pr

    intact, pr0 = score(modes0)
    log(f"intact aggregate = {intact['aggregate']:.4f}")

    # ⭐ pick the atoms by LIVE-SET USAGE, declared: the most-used modes of the well,
    # so the scoped delete is given its best chance to matter.  ⛔ Not chosen by
    # looking at any outcome.
    usage = np.bincount(codes_live[:, args.well], minlength=modes0[args.well].shape[0])
    order = np.argsort(-usage)
    picked = [int(j) for j in order[: int(args.n_atoms)]]
    log(f"well {args.well}: scope-deleting modes {picked} (usage {usage[picked].tolist()} "
        f"of {int(usage.sum())})")

    results = {}
    for j in picked:
        m = [a.copy() for a in modes0]
        m[args.well][j] = 0.0                      # the scoped edit, O(1)
        sc, pr = score(m)
        n_addr = int((codes[:, args.well] == j).sum())
        results[str(j)] = {
            "well": args.well, "mode": j,
            "live_rows_addressing_it": int(usage[j]),
            "dev_requests_addressing_it": n_addr,
            "aggregate_after": sc["aggregate"],
            "delta_aggregate": sc["aggregate"] - intact["aggregate"],
            "per_factor_after": sc["per_factor"],
            "frac_predictions_changed": float((pr != pr0).mean()),
            "frac_predictions_changed_among_addressed": (
                float((pr[codes[:, args.well] == j] != pr0[codes[:, args.well] == j]).mean())
                if n_addr else None),
        }
        log(f"  mode {j}: aggregate {intact['aggregate']:.4f} -> {sc['aggregate']:.4f} "
            f"(Δ={results[str(j)]['delta_aggregate']:+.4f}); {n_addr} dev requests addressed it; "
            f"{results[str(j)]['frac_predictions_changed']*100:.2f}% of all predictions moved")

    payload = {
        "schema": "chlu-hub-deletion/atom-scoped/1",
        "WARNING": ("⛔ SCOPED, NOT EXACT. The other modes of this well were fitted WITH "
                    "mode j present and the address book was counted over codes that "
                    "included it; a canonical rebuild would move all of them. This is "
                    "reported as a contrast to the exact operator, never as a delete."),
        "tag": args.tag, "build_digest": bman["digest"]["store_digest"],
        "well": args.well, "atoms": picked, "intact": intact, "scoped": results,
        "n_dev": int(rows.size), "seed": args.seed,
        "wall_s": time.perf_counter() - t0,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = d / "atom_scoped.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    log(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
