"""Experiment: ⭐ **three-point attribution** — WHERE does ``scale`` get lost?

⭐ **DIAL DECLARATION — *attribution*.**
- **Dial:** attribution.  ⛔ **No fix, no new mechanism, no arm.**  This spoke locates
  a known loss (R31/R34), it does not repair it.
- **Laundering control:** the **same probe, same protocol, same rows** at all three
  points.  ⛔ A different probe at each stage would measure the probes, not the
  pipeline.  Plus the **random-code control** at the code point.
- **Falsifies:** nothing of ours — ⭐ both outcomes are informative by design, and
  which one is returned decides where the next wave's effort goes.
- **Does NOT falsify:** a code-level number below the encoder's.  A 30-bit channel
  reconstructing a 2048-d vector loses information *by construction*; the question
  is **which** information, not whether any.

⛔ **What the path actually is** (R40.B, binding on every description of this
measurement): a **product quantiser + count table** — correlation-partitioned wells ×
per-well **k-means** codebook; address book = per-factor-value mode histograms,
naive-Bayes, ``argmax`` per well; read = codebook copy-back.  ⛔ No landscape, no
particle, no settling, no inter-well coupling.  ⛔ "k-particle launch read" names an
``argmax`` over a count table and is ⛔ not written anywhere here.

**The measurement, on the SAME rows with the SAME probe (PREREG §1):**

1. ``encoder`` — the frozen 2048-d feature vector;
2. ``well``    — one well's sub-vector, ``K`` fits, headline ``max_k``;
3. ``code``    — the one-hot of the integer K-tuple after k-means + ``argmax``;
   + ``code_random`` — a uniform code of the SAME width (the mandatory control).

⭐ **Headline rows are the SPREAD grid** — rows spread over the whole factor space and
never fitted by the store.  The **corner** (Montero's 72 held-out ``B`` rows and R16's
matched-seen companion) is reported **separately**; ⛔ a corner-only measurement
overstates the loss, which is global (R34).

⛔ **Every artifact is LOADED, never fitted:** the encoder and feature cache come from
``gen-feature-bridge``, the grouping from ``gen-commitment-collapse``'s cache
(``labels_vec``) and the store from ``gen-read-compose``'s per-seed checkpoint, with
its parameter hash asserted against the commitment manifest.  ⛔ Nothing here trains
anything on the generative path; the only thing fitted is the **measurement probe**.

⚠ **Two environment facts** (R31/R34, both paid for already): a worktree has no
``.claude/``, so ``$CHLU_GEN_DATA_ROOT`` **and** ``$CHLU_CLAUDE_ROOT`` must be set when
running from one, or the real-data path silently skips / the artifacts are not found.

Runnable::

    python -m chlu.experiments.exp_gen_attribution --dataset shapes3d --pretext recon
    python -m chlu.experiments.exp_gen_attribution --quick   # ⚠ plumbing smoke only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, gen_attribution_config, get_default_config  # noqa: E402
from ..core.gen_bridge import PRETEXTS, load_feature_cache, param_hash  # noqa: E402
from ..core.gen_store import address_codes, well_channels_from_labels  # noqa: E402
from ..eval.gen_attribution import (  # noqa: E402
    CONTROL_POINT,
    attribution_verdict,
    chance_rates,
    lookup_upper_bound,
    majority_baseline,
    mi_table,
    n_values_realised,
    onehot_codes,
    per_factor_rows,
    probe_point,
    random_codes,
    retention,
    summarize_across_seeds,
)
from ..eval.gen_probe import factor_labels  # noqa: E402
from ..experiments.exp_gen_compose import _claude_root, _write_json  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from ..utils.checkpoints import load_model  # noqa: E402

#: The row sets the probe is SCORED on.  ⭐ ``SPREAD`` is the headline (the loss is
#: global, R34); ``CORNER_B`` / ``CORNER_MS`` are reported separately and never
#: pooled into it.
EVAL_SETS = ("SPREAD", "CORNER_B", "CORNER_MS")


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------


def _cfg(config: CHLUConfig):
    return gen_attribution_config(config)


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only.  ⛔ A ``--quick`` number is NEVER reported."""
    c = _cfg(config)
    c.seeds = list(c.quick_seeds)
    c.n_spread_train = int(c.quick_n_spread_train)
    c.n_spread_test = int(c.quick_n_spread_test)
    c.probe_steps = int(c.quick_probe_steps)
    c.max_matched_seen = int(c.quick_max_matched_seen)
    c.verify_cache_checksum = bool(c.quick_verify_cache_checksum)
    return config


def attrib_out_root(config: CHLUConfig) -> Path:
    c = _cfg(config)
    if c.out_root:
        return Path(c.out_root).expanduser()
    return _claude_root(config) / "outputs" / "gen-loss-attribution"


def _abs(config: CHLUConfig, p: str) -> Path:
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else _claude_root(config) / q


# ---------------------------------------------------------------------------
# 1. ⛔ The frozen artifacts — loaded and ASSERTED, never rebuilt
# ---------------------------------------------------------------------------


def load_grouping(config: CHLUConfig, dataset: str, pretext: str, seed: int) -> np.ndarray:
    """``gen-commitment-collapse``'s cached ``labels_vec`` — the SAME wells R31 read.

    ⭐ Reusing it costs nothing and buys bit-identity with the codes every prior
    ruling reasoned about; re-deriving the grouping here would need a 32k-row
    correlation pass and could silently produce a different partition.
    """
    c = _cfg(config)
    d = _abs(config, c.commit_cache_pattern.format(
        dataset=dataset, pretext=pretext, seed=int(seed)))
    path = d / f"commit_cache_s{int(seed)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no commitment cache at {path} — ⛔ this spoke reuses "
            "gen-commitment-collapse's grouping so the wells are bit-identical to "
            "the ones R31 reasoned about"
        )
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["labels_vec"], np.int64)


def load_attrib_store(config: CHLUConfig, dataset: str, pretext: str, seed: int,
                      labels_vec: np.ndarray, log: Callable = print):
    """Load ``gen-read-compose``'s per-seed store and prove it is the shipped one.

    Three assertions, because each alone is weak: the parameter **hash** against
    ``gen-commitment-collapse``'s manifest, and the well channels against the
    grouping re-derived from ``labels_vec`` — ⛔ not just their sizes.
    """
    c = _cfg(config)
    path = _abs(config, c.store_checkpoint_source.format(
        dataset=dataset, pretext=pretext, seed=int(seed)))
    if not path.exists():
        raise FileNotFoundError(
            f"no store checkpoint at {path} — ⛔ this spoke reads gen-read-compose's "
            "stores rather than training its own"
        )
    store = load_model(path)
    h = param_hash(store)
    man = (_abs(config, c.commit_cache_pattern.format(
        dataset=dataset, pretext=pretext, seed=int(seed)))
        / f"commit_manifest_s{int(seed)}.json")
    expected = None
    if man.exists():
        with open(man) as fh:
            expected = json.load(fh).get("store_param_hash")
    if c.assert_store_hash and expected and expected != h:
        raise AssertionError(
            f"store at {path} hashes {h} but gen-commitment-collapse recorded "
            f"{expected} — ⛔ this is not the store R31/R34 measured"
        )
    wc = well_channels_from_labels(labels_vec, int(store.n_wells))
    if tuple(store.well_channels) != tuple(wc):
        raise AssertionError(
            "the loaded store's well channels are not the grouping re-derived from "
            "the commitment cache's labels_vec — ⛔ different partitions"
        )
    log(f"  ⭐ store {path.name} hash={h[:16]}… K={store.n_wells} M={store.n_modes} "
        f"{store.bits_per_read:.0f} bits/read (hash + channels asserted)")
    return store, {"store_path": str(path), "store_param_hash": h,
                   "store_param_hash_expected": expected,
                   "K": int(store.n_wells), "n_modes": int(store.n_modes),
                   "dim": int(store.dim), "bits_per_read": float(store.bits_per_read),
                   "well_sizes": [len(w) for w in store.well_channels]}


def select_rows(config: CHLUConfig, dataset: str, split, values, seed: int) -> dict:
    """⭐ The SPREAD pool: ``split.train_rows \\ store_train`` — spread over the whole
    factor space and ⛔ never fitted by the store — plus the corner row sets.

    ⛔ The probe is trained ONLY on ``SPREAD_train``; nothing is ever fitted on the
    72 corner rows.
    """
    c = _cfg(config)
    rows = select_store_rows(split, config, int(seed))
    pool = np.setdiff1d(np.asarray(split.train_rows, np.int64), rows["store_train"])
    rng = np.random.default_rng(int(seed) * int(c.row_seed_mult) + int(c.row_seed_offset))
    n = min(int(c.n_spread_train) + int(c.n_spread_test), pool.size)
    take = rng.permutation(pool)[:n]
    n_tr = min(int(c.n_spread_train), n)
    ms = matched_seen_rows(dataset, split, values, rows["store_train"])
    if int(c.max_matched_seen) and ms.size > int(c.max_matched_seen):
        ms = np.sort(rng.permutation(ms)[: int(c.max_matched_seen)])
    return {
        "SPREAD_train": np.sort(take[:n_tr]),
        "SPREAD": np.sort(take[n_tr:]),
        "CORNER_B": np.asarray(rows["eval_b"], np.int64),
        "CORNER_MS": np.asarray(ms, np.int64),
        "store_train": rows["store_train"],
    }


# ---------------------------------------------------------------------------
# 2. One seed
# ---------------------------------------------------------------------------


def run_seed(config: CHLUConfig, dataset: str, pretext: str, seed: int,
             out_dir: Path, log: Callable = print) -> dict:
    """Probe ``scale``/``orientation`` (and every other factor) at all three points."""
    c = _cfg(config)
    t0 = time.time()
    bdir = bridge_dir(config, dataset, pretext, int(seed))
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    crows, feats, cman = load_feature_cache(bdir, verify=bool(c.verify_cache_checksum))
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    log(f"  bridge {bdir} · cache scope={cman.get('scope')!r} rows={crows.size}")

    from ..core.gen_bridge import dataset_module

    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)

    rowsets = select_rows(config, dataset, split, values, int(seed))
    config.gen_store.encode_chunk = int(c.encode_chunk)
    # ⭐ Seeds 1-2 wrote ``scope="used"`` caches, so most rows below are re-encoded
    # through the FROZEN encoder.  That a re-encoded row is numerically identical to
    # a cached one is ⛔ not assumed: 64 genuinely cached rows are re-encoded and
    # bit-identity is asserted before anything is measured.
    reencode = _assert_cache_reencodes(dataset, config, encoder, crows, feats, cman, log=log)
    X, Y, NVR = {}, {}, {}
    for name in ("SPREAD_train",) + EVAL_SETS:
        r = rowsets[name]
        log(f"    gathering {name}: {r.size} rows")
        X[name] = gather_features(dataset, config, encoder, crows, feats, cman, r, log=log)
        Y[name] = factor_labels(grid, r)
        NVR[name] = n_values_realised(Y[name], names, cards)

    labels_vec = load_grouping(config, dataset, pretext, int(seed))
    store, sinfo = load_attrib_store(config, dataset, pretext, int(seed), labels_vec, log)

    # --- the codes (⭐ the ONLY thing that reaches the read) -----------------
    codes = {k: address_codes(store, np.asarray(v, np.float32)) for k, v in X.items()}
    rnd = {
        k: random_codes(v.shape[0], int(store.n_wells), int(store.n_modes),
                        int(c.random_code_seed) + 977 * int(seed) + 13 * i)
        for i, (k, v) in enumerate(codes.items())
    }

    # --- ⛔ THE ONE PROBE, one key, at every point ---------------------------
    key = jax.random.PRNGKey(int(c.probe_seed_offset) + int(seed))

    def _fit(name, Ztr, Zev):
        return probe_point(
            name, Ztr, Y["SPREAD_train"],
            {k: (Zev[k], Y[k]) for k in EVAL_SETS},
            cards, names, c, key, log=log,
        )

    points = {}
    points["encoder"] = _fit("encoder", X["SPREAD_train"], {k: X[k] for k in EVAL_SETS})
    wells = []
    for k_i, w in enumerate(store.well_channels):
        idx = np.asarray(w, np.int64)
        wells.append(_fit(
            f"well{k_i}", X["SPREAD_train"][:, idx], {k: X[k][:, idx] for k in EVAL_SETS}
        ))
    points["well"] = wells
    M = int(store.n_modes)
    points["code"] = _fit(
        "code", onehot_codes(codes["SPREAD_train"], M),
        {k: onehot_codes(codes[k], M) for k in EVAL_SETS},
    )
    points[CONTROL_POINT] = _fit(
        CONTROL_POINT, onehot_codes(rnd["SPREAD_train"], M),
        {k: onehot_codes(rnd[k], M) for k in EVAL_SETS},
    )

    # --- capacity bounds + the free instrument ------------------------------
    bounds = {}
    if bool(c.lookup_bound):
        for k in EVAL_SETS:
            bounds[k] = {
                "code": lookup_upper_bound(codes["SPREAD_train"], Y["SPREAD_train"],
                                           codes[k], Y[k], names),
                "code_random": lookup_upper_bound(rnd["SPREAD_train"], Y["SPREAD_train"],
                                                  rnd[k], Y[k], names),
            }
    mi = {k: mi_table(codes[k], Y[k], cards, names) for k in ("SPREAD_train",) + EVAL_SETS}

    # --- baselines + verdicts ----------------------------------------------
    chance = chance_rates(cards, names)
    majority = {k: majority_baseline(Y["SPREAD_train"], Y[k], names) for k in EVAL_SETS}

    def _best_well(evalset, nm):
        vals = [w["eval"][evalset]["per_factor"][nm] for w in wells]
        j = int(np.argmax(vals))
        return float(vals[j]), j

    tables, verdicts = {}, {}
    for es in EVAL_SETS:
        enc = points["encoder"]["eval"][es]["per_factor"]
        cod = points["code"]["eval"][es]["per_factor"]
        crn = points[CONTROL_POINT]["eval"][es]["per_factor"]
        bw = {nm: _best_well(es, nm) for nm in names}
        tables[es] = per_factor_rows(
            points["encoder"]["eval"][es], NVR[es], names,
            acc_encoder={nm: enc[nm] for nm in names},
            acc_best_well={nm: bw[nm][0] for nm in names},
            best_well_index={nm: float(bw[nm][1]) for nm in names},
            acc_code={nm: cod[nm] for nm in names},
            acc_code_random={nm: crn[nm] for nm in names},
            chance={nm: chance[nm] for nm in names},
            majority={nm: majority[es][nm] for nm in names},
            retention_best_well={nm: retention(bw[nm][0], enc[nm], chance[nm]) for nm in names},
            retention_code={nm: retention(cod[nm], enc[nm], chance[nm]) for nm in names},
            mi_max_frac={nm: mi[es]["per_factor"][nm]["max_fraction_of_entropy"]
                         for nm in names},
            lookup_code=({nm: bounds[es]["code"]["per_factor"][nm] for nm in names}
                         if bounds else {nm: float("nan") for nm in names}),
            lookup_code_random=({nm: bounds[es]["code_random"]["per_factor"][nm]
                                 for nm in names} if bounds else
                                {nm: float("nan") for nm in names}),
        )
        verdicts[es] = {
            nm: attribution_verdict(
                enc[nm], cod[nm], chance[nm], majority[es][nm], crn[nm],
                float(c.world_a_encoder_floor), float(c.world_a_code_slack),
            )
            for nm in names
        }

    rec = {
        "schema": "chlu-gen-attribution/1",
        "seed": int(seed),
        "dataset": dataset,
        "pretext": pretext,
        "dial": "attribution — ⛔ no fix, no new mechanism, no arm",
        "path_description": (
            "product quantiser + count table: correlation-partitioned wells x per-well "
            "k-means codebook; address book = per-factor-value mode histograms, "
            "naive-Bayes, argmax per well; read = codebook copy-back. ⛔ No landscape, "
            "no particle, no settling, no inter-well coupling on this path."
        ),
        "factor_names": names,
        "cardinalities": [int(x) for x in cards],
        "bridge_dir": str(bdir),
        "encoder_param_hash": enc_hash,
        "cache": {"scope": cman.get("scope"), "n_rows": int(crows.size),
                  "D": int(cman["D"]), "reencode_check": reencode},
        "store": sinfo,
        "rows": {k: int(np.asarray(v).size) for k, v in rowsets.items()},
        "n_values_realised": NVR,
        "chance": chance,
        "majority": majority,
        "points": points,
        "well_sizes": [len(w) for w in store.well_channels],
        "lookup_bounds": bounds,
        "mi": mi,
        "tables": tables,
        "verdicts": verdicts,
        "distinct_codes": {k: int(np.unique(v, axis=0).shape[0]) for k, v in codes.items()},
        "wall_clock_s": float(time.time() - t0),
        "complete": True,
    }
    _write_json(rec, Path(out_dir) / f"attrib_s{int(seed)}.json")
    return rec


# ---------------------------------------------------------------------------
# 3. Across seeds
# ---------------------------------------------------------------------------


def summarize(records: list, config: CHLUConfig) -> dict:
    """3-seed roll-up + ⭐ **the one-line verdict**, in the pre-registered words."""
    c = _cfg(config)
    names = records[0]["factor_names"]
    out = {"n_seeds": len(records), "seeds": [r["seed"] for r in records],
           "factor_names": names, "by_eval_set": {}}
    for es in EVAL_SETS:
        per_factor = {}
        for nm in names:
            rows = [r["tables"][es][nm] for r in records]
            per_factor[nm] = {
                "n_values_realised": sorted({int(x["n_values_realised"]) for x in rows}),
                "degenerate": bool(all(x["degenerate"] for x in rows)),
                "chance": float(rows[0]["chance"]),
                "majority": summarize_across_seeds([x["majority"] for x in rows]),
                "encoder": summarize_across_seeds([x["acc_encoder"] for x in rows]),
                "best_well": summarize_across_seeds([x["acc_best_well"] for x in rows]),
                "code": summarize_across_seeds([x["acc_code"] for x in rows]),
                "code_random": summarize_across_seeds([x["acc_code_random"] for x in rows]),
                "lookup_code": summarize_across_seeds([x["lookup_code"] for x in rows]),
                "lookup_code_random": summarize_across_seeds(
                    [x["lookup_code_random"] for x in rows]),
                "retention_best_well": summarize_across_seeds(
                    [x["retention_best_well"] for x in rows]),
                "retention_code": summarize_across_seeds([x["retention_code"] for x in rows]),
                "mi_max_fraction_of_entropy": summarize_across_seeds(
                    [x["mi_max_frac"] for x in rows]),
                "verdict_per_seed": [r["verdicts"][es][nm]["verdict"] for r in records],
            }
        out["by_eval_set"][es] = per_factor

    # ⭐ THE VERDICT — rendered on the SPREAD grid (the loss is global, R34) and on
    # the pre-registered factors only.
    head = {}
    for nm in list(c.verdict_factors):
        s = out["by_eval_set"]["SPREAD"][nm]
        v = attribution_verdict(
            s["encoder"]["mean"], s["code"]["mean"], s["chance"],
            s["majority"]["mean"], s["code_random"]["mean"],
            float(c.world_a_encoder_floor), float(c.world_a_code_slack),
        )
        v["per_seed_verdicts"] = s["verdict_per_seed"]
        v["retention_best_well"] = s["retention_best_well"]["mean"]
        head[nm] = v
    words = {
        "World A": "World A — the STORE drops it",
        "World B": "World B — the store is INNOCENT",
        "in-between": "in-between (⛔ reported as in-between, never rounded)",
    }
    uniq = sorted({v["verdict"] for v in head.values()})
    out["verdict"] = {
        "per_factor": head,
        "one_line": " · ".join(f"{nm}: {words[head[nm]['verdict']]}" for nm in head),
        "unanimous": len(uniq) == 1,
    }
    return out


def make_plot(summary: dict, path: Path) -> Path:
    """Retention at each point, ``scale``/``orientation`` vs the hue positive control."""
    es = summary["by_eval_set"]["SPREAD"]
    names = [n for n in summary["factor_names"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    pts = ["encoder", "best_well", "code", "code_random"]
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        vals = [es[nm][p]["mean"] for p in pts]
        errs = [es[nm][p]["std"] for p in pts]
        axes[0].bar(np.arange(len(pts)) + i * w, vals, w, yerr=errs, label=nm, capsize=2)
        axes[0].axhline(es[nm]["chance"], color="k", lw=0.4, ls=":")
    axes[0].set_xticks(np.arange(len(pts)) + 0.4)
    axes[0].set_xticklabels(pts)
    axes[0].set_ylabel("probe accuracy (SPREAD test)")
    axes[0].set_title("same probe, three points (+ random-code control)")
    axes[0].legend(fontsize=6, ncol=2)
    for nm in names:
        axes[1].plot([0, 1, 2],
                     [1.0, es[nm]["retention_best_well"]["mean"],
                      es[nm]["retention_code"]["mean"]], "o-", label=nm)
    axes[1].axhline(0.0, color="k", lw=0.5)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["encoder", "best well", "code"])
    axes[1].set_ylabel("retention of above-chance signal")
    axes[1].set_title("where the factor is lost")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_experiment_gen_attribution(config: Optional[CHLUConfig] = None,
                                   out_dir: Optional[str] = None,
                                   log: Callable = print) -> dict:
    config = config or get_default_config()
    c = _cfg(config)
    root = Path(out_dir) if out_dir else attrib_out_root(config)
    runs = root / "runs" / f"{c.dataset}_{c.pretext}"
    runs.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in list(c.seeds):
        path = runs / f"attrib_s{int(seed)}.json"
        if path.exists():
            with open(path) as fh:
                rec = json.load(fh)
            if rec.get("complete"):
                log(f"  seed {seed}: artefact HIT {path.name} — skipped")
                records.append(rec)
                continue
        log(f"  seed {seed}: running")
        records.append(run_seed(config, c.dataset, c.pretext, int(seed), runs, log=log))
    summary = summarize(records, config)
    _write_json(summary, root / "summary.json")
    if bool(c.make_plots):
        make_plot(summary, root / "attribution.png")
    log("⭐ VERDICT: " + summary["verdict"]["one_line"])
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=None)
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS) + [None])
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    config = get_default_config()
    c = _cfg(config)
    if args.dataset:
        c.dataset = args.dataset
    if args.pretext:
        c.pretext = args.pretext
    if args.seeds:
        c.seeds = list(args.seeds)
    if args.quick:
        apply_quick(config)
    run_experiment_gen_attribution(config, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
