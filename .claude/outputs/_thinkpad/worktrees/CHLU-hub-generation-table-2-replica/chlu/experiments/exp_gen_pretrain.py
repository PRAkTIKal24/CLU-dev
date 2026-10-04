"""⭐ ``gen-encoder-pretrain`` — does a reconstruction pretext KEEP ``scale`` and
``orientation``?

## DIAL DECLARATION
- **Dial:** the encoder's pretext.  ⛔ The store is **not** touched and the encoder
  stays **frozen** for phase 1; this produces a *candidate*, not a pipeline change.
- **Laundering controls (two, both full rows on every table):** the current
  ``recon`` encoder — the shipped ``gen_bridge.train_encoder(pretext="recon")``
  code path, called verbatim — and ``random_init``, an untrained conv stack, which
  is the floor any "the encoder learned it" sentence must clear.  Plus
  ``recon_control_2x``, the **compute** control: ⛔ a win bought with extra steps
  is not a representation win.
- **Falsifies:** ``scale`` **and** ``orientation`` ≥ ``pass_bar`` (0.90) on the
  **spread grid**, on SEEN data, for some variant.  ⛔ If no variant clears it,
  the result is FAIL — report and stop; do not touch the store.
- **Does NOT falsify:** losing on the hue factors (saturated for every arm); a
  **corner-only** pass (the corner is 216 rows realising 3 of 8 ``scale`` values —
  ⛔ never substituted for the grid); the VAE comparison arm trailing the AE
  headline (pre-registered as expected).

## What a new encoder VOIDS
⛔ A changed encoder changes the feature cache, and **every phase-1 number is
computed on that cache.**  This experiment therefore writes to **fresh paths**
and overwrites, re-fits and invalidates nothing.  ``chlu/core/gen_bridge.py`` is
read-only to this spoke; the shipped cache stays bit-identical.

Usage::

    python -m chlu.experiments.exp_gen_pretrain --quick        # ⚠ plumbing only
    python -m chlu.experiments.exp_gen_pretrain --out <dir> --seeds 0 1 2
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

from ..config import CHLUConfig, load_config  # noqa: E402
from ..core.gen_bridge import encode_images, param_hash, stream_images
from ..core.gen_encoders import (
    PRETRAIN_VARIANTS,
    PROBE_KINDS,
    bar_verdict,
    fit_feature_probe,
    realised_values,
    standardize,
    standardize_stats,
    train_variant,
    variant_spec,
)
from ..eval.gen_probe import probe_accuracy, summarize_per_factor, summarize_seeds
from ..experiments.exp_gen_bridge import _data_root, load_split
from ..eval.gen_probe import factor_labels

SCHEMA = "chlu-gen-pretrain/1"

#: The two pools.  ⚠ trap 2: a CORNER-only number will mislead — the loss this
#: spoke chases is GLOBAL, so the SPREAD grid is the test and the corner is a
#: separate, smaller, differently-supported column.
POOLS = ("grid", "corner")


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — a quick run's probe accuracy is NEVER reported."""
    gp = config.gen_pretrain
    gp.variants = list(gp.quick_variants)
    gp.seeds = list(gp.quick_seeds)
    gp.enc_steps = int(gp.quick_enc_steps)
    gp.probe_steps = int(gp.quick_probe_steps)
    gp.n_enc_train = int(gp.quick_n_enc_train)
    gp.n_probe_train = int(gp.quick_n_probe_train)
    gp.n_probe_eval = int(gp.quick_n_probe_eval)
    return config


def flag_provenance(config: CHLUConfig, quick: bool) -> dict:
    """Every non-default knob in force, for the report's flag-provenance table."""
    gp = config.gen_pretrain
    return {
        "quick": bool(quick),
        "dataset": str(gp.dataset),
        "variants": list(gp.variants),
        "seeds": [int(s) for s in gp.seeds],
        "n_enc_train": int(gp.n_enc_train),
        "n_probe_train": int(gp.n_probe_train),
        "n_probe_eval": int(gp.n_probe_eval),
        "enc_channels": list(gp.enc_channels),
        "hires_channels": list(gp.hires_channels),
        "enc_kernel": int(gp.enc_kernel),
        "enc_stride": int(gp.enc_stride),
        "enc_groups": int(gp.enc_groups),
        "enc_steps": int(gp.enc_steps),
        "enc_batch": int(gp.enc_batch),
        "enc_lr": float(gp.enc_lr),
        "dec_channels": list(gp.dec_channels),
        "dec_groups": int(gp.dec_groups),
        "deep_width_mult": float(gp.deep_width_mult),
        "deep_blocks_per_stage": int(gp.deep_blocks_per_stage),
        "mae_mask_ratio": float(gp.mae_mask_ratio),
        "mae_patch": int(gp.mae_patch),
        "mae_loss_on_masked": bool(gp.mae_loss_on_masked),
        "vae_beta": float(gp.vae_beta),
        "compute_control_mult": int(gp.compute_control_mult),
        "probe_kinds": list(gp.probe_kinds),
        "probe_hidden": int(gp.probe_hidden),
        "probe_steps": int(gp.probe_steps),
        "probe_batch": int(gp.probe_batch),
        "probe_lr": float(gp.probe_lr),
        "pass_bar": float(gp.pass_bar),
        "bar_factors": list(gp.bar_factors),
        "jax_version": str(jax.__version__),
    }


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def corner_rows(dataset: str, split, values: np.ndarray) -> np.ndarray:
    """The **seen** rows inside Montero's photometric corner (``matched_seen``)."""
    from .exp_gen_store import matched_seen_rows

    return matched_seen_rows(dataset, split, values, np.zeros(0, np.int64))


def select_pretrain_rows(split, corner: np.ndarray, cfg, seed: int) -> dict:
    """Deterministic, **arm-independent** rows: every variant sees exactly these.

    ``enc_train`` is drawn from ``split.train_rows`` exactly as the shipped
    ``select_rows`` does (⛔ the corner is *not* excluded from it — the shipped
    encoder sees corner rows and the control must too).  The probe pools are then
    drawn from what is left **with the corner removed**, so the SPREAD grid column
    is corner-free and the CORNER column is disjoint from every fit set.
    """
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(np.asarray(split.train_rows, np.int64))
    n_enc = min(int(cfg.n_enc_train), perm.size)
    enc_train = perm[:n_enc]
    rest = perm[n_enc:]
    corner = np.asarray(corner, np.int64)
    spread = rest[~np.isin(rest, corner)]
    n_pt = min(int(cfg.n_probe_train), spread.size)
    n_pe = min(int(cfg.n_probe_eval), spread.size - n_pt)
    corner_eval = np.setdiff1d(corner, enc_train)
    return {
        "enc_train": np.sort(enc_train),
        "probe_train": np.sort(spread[:n_pt]),
        "probe_eval": np.sort(spread[n_pt : n_pt + n_pe]),
        "corner_eval": np.sort(corner_eval) if bool(cfg.include_corner) else np.zeros(0, np.int64),
        "n_corner_dropped_in_enc_train": int(corner.size - corner_eval.size),
    }


def assert_rows_disjoint(rows: dict) -> dict:
    """⛔ The probe may not be scored on a row it was fitted on, ever."""
    out = {}
    pairs = [("enc_train", "probe_train"), ("enc_train", "probe_eval"),
             ("probe_train", "probe_eval"), ("probe_train", "corner_eval"),
             ("enc_train", "corner_eval")]
    for a, b in pairs:
        n = int(np.intersect1d(rows[a], rows[b]).size)
        if n:
            raise AssertionError(f"rows {a!r} and {b!r} overlap in {n} rows")
        out[f"{a}|{b}"] = 0
    return out


# ---------------------------------------------------------------------------
# One (variant, seed)
# ---------------------------------------------------------------------------


def run_variant_seed(
    variant: str,
    seed: int,
    pools: dict,
    labels: dict,
    rows: dict,
    config: CHLUConfig,
    out_dir: Path,
    log: Callable = print,
) -> dict:
    """Train one arm, freeze it, probe its features on both pools, verdict."""
    cfg = config.gen_pretrain
    grid_names = pools["factor_names"]
    spec = variant_spec(variant, cfg)

    t0 = time.time()
    key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_909)  # the bridge's chain
    key, ek = jax.random.split(key)
    log(f"  [{variant}] fitting encoder ({spec.family}, x{spec.steps_mult} steps) …")
    encoder, aux, hist, meta = train_variant(
        pools["enc_train"], variant, cfg, ek, bridge_cfg=config.gen_bridge, log=log
    )
    t_train = time.time() - t0

    # ⛔ FROZEN from here: one forward pass, no gradients, no adaptation.
    t1 = time.time()
    feats = {
        name: encode_images(encoder, pools[name], int(cfg.eval_batch))
        for name in ("probe_train", "probe_eval", "corner_eval")
        if pools[name].shape[0]
    }
    t_encode = time.time() - t1

    stats = standardize_stats(feats["probe_train"])
    Z = {k: standardize(v, stats) for k, v in feats.items()}
    cards = pools["cardinalities"]

    probes, verdicts = {}, {}
    for kind in [k for k in cfg.probe_kinds if k in PROBE_KINDS]:
        t2 = time.time()
        pkey = jax.random.PRNGKey(int(seed) * 104_729 + 20_260_914)
        probe, phist = fit_feature_probe(
            Z["probe_train"], labels["probe_train"], cards, cfg, pkey, kind=kind
        )
        rec = {
            "kind": kind,
            "hidden": int(cfg.probe_hidden) if kind == "mlp" else 0,
            "steps": int(cfg.probe_steps),
            "param_hash": param_hash(probe),
            "loss_first": float(phist[0]) if phist else None,
            "loss_final": float(phist[-1]) if phist else None,
            "fit_seconds": float(time.time() - t2),
            # ⭐ train accuracy too: a probe that cannot fit its own fit set is not
            # measuring the features, it is measuring an optimisation failure.
            "train": probe_accuracy(probe, Z["probe_train"], labels["probe_train"],
                                    grid_names, int(cfg.eval_batch)),
            "grid": probe_accuracy(probe, Z["probe_eval"], labels["probe_eval"],
                                   grid_names, int(cfg.eval_batch)),
        }
        if "corner_eval" in Z:
            rec["corner"] = probe_accuracy(probe, Z["corner_eval"], labels["corner_eval"],
                                           grid_names, int(cfg.eval_batch))
        probes[kind] = rec
        verdicts[kind] = {
            "grid": bar_verdict(rec["grid"]["per_factor"], cfg.bar_factors, cfg.pass_bar),
            **({"corner": bar_verdict(rec["corner"]["per_factor"], cfg.bar_factors,
                                      cfg.pass_bar)} if "corner" in rec else {}),
        }
        log(f"    [{variant}/{kind}] grid " + " ".join(
            f"{f}={rec['grid']['per_factor'][f]:.4f}" for f in cfg.bar_factors)
            + f"  -> {verdicts[kind]['grid']['verdict']}")

    result = {
        "schema": SCHEMA,
        "dataset": str(cfg.dataset),
        "variant": variant,
        "seed": int(seed),
        "encoder": meta,
        "rows": {k: int(np.asarray(v).size) for k, v in rows.items()
                 if isinstance(v, np.ndarray)},
        "n_corner_dropped_in_enc_train": int(rows["n_corner_dropped_in_enc_train"]),
        "pools": {
            "grid": {"n": int(pools["probe_eval"].shape[0]),
                     "realised": realised_values(labels["probe_eval"], grid_names)},
            "corner": ({"n": int(pools["corner_eval"].shape[0]),
                        "realised": realised_values(labels["corner_eval"], grid_names)}
                       if pools["corner_eval"].shape[0] else None),
            "probe_train": {"n": int(pools["probe_train"].shape[0]),
                            "realised": realised_values(labels["probe_train"], grid_names)},
        },
        "probes": probes,
        "bar": verdicts,
        "timing": {"train_seconds": float(t_train), "encode_seconds": float(t_encode)},
        "flags": flag_provenance(config, quick=False),
    }
    _write_json(result, out_dir / "result.json")
    log(f"  [{variant}] done in {t_train:.1f}s train / {t_encode:.1f}s encode")
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list, config: CHLUConfig) -> dict:
    """Across-seed summary per (variant, probe kind, pool) + the PASS/FAIL call."""
    cfg = config.gen_pretrain
    names = list(results[0]["probes"][list(results[0]["probes"].keys())[0]]["grid"]["per_factor"])
    by_variant = {}
    for variant in dict.fromkeys(r["variant"] for r in results):
        rs = [r for r in results if r["variant"] == variant]
        entry = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs],
                 "encoder": {k: rs[0]["encoder"][k] for k in
                             ("family", "note", "is_control", "is_comparison_arm",
                              "enc_channels", "dec_channels", "dec_blocks_per_stage",
                              "steps", "steps_mult", "encoder_param_count",
                              "pretext_head_param_count", "D", "feat_shape", "objective")},
                 "loss_final": summarize_seeds(
                     [{"v": r["encoder"]["loss_final"] if r["encoder"]["loss_final"] is not None
                       else float("nan")} for r in rs], "v"),
                 "train_seconds": summarize_seeds(
                     [{"v": r["timing"]["train_seconds"]} for r in rs], "v"),
                 "by_probe": {}}
        for kind in rs[0]["probes"]:
            k_entry = {}
            for pool in POOLS:
                if pool not in rs[0]["probes"][kind]:
                    continue
                per_seed = [r["probes"][kind][pool] for r in rs]
                k_entry[pool] = {
                    "aggregate": summarize_seeds(per_seed, "aggregate"),
                    "per_factor": summarize_per_factor(per_seed, names),
                    "n": int(per_seed[0]["n"]),
                    "realised": rs[0]["pools"][pool]["realised"],
                    "bar": bar_verdict(
                        {f: float(np.mean([p["per_factor"][f] for p in per_seed]))
                         for f in names}, cfg.bar_factors, cfg.pass_bar),
                    "per_seed_bar": [
                        bar_verdict(p["per_factor"], cfg.bar_factors, cfg.pass_bar)["verdict"]
                        for p in per_seed],
                }
            entry["by_probe"][kind] = k_entry
        by_variant[variant] = entry

    headline = "linear"
    clears = [v for v, e in by_variant.items()
              if headline in e["by_probe"]
              and e["by_probe"][headline].get("grid", {}).get("bar", {}).get("verdict") == "PASS"]
    mlp_only = [v for v, e in by_variant.items()
                if "mlp" in e["by_probe"]
                and e["by_probe"]["mlp"].get("grid", {}).get("bar", {}).get("verdict") == "PASS"
                and v not in clears]
    return {
        "schema": SCHEMA,
        "factor_names": names,
        "headline_probe": headline,
        "bar": float(cfg.pass_bar),
        "bar_factors": list(cfg.bar_factors),
        "by_variant": by_variant,
        "variants_clearing_bar_grid_linear": clears,
        "variants_clearing_bar_grid_mlp_only": mlp_only,
        "verdict": "PASS" if clears else "FAIL",
        "note": (
            "at least one variant reaches the bar on BOTH bar factors on the SPREAD "
            "grid with the headline LINEAR probe"
            if clears else
            "⛔ NO variant clears the bar on the spread grid with the headline linear "
            "probe. Per the pre-registered stop rule: report and STOP — the store is "
            "not touched and no cache is promoted."
        ),
        "flags": flag_provenance(config, quick=False),
    }


def write_tables(summary: dict, path: Path) -> Path:
    """A markdown table per (probe, pool) with ``n_values_realised`` on every row."""
    names = summary["factor_names"]
    lines = [f"# gen-encoder-pretrain — bar {summary['bar']} on {summary['bar_factors']}",
             "", f"**VERDICT: {summary['verdict']}** — {summary['note']}", ""]
    for kind in ("linear", "mlp"):
        for pool in POOLS:
            rows = [(v, e) for v, e in summary["by_variant"].items()
                    if kind in e["by_probe"] and pool in e["by_probe"][kind]]
            if not rows:
                continue
            realised = rows[0][1]["by_probe"][kind][pool]["realised"]
            n = rows[0][1]["by_probe"][kind][pool]["n"]
            lines += [f"## probe={kind} · pool={pool} (n={n})", "",
                      "| n_values_realised | " + " | ".join(
                          str(realised[f]["n_values_realised"]) for f in names) + " | | |",
                      "| chance | " + " | ".join(
                          f"{realised[f]['chance']:.3f}" for f in names) + " | | |",
                      "| variant | " + " | ".join(names) + " | aggregate | bar |",
                      "|" + "---|" * (len(names) + 3)]
            for v, e in rows:
                pf = e["by_probe"][kind][pool]["per_factor"]
                agg = e["by_probe"][kind][pool]["aggregate"]
                bar = e["by_probe"][kind][pool]["bar"]["verdict"]
                lines.append(
                    f"| {v} | " + " | ".join(f"{pf[f]['mean']:.4f} ± {pf[f]['std']:.4f}"
                                             for f in names)
                    + f" | {agg['mean']:.4f} | **{bar}** |")
            lines.append("")
    lines += ["## compute (⭐ a compute win is not a representation win)", "",
              "| variant | steps | enc params | pretext-head params | D | train s (mean) |",
              "|---|---|---|---|---|---|"]
    for v, e in summary["by_variant"].items():
        m = e["encoder"]
        lines.append(f"| {v} | {m['steps']} | {m['encoder_param_count']} | "
                     f"{m['pretext_head_param_count']} | {m['D']} | "
                     f"{e['train_seconds']['mean']:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_gen_pretrain(
    config: Optional[CHLUConfig] = None,
    out_dir: Optional[str | Path] = None,
    quick: bool = False,
    log: Callable = print,
) -> dict:
    config = config or CHLUConfig()
    if quick:
        config = apply_quick(config)
    cfg = config.gen_pretrain
    out = Path(out_dir or cfg.out_dir or "runs/gen_pretrain")
    out.mkdir(parents=True, exist_ok=True)
    dataset = str(cfg.dataset)

    log(f"⭐ gen-encoder-pretrain · dataset={dataset} · variants={list(cfg.variants)} "
        f"· seeds={list(cfg.seeds)} · REDUCED SCALE "
        f"(n_enc_train={cfg.n_enc_train}, enc_steps={cfg.enc_steps})")
    split, values = _split_and_values(dataset, config, log)
    corner = corner_rows(dataset, split, values)
    log(f"  corner (matched-seen, SEEN): {corner.size} rows")

    results = []
    for seed in [int(s) for s in cfg.seeds]:
        rows = select_pretrain_rows(split, corner, cfg, seed)
        assert_rows_disjoint(rows)
        todo = [v for v in cfg.variants
                if not (bool(cfg.skip_if_exists)
                        and (out / v / f"seed_{seed}" / "result.json").exists())]
        done = [v for v in cfg.variants if v not in todo]
        for v in done:
            results.append(json.loads((out / v / f"seed_{seed}" / "result.json").read_text()))
        if not todo:
            log(f"  seed {seed}: all {len(done)} variants already on disk — skipping stream")
            continue
        log(f"  seed {seed}: rows enc_train={rows['enc_train'].size} "
            f"probe_train={rows['probe_train'].size} probe_eval={rows['probe_eval'].size} "
            f"corner_eval={rows['corner_eval'].size} "
            f"(corner rows dropped for landing in enc_train: "
            f"{rows['n_corner_dropped_in_enc_train']})")
        pools, labels = _stream_pools(dataset, config, split, rows, log)
        for variant in todo:
            vdir = out / variant / f"seed_{seed}"
            vdir.mkdir(parents=True, exist_ok=True)
            results.append(run_variant_seed(variant, seed, pools, labels, rows,
                                            config, vdir, log))
        del pools  # ⚠ 16 GB machine: ⛔ never hold two seeds' image pools

    order = {v: i for i, v in enumerate(cfg.variants)}
    results.sort(key=lambda r: (order.get(r["variant"], 99), r["seed"]))
    summary = aggregate(results, config)
    _write_json(summary, out / "summary.json")
    write_tables(summary, out / "tables.md")
    log(f"⭐ VERDICT: {summary['verdict']} — {summary['note']}")
    if bool(cfg.make_plots):
        try:
            _plot_bar(summary, out / "bar_scale_orientation.png")
        except Exception as exc:  # pragma: no cover - plotting is never load-bearing
            log(f"  ⚠ plot skipped: {exc}")
    return summary


def _split_and_values(dataset: str, config: CHLUConfig, log: Callable):
    """``(split, latent values)`` — the corner predicate needs the raw values."""
    from ..core.gen_bridge import dataset_module

    split = load_split(dataset, config, log)
    values = dataset_module(dataset).load_labels(_data_root(config))
    return split, values


def _stream_pools(dataset: str, config: CHLUConfig, split, rows: dict, log: Callable):
    """ONE HDF5 pass per seed for every pool; every arm of that seed reuses it."""
    cfg = config.gen_pretrain
    keys = ("enc_train", "probe_train", "probe_eval", "corner_eval")
    allrows = np.concatenate([rows[k] for k in keys])
    t = time.time()
    imgs = stream_images(dataset, allrows, _data_root(config), int(cfg.cache_block))
    log(f"    streamed {allrows.size} images in {time.time() - t:.1f}s")
    pools, labels, off = {}, {}, 0
    for k in keys:
        n = int(rows[k].size)
        pools[k] = imgs[off : off + n]
        labels[k] = factor_labels(split.grid, rows[k])
        off += n
    pools["factor_names"] = list(split.grid.names)
    pools["cardinalities"] = tuple(int(c) for c in split.grid.cardinalities)
    return pools, labels


def _write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _plot_bar(summary: dict, path: Path) -> Path:
    factors = summary["bar_factors"]
    variants = list(summary["by_variant"])
    fig, axes = plt.subplots(1, len(factors), figsize=(6 * len(factors), 4), squeeze=False)
    for ax, f in zip(axes[0], factors, strict=True):
        means, errs, ch = [], [], None
        for v in variants:
            g = summary["by_variant"][v]["by_probe"]["linear"]["grid"]
            means.append(g["per_factor"][f]["mean"])
            errs.append(g["per_factor"][f]["std"])
            ch = g["realised"][f]["chance"]
        ax.bar(range(len(variants)), means, yerr=errs, color="#4477aa")
        ax.axhline(summary["bar"], color="crimson", ls="--", label=f"bar {summary['bar']}")
        ax.axhline(ch, color="grey", ls=":", label=f"chance {ch:.3f}")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{f} · spread grid · linear probe")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="gen-encoder-pretrain: candidate encoders")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--variants", type=str, nargs="+", default=None,
                   choices=list(PRETRAIN_VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--enc-steps", type=int, default=None)
    p.add_argument("--n-enc-train", type=int, default=None)
    p.add_argument("--probe-kinds", type=str, nargs="+", default=None, choices=list(PROBE_KINDS))
    p.add_argument("--no-skip", action="store_true", help="⛔ recompute even if a result exists")
    p.add_argument("--quick", action="store_true", help="⚠ plumbing smoke only")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = load_config(args.config) if args.config else CHLUConfig()
    gp = config.gen_pretrain
    if args.dataset:
        gp.dataset = args.dataset
    if args.variants:
        gp.variants = list(args.variants)
    if args.seeds:
        gp.seeds = list(args.seeds)
    if args.enc_steps:
        gp.enc_steps = int(args.enc_steps)
    if args.n_enc_train:
        gp.n_enc_train = int(args.n_enc_train)
    if args.probe_kinds:
        gp.probe_kinds = list(args.probe_kinds)
    if args.no_skip:
        gp.skip_if_exists = False
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_experiment_gen_pretrain(
        _config_from_args(args), out_dir=args.out, quick=bool(args.quick)
    )
    return 0 if summary else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
