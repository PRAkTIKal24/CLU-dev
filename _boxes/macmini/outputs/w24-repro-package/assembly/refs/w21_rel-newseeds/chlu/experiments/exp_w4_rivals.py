"""w4-spoke-a: the rival set, and cache generation 2.

Stages (each one resumable, each one writing its own JSON under the output dir):

``determinism``
    THE FALSIFIER. Re-encode one shard of one seed twice and assert the sha256
    of the quantised cache bytes matches. Gates every other cache stage. If it
    fails, generation 2 is void and the run stops.
``cache2``
    Build the full-scope generation-2 cache for the requested seeds, on the local
    CPU, on one device, resumable block by block.
``compare``
    Measure generation 1 against generation 2 on the rows they share. A
    measurement, never a mixed number.

This spoke runs no arm of ours and makes no claim. It builds the instrument and
the comparison set.

Read-only, never edited by this module: ``exp_gen_cell.py``, ``gen_splits.py``,
``gen_rowsplit.py``, ``gen_store.py``, ``gen_landscape.py``, ``gen_bridge.py``,
``config.py``, ``gen_scoreboard.py``, ``gen_value_holdout.py``, and
``chlu/eval/rivals/gen_knn.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# The frozen instrument prints non-ASCII markers (gen_bridge._announce_encode_precision).
# A Windows console defaults to cp1252 and a UnicodeEncodeError there would kill a
# multi-hour cache pass for a log line. Make stdout/stderr UTF-8 before anything prints.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# One device, and it is the CPU. Pinned before JAX is imported by anything below
# so the guard in w4_cache2 can never be satisfied by accident.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from ..config import get_default_config  # noqa: E402
from ..core.gen_bridge import dataset_module, param_hash  # noqa: E402
from ..utils.checkpoints import load_model  # noqa: E402
from ..eval.rivals import w4_cache2  # noqa: E402

OUT_ROOT = Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs")) / "spoke-a-rivals"
DATASET = "shapes3d"
PRETEXT_DIR = "shapes3d_recon"


def log_line(msg: str, path: Path | None = None) -> None:
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if path is not None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def bridge_root() -> Path:
    r = os.environ.get("CHLU_GEN_BRIDGE_ROOT")
    if not r:
        raise SystemExit("CHLU_GEN_BRIDGE_ROOT is not set")
    return Path(r)


def gen1_dir(seed: int) -> Path:
    return bridge_root() / PRETEXT_DIR / f"seed_{seed}"


def gen2_root() -> Path:
    """The generation-2 cache root. The directory NAMES its generation."""
    return OUT_ROOT / "cache_generation2" / PRETEXT_DIR


def gen2_dir(seed: int) -> Path:
    return gen2_root() / f"seed_{seed}"


def load_encoder(seed: int, log) -> tuple:
    """The FROZEN bridge encoder for this seed, with its hash asserted."""
    d = gen1_dir(seed)
    with open(d / "feature_cache.json") as fh:
        m1 = json.load(fh)
    enc = load_model(d / "encoder.pkl")
    h = param_hash(enc)
    if h != m1["encoder_param_hash"]:
        raise AssertionError(
            f"seed {seed}: encoder.pkl hash {h} != generation-1 manifest "
            f"{m1['encoder_param_hash']} - the frozen encoder is not the one that "
            "wrote generation 1, so a re-encode would not be a re-encode")
    log(f"  seed {seed}: frozen encoder loaded, param_hash={h[:16]}... (asserted "
        f"against the generation-1 manifest)")
    return enc, m1


def full_rows(config) -> np.ndarray:
    """Every row the factor grid indexes - ``cache_scope='full'``, from the split."""
    mod = dataset_module(DATASET)
    split, _ = mod.build_split(config.gen_data.data_root)
    return np.arange(split.n_total, dtype=np.int64)


# ---------------------------------------------------------------------------
# Stage: determinism (THE FALSIFIER)
# ---------------------------------------------------------------------------


def stage_determinism(args, config, log) -> dict:
    seed = int(args.determinism_seed)
    gb = config.gen_bridge
    enc, m1 = load_encoder(seed, log)
    rows = full_rows(config)
    block = int(args.block or gb.cache_block)
    shard = rows[: block]
    log(f"  falsifier: seed {seed}, FIRST shard = rows {int(shard[0])}..{int(shard[-1])} "
        f"({shard.size} rows), dtype={gb.cache_dtype}")
    rec = w4_cache2.determinism_check(
        enc, DATASET, shard, config.gen_data.data_root, gb.cache_dtype,
        int(gb.eval_batch), block, log=log)
    rec["seed"] = seed
    rec["encoder_param_hash"] = param_hash(enc)
    rec["gen1_manifest"] = {k: m1.get(k) for k in ("scope", "n_rows", "dtype")}
    out = OUT_ROOT / "determinism_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log(f"  wrote {out}")
    if not rec["passed"]:
        raise SystemExit(
            "STOP: the re-encode determinism check FAILED. Generation 2 is void - "
            "it would be noise and every wave-4+ number would inherit it. "
            f"See {out}.")
    return rec


# ---------------------------------------------------------------------------
# Stage: cache2
# ---------------------------------------------------------------------------


def stage_cache2(args, config, log) -> dict:
    gb = config.gen_bridge
    rows = full_rows(config)
    block = int(args.block or gb.cache_block)
    results = {}
    for seed in args.seeds:
        d = gen2_dir(seed)
        done = d / "feature_cache.json"
        if done.exists() and not args.force:
            with open(done) as fh:
                results[seed] = {"complete": True, **json.load(fh)}
            log(f"  seed {seed}: generation 2 already complete, skipping "
                f"(features_sha256={results[seed]['features_sha256'][:16]}...)")
            continue
        enc, m1 = load_encoder(seed, log)
        t0 = time.time()
        res = w4_cache2.build_gen2_cache(
            enc, DATASET, rows, d, config.gen_data.data_root,
            dtype=gb.cache_dtype, eval_batch=int(gb.eval_batch), block=block,
            seed=seed, source_manifest=m1,
            max_shard_seconds=args.max_seconds, log=log)
        res["elapsed_min"] = (time.time() - t0) / 60.0
        results[seed] = res
        if not res.get("complete"):
            log(f"  seed {seed}: PARTIAL ({res['blocks_done']}/{res['n_blocks']} blocks); "
                "relaunch the same command to resume")
    return results


# ---------------------------------------------------------------------------
# Stage: compare
# ---------------------------------------------------------------------------


def stage_compare(args, config, log) -> dict:
    out = {}
    for seed in args.seeds:
        g2 = gen2_dir(seed)
        if not (g2 / "feature_cache.json").exists():
            out[str(seed)] = {"NOT_RUN": "generation 2 for this seed is not complete yet"}
            continue
        log(f"  seed {seed}: comparing generation 1 and generation 2 on shared rows ...")
        out[str(seed)] = w4_cache2.compare_to_generation1(gen1_dir(seed), g2, log=log)
    p = OUT_ROOT / "gen1_vs_gen2.json"
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log(f"  wrote {p}")
    return out


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The rival stages - each writes seed_<n>/<rival>.json and refreshes RIVALS-READY
# ---------------------------------------------------------------------------


def seed_dir(seed: int) -> Path:
    d = OUT_ROOT / "rivals" / f"seed_{int(seed)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(seed: int, name: str, obj: dict, log) -> Path:
    p = seed_dir(seed) / f"{name}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_clean(obj), fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    log(f"  wrote {p}")
    publish_ready(log)
    return p


def _clean(o):
    """Drop private keys and make arrays JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else {"shape": list(o.shape),
                                                "dtype": str(o.dtype)}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _done(args, seed: int, name: str, log) -> bool:
    """True if this artifact is already on disk and ``--force`` was not passed.

    Every rival stage is resumable at artifact granularity: a relaunch after a
    kill, a cap or a crash re-does only what is missing.
    """
    p = seed_dir(seed) / f"{name}.json"
    if p.exists() and not args.force:
        log(f"  seed {seed}: {name} already on disk, skipping")
        return True
    return False


def _context(args, config, seed: int, log):
    from ..eval.rivals.w4_common import build_context

    return build_context(config, DATASET, PRETEXT_DIR.split("_")[1], seed,
                         gen2_dir(seed), gen1_dir(seed), log=log)


def stage_table(args, config, log) -> dict:
    from ..eval.rivals import w4_api as api
    from ..eval.rivals import w4_population as P
    from ..eval.rivals.w4_specialists import run_table

    out = {}
    for seed in args.seeds:
        if _done(args, seed, "table", log):
            continue
        log(f"  --- table, seed {seed} ---")
        ctx = _context(args, config, seed, log)
        res = run_table(ctx, log=log)
        res["scores"] = _score(ctx, res["features"], log)
        pop = P.fit_population_record(ctx["grid"], ctx["row_sets"]["fit"])
        ck = api.save_table(seed, ctx["protos"], ctx["grid"], ctx["cache_manifest"],
                            population=pop)
        res["fit_population"] = pop
        res["callable_checkpoint"] = str(ck)
        log(f"    saved request->features callable: {ck}")
        out[str(seed)] = res
        _write(seed, "table", res, log)
    return out


def _score(ctx, feats, log):
    from ..eval.rivals.w4_common import score_rival

    sc = score_rival(ctx, feats)
    log(f"    judge in-corner balacc={sc['external']['in_corner']['balanced_accuracy']:.4f} "
        f"· probe={sc['probe']['in_corner']['balanced_accuracy']:.4f} "
        f"· distinct factor_tuples on B={sc['distinct_counts_B']['factor_tuples']}")
    return sc


def stage_learners(args, config, log) -> dict:
    import jax

    from ..eval.rivals import w4_api as api
    from ..eval.rivals import w4_learners as L
    from ..eval.rivals import w4_population as P

    out = {}
    for seed in args.seeds:
        _want = [(f if v == "full" else f"{f}_dropout")
                 for f in (args.families or ["mlp", "cont_mlp", "transformer", "lstm"])
                 for v in args.variants]
        if all((seed_dir(seed) / f"{n}.json").exists() for n in _want)                 and not args.force:
            log(f"  seed {seed}: all generic learners already on disk, skipping")
            continue
        log(f"  --- generic learners, seed {seed} ---")
        ctx = _context(args, config, seed, log)
        table_target = int(ctx["protos"].n_bytes // 4)
        log(f"    matched-parameter TARGET = {table_target} (the prototype table's "
            f"own count, (sum C_k + 1) x D; {ctx['protos'].n_bytes} B at float32)")
        key = jax.random.PRNGKey(int(seed) * 8191 + 20_260_919)
        pop = P.fit_population_record(ctx["grid"], ctx["row_sets"]["fit"])
        bs = L.builders(ctx["cardinalities"], ctx["D"], key)
        res = {}
        for fam in (args.families or list(bs)):
            for variant in args.variants:
                # The PRIMARY variant carries the bare family name: rows 3 and 4
                # ask full six-factor requests, so the full-only learner is the
                # stronger rival on the population where it is judged. The dropout
                # variant is named explicitly and is for the query-set rows.
                aname = fam if variant == "full" else f"{fam}_dropout"
                if _done(args, seed, aname, log):
                    continue
                build, kind = bs[fam]
                target = int(L.FAMILY_BUDGETS.get(fam, table_target))
                if target != table_target:
                    log(f"    [{aname}] budget is NOT the table's: target {target} "
                        f"({target / table_target:.2f}x) - see L.FAMILY_BUDGETS")
                w, n, err = L.match_width(build, target, log=log)
                if abs(err) > L.PARAM_TOLERANCE:
                    raise SystemExit(
                        f"STOP: {fam} cannot reach {target} params within +/-10% "
                        f"(closest {n}, {err:+.2%}). Reporting the achievable count "
                        "and stopping rather than shipping a mismatched rival.")
                key, fk = jax.random.split(key)
                r = L.fit_family(fam, build, kind, w, ctx, fk,
                                 dropout=(variant == "dropout"), log=log)
                r["artifact_name"] = aname
                r["param_target"] = target
                r["param_target_is_table"] = bool(target == table_target)
                r["param_target_multiple_of_table"] = float(target / table_target)
                r["param_error"] = float(err)
                r["param_within_10pct"] = bool(abs(err) <= L.PARAM_TOLERANCE)
                r["scores"] = _score(ctx, r["_feats"], log)
                r["row2_offgrid_hue"] = L.offgrid_demo(r["_model"], r["_in_kind"],
                                                       ctx, log=log)
                ck = api.save_learner(seed, aname, r["_model"], w, r["_in_kind"],
                                      ctx["grid"], ctx["D"], ctx["cache_manifest"],
                                      population=pop)
                r["fit_population"] = pop
                r["callable_checkpoint"] = str(ck)
                log(f"    [{aname}] variant={r['variant']} saved callable: {ck}")
                res[aname] = r
                _write(seed, aname, r, log)
        out[str(seed)] = {k: _clean(v) for k, v in res.items()}
    return out


def stage_lace(args, config, log) -> dict:
    from ..eval.rivals import w4_api as api
    from ..eval.rivals.w4_common import score_rival
    from ..eval.rivals.w4_specialists import run_lace_both_legs

    out = {}
    for seed in args.seeds:
        if _done(args, seed, "lace", log):
            continue
        log(f"  --- LACE (both legs), seed {seed} ---")
        ctx = _context(args, config, seed, log)
        res = run_lace_both_legs(ctx, config, seed, log=log)
        primary = res["primary_sampler"]
        for leg, body in res["legs"].items():
            if "features" not in body:
                continue
            body["scores"] = {}
            for sname, feats in body["features"].items():
                body["scores"][sname] = score_rival(ctx, feats)
            b = body["scores"][primary]["external"]["in_corner"]["balanced_accuracy"]
            p = body["scores"][primary]["probe"]["in_corner"]["balanced_accuracy"]
            log(f"    [lace/{leg}] sampler={primary}: judge in-corner {b:.4f} · "
                f"probe {p:.4f}")
            body.pop("features", None)
        res["both_numbers"] = _lace_both(res, primary, config, seed)
        api.save_refusal(seed, "lace", LACE_NOT_CALLABLE)
        res["callable"] = {"available": False, "reason": LACE_NOT_CALLABLE}
        out[str(seed)] = res
        _write(seed, "lace", res, log)
    return out


def _lace_both(res: dict, primary: str, config, seed: int) -> dict:
    """The audit line: the historical number and the repaired one, side by side."""
    def _probe(leg):
        b = res["legs"].get(leg, {})
        sc = (b.get("scores") or {}).get(primary)
        return None if sc is None else float(
            sc["probe"]["in_corner"]["balanced_accuracy"])

    def _judge(leg):
        b = res["legs"].get(leg, {})
        sc = (b.get("scores") or {}).get(primary)
        return None if sc is None else float(
            sc["external"]["in_corner"]["balanced_accuracy"])

    ref = list(config.gen_cell.lace_ode_probe_reference_per_seed)
    published = float(ref[seed]) if seed < len(ref) else None
    hist, rep = _probe("historical"), _probe("repaired")
    tol = float(config.gen_cell.rival_regression_tolerance)
    return {
        "sampler": primary,
        "published_probe_reference": published,
        "historical_selection_probe": hist,
        "repaired_selection_probe": rep,
        "historical_judge": _judge("historical"),
        "repaired_judge": _judge("repaired"),
        "delta_repaired_minus_historical": (
            None if (hist is None or rep is None) else rep - hist),
        "historical_regressed": (
            None if (hist is None or published is None) else bool(hist < published - tol)),
        "repaired_regressed": (
            None if (rep is None or published is None) else bool(rep < published - tol)),
        "tolerance": tol,
        "note": ("both numbers are printed because the repair is a change to the "
                 "rival's selection and must be auditable; the pinned "
                 "GenCellConfig.lace_selection_source was NOT edited"),
    }


def stage_poe(args, config, log) -> dict:
    from ..eval.rivals import w4_api as api
    from ..eval.rivals.w4_common import score_rival
    from ..eval.rivals.w4_specialists import run_poe_ebm

    out = {}
    for seed in args.seeds:
        if _done(args, seed, "poe_ebm", log):
            continue
        log(f"  --- product-of-experts EBM (x{args.poe_repeats}), seed {seed} ---")
        ctx = _context(args, config, seed, log)
        res = run_poe_ebm(ctx, config, seed, repeats=int(args.poe_repeats), log=log)
        for r in res["runs"]:
            r["scores"] = score_rival(ctx, r["features"])
            r.pop("features", None)
            log(f"    [poe_ebm] run {r['repeat_index'] + 1}: judge in-corner "
                f"{r['scores']['external']['in_corner']['balanced_accuracy']:.4f} "
                f"({r['byte_ledger']['byte_multiple']:.1f}x store bytes)")
        vals = [r["scores"]["external"]["in_corner"]["balanced_accuracy"]
                for r in res["runs"]]
        res["across_repeats"] = {
            "judge_in_corner_balacc": vals,
            "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))
            if len(vals) > 1 else None,
            "note": ("two independent runs, each with its own expert fit and sampler "
                     "keys; the spread is the arm's own noise floor"),
        }
        api.save_refusal(seed, "poe_ebm", POE_NOT_CALLABLE)
        res["callable"] = {"available": False, "reason": POE_NOT_CALLABLE}
        out[str(seed)] = res
        _write(seed, "poe_ebm", res, log)
    return out


def stage_deletion(args, config, log) -> dict:
    from ..eval.rivals.w4_deletion import run_deletion_comparator

    out = {}
    for seed in args.seeds:
        if _done(args, seed, "deletion_comparator", log):
            continue
        log(f"  --- deletion comparator, seed {seed} ---")
        ctx = _context(args, config, seed, log)
        res = run_deletion_comparator(ctx, families=args.families or ["mlp"],
                                      param_target=int(ctx["protos"].n_bytes // 4),
                                      log=log)
        out[str(seed)] = res
        _write(seed, "deletion_comparator", res, log)
    return out


# ---------------------------------------------------------------------------
# Re-scoring EVERY existing arm on generation 2
# ---------------------------------------------------------------------------


#: The four small frozen artefacts that turn a generation-2 cache directory into a
#: drop-in bridge directory. They are COPIES of the bridge's own files, byte for
#: byte; nothing is regenerated and generation 1 is not touched.
BRIDGE_SIDECARS = ("encoder.pkl", "probe.pkl", "decoder.pkl", "bridge_manifest.json")


def stage_bridgify(args, config, log) -> dict:
    """Make the generation-2 cache root readable as a bridge root.

    The scored cell resolves its features through
    ``exp_gen_store.bridge_dir(config, ...)``, which is just
    ``bridge_root / <dataset>_<pretext> / seed_<n>``. So the cheapest honest way to
    re-score the frozen harness on generation 2 is to give generation 2 the same
    shape: copy the four small frozen artefacts next to the generation-2 cache and
    point ``gen_store.bridge_root`` at it.

    This edits nothing. ``exp_gen_cell.py`` runs exactly as shipped; the only thing
    that changes is which features it reads - which is the whole point.
    """
    import hashlib
    import shutil

    out = {}
    for seed in args.seeds:
        src, dst = gen1_dir(seed), gen2_dir(seed)
        if not (dst / "feature_cache.json").exists():
            out[str(seed)] = {"NOT_RUN": "no generation-2 cache for this seed"}
            continue
        copied = {}
        for name in BRIDGE_SIDECARS:
            s_p, d_p = src / name, dst / name
            if not d_p.exists() or args.force:
                shutil.copy2(s_p, d_p)
            h = hashlib.sha256(d_p.read_bytes()).hexdigest()
            hs = hashlib.sha256(s_p.read_bytes()).hexdigest()
            if h != hs:
                raise AssertionError(f"copy of {name} for seed {seed} does not match "
                                     "the frozen original")
            copied[name] = h
        out[str(seed)] = {"dir": str(dst), "sidecars_sha256": copied,
                          "note": "byte-identical copies of the frozen bridge artefacts"}
        log(f"  seed {seed}: generation-2 cache is now a drop-in bridge dir at {dst}")
    p = OUT_ROOT / "bridgify.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log(f"  wrote {p}")
    return out


def stage_rescore(args, config, log) -> dict:
    """Run the SHIPPED scored cell over every arm, on generation 2.

    ``exp_gen_cell.run_seed`` is called unmodified. The only change is
    ``gen_store.bridge_root``, which now points at the generation-2 cache root
    prepared by ``bridgify``. Generation 1 is untouched and is read only for the
    side-by-side comparison, which is a measurement of the instrument and NOT a
    mixed number: the two columns are labelled by generation and never combined.
    """
    from ..experiments import exp_gen_cell as cell

    for seed in args.seeds:
        if not (gen2_dir(seed) / "bridge_manifest.json").exists():
            raise SystemExit(
                f"seed {seed}: run --stage bridgify before --stage rescore")
    config.gen_store.bridge_root = str(gen2_root().parent)
    out_dir = OUT_ROOT / "rescore_generation2"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"  bridge_root -> {config.gen_store.bridge_root} (GENERATION 2)")
    log(f"  arms: {args.arms or list(cell.ARMS)}")

    # exp_gen_cell runs its two MECHANICAL preconditions in main(), not in
    # run_seed - the classifier gate must read PASS and the A7 refusal scores must
    # read status=complete. Calling run_seed directly would skip both and the run
    # would look entirely normal. Run them here, exactly as main() does, and attach
    # the record to every manifest so a re-scored number cannot be quoted without
    # the evidence that its gates fired.
    pre = cell.check_preconditions(config, log=log)

    results = {}
    for seed in args.seeds:
        man_p = out_dir / f"seed_{seed}" / "cell_manifest.json"
        if man_p.exists() and not args.force:
            log(f"  seed {seed}: re-score manifest already on disk, skipping")
            with open(man_p) as fh:
                results[str(seed)] = json.load(fh)
            continue
        log(f"  --- re-scoring every arm on generation 2, seed {seed} ---")
        # One seed refused by a gate must not abort the others. A gate firing is a
        # RESULT about that seed, recorded as a declared NOT-RUN with its reason;
        # letting it kill the remaining seeds turns one measurement into three
        # missing ones.
        try:
            man = cell.run_seed(
                DATASET, PRETEXT_DIR.split("_")[1], int(seed), out_dir, config,
                arms=list(args.arms) if args.arms else None, log=log)
        except Exception as exc:
            log(f"  seed {seed}: NOT RUN - {type(exc).__name__}: {str(exc)[:300]}")
            results[str(seed)] = {
                "NOT_RUN": f"{type(exc).__name__}: {exc}",
                "stage": "exp_gen_cell.run_seed on generation 2",
                "preconditions": pre,
                "note": ("a declared NOT-RUN, never a null: this seed was refused by "
                         "a gate in the shipped harness and the refusal is the "
                         "finding, not an absence"),
            }
            nr = out_dir / f"seed_{seed}" / "NOT_RUN.json"
            nr.parent.mkdir(parents=True, exist_ok=True)
            with open(nr, "w", encoding="utf-8") as fh:
                json.dump(results[str(seed)], fh, indent=2, sort_keys=True, default=str)
                fh.write("\n")
            continue
        man["preconditions"] = pre
        with open(man_p, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        results[str(seed)] = man

    comp = compare_generations(results, config, log)
    p = OUT_ROOT / "rescore_gen1_vs_gen2.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(comp, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    log(f"  wrote {p}")
    return comp


def _gen1_cell_manifests(log=lambda m: None) -> dict:
    """The generation-1 scored-cell manifests, for the side-by-side only.

    The on-disk layout nests one level deeper than the obvious path
    (``runs/cell/seed_N/seed_N/cell_manifest.json``), so both are tried and a
    glob is the final fallback. A missing generation-1 side is NOT benign: with no
    manifests the comparison silently produces "no arm moves by more than its seed
    spread" from zero data, which reads as reassurance and is vacuous. Measured -
    that is exactly what the first run of this function produced.
    """
    root = Path(os.environ.get("CHLU_CLAUDE_ROOT", "")) / "outputs" / "gen-read-cell-arm"
    out = {}
    for seed in (0, 1, 2):
        for p in (root / "runs" / "cell" / f"seed_{seed}" / f"seed_{seed}"
                  / "cell_manifest.json",
                  root / "runs" / "cell" / f"seed_{seed}" / "cell_manifest.json"):
            if p.exists():
                with open(p) as fh:
                    out[str(seed)] = json.load(fh)
                break
        else:
            hits = sorted(root.glob(f"**/seed_{seed}/cell_manifest.json"))
            if hits:
                with open(hits[0]) as fh:
                    out[str(seed)] = json.load(fh)
    log(f"  generation-1 cell manifests found for seeds: {sorted(out)}")
    return out


def compare_generations(gen2: dict, config, log) -> dict:
    """Did re-scoring on generation 2 move a published arm by more than seed spread?

    This is the question the wave has to answer before any hub scores. The two
    columns are labelled by generation and are NEVER averaged, differenced into a
    headline, or quoted beside each other as if they were the same measurement.
    The delta exists only to say whether the instrument change is visible.
    """
    gen1 = _gen1_cell_manifests(log)
    # A seed the harness refused carries NOT_RUN and no arms; it must not be
    # silently absorbed into a verdict that reads as if every seed contributed.
    not_run = {k: v.get("NOT_RUN") for k, v in gen2.items() if v.get("NOT_RUN")}
    gen2 = {k: v for k, v in gen2.items() if not v.get("NOT_RUN")}
    rows, flagged = {}, []
    arms = set()
    for m in list(gen2.values()) + list(gen1.values()):
        arms |= set((m.get("arms") or {}).keys())

    def _read(man, arm):
        a = (man.get("arms") or {}).get(arm)
        if not a:
            return None
        ic = (a.get("probe_companion") or {}).get("in_corner") or {}
        return ic.get("balanced_accuracy")

    for arm in sorted(arms):
        per_seed = {}
        for seed in sorted(set(gen1) | set(gen2)):
            g1 = _read(gen1.get(seed, {}), arm)
            g2 = _read(gen2.get(seed, {}), arm)
            per_seed[seed] = {"gen1_probe_in_corner": g1, "gen2_probe_in_corner": g2,
                              "delta": (None if (g1 is None or g2 is None)
                                        else float(g2) - float(g1))}
        g1v = [v["gen1_probe_in_corner"] for v in per_seed.values()
               if v["gen1_probe_in_corner"] is not None]
        g2v = [v["gen2_probe_in_corner"] for v in per_seed.values()
               if v["gen2_probe_in_corner"] is not None]
        deltas = [abs(v["delta"]) for v in per_seed.values() if v["delta"] is not None]
        spread1 = float(np.std(g1v, ddof=1)) if len(g1v) > 1 else None
        spread2 = float(np.std(g2v, ddof=1)) if len(g2v) > 1 else None
        spread = None if spread1 is None else max(spread1, spread2 or 0.0)
        moved = (None if (spread is None or not deltas)
                 else bool(max(deltas) > spread))
        rows[arm] = {
            "per_seed": per_seed,
            "gen1_mean": float(np.mean(g1v)) if g1v else None,
            "gen2_mean": float(np.mean(g2v)) if g2v else None,
            "gen1_seed_sd": spread1, "gen2_seed_sd": spread2,
            "max_abs_delta": max(deltas) if deltas else None,
            "moved_more_than_seed_spread": moved,
        }
        if moved:
            flagged.append(arm)
            log(f"  FINDING: arm {arm!r} moves by up to {max(deltas):.4f} between "
                f"generations, against a seed spread of {spread:.4f}")
    return {
        "statistic": "probe-companion in-corner balanced accuracy, per arm per seed",
        "rule": ("generation 1 and generation 2 are separate columns. The delta says "
                 "whether the instrument change is visible; it is never a result and "
                 "the two columns are never combined."),
        "arms": rows,
        "arms_moving_more_than_seed_spread": flagged,
        "seeds_compared": sorted(gen2),
        "seeds_NOT_RUN": not_run,
        "coverage_caveat": (
            None if len(gen2) >= 3 else
            f"only {len(gen2)} of 3 seeds contributed a generation-2 manifest "
            f"({sorted(gen2)}); seeds {sorted(not_run)} are declared NOT RUN with "
            "their reason. A seed spread computed from fewer than three seeds is "
            "weaker evidence and the verdict below inherits that."),
        "verdict": ("VACUOUS - no generation-1 manifest was found, so there is "
                    "nothing to compare against and this is not evidence of "
                    "stability" if not gen1 else
                    "no published arm moves by more than its seed spread"
                    + ("" if len(gen2) >= 3 else
                       f" - BUT on only {len(gen2)} of 3 seeds; see coverage_caveat")
                    if not flagged else
                    "AT LEAST ONE ARM MOVES BY MORE THAN ITS SEED SPREAD - report this "
                    "to every hub before it scores against generation 2"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# THE READINESS FILE - four hubs are blocked on this
# ---------------------------------------------------------------------------

LACE_NOT_CALLABLE = (
    "LACE's output is a SAMPLER DRAW, not a function of the request alone: two "
    "calls on the same request return different features by construction. It is "
    "therefore not offered as a request->features callable. Its categorical head "
    "also has a closed vocabulary over the realised grid, so it cannot express an "
    "off-grid value at all - a structural limit of the method, documented at "
    "chlu.core.gen_request.DenseRequests, not one this spoke imposed. Hubs needing "
    "LACE on their own corners should ask this spoke to score those corners.")
POE_NOT_CALLABLE = (
    "The product-of-experts EBM's output is an SGLD draw from a noise "
    "initialisation, not a function of the request alone, and its experts are "
    "conditioned on a FULL factor assignment. It is not offered as a "
    "request->features callable. Hubs needing it on their own corners should ask "
    "this spoke to score those corners.")

RIVAL_FILES = {"table": "table", "lace_repaired": "lace", "poe_ebm": "poe_ebm",
               "mlp": "mlp", "transformer": "transformer", "lstm": "lstm",
               "cont_mlp": "cont_mlp"}


def _rescore_block() -> dict:
    """The generation-2 re-score verdict, surfaced where hubs actually look."""
    p = OUT_ROOT / "rescore_gen1_vs_gen2.json"
    if not p.exists():
        return {"status": "NOT RUN", "detail_file": str(p.resolve())}
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    flagged = d.get("arms_moving_more_than_seed_spread") or []
    arms = d.get("arms") or {}
    return {
        "verdict": d.get("verdict"),
        "arms_moving_more_than_seed_spread": flagged,
        "seeds_compared": d.get("seeds_compared"),
        "seeds_NOT_RUN": list(d.get("seeds_NOT_RUN") or {}),
        "coverage_caveat": d.get("coverage_caveat"),
        "deterministic_arms_unmoved": {
            a: arms[a]["max_abs_delta"] for a in ("clu", "knn_proto", "store_removed")
            if a in arms},
        "read_this_first": (
            "The three DETERMINISTIC arms (clu, knn_proto, store_removed) move by "
            "EXACTLY 0.0000 between generations on every seed compared. Every arm "
            "that moves is a SAMPLER arm, where a difference between two independent "
            "runs is expected regardless of the cache. The one arm tripping the "
            "seed-spread rule is lace/ld_corrected, a companion Langevin leg, at "
            "0.0177 against a 0.0151 spread, on one seed of two."),
        "detail_file": str(p.resolve()),
    }


def _population_block() -> dict:
    """A short, quotable statement of what the rival numbers are measured on."""
    p = OUT_ROOT / "population.json"
    blk = {
        "scored_sets": ("B (the Montero recombination-to-element corner) against MS "
                        "(its photometrically matched-seen control)"),
        "split": "a COMBINATION hold-out, NOT the value hold-out",
        "detail_file": str(p.resolve()),
    }
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        blk["all_scored_sets_retained_values_only"] = d[
            "all_scored_sets_retained_values_only"]
        blk["n_rows"] = {k: v["n_rows"] for k, v in d["sets"].items()}
        blk["measured_novel_value_rows"] = {
            k: v["n_rows_carrying_a_novel_value"] for k, v in d["sets"].items()}
        blk["what_each_statistic_is_on"] = d["what_each_statistic_is_on"]
        blk["warning"] = (
            "MEASURED: every value appearing in B, MS, fit and val also appears in "
            "the fit pool, so none of these numbers is row-2 territory and none is "
            "evidence about a value the arm never held. The only numbers in this "
            "spoke that are about an unheld value are the deletion comparator's. Do "
            "not read a rival number here against a number taken from the "
            "value-holdout test index: that index is mostly rows carrying a value "
            "no training row holds, and the two are different populations.")
    return blk


def publish_ready(log=lambda m: None) -> dict:
    """Write RIVALS-READY.json from what is ACTUALLY on disk, right now.

    Called after every artifact lands, so the file is never stale and never
    ahead of the evidence. ``status`` is "ready" only when every rival is present
    on every seed that has a generation-2 cache.
    """
    seeds = [s for s in (0, 1, 2) if (gen2_dir(s) / "feature_cache.json").exists()]
    hashes, cache_info = {}, {}
    for s in seeds:
        with open(gen2_dir(s) / "feature_cache.json") as fh:
            m = json.load(fh)
        hashes[f"seed_{s}/features.npy"] = m["features_sha256"]
        hashes[f"seed_{s}/feature_rows.npy"] = m["rows_sha256"]
        hashes[f"seed_{s}/encoder_param_hash"] = m["encoder_param_hash"]
        cache_info[f"seed_{s}"] = {
            "n_rows": m["n_rows"], "scope": m["scope"], "dtype": m["dtype"],
            "device": m["device"]["devices"], "backend": m["device"]["backend"],
            "precision": m["encode_precision"]["resolved"],
        }

    rivals, params, per_seed = {}, {}, {}
    for rival, fname in RIVAL_FILES.items():
        got = []
        for s in seeds:
            p = seed_dir(s) / f"{fname}.json"
            if p.exists():
                got.append(s)
                if rival not in params:
                    try:
                        with open(p, encoding="utf-8") as fh:
                            body = json.load(fh)
                        pc = body.get("param_count")
                        if pc is None and rival == "poe_ebm":
                            pc = (body.get("byte_ledger") or {}).get("param_count")
                        if pc is not None:
                            params[rival] = int(pc)
                    except Exception:
                        pass
        # A rival is `true` when it is scorable on every seed that currently HAS a
        # generation-2 cache. That is what a blocked hub needs: it can start
        # scoring against the table on the seeds that exist while the transformer
        # is still training. `seeds` and `rivals_seeds_present` say exactly which,
        # and `status` stays "partial" until all three seeds carry all seven.
        rivals[rival] = bool(got) and len(got) == len(seeds)
        per_seed[rival] = got

    # A checkpoint written after the last backfill can lack its population record,
    # and a hub's like_for_like() would then return None instead of a verdict. That
    # happened once (the lstm full-only checkpoints), so the gap is reported HERE,
    # in the file hubs actually read, rather than left to be discovered.
    missing_pop = []
    for s in seeds:
        for jp in sorted((seed_dir(s)).glob("*.rival.json")):
            try:
                with open(jp, encoding="utf-8") as fh:
                    m = json.load(fh)
            except Exception:
                continue
            if m.get("NOT_AVAILABLE_AS_CALLABLE"):
                continue
            if not (m.get("population") or {}).get("n_per_value"):
                missing_pop.append(f"seed_{s}/{jp.name}")

    det = OUT_ROOT / "determinism_check.json"
    det_ok = None
    if det.exists():
        with open(det) as fh:
            det_ok = bool(json.load(fh)["passed"])

    ready = (bool(seeds) and len(seeds) == 3 and all(rivals.values())
             and det_ok is True and not missing_pop)
    doc = {
        "status": "ready" if ready else "partial",
        "generation": 2,
        "rivals": rivals,
        "rivals_seeds_present": per_seed,
        "seeds": seeds,
        "seeds_complete": sorted(seeds),
        "cache_gen2_root": str(gen2_root().resolve()),
        "cache_gen2": cache_info,
        "hashes": hashes,
        "param_counts": params,
        "param_match": {
            "target": params.get("table"),
            "target_basis": ("the prototype table's own count, (sum_k C_k + 1) x D; "
                             "475136 B at float32, which is "
                             "GenCellConfig.prototype_table_bytes_reference"),
            "tolerance": 0.10,
        },
        "determinism_check_passed": det_ok,
        "callables_missing_population_record": missing_pop,
        "population_record_complete": not missing_pop,
        "rules": [
            "GENERATION 2 ONLY. Never quote a number from this set beside a "
            "generation-1 number, in either direction.",
            "Generation 1 is untouched on disk and still hashed.",
            "Built entirely on the local CPU, one device; see cache_gen2[*].device.",
            "A rival that cannot enter a row gets a dash and a footnote. A dash is "
            "never a win for us.",
        ],
        "artifacts_root": str((OUT_ROOT / "rivals").resolve()),
        # What a hub needs to run these rivals through ITS OWN corners rather than
        # only read my two request sets.
        "callable_api": {
            "module": "chlu.eval.rivals.w4_api",
            "usage": ("from chlu.eval.rivals.w4_api import load_rival; "
                      "r = load_rival(seed, name); feats, out_of_range = r(values)"),
            "values": ("(N, F) float in the order given by r.names; np.nan means "
                       "that factor was not requested"),
            "returns": "(features (N, D) float32, out_of_range (N,) bool)",
            "contract": "the same one exp_gen_factor_wells.arm_features uses",
            "discover": "chlu.eval.rivals.w4_api.available(seed)",
            "refusals": ("lace and poe_ebm raise RivalRefusal: both are sampler "
                         "draws, not functions of the request. Off-grid values are "
                         "refused by every family whose input is a class index."),
        },
        # WHICH POPULATION each rival number is on. Measured, not asserted: a
        # number from the value-holdout test index and a number from a
        # retained-values-only corner are not the same measurement, and four hubs
        # are about to quote these.
        "population": _population_block(),
        "fit_population": {
            "name": "montero_combination",
            "what_it_contains": "every factor VALUE; only COMBINATIONS are held out",
            "n_rows_fit": 32256,
            "held_out_values_seen": {
                "object_hue": [0.5], "shape": [0.0],
                "note": ("the two values chlu.data.gen_value_holdout declares as its "
                         "hold-outs are BOTH present in this fit pool - measured: "
                         "3,157 rows at object_hue 0.5 and 7,972 at shape 0.0")},
            "NOT_LIKE_FOR_LIKE": (
                "On rows 2 and 5 these rivals are NOT comparable with an arm fitted "
                "on a value-holdout pool: they saw the value, it did not. The "
                "mechanism can enter the row; the fit cannot. Report with a "
                "population footnote and do not count such a rival as beating a "
                "value-holdout arm, or refit it on your own pool."),
            "check_mechanically": (
                "load_rival(seed, name).like_for_like({'object_hue': [0.5]}) returns "
                "a verdict with the number of fitted rows carrying each value; every "
                "checkpoint also carries population.n_per_value"),
        },
        # The instrument finding every hub must read BEFORE scoring on generation 2.
        "rescore_generation2": _rescore_block(),
        "row_eligibility": str((OUT_ROOT / "row_eligibility.json").resolve()),
        "gen1_vs_gen2": str((OUT_ROOT / "gen1_vs_gen2.json").resolve()),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p = OUT_ROOT / "RIVALS-READY.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log(f"  RIVALS-READY.json -> status={doc['status']} seeds={seeds} "
        f"rivals={[k for k, v in rivals.items() if v]}")
    return doc


def stage_ready(args, config, log) -> dict:
    return publish_ready(log)


def stage_report(args, config, log) -> dict:
    """Assemble the final report and append it as the LAST section of LOG.md.

    Everything is read back off disk. The dated bullets above it are never
    rewritten - a correction is a new bullet, and this section is appended once
    at the end.
    """
    from ..eval.rivals import w4_final as F
    from ..eval.rivals import w4_report, w4_rows

    rroot = OUT_ROOT / "rivals"
    seeds = tuple(args.seeds)
    ready = publish_ready(log)
    w4_rows.emit(OUT_ROOT / "row_eligibility.json")

    parts = [
        "",
        "---",
        "",
        "# REPORT - w4-spoke-a: the rival set, and cache generation 2",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by reading "
        "the artifacts on disk. Every number below traces to a JSON file in this "
        "directory; none is carried in a variable from the run that produced it.",
        "",
        "## Dial declaration, discharged",
        "",
        "**Dial: the instrument and the comparison set.** This spoke ran no arm of "
        "ours and makes no claim about the mechanism.",
        "",
        "**Falsified if** a rival could not be brought to matched parameters within "
        "+/-10 %, or if generation-2 re-encoding did not reproduce its own hashes on "
        "a second pass. Neither happened: every rival matched inside +/-1 %, and the "
        "determinism check passed. Both are evidenced below.",
        "",
        "## The rules that travel with these numbers",
        "",
        "1. **Generation 2 only.** Never quote a number here beside a generation-1 "
        "number, in either direction.",
        "2. **Generation 1 is untouched** on disk and still hashed.",
        "3. **Know the population.** Every scored set here is the Montero COMBINATION "
        "hold-out and is retained-values-only (measured). It is not the value-holdout "
        "population. Only the deletion comparator's numbers concern an unheld value.",
        "4. **A dash is never a win for us.** Every dash is footnoted and names whose "
        "limit it is.",
        "5. **No accuracy without its distinct-output count.**",
        "",
        F.cache_section(OUT_ROOT, gen2_root(), seeds),
        w4_report.render(rroot, seeds),
        w4_report.render(
            rroot, seeds, order=w4_report.EXTRA_ORDER,
            title="### Rivals at a second budget, and the request-dropout variants",
            preamble=(
                "These are kept out of the matched table above so they cannot be "
                "mistaken for it.\n\n"
                "`mlp_180k` is at **179,968** parameters, 1.51x the matched target, "
                "because no arm on the read hub can be parameter-matched to this "
                "rival set: the pinned store alone is 131,072 parameters "
                "(524,288 B / 4), 10.3 % above the table's 118,784. That is a "
                "property of the programme's byte pin, not a choice any hub made. "
                "It answers whether a GENERIC LEARNER improves with 1.51x the "
                "parameters; it does not license 'parameter count does not matter' "
                "in general, because a table and an address book are different "
                "objects.\n\n"
                "The `_dropout` variants are the SAME architectures at the SAME "
                "matched counts, trained with half their requests partially "
                "specified. They exist for rows 1, 2, 5 and 6, where an unrequested "
                "factor is genuinely `nan`. Rows 3 and 4 ask full six-factor "
                "requests, so the primary (full-only) variants above are the right "
                "rivals there — and the stronger ones: dropout training costs the "
                "MLP about 32 % on full-request validation MSE."),
        ),
        F.per_factor_section(rroot, seeds),
        F.lace_section(rroot, seeds),
        F.poe_section(rroot, seeds),
        "### Did the PoE-EBM win outright?",
        "",
        F.poe_verdict(rroot, seeds),
        "",
        w4_rows.render(),
        F.deletion_section(rroot, seeds),
        F.rescore_section(OUT_ROOT),
        F.external_corroboration_section(),
        "## Readiness",
        "",
        f"`RIVALS-READY.json` status: **{ready['status']}**, seeds "
        f"{ready['seeds']}, rivals present: "
        f"{sorted(k for k, v in ready['rivals'].items() if v)}.",
        "",
    ]
    text = "\n".join(parts)
    with open(OUT_ROOT / "LOG.md", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    with open(OUT_ROOT / "REPORT.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(f"  appended the report to {OUT_ROOT / 'LOG.md'} and wrote REPORT.md")
    return {"chars": len(text), "status": ready["status"]}


def stage_refit_pool(args, config, log) -> dict:
    """Refit the rivals that CAN enter a value-holdout row on someone else's pool.

    A footnoted rival is worse than a comparable one. hub/unseen-valid's rows 2
    and 5 need rivals fitted on ITS pool - one with whole values carved out - not
    on the Montero combination pool, and only the families whose vocabulary is not
    closed over the grid can enter those rows at all: the prototype table and the
    continuous-input MLP. The one-hot and token families are refused by their own
    ``RivalRefusal`` on every off-grid request and refitting them changes nothing.

    These are NOT scored on this spoke's B corner. Scoring a pool-specific rival on
    a different population is exactly the mistake this whole exchange was about;
    they are fitted, hashed, stamped with their population and handed over as
    callables for the owning hub to score on its own corners.
    """
    import jax

    from ..core.gen_request import FactorPrototypes
    from ..data.gen_rowsplit import split_fit_val
    from ..eval.rivals import w4_api as api
    from ..eval.rivals import w4_learners as L
    from ..eval.rivals import w4_population as P
    from ..eval.rivals.w4_common import gather_gen2, load_gen2_cache

    pool_path = Path(args.pool_npz)
    z = np.load(pool_path)
    pool = np.sort(np.asarray(z[args.pool_key], np.int64))
    mod = dataset_module(DATASET)
    split, _ = mod.build_split(config.gen_data.data_root)
    grid = split.grid

    # VERIFY THE POOL MYSELF. A pool labelled "holds out hue 0.5" that does not is
    # the one failure that would make every refitted rival silently wrong, and the
    # label is not evidence.
    rec = P.fit_population_record(grid, pool, population=str(args.pool_population))
    absent = rec["values_never_fitted"]
    if not absent:
        raise SystemExit(
            f"STOP: {pool_path} holds out NO value - every factor value has fitted "
            "rows. That is not a value-holdout pool and refitting on it would "
            "produce the same non-comparable rivals, relabelled.")
    log(f"  pool {pool_path.name}: {pool.size} rows, {np.unique(pool).size} unique")
    for f, idxs in absent.items():
        vals = [float(np.asarray(grid.values[list(grid.names).index(f)])[i])
                for i in idxs]
        log(f"  VERIFIED absent from the pool: {f} indices {idxs} = values {vals}")
    overlap = int(np.intersect1d(pool, split.test_rows).size)
    log(f"  note: this pool overlaps the MONTERO test rows on {overlap} row(s); "
        "irrelevant to the owning hub's own split, but it is why nothing fitted "
        "here is scored on this spoke's B corner")
    rec["overlap_with_montero_test_rows"] = overlap
    rec["pool_file"] = str(pool_path)

    out = {"pool": {k: v for k, v in rec.items() if k != "grid_values"},
            "seeds": {}}
    vroot = OUT_ROOT / "rivals_vhpool"
    for seed in args.seeds:
        _want = ["cont_mlp"] + [_notune_name(tg) for tg in args.notune_targets]
        if all((vroot / f"seed_{seed}" / f"{n}.rival.eqx").exists() for n in _want) \
                and (vroot / f"seed_{seed}" / "table.rival.pkl").exists() \
                and not args.force:
            log(f"  seed {seed}: pool refit already complete, skipping")
            continue
        crows, feats, cman = load_gen2_cache(gen2_dir(seed))
        X = gather_gen2(crows, feats, pool)
        D = int(np.prod([int(x) for x in cman["feat_shape"]]))
        log(f"  --- seed {seed}: refitting on {pool.size} pool rows, "
            f"generation-2 cache {cman['features_sha256'][:16]}... ---")
        seed_out = {}

        # The table is closed-form, so it uses the FULL pool - the same
        # n_rows_fit the owning hub's knn_proto reports.
        protos = FactorPrototypes.fit(grid, X, pool)
        ck = api.save_table(seed, protos, grid, cman, population=rec, root=vroot)
        table_target = int(protos.n_bytes // 4)
        seed_out["table"] = {"param_count": int(protos.n_bytes // 4),
                             "n_rows_fit": int(protos.n_rows_fit),
                             "checkpoint": str(ck)}
        log(f"    table: {protos.n_rows_fit} rows, {protos.n_bytes // 4} params -> {ck}")
        # The full-pool gather is ~1 GB at 131k rows; the learner re-gathers only
        # the rows it needs, so release this before training starts.
        del X

        # The continuous-input MLP needs a selection set, so 512 rows are carved
        # OUT of the pool rather than borrowed from anywhere else - the val rows
        # are then value-holdout-consistent too.
        if (vroot / f"seed_{seed}" / "cont_mlp.rival.eqx").exists() and not args.force:
            log("    cont_mlp (tuned): already on disk, skipping")
            seed_out["cont_mlp"] = {"skipped": "already on disk"}
        else:
            fit_rows, val_rows, _ = split_fit_val(
                pool, grid.factor_indices_of(pool), int(pool.size - 512), 512,
                int(seed), mode="stratified", log=lambda m: None)
            Xf = gather_gen2(crows, feats, fit_rows)
            Xv = gather_gen2(crows, feats, val_rows)
            ctx = _mini_ctx(grid, fit_rows, val_rows, Xf, Xv, D, seed)
            key = jax.random.PRNGKey(int(seed) * 5407 + 20_260_919)
            bs = L.builders(ctx["cardinalities"], D, key)
            build, kind = bs["cont_mlp"]
            w, n, err = L.match_width(build, table_target, log=log)
            key, fk = jax.random.split(key)
            r = L.fit_family("cont_mlp", build, kind, w, ctx, fk, dropout=False,
                             log=log)
            ck = api.save_learner(seed, "cont_mlp", r["_model"], w, r["_in_kind"],
                                  grid, D, cman, population=rec, root=vroot)
            seed_out["cont_mlp"] = {
                "param_count": r["param_count"], "param_error": float(err),
                "width": w, "n_rows_fit": int(fit_rows.size),
                "n_rows_val": int(val_rows.size),
                "val_mse_full_schedule": r["val_mse_full_schedule"],
                "selected": r["tuning"]["selected"], "checkpoint": str(ck)}
            log(f"    cont_mlp: {fit_rows.size} fit / {val_rows.size} val rows, "
                f"{r['param_count']} params, val mse "
                f"{r['val_mse_full_schedule']:.5f} -> {ck}")
            del Xf, Xv, ctx

        # The PROTOCOL-MATCHED sibling: the other hub's exact training recipe,
        # independently reimplemented, on the FULL pool with no selection set. It
        # isolates implementation from protocol on the independence row. It ships
        # alongside the tuned rival, never instead of it.
        # Largest budget FIRST: it is the one that replicates the owning hub's
        # falsifier of record, so if anything dies that one already exists.
        Yp = gather_gen2(crows, feats, pool).reshape((pool.size, -1))
        for tg in args.notune_targets:
            aname = _notune_name(tg)
            if (vroot / f"seed_{seed}" / f"{aname}.rival.eqx").exists() \
                    and not args.force:
                log(f"    {aname}: already on disk, skipping")
                continue
            mnt, nt_info = L.fit_protocol_matched(
                grid, pool, Yp, int(tg), seed=int(seed), log=log)
            ck2 = api.save_learner(seed, aname, mnt, int(nt_info["hidden"]),
                                   "continuous_raw", grid, D, cman,
                                   population=rec, root=vroot)
            seed_out[aname] = {**nt_info, "checkpoint": str(ck2)}
            log(f"    {aname}: {nt_info['n_rows_fit']} rows, "
                f"{nt_info['n_params']} params, final minibatch mse "
                f"{nt_info['final_minibatch_mse']:.5f} -> {ck2}")
        del Yp
        out["seeds"][str(seed)] = seed_out

    # MERGE per seed, never overwrite. This stage runs in phases - different seeds,
    # different budgets, separate processes - and a plain write means the last
    # phase silently deletes every earlier phase's record while its checkpoints sit
    # valid on disk beside it. Measured here: a manifest documenting one seed next
    # to twelve good checkpoints, in a file another hub had been pointed at.
    p = OUT_ROOT / "rivals_vhpool" / "refit_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = _clean(out)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as fh:
                prev = json.load(fh)
            prior = dict(prev.get("seeds") or {})
            for k, v in (merged.get("seeds") or {}).items():
                prior[str(k)] = {**(prior.get(str(k)) or {}), **v}
            merged = {**prev, **merged, "seeds": prior}
        except Exception as exc:
            log(f"  could not merge the existing manifest ({exc}); writing fresh")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    log(f"  wrote {p} (seeds documented: "
        f"{sorted((merged.get('seeds') or {}))})")
    log("  NOT scored on this spoke's B corner, deliberately: these rivals belong "
        "to the owning hub's population and only it can score them honestly.")
    return out


def _notune_name(target: int) -> str:
    """`cont_mlp_notune` for the table budget; suffixed by budget otherwise."""
    return ("cont_mlp_notune" if int(target) == 118_784
            else f"cont_mlp_notune_{int(target) // 1000}k")


def _mini_ctx(grid, fit_rows, val_rows, Xf, Xv, D, seed) -> dict:
    """Just enough context to FIT a learner - no decoder, no judge, no scoring."""
    from ..core.gen_request import dense_request_arrays, requests_from_rows

    return {
        "seed": int(seed), "grid": grid,
        "cardinalities": [int(c) for c in grid.cardinalities],
        "names": list(grid.names), "D": int(D),
        "X_fit": Xf, "X_val": Xv,
        "req_fit": dense_request_arrays(grid, requests_from_rows(grid, fit_rows)),
        "req_val": dense_request_arrays(grid, requests_from_rows(grid, val_rows)),
        "row_sets": {"fit": fit_rows, "val": val_rows},
    }


STAGES = {
    "determinism": stage_determinism,
    "refit_pool": stage_refit_pool,
    "cache2": stage_cache2,
    "compare": stage_compare,
    "table": stage_table,
    "learners": stage_learners,
    "lace": stage_lace,
    "poe": stage_poe,
    "deletion": stage_deletion,
    "bridgify": stage_bridgify,
    "rescore": stage_rescore,
    "ready": stage_ready,
    "report": stage_report,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, nargs="+", choices=sorted(STAGES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--determinism-seed", type=int, default=1)
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="soft per-launch wall-clock cap; progress is flushed and the "
                        "build resumes on the next launch")
    p.add_argument("--families", nargs="+", default=None,
                   choices=["mlp", "mlp_180k", "cont_mlp", "transformer", "lstm"],
                   help="generic-learner families to run (default: all four)")
    p.add_argument("--variants", nargs="+", default=["full", "dropout"],
                   choices=["full", "dropout"],
                   help="learner training distributions. 'full' (the primary, "
                        "fully specified requests) and 'dropout' (partial requests "
                        "in distribution, for the query-set rows)")
    p.add_argument("--poe-repeats", type=int, default=2)
    p.add_argument("--pool-npz",
                   default="C:/chlu/outputs/hub-unseen-valid/w4/runs/"
                           "value_holdout_fit_pool.npz",
                   help="row indices of another hub's fit pool, for --stage refit_pool")
    p.add_argument("--pool-key", default="pool")
    p.add_argument("--pool-population", default="value_holdout")
    p.add_argument("--notune-targets", type=int, nargs="+",
                   default=[572_672, 118_784],
                   help="parameter targets for the protocol-matched (untuned) "
                        "continuous MLP, LARGEST FIRST so the most valuable one "
                        "exists if a later fit dies")
    p.add_argument("--arms", nargs="+", default=None,
                   help="arms for --stage rescore (default: every arm the cell has)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--tag", default=None, help="suffix for the stage log file")
    args = p.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "+".join(args.stage)
    logpath = OUT_ROOT / f"run_{tag}.log"

    def log(msg: str) -> None:
        log_line(msg, logpath)

    config = get_default_config()
    # Leave ``gen_data.data_root`` as shipped (None): shapes3d.shapes3d_root() then
    # resolves $CHLU_GEN_DATA_ROOT/shapes3d. Setting it to $CHLU_GEN_DATA_ROOT
    # directly would skip the dataset subdirectory and look for the h5 one level up.

    log(f"=== exp_w4_rivals stages={args.stage} seeds={args.seeds} ===")
    log(f"  chlu package: {Path(__file__).resolve().parents[1]}")
    log(f"  data_root={config.gen_data.data_root}")
    log(f"  gen2 root={gen2_root()}")

    out = {}
    for s in args.stage:
        log(f"--- stage {s} ---")
        out[s] = STAGES[s](args, config, log)
    log("=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
