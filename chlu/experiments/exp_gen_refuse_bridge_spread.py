"""w5.1 ``refuse`` (R-A retry) -- the BRIDGE-SEED SPREAD, printed beside the parent's.

⛔⛔ **No scored run.** Nothing here touches a held-out row that has not already been
scored. It re-measures the **spread** of two quantities the parent hub reported with a ±
that is narrower than the truth, under a **wider instrument**, and prints the two
instruments in ONE table so the reader never has to hold two numbers in their head.

**The two instruments.**

``fixed``   the parent's: store seed *s* varies the fit-pool draw and the atom selection;
            encoder, decoder and classifier are pinned at bridge seed 0 for all three.
``joint``   this retry's, and ``w5 hub/delete``'s: store seed *s* varies the fit-pool draw,
            the atom selection, the **encoder**, the **decoder** and the **classifier**
            together. ⛔ Neither instrument varies the channel→well partition, which is
            built once from ``PINS["seed"]``'s features and shared by every seed in both
            hubs -- so "joint" is five of six, in both, and that is exactly why the two
            hubs become comparable rather than complete.

**The two quantities**, chosen because the target file names them as the ones whose
reported ± is narrower than the truth:

1. **eviction byte counts** -- how many bytes of the deployed image an
   ``evict_term(object_hue = 0)`` changes, and how many it changes OUTSIDE the permitted
   ranges (the second is the claim; the first is provenance);
2. **novel-tuple counts** -- ``score_novelty_row``'s ``distinct_valid_novel_tuples`` over
   the survivors of each deletion, per refuser.

⛔ The saturated quantities (refusal 1.0000, delta-reject 0.0000, the 1-byte hole) are NOT
chased: they are bounded by construction and a wider instrument cannot move them. They are
printed once, as a **fixed point**, so that "did not move" is on the record as measured.

Scoring is the frozen harness's throughout: this module calls ``score_deletion`` from the
parent's own report module on BOTH artifact trees, so the metric provably cannot differ
between the two columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .exp_gen_factor_wells import _jdump
from .exp_gen_refuse import seed_dir
from .exp_gen_refuse_report import LEDGER, score_deletion

#: The refusers that carry a survivor novel-tuple count on both flavours.
NOVEL_REFUSERS = ("field_geom", "bitmap8", "pair_presence", "pair_deletion")


def _spread(xs) -> dict:
    """mean, the parent's ``spread``, the half-range, and min/max of a 3-seed sample.

    ⚠⚠ **The parent's ``±`` is the FULL RANGE, not a half-range and not an sd.** Read out
    of its own code, ``exp_gen_refuse_report._agg``: ``"spread": float(a.max() - a.min())``,
    printed as ``mean ± spread``. So its published ``2,317.3 ± 312`` denotes
    ``mean = 2317.3``, ``max - min = 312``, and the *interval* it writes,
    ``[2005.3, 2629.3]``, is about **twice as wide as the three numbers it saw**
    (``[2178, 2490]``). ⛔ That is a CONSERVATIVE convention, not an understated one, and
    the distinction decides this retry's verdict -- so the same statistic is computed here,
    under the same name, and the half-range and the raw three numbers travel beside it.
    """
    a = np.asarray([x for x in xs if x is not None], float)
    if a.size == 0:
        return {"n": 0, "mean": None, "spread": None, "half_range": None,
                "min": None, "max": None, "per_seed": []}
    return {"n": int(a.size), "mean": float(a.mean()),
            "spread": float(a.max() - a.min()),
            "half_range": float((a.max() - a.min()) / 2.0),
            "min": float(a.min()), "max": float(a.max()),
            "per_seed": [float(x) for x in a]}


def collect(out: Path, label: str, log: Callable = print) -> dict:
    """Pull both quantities (and the fixed points) out of one artifact tree."""
    sc = score_deletion(Path(out), log=log)
    got = sorted(sc)
    if not got:
        raise AssertionError(f"[{label}] no deletion.npz under {out} -- nothing to compare")
    rec = {"label": label, "out": str(out), "seeds_present": got,
           "evict_bytes": {}, "novel": {}, "fixed_points": {}, "provenance": {}}

    # -- instrument provenance, read from each seed's own deletion.json ---------
    meta = {}
    for s in got:
        m = json.loads((seed_dir(Path(out), s) / "deletion.json").read_text("utf-8"))
        meta[s] = m
        fc = m.get("feature_cache") or {}
        rec["provenance"][str(s)] = {
            "bridge_mode": m.get("bridge_mode", "pinned (implicit: artifact written "
                                                "before the mode switch existed)"),
            "bridge_seed": m.get("bridge_seed", 0),
            "features_sha256": fc.get("features_sha256"),
            "encoder_param_hash": fc.get("encoder_param_hash"),
            "cache_generation": fc.get("generation"), "cache_n_rows": fc.get("n_rows"),
            "cache_scope": fc.get("scope"),
            "decoder_param_hash": (m.get("judge") or {}).get("decoder_param_hash"),
            "classifier_param_hash": (m.get("judge") or {}).get("classifier_param_hash"),
        }

    # -- QUANTITY 1: eviction byte counts --------------------------------------
    ba = {s: sc[s]["evict"]["_proof"]["byte_assertion"] for s in got}
    rec["evict_bytes"] = {
        "changed": _spread([ba[s]["bytes_changed"] for s in got]),
        "outside_permitted": _spread(
            [ba[s]["bytes_changed_outside_permitted"] for s in got]),
        "n_items_evicted": _spread(
            [sc[s]["evict"]["_proof"]["n_items_evicted"] for s in got]),
        "read_rel_change_on_survivors": _spread(
            [sc[s]["evict"]["_proof"]["read_rel_change_on_surviving_requests"]
             for s in got]),
        # ⭐ The NEGATIVE CONTROL is a byte count too, and it is the one that shows the
        # instrument is not blind: a 1e-3 nudge to one address-book row must move
        # thousands of bytes and fail `assert_no_refit`. ⚠ `w5 hub/delete` measured it at
        # 5,735 / 5,726 / 5,783 and the parent at 5,735 / - / 5,762; if the two hubs are
        # now comparable this must land on the delete hub's three, so it is collected.
        "negative_control_bytes": _spread(
            [(meta[s].get("negative_control_refit") or {}).get("bytes_changed")
             for s in got]),
    }
    rec["per_seed_byte_table"] = {
        str(s): {
            "evict_bytes_changed": int(ba[s]["bytes_changed"]),
            "evict_bytes_outside": int(ba[s]["bytes_changed_outside_permitted"]),
            "n_items_evicted": int(sc[s]["evict"]["_proof"]["n_items_evicted"]),
            "hole_bytes_changed": int(
                sc[s]["hole"]["_proof"]["byte_assertion"]["bytes_changed"]),
            "hole_bytes_outside": int(
                sc[s]["hole"]["_proof"]["byte_assertion"]
                ["bytes_changed_outside_permitted"]),
            "total_image_bytes": int(ba[s]["total_image_bytes"]),
            "negative_control_bytes": (
                meta[s].get("negative_control_refit") or {}).get("bytes_changed"),
        } for s in got}

    # -- QUANTITY 2: novel-tuple counts ----------------------------------------
    for flav in ("hole", "evict"):
        rec["novel"][flav] = {}
        for nm in NOVEL_REFUSERS:
            vals = [sc[s][flav].get(nm, {}).get("survivors", {})
                    .get("distinct_valid_novel_tuples") for s in got]
            if all(v is None for v in vals):
                continue
            rec["novel"][flav][nm] = _spread(vals)

    # -- the fixed points, printed so "did not move" is measured, not assumed ---
    rec["fixed_points"] = {
        "HOLE bytes changed": _spread(
            [sc[s]["hole"]["_proof"]["byte_assertion"]["bytes_changed"] for s in got]),
        "HOLE bytes changed OUTSIDE permitted": _spread(
            [sc[s]["hole"]["_proof"]["byte_assertion"]["bytes_changed_outside_permitted"]
             for s in got]),
        "HOLE reject deleted conjunction, `field_geom`": _spread(
            [sc[s]["hole"]["field_geom"]["reject_deleted_post"] for s in got]),
        "HOLE accept never-held conjunction, `field_geom`": _spread(
            [sc[s]["hole"]["field_geom"]["accept_unseen_conjunction_post"] for s in got]),
        "HOLE delta-reject, `bitmap8` (THE BAR)": _spread(
            [sc[s]["hole"]["bitmap8"]["delta_reject"] for s in got]),
        "HOLE accept never-held conjunction, `pair_presence`": _spread(
            [sc[s]["hole"]["pair_presence"]["accept_unseen_conjunction_post"]
             for s in got]),
        "EVICT reject deleted value, `field_geom`": _spread(
            [sc[s]["evict"]["field_geom"]["reject_deleted_post"] for s in got]),
        "EVICT reject deleted value, `bitmap8`": _spread(
            [sc[s]["evict"]["bitmap8"]["reject_deleted_post"] for s in got]),
        "EVICT accept survivors, `field_geom`": _spread(
            [sc[s]["evict"]["field_geom"]["accept_survivors_post"] for s in got]),
    }
    rec["boolean_fixed_points"] = {
        "HOLE rendered features bit-identical": [
            bool(sc[s]["hole"]["_proof"]["rendered_features_bit_identical"])
            for s in got],
    }
    return rec


def _fmt(sp: dict, dp: int = 1) -> str:
    """``mean +/- spread`` in the PARENT's convention, so the columns are like-for-like."""
    if not sp or sp.get("mean") is None:
        return "-"
    m, h = sp["mean"], sp["spread"]
    if dp >= 4:
        return f"{m:.4f} +/- {h:.4f}"
    if dp == 0:
        return f"{m:,.0f} +/- {h:,.0f}"
    return f"{m:,.1f} +/- {h:,.0f}"


def _obs(sp: dict, dp: int = 1) -> str:
    """The three numbers themselves, as an observed range -- the actual evidence."""
    if not sp or sp.get("mean") is None:
        return "-"
    f = "{:,.4f}" if dp >= 4 else "{:,.0f}"
    return f"[{f.format(sp['min'])}, {f.format(sp['max'])}]"


def _widen(a: dict, b: dict) -> str:
    """Does the wider instrument's spread exceed the narrower one's?"""
    if a.get("spread") is None or b.get("spread") is None:
        return "-"
    ha, hb = a["spread"], b["spread"]
    if ha == 0 and hb == 0:
        return "both 0"
    if ha == 0:
        return "⭐ 0 -> nonzero"
    return f"x{hb / ha:.2f}"


def _moved(fixed: dict, joint: dict) -> str:
    """⭐⭐ THE TEST THE DIAL DECLARES: does the joint mean fall outside the parent's ±?

    ⛔ Deliberately asymmetric. The question the target file asks is whether the PARENT'S
    STATED INTERVAL covers the truth, so the parent's interval is the one the new central
    value is tested against -- not the other way round.

    ⚠ Two verdicts, because they can disagree and that difference is the whole finding.
    ``published`` tests against ``mean +/- spread`` -- what the parent actually wrote down,
    an interval about twice the width of what it saw. ``observed`` tests against the
    parent's own ``[min, max]``, the three numbers it actually had. A quantity inside the
    first and outside the second was covered only by the parent's generous notation.
    """
    if fixed.get("mean") is None or joint.get("mean") is None:
        return "-"
    m, sp = fixed["mean"], fixed["spread"]
    pub = (m - sp) - 1e-12 <= joint["mean"] <= (m + sp) + 1e-12
    obs = fixed["min"] - 1e-12 <= joint["mean"] <= fixed["max"] + 1e-12
    if pub and obs:
        return "inside both"
    if pub and not obs:
        return "⚠ inside published, OUTSIDE observed [min,max]"
    return "⛔ OUTSIDE the published +/-"


def collect_delete_hub(out: Path) -> dict:
    """``w5 hub/delete``'s own per-seed byte artifacts, read from ITS tree, not its LOG.

    ⭐ This is the cross-hub check the target file asks for. That hub varies pool, atom
    seed, encoder, decoder and classifier jointly; after ``--bridge-mode joint`` so do we,
    off the same store-seed index and the same two lines of pool arithmetic. So its
    ``evict`` byte counts and its negative control are a **prediction** for our joint arm,
    ⛔ not merely a similar quantity.

    It reads ``run1/seed_*/bytes.json``: ``evict.byte_assertion.bytes_changed``,
    ``bytes_changed_outside_permitted``, ``evict.n_items`` and
    ``negative_control_refit.bytes_changed``.
    """
    per = {}
    for s in (0, 1, 2):
        p = Path(out) / f"seed_{s}" / "bytes.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text("utf-8"))
        ev = d.get("evict") or {}
        ba = ev.get("byte_assertion") or {}
        per[str(s)] = {
            "evict_bytes_changed": ba.get("bytes_changed"),
            "evict_bytes_outside": ba.get("bytes_changed_outside_permitted"),
            "n_items_evicted": ev.get("n_items", ev.get("n_items_evicted")),
            "hole_bytes_changed": ((d.get("hole") or {}).get("byte_assertion")
                                   or {}).get("bytes_changed"),
            "hole_bytes_outside": ((d.get("hole") or {}).get("byte_assertion")
                                   or {}).get("bytes_changed_outside_permitted"),
            "negative_control_bytes": (
                d.get("negative_control_refit") or {}).get("bytes_changed"),
        }
    return per


def build(fixed_out: Path, joint_out: Path, dest: Path, log: Callable = print,
          delete_out: Optional[Path] = None, parent_out: Optional[Path] = None) -> dict:
    """Three trees, one table.

    ``fixed_out``  MY pinned arm -- the parent's configuration, rebuilt by me.
    ``joint_out``  MY joint arm  -- the only thing differing from ``fixed_out`` is which
                   bridge seed the store seed selects. ⭐ This pair is the dial, clean.
    ``parent_out`` the PARENT's own tree, printed as a third column. ⭐ It is what the
                   target file means by "your parent's fixed-instrument spread", and
                   ``parent`` vs ``mine-fixed`` is a **reproducibility** statement that
                   would otherwise hide inside the contrast as an environment term.
    """
    log("=== fixed instrument, MY build (the parent's configuration) ===")
    A = collect(Path(fixed_out), "mine-fixed (bridge seed 0 at every store seed)", log=log)
    log("=== joint instrument, MY build ===")
    B = collect(Path(joint_out), "mine-joint (bridge seed = store seed)", log=log)
    P = None
    if parent_out is not None:
        log("=== the PARENT's own tree ===")
        P = collect(Path(parent_out), "parent-fixed (as published)", log=log)

    # ⛔ ASSERTED, not assumed: the two trees must differ in the instrument at seeds 1/2
    # and COINCIDE at seed 0, or the comparison is measuring something else.
    ok = {}
    for s in ("0", "1", "2"):
        if s not in A["provenance"] or s not in B["provenance"]:
            continue
        pa, pb = A["provenance"][s], B["provenance"][s]
        same = (pa["features_sha256"] == pb["features_sha256"]
                and pa["decoder_param_hash"] == pb["decoder_param_hash"]
                and pa["classifier_param_hash"] == pb["classifier_param_hash"])
        ok[s] = {"instrument_identical": bool(same), "expected_identical": (s == "0")}
        if same != (s == "0"):
            raise AssertionError(
                f"seed {s}: instrument identical={same} but expected {s == '0'} -- the "
                "two columns are not the comparison this module claims to make")
    log(f"  instrument cross-check: {json.dumps(ok)}")

    lines = []
    W = lines.append
    W("### ⭐⭐ THE BRIDGE-SEED SPREAD -- the parent's fixed instrument beside the full one")
    W("")
    W("`fixed` = bridge seed 0 at every store seed (the parent). `joint` = bridge seed = "
      "store seed: encoder, decoder, classifier and cache move WITH the seed, which is "
      "`w5 hub/delete`'s convention.")
    W("Both columns are `mean +/- spread` over store seeds 0/1/2 in THE PARENT'S OWN "
      "convention -- ⚠ its `+/-` is the **full range** `max - min` "
      "(`exp_gen_refuse_report._agg`), not a half-range and not an sd, so the interval it "
      "writes is about twice the width of the three numbers it saw. Each column's observed "
      "`[min, max]` is printed beside it so either standard can be applied. Both columns "
      "are scored by the SAME frozen `score_deletion`. ⛔ Neither column is a scored run: "
      "both re-read artifacts of rows that were already scored.")
    W("")
    W("| quantity | PARENT, fixed (as published) | mine, FIXED (its config, my build) | "
      "⭐ mine, JOINT (the wider instrument) | observed [min,max]: parent / joint | "
      "spread ratio joint:parent | ⭐⭐ joint mean vs the parent's published +/- |")
    W("|---|---|---|---|---|---|---|")

    def _pick(path):
        a = A
        for k in path:
            a = a[k] if isinstance(a, dict) else {}
        p = P
        if p is not None:
            for k in path:
                p = p[k] if isinstance(p, dict) else {}
        b = B
        for k in path:
            b = b[k] if isinstance(b, dict) else {}
        return (p if p else a), a, b

    paths = [
        (("evict_bytes", "changed"), "EVICTION bytes changed", 0),
        (("evict_bytes", "outside_permitted"),
         "EVICTION bytes changed OUTSIDE permitted ⭐ THE CLAIM", 0),
        (("evict_bytes", "n_items_evicted"), "EVICTION items evicted", 1),
        (("evict_bytes", "read_rel_change_on_survivors"),
         "EVICTION read relative change on survivors", 4),
        (("evict_bytes", "negative_control_bytes"),
         "NEGATIVE CONTROL bytes (1e-3 nudge to one address-book row)", 0),
    ]
    for flav, tag in (("hole", "post-HOLE"), ("evict", "post-EVICTION")):
        for nm in NOVEL_REFUSERS:
            if nm in A["novel"][flav] and nm in B["novel"][flav]:
                paths.append((("novel", flav, nm),
                              f"NOVEL TUPLES {tag} - `{nm}` ({LEDGER[nm]:,} B)", 1))
    for path, name, dp in paths:
        p, a, b = _pick(path)
        W(f"| {name} | {_fmt(p, dp)} | {_fmt(a, dp)} | {_fmt(b, dp)} | "
          f"{_obs(p, dp)} / {_obs(b, dp)} | {_widen(p, b)} | {_moved(p, b)} |")

    W("")
    W('### The fixed points -- printed so that "did not move" is MEASURED, not assumed')
    W("")
    W("| quantity | parent, fixed | mine, fixed | ⭐ mine, joint |")
    W("|---|---|---|---|")
    for k in A["fixed_points"]:
        pv = _fmt(P["fixed_points"][k], 4) if P else "-"
        W(f"| {k} | {pv} | {_fmt(A['fixed_points'][k], 4)} | "
          f"{_fmt(B['fixed_points'][k], 4)} |")
    for k in A["boolean_fixed_points"]:
        pv = P["boolean_fixed_points"][k] if P else "-"
        W(f"| {k} | {pv} | {A['boolean_fixed_points'][k]} | "
          f"{B['boolean_fixed_points'][k]} |")

    W("")
    W("### ⭐ The per-seed byte table -- the three numbers, not their summary")
    W("")
    W("⚠ The mean and the spread of a byte count are **provenance, not a claim**: the count is "
      "the footprint of however many stored items happen to hold the evicted value under that "
      "pool and atom draw. ⭐ The claim is the column that reads **0** on every seed in every "
      "instrument.")
    W("")
    W("| tree | store seed | evict: bytes changed | evict: OUTSIDE ⭐ | items evicted | "
      "hole: bytes | hole: OUTSIDE ⭐ | negative control |")
    W("|---|---|---|---|---|---|---|---|")
    for rec in ([P] if P else []) + [A, B]:
        tag = rec["label"].split(" ")[0]
        for s, r in sorted(rec["per_seed_byte_table"].items()):
            W(f"| {tag} | {s} | {r['evict_bytes_changed']:,} | "
              f"**{r['evict_bytes_outside']}** | {r['n_items_evicted']} | "
              f"{r['hole_bytes_changed']} | **{r['hole_bytes_outside']}** | "
              f"{r['negative_control_bytes']:,} |")

    # ⭐ parent vs my-fixed is a REPRODUCIBILITY statement, kept out of the dial contrast.
    repro = None
    if P:
        cells, same = 0, 0
        for s in sorted(A["per_seed_byte_table"]):
            if s not in P["per_seed_byte_table"]:
                continue
            for q, v in A["per_seed_byte_table"][s].items():
                cells += 1
                same += int(P["per_seed_byte_table"][s].get(q) == v)
        nov = {}
        for flav in ("hole", "evict"):
            for nm in NOVEL_REFUSERS:
                if nm in A["novel"][flav] and nm in P["novel"][flav]:
                    cells += 1
                    ok_ = (A["novel"][flav][nm]["per_seed"]
                           == P["novel"][flav][nm]["per_seed"])
                    same += int(ok_)
                    nov[f"{flav}/{nm}"] = ok_
        repro = {"cells": cells, "identical": same, "all_identical": bool(same == cells),
                 "novel_per_seed_identical": nov}
        W("")
        W("### ⭐ Reproducibility of the PINNED arm -- my build against the parent's")
        W("")
        W(f"Same configuration, my machine, `OMP_NUM_THREADS=3`: **{same}/{cells}** "
          "per-seed cells identical (byte counts, item counts, negative control and every "
          "per-seed novel-tuple count). ⛔ Kept as a separate statement rather than folded "
          "into the dial contrast, because a reproducibility failure and an instrument "
          "effect are different things and must not be allowed to look alike.")

    # -- ⭐ the cross-hub comparability test ------------------------------------
    DH = collect_delete_hub(Path(delete_out)) if delete_out else {}
    verdict = None
    if DH:
        W("")
        W("### ⭐⭐ ARE THE TWO HUBS COMPARABLE NOW? -- the joint arm against "
          "`w5 hub/delete`'s own artifacts")
        W("")
        W("⛔ Read from ITS `run1/seed_*/bytes.json`, not from its LOG. After "
          "`--bridge-mode joint` the two hubs select the same pool draw, the same atom "
          "seed, the same encoder, the same decoder and the same classifier off the same "
          "store-seed index, so **its numbers are a prediction for mine**, not merely a "
          "similar quantity. The parent's fixed column is printed beside, to show what "
          "was and was not comparable before.")
        W("")
        W("| store seed | quantity | parent (fixed) | ⭐ this retry (joint) | "
          "`w5 hub/delete` | joint == delete? |")
        W("|---|---|---|---|---|---|")
        agree, total = 0, 0
        for s in ("0", "1", "2"):
            if s not in DH or s not in B["per_seed_byte_table"]:
                continue
            for q, lbl in (("evict_bytes_changed", "evict: bytes changed"),
                           ("evict_bytes_outside", "evict: bytes OUTSIDE ⭐"),
                           ("n_items_evicted", "items evicted"),
                           ("negative_control_bytes", "negative control bytes")):
                fa = A["per_seed_byte_table"].get(s, {}).get(q)
                bb = B["per_seed_byte_table"][s].get(q)
                dd = DH[s].get(q)
                same = (bb == dd)
                total += 1
                agree += int(same)
                W(f"| {s} | {lbl} | {fa:,} | **{bb:,}** | {dd:,} | "
                  f"{'⭐ **YES**' if same else '⛔ no'} |")
        verdict = {"cells_agreeing": agree, "cells_compared": total,
                   "all_agree": bool(agree == total)}
        W("")
        if verdict["all_agree"]:
            W(f"⭐⭐ **{agree}/{total} cells agree EXACTLY.** The two hubs' seed-1 and seed-2 "
              "numbers are now **comparable**, and the reason is mechanical rather than "
              "statistical: under `joint` they are the same store, built from the same "
              "bytes, so they may share a row.")
        else:
            W(f"⛔ **{agree}/{total} cells agree.** The two hubs are **NOT** comparable "
              "even under `joint`, and the disagreeing cells above say where. Something "
              "other than the bridge seed differs between the two builds and it must be "
              "found before any shared row is printed.")

    W("")
    W("### Instrument provenance -- which encoder/decoder/classifier each seed actually read")
    W("")
    W("| tree | store seed | bridge seed | features sha256 | encoder hash | decoder hash | "
      "classifier hash |")
    W("|---|---|---|---|---|---|---|")
    for rec in ([P] if P else []) + [A, B]:
        tag = rec["label"].split(" ")[0]
        for s, p in sorted(rec["provenance"].items()):
            W(f"| {tag} | {s} | {p['bridge_seed']} | "
              f"`{(p['features_sha256'] or '?')[:12]}` | "
              f"`{(p['encoder_param_hash'] or '?')[:12]}` | "
              f"`{(p['decoder_param_hash'] or '?')[:12]}` | "
              f"`{(p['classifier_param_hash'] or '?')[:12]}` |")

    md = "\n".join(lines) + "\n"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "BRIDGE-SPREAD.md").write_text(md, encoding="utf-8")
    _jdump({"parent_fixed": P, "mine_fixed": A, "mine_joint": B,
            "instrument_cross_check": ok, "delete_hub": DH,
            "comparability_verdict": verdict, "pinned_arm_reproducibility": repro},
           dest / "bridge_spread.json")
    log(md)
    return {"parent_fixed": P, "mine_fixed": A, "mine_joint": B, "cross_check": ok,
            "delete_hub": DH, "comparability_verdict": verdict,
            "pinned_arm_reproducibility": repro}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fixed-out", required=True)
    p.add_argument("--joint-out", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--delete-out", default=None,
                   help="w5 hub/delete's run tree, for the cross-hub comparability test")
    p.add_argument("--parent-out", default=None,
                   help="the PARENT hub's own run tree, printed as the published column")
    a = p.parse_args(argv)
    build(Path(a.fixed_out), Path(a.joint_out), Path(a.dest),
          delete_out=Path(a.delete_out) if a.delete_out else None,
          parent_out=Path(a.parent_out) if a.parent_out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
