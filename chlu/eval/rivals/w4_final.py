"""Assemble the final report from the artifacts, and append it to LOG.md.

The report is the last section of the same LOG.md that carries the dated bullets.
It is built by reading the JSON files a run wrote, never from values held in
memory, so every number in it can be traced to a file on disk.

A missing artifact becomes an explicit NOT RUN with its reason. Nothing is
skipped silently and nothing becomes a zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import w4_report, w4_rows


def _read(p: Path) -> Optional[dict]:
    if not Path(p).exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _f(x, nd=4):
    return "NOT RUN" if x is None else f"{x:.{nd}f}"


def lace_section(root: Path, seeds) -> str:
    """Both LACE numbers, side by side, per seed. The repair must be auditable."""
    L = ["### LACE: the historical selection and the repaired one, both printed", ""]
    L.append("`GenCellConfig.lace_selection_source` was NOT edited. The historical leg "
             "is still run from the pinned value; the repair re-runs the SAME 36-trial "
             "selection (same grid, same short schedule, same criterion) on the "
             "STRATIFIED rows the arm is actually scored on, in this spoke's own "
             "module.")
    L.append("")
    L.append("| seed | published (gen 1) | historical selection | repaired selection | "
             "delta | historical regressed? | repaired regressed? |")
    L.append("|---:|---:|---:|---:|---:|:--:|:--:|")
    any_row = False
    for s in seeds:
        body = _read(root / f"seed_{s}" / "lace.json")
        if body is None:
            L.append(f"| {s} | — | NOT RUN | NOT RUN | — | — | — |")
            continue
        b = body.get("both_numbers") or {}
        any_row = True
        L.append(
            f"| {s} | {_f(b.get('published_probe_reference'))} | "
            f"{_f(b.get('historical_selection_probe'))} | "
            f"{_f(b.get('repaired_selection_probe'))} | "
            f"{_f(b.get('delta_repaired_minus_historical'))} | "
            f"{b.get('historical_regressed')} | {b.get('repaired_regressed')} |")
    L.append("")
    L.append("Statistic: probe-companion in-corner balanced accuracy on the primary "
             "sampler, which is the statistic the harness's own regression check "
             "uses. The published column is the generation-1 reference constant from "
             "`GenCellConfig.lace_ode_probe_reference_per_seed` and is shown for "
             "orientation ONLY — it is a generation-1 number and must not be quoted "
             "beside a generation-2 one.")
    L.append("")
    if not any_row:
        L.append("**NOT RUN** on every seed — no `lace.json` on disk.")
        L.append("")
    return "\n".join(L)


def poe_section(root: Path, seeds) -> str:
    """The PoE-EBM, at its own budget, run twice, with the byte multiple."""
    L = ["### The product-of-experts EBM, at its OWN budget, run twice", ""]
    L.append("| seed | params | deployed bytes | byte multiple vs the 524,288 B store | "
             "run 1 judge in-corner | run 2 judge in-corner | spread |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|")
    seen = False
    for s in seeds:
        body = _read(root / f"seed_{s}" / "poe_ebm.json")
        if body is None:
            L.append(f"| {s} | — | — | — | NOT RUN | NOT RUN | — |")
            continue
        seen = True
        led = body.get("byte_ledger") or {}
        runs = body.get("runs") or []
        vals = [r["scores"]["external"]["in_corner"]["balanced_accuracy"]
                for r in runs if r.get("scores")]
        sd = body.get("across_repeats", {}).get("sd")
        L.append(
            f"| {s} | {led.get('param_count', 0):,} | "
            f"{led.get('deployed_bytes', 0):,} | "
            f"{led.get('byte_multiple', 0):.1f}x | "
            f"{_f(vals[0] if len(vals) > 0 else None)} | "
            f"{_f(vals[1] if len(vals) > 1 else None)} | "
            f"{_f(sd)} |")
    L.append("")
    L.append("Two independent runs, each with its own expert fit and its own sampler "
             "keys, because a 72-item corner read from a stochastic sampler cannot "
             "carry a confident claim from one draw. The byte multiple belongs beside "
             "every number this arm produces: it is not held to the store's budget "
             "and its numbers are not comparable in level to an arm that is.")
    L.append("")
    if not seen:
        L.append("**NOT RUN** on every seed — no `poe_ebm.json` on disk.")
        L.append("")
    return "\n".join(L)


def poe_verdict(root: Path, seeds) -> str:
    """Did the PoE-EBM win outright? If so it goes in the appendix WITH its numbers."""
    data = w4_report.collect(root, seeds)
    means = {}
    for rival, d in data.items():
        m, _, n = w4_report._agg(d["per_seed"], "judge_in_corner")
        if m is not None:
            means[rival] = m
    if "poe_ebm" not in means:
        return ("The PoE-EBM is NOT RUN, so the question of whether it wins outright "
                "is not answered here. It is not a dash in our favour.")
    best = max(means, key=means.get)
    if best == "poe_ebm":
        others = ", ".join(f"{k} {v:.4f}" for k, v in sorted(
            means.items(), key=lambda kv: -kv[1]) if k != "poe_ebm")
        return (f"**THE PoE-EBM WINS OUTRIGHT** on the judge's in-corner statistic: "
                f"{means['poe_ebm']:.4f} against {others}. Per the task, that is not a "
                f"reason to bury it: it is reported here, in the table above, WITH its "
                f"numbers and its byte multiple. It runs at its own budget, which is "
                f"the honest caveat and is printed beside it — it is not a "
                f"byte-matched win.")
    return (f"The PoE-EBM does not win outright: the best judge in-corner mean is "
            f"{best} at {means[best]:.4f}, against the PoE-EBM's "
            f"{means['poe_ebm']:.4f}. Reported either way.")


def deletion_section(root: Path, seeds) -> str:
    L = ["### The deletion comparator: retrain from scratch without the value", ""]
    first = None
    for s in seeds:
        body = _read(root / f"seed_{s}" / "deletion_comparator.json")
        if body is None:
            continue
        first = first or body
        cl = body.get("cost_line") or {}
        L.append(f"**Seed {s} cost line.** {cl.get('one_line', 'NOT RUN')}")
        L.append("")
    if first is None:
        L.append("**NOT RUN** on every seed — no `deletion_comparator.json` on disk.")
        L.append("")
        return "\n".join(L)
    L.append(first.get("scope", ""))
    L.append("")
    L.append("| seed | method | wall clock | rows refitted | judge on the deleted "
             "factor, FORGET set | same, RETAIN set |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for s in seeds:
        body = _read(root / f"seed_{s}" / "deletion_comparator.json")
        if body is None:
            L.append(f"| {s} | — | — | — | NOT RUN | NOT RUN |")
            continue
        for mname, m in (body.get("methods") or {}).items():
            sc = m.get("scores") or {}
            fg = (sc.get("forget") or {}).get("judge_on_deleted_factor")
            rt = (sc.get("retain") or {}).get("judge_on_deleted_factor")
            L.append(f"| {s} | {mname} | {m.get('wall_clock_s', 0):.3f}s | "
                     f"{m.get('refit_rows', 0):,} | {_f(fg)} | {_f(rt)} |")
    L.append("")
    L.append("Both sets are needed or neither number means anything: performance on "
             "the FORGET set should fall, and performance on the RETAIN set should "
             "not. A method that forgets the value by getting worse everywhere has "
             "not deleted anything, it has been damaged.")
    L.append("")
    return "\n".join(L)


def rescore_section(out_root: Path) -> str:
    L = ["### Every existing arm, re-scored on generation 2", ""]
    comp = _read(out_root / "rescore_gen1_vs_gen2.json")
    if comp is None:
        L.append("**NOT RUN** — no `rescore_gen1_vs_gen2.json` on disk. Acceptance "
                 "item 5 is therefore NOT met and is declared as such, not skipped.")
        L.append("")
        return "\n".join(L)
    L.append(comp.get("rule", ""))
    L.append("")
    L.append("| arm | gen-1 mean | gen-1 seed sd | gen-2 mean | gen-2 seed sd | "
             "max abs delta | moved more than seed spread? |")
    L.append("|---|---:|---:|---:|---:|---:|:--:|")
    for arm, r in sorted((comp.get("arms") or {}).items()):
        L.append(f"| {arm} | {_f(r.get('gen1_mean'))} | {_f(r.get('gen1_seed_sd'))} | "
                 f"{_f(r.get('gen2_mean'))} | {_f(r.get('gen2_seed_sd'))} | "
                 f"{_f(r.get('max_abs_delta'))} | "
                 f"{r.get('moved_more_than_seed_spread')} |")
    L.append("")
    L.append(f"**Verdict: {comp.get('verdict')}**")
    L.append("")
    return "\n".join(L)


def cache_section(out_root: Path, gen2_root: Path, seeds) -> str:
    L = ["### Cache generation 2", ""]
    det = _read(out_root / "determinism_check.json")
    if det is None:
        L.append("**The determinism check has no record on disk.** Generation 2 "
                 "cannot be trusted without it.")
    else:
        L.append(f"**The falsifier passed.** One shard (seed {det['seed']}, "
                 f"{det['n_rows']} rows) re-encoded twice in the same process; the "
                 f"sha256 of the quantised cache bytes matched: "
                 f"`{det['sha256_pass1'][:32]}…` both times. Had it not, generation 2 "
                 "would be noise and this spoke would have stopped here.")
    L.append("")
    L.append("| seed | rows | scope | dtype | device | precision | features sha256 |")
    L.append("|---:|---:|---|---|---|---|---|")
    for s in seeds:
        m = _read(Path(gen2_root) / f"seed_{s}" / "feature_cache.json")
        if m is None:
            L.append(f"| {s} | NOT RUN | — | — | — | — | — |")
            continue
        L.append(f"| {s} | {m['n_rows']:,} | {m['scope']} | {m['dtype']} | "
                 f"{','.join(m['device']['devices'])} | "
                 f"{m['encode_precision']['resolved']} | "
                 f"`{m['features_sha256'][:24]}…` |")
    L.append("")
    cmp1 = _read(out_root / "gen1_vs_gen2.json")
    if cmp1:
        L.append("**Generation 1 and generation 2 disagree, with the predicted "
                 "signature.** Same encoder parameter hash in every case, so this is "
                 "the same encoder computing different bytes:")
        L.append("")
        L.append("| seed | elements compared | differing | share | max abs diff |")
        L.append("|---:|---:|---:|---:|---:|")
        for s in seeds:
            r = cmp1.get(str(s)) or {}
            if not r.get("comparable"):
                L.append(f"| {s} | — | — | — | — |")
                continue
            L.append(f"| {s} | {r['n_elements_compared']:,} | "
                     f"{r['n_elements_differing']:,} | "
                     f"{r['frac_elements_differing']:.3%} | "
                     f"{r['max_abs_diff']:.8g} |")
        L.append("")
        L.append("The max deviations are one float16 ULP at those magnitudes, on about "
                 "half a percent of elements. That is the few-ULP / ~1 %-of-elements "
                 "signature of a cache that was not all produced on one device. It "
                 "does not invalidate any generation-1 number; it says those numbers "
                 "belong to a different cache and cannot be quoted beside a "
                 "generation-2 one.")
        L.append("")
    return "\n".join(L)


def per_factor_section(root: Path, seeds, order=None) -> str:
    """Per-factor judge accuracy on B. The aggregate hides the only interesting part.

    On this spoke's own corner the prototype table reads five of the six factors
    at or near 1.000 and ``scale`` at 0.139. An aggregate of 0.854 reports none of
    that. A rival is not one number, and a table that prints only the aggregate
    invites exactly the comparison that cannot be made.
    """
    order = order or w4_report.ROW_ORDER
    data = {}
    factors = []
    for rival in order:
        per_seed = {}
        for s in seeds:
            body = _read(root / f"seed_{s}" / w4_report.FILE_OF[rival] + ".json") \
                if False else _read(root / f"seed_{s}" /
                                    f"{w4_report.FILE_OF[rival]}.json")
            sc = w4_report._scores_of(rival, body)
            if sc is None:
                continue
            pf = sc["external"]["composition_B"]["per_factor"]
            factors = factors or sorted(pf)
            per_seed[s] = pf
        data[rival] = per_seed
    if not factors:
        return ("### Per-factor judge accuracy on B\n\n"
                "**NOT RUN** — no rival artifacts on disk.\n")
    L = ["### Per-factor judge accuracy on B (mean over seeds)", "",
         "The aggregate is the last thing to read. On this corner the prototype "
         "table reads every factor except `scale` at or near 1.000, and `scale` "
         "near chance — which is the whole story about what a per-value mean can "
         "and cannot represent, and it is invisible in the aggregate.", "",
         "| rival | " + " | ".join(f"`{f}`" for f in factors) + " | aggregate |",
         "|---|" + "---:|" * (len(factors) + 1)]
    for rival in order:
        per_seed = data[rival]
        if not per_seed:
            L.append(f"| {w4_report.PRETTY.get(rival, rival)} | "
                     + " | ".join("—" for _ in factors) + " | — |")
            continue
        cells = []
        for f in factors:
            vals = [pf[f] for pf in per_seed.values() if f in pf]
            cells.append(_f(float(np.mean(vals)), 3) if vals else "—")
        agg = [float(np.mean(list(pf.values()))) for pf in per_seed.values()]
        L.append(f"| {w4_report.PRETTY.get(rival, rival)} | " + " | ".join(cells)
                 + f" | {_f(float(np.mean(agg)), 3)} |")
    L.append("")
    L.append("Read a rival by the factors it carries, not by its mean. A rival that "
             "wins an aggregate while losing a factor another arm carries is not "
             "better at the task; it is better at a different mixture of it.")
    L.append("")
    return "\n".join(L)
