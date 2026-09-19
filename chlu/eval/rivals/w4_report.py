"""Assemble the rival table from the artifacts on disk.

Every number in the rendered table is read back out of a JSON file that a run
wrote, never carried in a variable from the run that produced it. A table
rendered from memory is a table nobody can audit, and a reference constant
derived from a rendered table is the error ``gen-external-classifier`` 11.4
flagged - so this goes one way only: JSON on disk -> markdown.

Conventions this module enforces rather than hopes for:

* A missing rival is a DASH with a footnote, never a zero and never an omission.
* An accuracy is never printed without its distinct-output count.
* Every row carries its parameter count and, where it differs from the matched
  budget, its byte multiple.
* Generation 2 only. The generation is printed in the header of every table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

#: Display order: the incumbent first, then the other specialists, then the
#: generic learners.
ROW_ORDER = ("table", "lace_repaired", "poe_ebm",
             "mlp", "cont_mlp", "transformer", "lstm")

#: Rivals that are NOT at the matched budget, or are a second training
#: distribution of one that is. They go in their own table so a reader cannot
#: mistake them for the matched set.
EXTRA_ORDER = ("mlp_180k", "mlp_dropout", "mlp_180k_dropout", "cont_mlp_dropout",
               "transformer_dropout", "lstm_dropout")

PRETTY = {
    "table": "prototype table (incumbent)",
    "lace_repaired": "LACE (selection repaired)",
    "poe_ebm": "product-of-experts EBM",
    "mlp": "MLP (one-hots)",
    "cont_mlp": "continuous-input MLP",
    "transformer": "transformer (6 tokens, 2 layers)",
    "lstm": "LSTM (6 tokens)",
    "mlp_180k": "MLP at 179,968 params (the read hub's arm budget)",
    "mlp_dropout": "MLP, request-dropout variant",
    "mlp_180k_dropout": "MLP at 179,968 params, request-dropout variant",
    "cont_mlp_dropout": "continuous-input MLP, request-dropout variant",
    "transformer_dropout": "transformer, request-dropout variant",
    "lstm_dropout": "LSTM, request-dropout variant",
}

FILE_OF = {"table": "table", "lace_repaired": "lace", "poe_ebm": "poe_ebm"}
FILE_OF.update({k: k for k in
                ("mlp", "cont_mlp", "transformer", "lstm") + EXTRA_ORDER})


def _load(root: Path, seed: int, name: str) -> Optional[dict]:
    p = root / f"seed_{seed}" / f"{name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _scores_of(rival: str, body: dict, primary_sampler: str = "ode") -> Optional[dict]:
    """The one scores block this rival's row is read from."""
    if body is None:
        return None
    if rival == "lace_repaired":
        leg = (body.get("legs") or {}).get("repaired") or {}
        return (leg.get("scores") or {}).get(primary_sampler)
    if rival == "poe_ebm":
        runs = body.get("runs") or []
        return runs[0].get("scores") if runs else None
    return body.get("scores")


def _fmt(x, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def collect(root: Path, seeds=(0, 1, 2), order=None) -> dict:
    """Every rival's per-seed numbers, plus what is missing and why."""
    out = {}
    for rival in (order or ROW_ORDER):
        per_seed, notes = {}, []
        for s in seeds:
            body = _load(root, s, FILE_OF[rival])
            sc = _scores_of(rival, body)
            if sc is None:
                per_seed[s] = None
                notes.append(f"seed {s}: NOT RUN (no artifact on disk)")
                continue
            row = {
                "judge_in_corner": sc["external"]["in_corner"]["balanced_accuracy"],
                "judge_in_corner_auroc": sc["external"]["in_corner"].get("auroc"),
                "probe_in_corner": sc["probe"]["in_corner"]["balanced_accuracy"],
                "judge_composition_B": sc["external"]["composition_B"]["aggregate"],
                "distinct_factor_tuples_B": sc["distinct_counts_B"]["factor_tuples"],
                "n_B": sc["distinct_counts_B"]["n"],
            }
            if rival == "poe_ebm":
                row["byte_multiple"] = (body.get("byte_ledger") or {}).get(
                    "byte_multiple")
                row["param_count"] = (body.get("byte_ledger") or {}).get("param_count")
            else:
                row["param_count"] = body.get("param_count")
                row["param_error"] = body.get("param_error")
            per_seed[s] = row
        out[rival] = {"per_seed": per_seed, "notes": notes}
    return out


def _agg(per_seed: dict, key: str):
    vals = [v[key] for v in per_seed.values() if v is not None and v.get(key) is not None]
    if not vals:
        return None, None, 0
    return (float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else None, len(vals))


def render(root: Path, seeds=(0, 1, 2), order=None, title=None,
           preamble=None) -> str:
    """The rival table, as markdown, read back from disk."""
    order = order or ROW_ORDER
    data = collect(root, seeds, order)
    L = []
    L.append(title or ("### The rival set on GENERATION 2 "
                       f"({len(seeds)} seeds: "
                       f"{', '.join(str(s) for s in seeds)})"))
    L.append("")
    L.append("Judge = the frozen external classifier. Probe = its companion, on the "
             "same renders. `in-corner` is balanced accuracy on the held-out corner "
             "against its matched-seen control; chance is 0.5. `distinct` is distinct "
             "factor tuples the judge reads on B, out of 72 - an accuracy is never "
             "printed without it.")
    if preamble:
        L.append("")
        L.append(preamble)
    L.append("")
    L.append("| rival | params | judge in-corner | probe in-corner | judge comp-B | "
             "distinct/72 | seeds |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for rival in order:
        d = data[rival]
        ps = d["per_seed"]
        jm, jsd, n = _agg(ps, "judge_in_corner")
        pm, psd, _ = _agg(ps, "probe_in_corner")
        cm, _, _ = _agg(ps, "judge_composition_B")
        dm, _, _ = _agg(ps, "distinct_factor_tuples_B")
        pc = next((v.get("param_count") for v in ps.values()
                   if v is not None and v.get("param_count")), None)
        bm = next((v.get("byte_multiple") for v in ps.values()
                   if v is not None and v.get("byte_multiple")), None)
        pcs = "—" if pc is None else f"{pc:,}" + (f" ({bm:.1f}x)" if bm else "")
        js = "—" if jm is None else (f"{jm:.4f}" + (f" ± {jsd:.4f}" if jsd else ""))
        L.append(f"| {PRETTY.get(rival, rival)} | {pcs} | {js} | {_fmt(pm)} | {_fmt(cm)} | "
                 f"{'—' if dm is None else f'{dm:.1f}'} | {n} |")
    L.append("")
    missing = [f"{PRETTY.get(r, r)}: " + "; ".join(data[r]["notes"])
               for r in order if data[r]["notes"]]
    if missing:
        L.append("Dashes above, with their reasons - a dash is never a win for us:")
        L.append("")
        for m in missing:
            L.append(f"- {m}")
        L.append("")
    return "\n".join(L)
