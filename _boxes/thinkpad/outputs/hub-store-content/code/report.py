"""hub/store-content — turn the run JSONs into the tables that go in LOG.md.

⛔ Reads only what the runs wrote; it computes no new model quantity.  The paired
estimand comes from ``estimand.paired_verdict`` and the margins come from the PREREG,
never from here.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimand import paired_verdict            # noqa
from instrument import FACTORS                 # noqa

OUT = Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs")) / "hub-store-content"
#: declared in PREREG-hub-store-content-1.md §3, ⛔ before the run
MARGIN_AGG = 0.02
MARGIN_FACTOR = 0.05
#: chance level per factor on Shapes3D, for reading the table honestly
CHANCE = {"floor_hue": 1 / 10, "wall_hue": 1 / 10, "object_hue": 1 / 10,
          "scale": 1 / 8, "shape": 1 / 4, "orientation": 1 / 15}


def load(tag: str) -> dict:
    d = {}
    p = OUT / "runs" / tag
    for f in sorted(p.glob("*.json")):
        if f.name in ("scoreboard.json", "verdicts.json"):
            continue
        d[f.stem] = json.load(open(f))
    return d


def flat_table(runs: dict, row: str = "full") -> str:
    hdr = ("| arm | D | N | bytes | mix | agg | exact | " +
           " | ".join(f"**{f}**" if f in ("scale", "orientation") else f for f in FACTORS) + " |")
    sep = "|" + "---|" * (7 + len(FACTORS))
    lines = [hdr, sep]
    for name, r in runs.items():
        rr = r["rows"].get(row)
        if rr is None:
            continue
        sp = r.get("spec", {})
        led = r.get("byte_ledger", {})
        mx = (r.get("grouping") or {}).get("mixing")
        pf = rr["per_factor"]
        lines.append(
            f"| `{name}` | {sp.get('D','-')} | {sp.get('n_atoms','-')} | "
            f"{led.get('deployed_bytes', 0):,} | "
            f"{('%.2f' % mx) if isinstance(mx, (int, float)) else '-'} | "
            f"{rr['aggregate']:.4f} | "
            f"{rr['exact_match']:.4f} | " +
            " | ".join(f"{pf[f]:.4f}" for f in FACTORS) + " |")
    lines.append("| _chance_ | | | | | " + f"{np.mean(list(CHANCE.values())):.4f} | | " +
                 " | ".join(f"{CHANCE[f]:.4f}" for f in FACTORS) + " |")
    return "\n".join(lines)


def verdicts(runs: dict, pairs, row: str = "full") -> list:
    out = []
    for arm, ctl in pairs:
        if arm not in runs or ctl not in runs:
            continue
        a = runs[arm]["rows"][row].get("per_corner")
        b = runs[ctl]["rows"][row].get("per_corner")
        if not a or not b or len(a) < 6:
            continue
        out.append(paired_verdict(a, b, "aggregate", MARGIN_AGG, arm, ctl))
        for f in ("scale", "orientation", "shape"):
            out.append(paired_verdict(a, b, "", MARGIN_FACTOR, arm, ctl, factor=f))
        out.append(paired_verdict(a, b, "distinct_valid_novel_tuples", 0.0, arm, ctl))
    return out


def verdict_table(vs: list) -> str:
    lines = ["| arm vs control | metric | arm | control | paired Δ | 95 % CI | W/T/L | p | margin | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for v in vs:
        lines.append(
            f"| `{v['arm']}` vs `{v['control']}` | {v['metric']} | {v['arm_mean']:.4f} | "
            f"{v['control_mean']:.4f} | **{v['paired_mean_delta']:+.4f}** | "
            f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
            f"{v['won']}/{v['tied']}/{v['lost']} | {v['wilcoxon_p_two_sided']:.2e} | "
            f"{v['declared_margin']:.2f} | "
            f"{'WIN' if v['verdict'].startswith('beats') else 'no'} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--row", default="full")
    a = ap.parse_args(argv)
    runs = load(a.tag)
    print(f"### {a.tag} — row `{a.row}`\n")
    print(flat_table(runs, a.row))
    pairs = [("WIDEN", "P"), ("WIDEN", "WIDEN_SHUF"), ("WIDEN", "basis_head"),
             ("WIDEN", "knn_proto"), ("WIDEN_MIX", "P"), ("WIDEN_MIX", "WIDEN_MIX_SHUF"),
             ("basis_head@WIDEN", "basis_head"), ("P", "basis_head")]
    vs = verdicts(runs, pairs, a.row)
    if vs:
        print("\n" + verdict_table(vs))
        json.dump(vs, open(OUT / "runs" / a.tag / "verdicts.json", "w"), indent=1)
    # the charter floors, per arm
    print("\n#### charter §C floors")
    print("| arm | full | atom_mean_lookup | row_lookup | atoms_shuffled | §C.1 | §C.2 |")
    print("|---|---|---|---|---|---|---|")
    for name, r in runs.items():
        rows = r["rows"]
        if "atom_mean_lookup" not in rows:
            continue
        f = rows["full"]["aggregate"]
        m, l, s = (rows["atom_mean_lookup"]["aggregate"], rows["row_lookup"]["aggregate"],
                   rows["atoms_shuffled"]["aggregate"])
        print(f"| `{name}` | {f:.4f} | {m:.4f} | {l:.4f} | {s:.4f} | "
              f"{'PASS' if f > max(m, l) else 'FAIL - decorative'} | "
              f"{'PASS' if s < 0.5 * f else 'FAIL - did not collapse'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
