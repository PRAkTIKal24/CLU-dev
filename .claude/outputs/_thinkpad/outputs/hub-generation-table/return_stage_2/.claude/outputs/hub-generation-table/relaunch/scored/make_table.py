"""The generation table as it now stands, WITH THE 0.5278 CONSTANT-OUTPUT FLOOR DRAWN ON IT.

Every bar is annotated with its distinct-factor-tuple count (the launch addendum's standing
rule: an aggregate without its tuple count is void) and its deployed byte ledger.
Rows below the floor are the ones a near-constant output would beat.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

O = os.environ["CHLU_OUT3"].replace("/c/", "C:/", 1)
cell = json.load(open(os.path.join(O, "scored", "cell", "seed_0", "cell_manifest.json"),
                     encoding="utf-8"))
arm = json.load(open(os.path.join(O, "scored", "arm", "arm_results.json"), encoding="utf-8"))

FLOOR = 0.5278
rows = []


def add(label, agg, tup, byt, kind):
    rows.append({"label": label, "agg": float(agg), "tup": tup, "bytes": byt, "kind": kind})


for a, r in cell["arms"].items():
    add(a, r["external"]["composition_B"]["aggregate"],
        r["distinct"]["B"]["factor_tuples"],
        r.get("byte_ledger", {}).get("deployed_bytes"), "rival")
L2 = arm["rows"]
add("THE ARM (shared-basis, NMF)", L2["arm"]["composition_B"]["aggregate"],
    L2["arm"]["distinct_B"]["factor_tuples"], L2["arm"]["byte_ledger"]["total_bytes"], "arm")
add("clu_linear  (shipped store,\nSHIPPED BYTES, read verb only)",
    L2["clu_linear"]["composition_B"]["aggregate"],
    L2["clu_linear"]["distinct_B"]["factor_tuples"],
    L2["clu_linear"]["byte_ledger"]["total_bytes"], "arm")
for k, lab, kind in (("control_atoms_medoid", "C.1 atoms_medoid (stored rows)", "ctrl"),
                     ("control_atoms_kpp", "C.1 atoms_kpp (adversarial)", "ctrl"),
                     ("control_pca_affine", "R4 pca_affine (CONTROL ONLY)", "ctrl"),
                     ("control_random", "D-matched random null", "ctrl"),
                     ("control_mean", "C.2 store_removed (K=64)", "ctrl")):
    add(lab, L2[k]["composition_B"]["aggregate"], L2[k]["distinct_B"]["factor_tuples"],
        L2[k]["byte_ledger"].get("total_bytes"), kind)
add("best constant output", FLOOR, 1, 0, "floor")

rows.sort(key=lambda r: r["agg"])
ceil_v = cell["ceiling"]["external"]["composition_B"]["aggregate"]

COL = {"arm": "#2b6cb0", "rival": "#718096", "ctrl": "#b7791f", "floor": "#c53030"}
fig, ax = plt.subplots(figsize=(11.5, 8.2))
y = np.arange(len(rows))
ax.barh(y, [r["agg"] for r in rows],
        color=[COL[r["kind"]] for r in rows], height=0.68,
        edgecolor="white", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
ax.axvline(FLOOR, color=COL["floor"], lw=2.2, ls="--", zorder=5)
ax.text(FLOOR + 0.006, len(rows) - 0.4,
        f"constant-output floor {FLOOR}\n(1 distinct tuple of 72)",
        color=COL["floor"], fontsize=9, va="top", fontweight="bold")
ax.axvline(ceil_v, color="#2f855a", lw=1.6, ls=":", zorder=5)
ax.text(ceil_v - 0.006, 0.2, f"decoder ceiling {ceil_v:.4f}", color="#2f855a",
        fontsize=9, ha="right", rotation=90, va="bottom")
ax.axvspan(0, FLOOR, color=COL["floor"], alpha=0.05, zorder=0)

for i, r in enumerate(rows):
    b = "" if not r["bytes"] else f"   {r['bytes']:,} B"
    ax.text(r["agg"] + 0.008, i, f"{r['agg']:.4f}   ({r['tup']} tuples/72){b}",
            va="center", fontsize=8.2)

ax.set_xlim(0, 1.30)
ax.set_xlabel("composition-B aggregate  (external classifier, eval_b, 72 images, seed 0)")
ax.set_title("The generation table on FORGIS_14 (Windows AMD64), seed 0 — scored 1/3\n"
             "every bar carries its distinct-tuple count; the dashed line is what emitting "
             "ONE picture 72 times scores", fontsize=10.5)
ax.grid(axis="x", alpha=0.25, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
out = os.path.join(O, "scored", "generation_table_seed0.png")
fig.savefig(out, dpi=170)
print("wrote", out)
below = [r["label"].replace("\n", " ") for r in rows if r["agg"] < FLOOR and r["kind"] != "floor"]
print(f"rows BELOW the constant-output floor ({len(below)}): {below}")
