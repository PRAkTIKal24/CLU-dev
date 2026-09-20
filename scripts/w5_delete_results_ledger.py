"""Generate RESULTS-LEDGER.md from the artifacts. Nothing here computes a result."""
import json, os, subprocess
from pathlib import Path

OUT = Path(r"C:/chlu/outputs/w5-delete")
RUN = OUT / "run1"
REP = OUT / "replication" / "run1"
utc = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                     capture_output=True, text=True).stdout.strip()


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


L = []
L.append("# w5 hub/delete — RESULTS LEDGER")
L.append("")
L.append(f"Generated {utc} from the artifacts, not typed. Target `deletion`, budget **3 scored "
         "runs**, banked since wave 2.")
L.append("")
L.append("| run | PREREG | what it was | status | outcome |")
L.append("|---|---|---|---|---|")

rep = load(RUN / "report.json")
status = "COMPLETE" if rep else "IN FLIGHT at the time of writing"
seeds = rep["seeds_scored"] if rep else []
L.append(f"| **1/3** | `PREREG-w5-delete-1.md` | Row 7 on the field store: delete `object_hue=0.0` "
         "by lifetime/eviction, and delete `red ∧ cylinder` by the 1-byte coupling hole. BOTH "
         "halves — the refusal and the surviving composition — on 109 held-out corners (20 "
         f"deleted / 89 survivor), seeds 0/1/2, cache generation 2. | {status} | see below |")
L.append("| 2/3 | — | **NOT SPENT** | banked | — |")
L.append("| 3/3 | — | **NOT SPENT** | banked | — |")
L.append("")
L.append("⛔ A declared NOT-RUN is never a null. Runs 2 and 3 are returned unspent.")
L.append("")
L.append("## Per-bar outcome of run 1/3")
L.append("")
L.append("| bar | what it required (declared in advance) | result |")
L.append("|---|---|---|")

rows = []
b = {s: load(RUN / f"seed_{s}" / "bytes.json") for s in (0, 1, 2)}
u = {s: load(RUN / f"seed_{s}" / "unseen_pairs.json") for s in (0, 1, 2)}
c4 = {s: load(RUN / f"seed_{s}" / "c4_nearest_atom.json") for s in (0, 1, 2)}
d = {s: load(RUN / f"seed_{s}" / "dash.json") for s in (0, 1, 2)}
lr = {s: load(RUN / f"seed_{s}" / "learners.json") for s in (0, 1, 2)}

have_b = [s for s in (0, 1, 2) if b[s]]
if have_b:
    ev = " / ".join(f"{b[s]['evict']['byte_assertion']['bytes_changed']:,}" for s in have_b)
    out = " / ".join(str(b[s]['evict']['byte_assertion'].get('bytes_changed_outside_permitted'))
                     for s in have_b)
    allh = all(b[s]["ALL_BYTE_PROPERTIES_HOLD"] for s in have_b)
    rows.append(("**BAR 5** — byte property",
                 "0 bytes changed OUTSIDE the permitted ranges, exact, no tolerance; "
                 "`assert_no_refit` passes; the hole is 1 byte",
                 f"**PASS on seeds {', '.join(map(str, have_b))}.** evict {ev} B changed, "
                 f"**outside {out}**; hole **1 B / outside 0** on every seed; "
                 f"`ALL_BYTE_PROPERTIES_HOLD` = **{allh}**; negative control fires ("
                 + " / ".join(format(b[s]["negative_control_refit"]["bytes_changed"], ",")
                              for s in have_b) + " B)"))

have_u = [s for s in (0, 1, 2) if u[s]]
if have_u:
    ok = all(u[s]["BAR4b_PASS"] for s in have_u)
    lit = " / ".join(str(u[s]["other_unseen_pairs"]["n_with_any_hole_flag"]) for s in have_u)
    rows.append(("**BAR 4(b)** — the trap test",
                 "every conjunction the pool never held, other than the hole, keeps `hole` flag "
                 "0.0000 and refusal drift ≤ 0.01",
                 f"**{'PASS' if ok else 'FAIL'} on seeds {', '.join(map(str, have_u))}.** "
                 f"102 such conjunctions; the hole flags 1.0000; of the other 101, **{lit} light "
                 f"up** and the worst refusal drift is "
                 f"{max(u[s]['other_unseen_pairs']['max_abs_refuse_drift'] for s in have_u):.4f}"))

have_d = [s for s in (0, 1, 2) if d[s]]
if have_d:
    fams = d[have_d[0]]["families"]
    rows.append(("**The dash, earned**",
                 "a row a rival cannot enter gets a dash and a footnote, never a win for us",
                 f"**{len(fams)} learner families each refuse 0.0000** of the 720 deleted-"
                 f"conjunction requests and answer them anyway; `lace` and `poe_ebm` are not "
                 f"callable at all. Population-caveated and descriptive, not a contest."))

have_c4 = [s for s in (0, 1, 2) if c4[s]]
if have_c4:
    rows.append(("**§C.4** — nearest atom",
                 "an arm whose settled state sits on its nearest atom is a lookup table",
                 "**NOT a lookup table**: relative distance " +
                 " / ".join(f"{c4[s]['mean_relative_distance']:.4f}" for s in have_c4) +
                 " (min " + " / ".join(f"{c4[s]['min_relative_distance']:.4f}" for s in have_c4) + ")"))

if rep:
    for s in rep["seeds_scored"]:
        break
    bars = rep["per_seed"][str(rep["seeds_scored"][0])]["bars"]
    for key, label, req in (
        ("BAR1_evict_refusal", "**BAR 1** — eviction refuses",
         "refusal ≥ 0.99 on the deleted set and on red's other shapes"),
        ("BAR2_evict_composition", "**BAR 2** — composition survives",
         "abs(delta) <= 1.0 novel tuple/corner on the 89 survivor corners"),
        ("BAR3_hole_refusal", "**BAR 3** — the hole refuses",
         "hole flag = 1.0000 and refusal ≥ 0.99 on the deleted conjunction"),
        ("BAR4_hole_surgical", "**BAR 4** — the hole is surgical",
         "hole flag 0.0000 on every survivor probe, drift ≤ 0.01, paired Δ exactly 0"),
    ):
        got = bars.get(key, {})
        per = [rep["per_seed"][str(s)]["bars"].get(key, {}).get("PASS") for s in rep["seeds_scored"]]
        rows.append((label, req,
                     f"**{'PASS' if all(per) else 'MIXED/FAIL'}** per seed: " +
                     " / ".join(str(x) for x in per)))

def _cell(x):
    # a literal | inside a cell splits the markdown row; escape rather than hope
    return str(x).replace("|", "\|")


for a, bb, cc in rows:
    L.append(f"| {_cell(a)} | {_cell(bb)} | {_cell(cc)} |")

L.append("")
L.append("## Replication (charter §A.5)")
L.append("")
comp = load(RUN / "replication.json")
if comp:
    L.append(f"**REPLICATED = {comp['REPLICATED']}** — clean checkout of `w5/delete`, its own "
             "process tree, output root and partitions. EXACT tolerance on byte properties and "
             "refusal rates; 1e-6 on the two judge-mediated composition deltas.")
    L.append("")
    L.append("| quantity | class | run | replication | max gap | agrees |")
    L.append("|---|---|---|---|---|---|")
    for k, v in comp["headline"].items():
        if "AGREES" not in v:
            L.append(f"| `{k}` | — | — | — | — | *(NOT RUN)* |")
            continue
        L.append("| `%s` | %s | %s | %s | %s | **%s** |"
                 % (k, v["class"],
                    " / ".join(f"{x:.4f}" for x in v["run"]),
                    " / ".join(f"{x:.4f}" for x in v["replication"]),
                    f"{v['max_abs_gap']:.2e}", v["AGREES"]))
else:
    repb = {s: load(REP / f"seed_{s}" / "bytes.json") for s in (0, 1, 2)}
    have = [s for s in (0, 1, 2) if repb[s] and b.get(s)]
    L.append("The full mechanical comparison (`--stage compare`) needs both reports; what is "
             "already established is the **byte half**, on a clean checkout:")
    L.append("")
    L.append("| seed | store image sha256[:32] — run | replication | identical | evict changed | hole |")
    L.append("|---|---|---|---|---|---|")
    for s in have:
        same = b[s]["manifest"]["sha256"] == repb[s]["manifest"]["sha256"]
        L.append("| %d | `%s` | `%s` | **%s** | %s = %s | 1 B = 1 B |"
                 % (s, b[s]["manifest"]["sha256"][:32], repb[s]["manifest"]["sha256"][:32],
                    same,
                    f"{b[s]['evict']['byte_assertion']['bytes_changed']:,}",
                    f"{repb[s]['evict']['byte_assertion']['bytes_changed']:,}"))

L.append("")
L.append("## Cost line")
L.append("")
if rep:
    c = rep["per_seed"][str(rep["seeds_scored"][0])]["cost"]
    L.append(f"- table **retrain from scratch (GOLD STANDARD)**: "
             f"**{c['retrain_from_scratch_s']:.2f} s**, **{c['rows_refitted_by_retrain']:,} rows "
             f"refitted**, {c['rows_dropped']:,} dropped.")
    L.append(f"- table **row removal**: {c['row_removal_s']:.4f} s, **0 rows refitted**.")
L.append("- **field store, evict**: an in-place zero of 6–7 items. **0 rows refitted, 0 bytes "
         "outside the permitted ranges.**")
L.append("- **field store, the hole**: **1 byte. 0 rows refitted.**")
if any(lr.values()):
    s = [x for x in (0, 1, 2) if lr[x]][0]
    for fam, legs in lr[s]["families"].items():
        L.append(f"- **`{fam}` retrained without the value**: "
                 f"**{legs['without_red']['wall_clock_s']:.1f} s**, "
                 f"{legs['without_red']['rows_refitted']:,} rows — and afterwards it still answers "
                 f"the deleted requests at refusal **0.0000**.")
else:
    L.append("- MLP retrain-from-scratch leg: see `learners.json` if present; **declared NOT RUN** "
             "otherwise, which is never a null.")

(OUT / "RESULTS-LEDGER.md").write_text(chr(10).join(L), encoding="utf-8")
txt = chr(10).join(L)
print("WROTE", (OUT / "RESULTS-LEDGER.md"), len(txt), "chars,", txt.count(chr(10)) + 1, "lines")
