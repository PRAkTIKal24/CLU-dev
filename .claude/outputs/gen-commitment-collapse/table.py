"""Regenerate every table in gen-commitment-collapse.md from the on-disk manifests."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
RUNS = R / "runs" / "shapes3d_recon"
SEEDS = [0, 1, 2]
man = {s: json.load(open(RUNS / f"seed_{s}" / f"commit_manifest_s{s}.json")) for s in SEEDS}
swp = {s: json.load(open(RUNS / f"seed_{s}" / f"sweep_s{s}.json"))["sweep"] for s in SEEDS}
sco = {s: json.load(open(RUNS / f"seed_{s}" / f"scores_s{s}.json")) for s in SEEDS}


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} ± {v.std():.4f}"


def msi(v):
    v = np.asarray(v, float)
    return f"{v.mean():.1f} ± {v.std():.1f}"


out = []
P = out.append
P("# gen-commitment-collapse — tables (regenerated from disk)\n")

P("## 1. Diagnosis — posterior, per set, per seed (⛔ is it the posterior or the argmax?)\n")
P("| set | mean top-1 | mean top1−top2 margin | entropy bits/well (max 6) | degenerate? | argmax distinct |")
P("|---|---|---|---|---|---|")
for k in ("A", "B", "MS"):
    d = [man[s]["diagnosis"][k] for s in SEEDS]
    P(f"| **{k}** | {ms([x['mean_top1_confidence'] for x in d])} | "
      f"{ms([x['mean_top1_minus_top2_margin'] for x in d])} | "
      f"{ms([x['mean_entropy_bits_per_well'] for x in d])} | "
      f"{'yes' if any(x['degenerate'] for x in d) else '**no**'} | "
      + " / ".join(str(man[s]['baseline_argmax_distinct'][k]) for s in SEEDS)
      + f" of {man[0]['rows'][k]} |")

P("\n## 2. ⭐ Factor leverage — which factor moves the committed code (seed-mean)\n")
names = json.load(open(R / "codes_B_seed0.json"))["factor_names"]
for k in ("A", "B", "MS"):
    P(f"\n**{k}** — `code_flip_rate` (edit one factor ⇒ does the K-tuple move?) / "
      "`n_codes_from_factor_alone` / evidence share\n")
    P("| factor | " + " | ".join(names) + " |")
    P("|---|" + "---|" * len(names))
    lev = [[r for r in man[s]["factor_leverage"][k]["per_factor"]] for s in SEEDS]
    ev = [man[s]["factor_evidence"][k]["evidence_share_per_factor"] for s in SEEDS]
    P("| flip rate | " + " | ".join(
        f"{np.mean([lev[i][f]['code_flip_rate'] for i in range(3)]):.2f}"
        for f in range(len(names))) + " |")
    P("| codes alone | " + " | ".join(
        f"{np.mean([lev[i][f]['n_codes_from_factor_alone'] for i in range(3)]):.1f}"
        for f in range(len(names))) + " |")
    P("| evidence share | " + " | ".join(
        f"{np.mean([ev[i][f] for i in range(3)]):.3f}" for f in range(len(names))) + " |")

P("\n## 3. Code-tuple overlap (R29 Q2's owed number), argmax, per seed\n")
P("| pair | seed 0 | seed 1 | seed 2 |")
P("|---|---|---|---|")
for key, lab in (("B_in_A", "|B ∩ A|"), ("B_in_MS", "|B ∩ MS|"), ("MS_in_A", "|MS ∩ A|")):
    P(f"| **{lab}** | " + " | ".join(
        f"{man[s]['code_overlap'][key]['n_intersection']} of "
        f"{man[s]['code_overlap'][key]['n_distinct_b']}" for s in SEEDS) + " |")

P("\n## 4. The sweep — distinct codes, ⛔ every row with its MS companion (3-seed mean)\n")
P("| config | B / 72 | MS / ~198 | A / 512 | realised width | abstain (B) |")
P("|---|---|---|---|---|---|")
keys = [r["config"] for r in swp[0]]
rows = []
for i, key in enumerate(keys):
    b = [swp[s][i]["sets"]["B"]["n_distinct"] for s in SEEDS]
    m_ = [swp[s][i]["sets"]["MS"]["n_distinct"] for s in SEEDS]
    a = [swp[s][i]["sets"]["A"]["n_distinct"] for s in SEEDS]
    ch = swp[0][i]["channel"]
    ab = np.mean([swp[s][i]["sets"]["B"]["abstain_rate"] for s in SEEDS])
    rows.append((key, np.mean(b), np.mean(m_), np.mean(a), ch, ab))
    flag = " ⛔" if ch["exceeds_design_constant"] else ""
    P(f"| `{key}` | {msi(b)} | {msi(m_)} | {msi(a)} | {ch['pointer_bits']:.1f} bits{flag} | {ab:.2f} |")

P("\n## 5. ⭐ The accuracy leg — the pre-registered shortlist, 3 seeds\n")
P("| config | B distinct | MS distinct | in-corner `shape` balacc | AUROC | A aggregate | width | PASS? |")
P("|---|---|---|---|---|---|---|---|")
for key in sco[0]["configs"]:
    c = [sco[s]["configs"][key] for s in SEEDS]
    ch = c[0]["channel"]
    flag = " ⛔" if ch["exceeds_design_constant"] else ""
    P(f"| `{key}` | {msi([x['n_distinct']['B'] for x in c])} | "
      f"{msi([x['n_distinct']['MS'] for x in c])} | "
      f"**{ms([x['in_corner_discrimination']['balanced_accuracy'] for x in c])}** | "
      f"{ms([x['in_corner_discrimination']['auroc'] for x in c])} | "
      f"{ms([x['composition_A']['aggregate'] for x in c])} | {ch['pointer_bits']:.1f} bits{flag} | "
      f"{'⭐ PASS' if all(x['PASS'] for x in c) else 'FAIL'} |")
P(f"| *target* `knn_proto`/`proto` | 72 / 72 | — | *{sco[0]['pass_target']['balacc']:.4f} ± "
  f"{sco[0]['pass_target']['spread']:.4f}* | — | — | unbounded | — |")

P("\n### per-factor probe accuracy on B (3-seed mean)\n")
facs = list(sco[0]["configs"][list(sco[0]["configs"])[0]]["composition_B"]["per_factor"])
P("| config | " + " | ".join(facs) + " |")
P("|---|" + "---|" * len(facs))
for key in sco[0]["configs"]:
    c = [sco[s]["configs"][key] for s in SEEDS]
    P(f"| `{key}` | " + " | ".join(
        f"{np.mean([x['composition_B']['per_factor'][f] for x in c]):.3f}" for f in facs) + " |")

txt = "\n".join(out) + "\n"
(R / "tables.md").write_text(txt)
print(txt)
