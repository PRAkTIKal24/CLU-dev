# doc-curator-c2w8-pass1-fold — the C2W8 pass-1 fold + rider 3b (the `cl-encoder` read-out amendment)

**Campaign 2, wave C2W8 PASS 2. Agent:** doc-curator. **NO worktree** (docs + registries only).
Writes `.claude/outputs/doc-curator-c2w8-pass1-fold.md`. **Spawns NOW** — its scope is **pass 1 only,
which is closed, merged and green**; the pass-2 fold is a **separate later pass** and is explicitly
out of scope here (nothing from wt1/wt2/wt3 is measured yet).

**Binding documents:** `.claude/outputs/c2w8-well-lifecycle.md` + `census.json` + `ERRATA-C2W8.md` ·
`.claude/outputs/c2w8-cifar-strong-phi.md` · the two `[C2W8]` §10 entries (scoping + review) ·
`PREREG-C2W8.md` and `PREREG-C2W8-PASS2.md` (context; **neither is ever edited**) · charter §A21 /
§A26.6 / §A28 · `AGENT_PROTOCOL.md` §5.

## ⛔⛔ READ THIS FIRST — the sweep hazard that nearly shipped a false all-clear

`[C2W6-CLOSE-FINAL]` established, and the C2W6 Hub reproduced first-hand: **a directory-scoped
`Grep` over `.claude/` silently returns nothing because `.claude/**` is gitignored** — while the same
pattern against a **named file** returns hits. ⛔ **STANDING: any "I swept `.claude/` and found
nothing" claim is VOID unless the sweep was per-file, or used a tool that ignores gitignore
(`grep -rn` via Bash does not skip ignored files).** ⛔ **Positive-control every negative sweep** —
search first for a string you know is present in the same scope, and **report the control's hit count
beside every "no hits" claim**. ⚠ The C2W8 Hub tripped the same wire in its own review entry (it
asserted "grep-checked" without running it, caught it, and corrected it on the record) — **do not
make it three.**

## The fold (items 1–6)

**1 — ⭐ RIDER 3b, the `cl-encoder` amendment (the directive names this one explicitly).**
`cl-encoder`'s arm-isolation table concludes *"reconstruction bought nothing"* (convae 0.238 **below**
randconv 0.244, −0.006, at its pre-sweep defaults). **That is a READ-OUT artifact.** At the measured
read-out (`keep` / plain-PCA / L2 / `d=256`) `convae` reaches CLU **0.267 ± 0.006** and kNN-ring
**0.277 ± 0.003** — **+0.054 over randconv**. ⇒ **the objective/architecture decomposition in that
table is NOT safe to quote.** File a dated amendment (C-3 dated-banner precedent: body untouched,
correction appended) and add the never-quote line. ⛔ Do **not** restate it as "reconstruction helps"
— the honest form is *the sign of the objective comparison depends on the read-out, and the banked
table's read-out was not the measured one.*

**2 — the pass-1 findings into the registries**, each with its measured form:
- ⛔ **`population_eroded_not_attractor` is MISLABELLED and must be corrected wherever it is carried:**
  those wells carry depth **0.4–2.2**, not depth → 0, so **Add.9 §A28.3 mechanic 1 (erosion drives
  depth to zero) does not describe them.** They are wells that **exist but are unaddressable at the
  query scale** — the **w25/N100 reach condition** (`s ≳ σ_q`), a different mechanism with a
  different fix. ⛔ **No artifact may cite this population as erosion evidence.**
- **The K1 gate was licensed by neither leg** (Hub-verified): `P` suppressed — **37/48 wells never
  read (~77 %, an order of magnitude ABOVE the 0.05 bar) yet only 1/48 passes `is_attractor`, 36 of
  those 37 excluded solely by the capture test**; on 2/3 seeds every `capture_radius` is exactly 0.0
  **and** `θ_att` = max depth over all wells ⇒ **two conjuncts forced false, `P = 0` a tautology**.
  `M`'s geometric leg never binds (`R_cert` **10.3–11.2×** the key spacing; `payload_dist` **exactly
  0.0** on all 28/29/29 pairs; monitor **#3 refusal rate 0.000**) ⇒ **`M` ≡ the same-class pair rate**.
- ⭐ **The diagnosis:** foreign contribution exceeds own on **45/48** wells; **C3 locality HOLDS in
  parameter space (violation 0.000, exact) and FAILS in function space (78–84 % of writes raise the
  foreign contribution)** — *a write touches only its own atom block, but atoms have TAILS; local in
  parameters is not local in the landscape.*
- **`capture_radius` exactly 0.000 on 47/48 while `λ_min > 0` everywhere (0.791–8.873)** ⇒ **positive
  curvature is necessary and NOT sufficient** (SC-6's lesson reproduced).

**3 — BANK, DO NOT RE-LITIGATE (the directive's own list):**
- ⭐ **The trash region's first use SHIPPED** — OFF, **bit-identical and parameter-count-identical**,
  with real designed negatives. A C1 artifact finally live. ⭐ **The compact-gate lesson is now
  load-bearing doctrine:** `γ_φ` needed a **compact** gate (exactly zero beyond `r_k`) rather than a
  sigmoid, because **a sigmoid tail makes a "local" change global** — and pass 2 applies that lesson
  one level down, to the atoms.
- **The Split-CIFAR strong-φ re-price stands WITH its mandatory provenance sentence and is NEVER
  quoted without it.** ⛔ **A re-price SCOPES the null to the feature space; it does NOT retire it.**
  Registered form: *"Split-CIFAR was a null at frozen-PCA φ; re-priced at strong φ (arm named, bytes
  ledgered), it reads X."* Numbers: `simclr` **0.31912** (5 seeds) vs PCA reference **0.16080**
  (3 seeds) = **+0.158 paired**; **−0.01424 below its own kNN-in-φ launder, 0/5 seeds positive**.
- ⛔⛔ **Carry the wave's sharpest laundering statement forward:** *at every strong-φ arm the settle
  equals its own same-keys kNN to within **±0.0007*** (randconv +0.0003 ± 0.0003 · convae
  −0.0007 ± 0.0007 · simclr +0.0002 ± 0.0004) **vs −0.048 at PCA φ** — **in a strong feature space the
  damped settle IS nearest-key indexing**, and the residual deficit is the **buffer's admission
  policy, not the read**. Sixth consecutive laundering confirmation.

**4 — the K6 cross-reference slip, banked earlier this wave and still open.** Add.9 **§A27.2**
registers the reader precondition as *"K6, §A28.3"* — but **§A28.3 is the well-lifecycle ruling**, not
a reader-class ruling; K6's natural home is **§A26.3**'s program-wide reader-class rule. Reads as a
citation slip. **File the pointer correction** (or, if you judge it the Advisor's to make, file it as
a flagged registry note with the diagnosis, and say which you did).

**5 — CM-23(q) travel rule, re-confirmed:** **+0.510 / −0.153 vs iCaRL / −0.036 laundered** travel
together, with the CIFAR null as the scope clause and the Head's Addendum-2 ruling that this **does
not count as an external benchmark won**. ⚠ New, from pass 1: at `simclr` the CLU **no longer wins
the rehearsal-free class** (`wins_rehearsal_free_class = false`; the PCA arm's win was a razor
**0.0008** over `finetune`) — the strong φ lifted the baselines too. **Record it; it does not touch
the Split-MNIST +0.510.**

**6 — two Hub self-corrections, filed as process record** (both the same defect class — *a claim
assumed true from its plausibility*): an **unrun "grep-checked" assertion** in a review entry, and a
**background launcher's completion mistaken for the job's completion** (a 3-byte log; caught, and the
verdict withheld until the real 1443/0 landed with HEAD re-verified). ⭐ Standing: **a background
launcher completing is not the job completing** — the quotable evidence is the summary line plus a
HEAD re-check, never the notification.

## ⛔ Out of scope (do not touch)
Pass-2 results (nothing is measured yet) · `PREREG-C2W8.md` / `PREREG-C2W8-PASS2.md` (**never
edited**; errata are dated blocks) · any code file · any C2W6/C2W7 in-flight curator target.

## Acceptance
1. Rider 3b amendment filed with the never-quote line, as a dated banner.
2. All pass-1 items in §2 folded with their **measured** forms; the mislabel corrected everywhere it
   is carried.
3. Every negative sweep reported **with its positive control's hit count**, per-file.
4. Your report's **first 10 lines** name anything you could not close and why.
5. ⛔ Declared NOT-RUNs / out-of-scope items listed as such, never as nulls.
