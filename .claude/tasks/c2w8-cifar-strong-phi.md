# c2w8-cifar-strong-phi — the Split-CIFAR retry, registered as a RE-PRICE of the CIFAR null

**Campaign 2, wave C2W8 ("Consolidation + trash"). Agent:** experiment-engineer.
**Worktree 2 of ≤3.** Branch **`c2w8-cifar-strong-phi`** from `main @ d70898b`, worktree `../CHLU-c2w8-phi`.
Writes `.claude/outputs/c2w8-cifar-strong-phi.md` + artifacts to `.claude/outputs/c2w8-cifar-strong-phi/`.
**Spawns NOW, in parallel with `c2w8-well-lifecycle` (wt1) — zero file overlap, declared below.**
**Budget:** ≈ 0.5 day plumbing + ≈ 4–8 h of measured runs. Price the encoder-fit cost **before**
running the sweep; **cut arms before cutting seeds, and declare the cut.**

**Binding documents, read first:**
- `.claude/outputs/c2w8-well-lifecycle/PREREG-C2W8.md` — **§7 (the launder-margin correction),
  §8 (your mandatory provenance), §6 N7/N8 (your registered predictions), §9 (NOT-RUNs).**
  You implement the registered predictions; you do not re-derive or re-tune them.
- charter **§A21 C2W8 row** (the benchmark's Head ratification) · **§A4.3** (the strong-φ policy —
  your specification: identical φ for CLU / baselines / launder, φ params in the byte ledger on
  **all** arms).
- `.claude/outputs/` w25 CL-entry artifacts + `claims_matrix.md` **CM-23(q)** (the three sentences
  that always travel together) · `PREREG_CL_PHI.md` (the `task1_only` regime is **binding**;
  `phi_dim ≥ 16`).
- intervention doc **§6** (benchmark criteria) and **§8.3** (no primary claim where the competition
  is absent by construction — this is an "and also" wave, and CL is supplementary by Head ruling).

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial:** none as a new claim — this is a **re-price of a banked null** on the isolation/lifetimes
  benchmark. ⛔ No paper number; no new benchmark entry.
- **Laundering control:** **kNN-in-φ at matched memory** (N89 / CM-22(i)) on **every** cell, computed
  in the **same** φ the arm uses. ⚠ A strong φ makes the launder stronger too — that is the point of
  the control and it is not a reason to weaken it.
- **Falsifies the re-price:** strong φ leaves Split-CIFAR ACC statistically unmoved from the banked
  PCA-φ null ⇒ the null's diagnosed cause (feature space) is **wrong** and the null re-prices to
  the discipline, which is a materially more important finding than a lift. Report it as such.
- **Does NOT falsify:** losing to replay/iCaRL (never claimed under any outcome, CM-23(q)); a
  negative launder margin (that is N8's registered prediction, prior 0.85).
- ⛔ **N94** on every cell; anything below the undemoted floor is labelled **non-promotable**.

---

## FILE OWNERSHIP (declared)

**You own and may modify:**
`chlu/experiments/exp_cl_entry.py` · `chlu/experiments/phi_encoders.py` ·
`chlu/experiments/exp_phi_read_in.py` · `chlu/experiments/exp_phi_stream.py` ·
`tests/test_cifar_strong_phi.py` (**new**) · `chlu/config.py` (**additive only**, and ⚠ **coordinate**:
your sibling also appends there — append at the **end of the CL block**, touch no existing default,
and expect the Hub to resolve a trivial adjacency conflict at merge).

⛔ **DO NOT TOUCH:** `chlu/core/controller.py` · `chlu/core/clu_system.py` ·
`chlu/core/friction_field.py` · `chlu/core/well_lifecycle.py` · `chlu/experiments/usage_telemetry.py` ·
`chlu/experiments/exp_well_lifecycle.py` (**all `c2w8-well-lifecycle`, wt1**) · the C2W6 files
(`train_cluformer.py`, `blocks.py`, `scripts/csf3/`, `exp_anti_erosion.py`) · the C2W7 files
(`multiplicity_read.py`, `monitors.py`, `factored_store.py`, `multiwell_read.py`).

⚠ Work **in your worktree**, never in the shared main checkout.

---

## The task

**1 — Plumbing (small; most of it already exists).** `exp_cl_entry.py` already dispatches
`cfg.phi_arm` through `build_read_in`, and `phi_encoders.py` already implements
`ENCODER_ARMS = ("randconv", "convae", "simclr")` — the §A4.3 strong-φ family. Your job is to make
the **Split-CIFAR-10 class-IL stream run end-to-end on each arm** under the binding `task1_only`
regime, with the encoder fit **on task-1 classes only** and then frozen for the whole stream.
⛔ **`generic_frozen` is a declared upper bound and is never a headline** (w24 ruling, binding);
`online` stays the unrun stub.

**2 — The cells.** Split-CIFAR-10, class-IL, rehearsal-free, ≥ 3 seeds (≥ 5 on the headline margin):
- **arms:** `randconv` · `convae` · `simclr` · the banked frozen-PCA φ as the **reference row**;
- **every arm** carries: the CLU store's ACC + forgetting/BWT (GEM formulas), the **kNN-in-φ launder
  at matched memory in the same φ**, and the rehearsal-free baseline table already in
  `cl_baselines.py` (EWC / SI / LwF as the **known nulls** — ⛔ never presented as a CLU win);
- **byte ledger on every arm including the launder**, with **φ params counted** (§A4.3 — φ bytes are
  the whole point of the fairness invariant here; a strong encoder that is off the ledger is the
  matched-bytes violation in its most obvious form).

**3 — The two registered readings (prereg §6).**
- **N7:** does strong φ lift CLU ACC by **≥ +0.10** over the banked PCA-φ null? (prior 0.55)
- **N8:** does CLU beat its **own** kNN-in-φ launder on Split-CIFAR? (prior **0.15**)
⭐ **N8 is the one that matters.** N7 without N8 means the feature space was the null's cause **and**
the store still adds nothing over the trivial substitute in that better space — which is a clean,
publishable scope clause, not a disappointment. Report both, with signs, SEs and seed counts.

**4 — ⛔ THE MANDATORY PROVENANCE (prereg §8, non-negotiable).** Every number you report, and every
sentence anyone could lift from your report, carries this form:
> *"Split-CIFAR was a null at frozen-PCA φ; re-priced at strong φ (arm named, bytes ledgered), it
> reads X."*
⛔ **It is NEVER quoted without that provenance, in any artifact, draft or table.** A re-price is not
a new benchmark entry, and a favourable X does **not** retire the null — it scopes it to the feature
space, which is exactly what was diagnosed. Put this sentence in your report's **§1**, not a
footnote.

**5 — ⛔ The CM-23(q) travel rule.** If you quote the Split-MNIST side at all for context, the three
sentences travel together: **+0.510 over the rehearsal-free class · −0.153 vs iCaRL · −0.036
LAUNDERED**, with the CIFAR null as the scope clause and the Head's ruling that this **does not count
as an external benchmark won**. ⛔ "+0.510" never appears without "−0.036 laundered" in the same
paragraph.

---

## Acceptance (mechanical)

1. Split-CIFAR-10 class-IL runs end-to-end on all three strong-φ arms under `task1_only`, ≥ 3 seeds.
2. Every cell carries its kNN-in-φ launder **in the same φ**, its baseline table, and its byte
   ledger **with φ params counted**.
3. N7 and N8 are each reported with sign, SE and seed count, against the registered priors.
4. The §8 provenance sentence appears in your report's §1.
5. Full suite green on your branch, with the count arithmetic stated.
6. Your report's **first 10 lines** name any downstream reconciliation list (protocol §5 corollary).
7. **Declared NOT-RUNs are listed as NOT-RUNs, never as nulls.**

## Honesty clauses carried
If an encoder arm will not fit in budget, cut the **arm** (declared) before you cut seeds — a
one-seed arm is not a result and the C2W4 rescue-gate lesson (any gate whose control has
learned-init variance is underpowered at n=3) applies directly to encoder inits here. If the strong
φ leaves the null unmoved, that is the **more** important outcome and it gets the report's headline.
⛔ You never push `origin`; the Hub handles integration and `clu-dev`.
