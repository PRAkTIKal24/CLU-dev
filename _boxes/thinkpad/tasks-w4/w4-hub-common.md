# WAVE 4 — COMMON RULES FOR ALL FOUR HUBS
**From the Advisor's wave plan (Head-ruled 2026-09-19). Binds `hub/read`, `hub/unseen-valid`,
`hub/placement`, `hub/field-store`.** Boot: `autonomous-hub-charter.md` → this file → your target file.
⛔ Nothing else. Everything you need from waves 2–3 is in §4.

## 1. ⭐⭐ THE COMPARISON — this is the change from wave 3
**Every result row is scored against the FULL RIVAL SET**, built by Spoke A: prototype table · repaired
LACE · product-of-experts EBM (own budget, multiple printed) · **MLP · transformer · LSTM ·
continuous-input MLP**, all at matched total parameters ±10 % with the count printed, 3 seeds.

⛔⛔ **A win over an earlier CLU configuration is an ABLATION, and is labelled so in every table.
A win over a RIVAL is a RESULT.** ⭐ Wave 3's win was over `basis_head` — an ablation by this rule.
The generic learners are the new bar and nobody has beaten them yet.
⛔ Rows a rival cannot enter (5–7) get a **dash and a footnote**. ⛔ A dash is never a win for us.
**Like-for-like ablation on every row:** the correlation-split store + linear head.
**Landscape-removed and landscape-shuffled controls on every arm, every row.**

## 2. Scored runs, seeds, and the gate you are blocked on
- **3 scored runs per named TARGET.** ⭐ **Replications of the same PREREG are FREE** (Advisor ruling —
  this reverses wave 3's ambiguity). All scored runs reported; the best is never reported alone.
- ⛔⛔ **Everything is scored on cache GENERATION 2 (seeds 0, 1, 2).** ⛔ **Generation-1 numbers (waves 2
  and 3) are NEVER mixed with it, in either direction.** ⭐ **Three seeds before any number is a result.**
- ⭐ **MECHANICAL PRECONDITION for any scored run:**
  `/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` exists and the rivals you need are `true` in it.
  ⭐ **DEV work starts immediately — do not idle waiting.** Poll the file; it goes `partial` before `ready`.
- Append your row to `/c/chlu/SCORED-LEDGER.md` **before** a scored run starts.

## 3. The scoreboard — the Head's taxonomy, all seven rows
**1** seen · **2** unseen-valid · **3** two-way · **4** many-way · **5** compositional refusal ·
**6** invalid-request refusal · **7** deletion (incl. refuse-after-deletion and the red-cylinder
conjunction). Fill all seven; you are judged on your target rows; the rest are reported, never claimed.
⭐ **Row 2 uses the declared SECOND JUDGE (the oracle-referenced readout) and row 2 only.**
**Estimand: ≥ 49 corners, paired, margin declared in the PREREG before the first held-out number.**
⛔ A single cell never decides a target.

## 4. ⛔ KNOWN — do not rediscover, do not re-litigate
1. ⭐ **The read is the bottleneck, measured three ways.** From identical content: `knn_proto` reads
   `orientation` at **0.9320**, `basis_head` **0.5142**, the addressed read **0.1828**. `clu_span` ties
   a same-size **PCA dictionary** at +0.0000 Δ. Changing the read verb (argmax → linear span) was the
   single largest effect in the programme (+0.2014, 9 → 27 tuples at a byte-identical ledger).
2. ⭐ **Factor-aligned wells win many-way composition** (+11.612 tuples/corner over `basis_head`,
   49 corners, p = 9.3e-06) and **tie two-way** (+1.200, p = 0.322). The win is **alignment, not
   balance**: equal wells cost a correlation partition 14 tuples/corner.
3. ⛔ **Learned well placement from reconstruction alone was a NULL, and unfreezing the encoder alone
   made it worse.** ⛔ Do not start there.
4. ⛔ **The encoder limits factor alignment:** only **12 of 1984** live channels are top-ranked by
   `shape` and **4** by `scale`; three of five wells go to hue.
5. ⛔ **Top-1 judging cannot separate a perfect arm from a perfect lookup on row 2** — usable span
   exactly **0.0000 Δ**. Both lookup diagnostics (`SNAPFRAC` null 0.6680; `d_near` 0.187 < `d_req`
   0.313) fire on a *perfect* render. ⭐ Hence the second judge, row 2 only.
6. ⛔ **The frozen `row2_unseen.json` realises only 48 corners** — below the ≥49 estimand. Regenerate.
7. ⛔ **Refusal is winnable at 8 bytes** (per-factor presence bitmap + bracket check: rows 1,2,5,6 all
   1.0000). ⭐ The exception, and it is exact: **a per-factor gate cannot express a conjunction deletion
   at all**, and the joint table that can costs every row-3/4 request.
8. ⛔ **Dreaming: BANKED. Admission gate as built: RETIRED. Earlier-layer content: RETIRED**
   (0 of 49 corners, p = 1.1e-09). ⛔ None of these is reopened in wave 4.
9. ⚠ **At a 2^17-byte budget with a fixed 2048-dim render interface, `N·D = 2^17` forces content to be
   paid for in atoms.** Moving the content dial means moving the render interface first.
10. ⚠ **Eligible deletion targets on this decoder:** `object_hue` yes; ⛔ `shape = cube` NO (it is the
    degradation attractor for incoherent features); ⛔ `scale`, `orientation` no (physical defaults).

## 5. Labels, and the claim you may not make
⭐ **Labels are used ONCE at fit time to align wells, and at the address book.** ⭐⭐ **Declare this in
every table caption.** ⛔ **Zero-supervision structure is BANKED — it is not claimed in wave 4.**

## 6. Process
One launcher per hub; write `START_UTC.txt` first and refuse to start if a live one exists.
`LOG.md` in dated bullets throughout, never retrospectively rewritten; corrections are new bullets;
the final report is the last section of that same file. ⛔ **A declared NOT-RUN is never a null.**
Local CPU (4 threads); ⭐ **GPU is cloud-only and only for a run whose best case is a headline number** —
write the three justification lines into `LOG.md` first, or you do not have a case.
Cap memory and per-corner wall clock on every launch; make every run resumable.
Return needs BOTH a report AND a `git bundle` — ⛔ `git status` does not show gitignored outputs, so
verify with a directory listing. ⭐ **Bundle the branch WHOLE.**
