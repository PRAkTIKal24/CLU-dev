# ERRATA — `PREREG-Bprime.md`

**Filed 2026-07-31 (C2W4) by `doc-curator-c2w3-sync`, per the Hub ruling recorded in the `[C2W3]` §10 entry, reconciliations 2 and 3.**

> ⛔⛔ **`PREREG-Bprime.md` IS NOT EDITED, AND MUST NOT BE.** A pre-registration whose text is revised after the fact stops being one. This file sits **beside** the prereg, points **at** it, and names the wrong sentence, the corrected statement, the affected cells and the **direction** of each error. Every downstream site (draft, registry, report) takes its wording from **here**, not from the prereg's original sentence.
>
> **Scope:** two errata, **E1** (the byte law) and **E2** (MUNKEY's venue). Both were found in C2W3, by our own spokes, against our own published statements. ⭐ **Neither erratum falsifies the result it corrects** — E1's theorem stands and its error is *conservative*; E2's narrowing stands and is *weakened as a threat to us*, not strengthened.
>
> **A third live erratum, E3 (monitor #6's "27 post-repair"), is NOT filed here** — it is a harness/instrument count, not a B′ prereg statement. It lives in `negative_results.md` (**N169**) and `claims_matrix.md` §0.6, and it is **PROVISIONAL pending `harness-debt`'s one-time re-score diff**.

---

## E1 — The byte law holds in **24 of 28** cells, not 28 of 28. The corrected law is exact in all 28.

**Affected statement.** `PREREG-Bprime.md` **§7**, and every site inheriting it, states the two-sided byte ledger's closed form as *"verified to 1e-9 in all 28 cells."*

**Published (WRONG) closed form**

```
ratio = atoms_per_item·(dim + 2)/dim  +  d/dim        =  1.4·A + 0.8
```

**Corrected (EXACT) closed form** — `bprime-theory` **T1.1/T1.2**, exact rational arithmetic, **0 ulp**, `t1_bytes.py`:

```
ratio  =  [ A·(D + 2)  +  d ] / (d + m)
```

with `A ≡ N_at/K` atoms per live item, `D = d + m + n_spectator` the **store** dimension, `d` the address dim, `m` the payload dim.

**What is actually wrong, and where it bites**

| item | published | corrected | source |
|---|---|---|---|
| cells in which the **shipped/published** law reproduces the measured ledger | 28 / 28 "to 1e-9" | ⛔ **24 / 28** | `bprime-theory` T1.2 |
| cells in which the **corrected** law reproduces it | — | ✅ **28 / 28, exact in rational arithmetic (0 ulp)** | `bprime-theory` T1.2 |
| the four `manifold` (`n_spectator = 1`) cells | **43.33×** | ⛔ **52.00× measured** (**+8.6667, +20 %**) | `bprime-theory` T1.2; Hub re-derivation, `[C2W3]` §10 |
| the architectural floor at `n_spec = 1` | printed **2.00×** by `floor_note` | ⛔ **2.40×** — the floor **RISES** | `bprime-theory` T1.2 |
| the floor at `n_spec = 0` | **2.20×** | ✅ **2.20×, unchanged** | — |
| the measured minimum ratio anywhere | **2.28×** | ✅ **2.28×, unchanged** (it is an `n_spec = 0` cell) | — |
| shell-atom floors | — | **2.40× / 2.60×** (`×9/8` surcharge on the atom term, `+1/(D+2) = 12.5 %`; `52.00 → 58.40×` exactly) | `bprime-theory` T1.2 |

**Cause.** `chlu/experiments/memory_gym.py::byte_ratio_law` divides by the **store** dimension `D` where the launder row is `(d + m)` floats. The bug is **invisible whenever `n_spectator = 0`** — which is the only geometry the unit test covers (`tests/test_memory_gym.py:118-141` passes `n_spectator = 0` literally; the end-to-end test is parametrised over `aggregate`/`recency` only).

### ⭐ Direction of the error, and what therefore SURVIVES

⭐⭐ **The error is CONSERVATIVE. The store costs *more* relative to the table than we published, so no claim of ours was inflated.**

- ✅ **The byte-floor theorem STANDS** — it is an accounting identity over the parameter leaves, not a fit. What was wrong is the *published verification sentence* and the *shipped formula*, not the theorem (`bprime-theory` §8: *"fired on the published wording, not on the theorem"*).
- ✅ ⭐ **`PREREG-Bprime.md` §7's reuse licence STANDS. `bprime-rivals` does NOT need to re-measure.**
- ✅ The **direction** of every dividend/byte statement is unchanged; the floor moves **up** at `n_spec = 1`, never down.

### Replacement sentence (use this verbatim wherever the old one appears)

> *"The two-sided byte ledger obeys `ratio = [A(D+2) + d]/(d+m)` exactly, verified in **all 28** C2W1 cells in exact rational arithmetic (0 ulp). ⚠ The **previously published** closed form `1.4·A + 0.8` and the shipped `byte_ratio_law` reproduce the measured ledger in **24 of 28** cells; they are wrong on the four `n_spectator = 1` (`manifold`) cells, which measure **52.00×** against a published **43.33×** (+20 %), and they understate the architectural floor there (**2.40×**, printed as 2.00×). The error is **conservative** — the store costs more relative to the table than published — so no claim was inflated."*

### Sites, and their disposition

| site | disposition |
|---|---|
| `PREREG-Bprime.md` §7 | ⛔ **NOT EDITED** (Hub ruling). This file is the correction of record. |
| `.claude/outputs/memory-gym-v0.md` §2 / §3.1 / `PREREG-B1` | ⚠ **dated erratum banner appended at the head of the file** by this pass, pointing here. The measured numbers in the body are untouched (they were always right — it is the *law* that was wrong). |
| `.claude/outputs/track2-admissibility.md` §2 (and its Table row for **CLU**) | ⚠ **dated erratum banner appended at the head of the file**, pointing here. |
| charter **§A2.3** | ✅ **VERIFIED, NOT EDITED** — the Advisor has already annotated it in place (ADDENDUM 3 §A12, *"Errata handled: §A2.3 byte-law erratum … annotated in place (Advisor, this date)"*). The charter is the Advisor's document. |
| `negative_results.md` | **N167** (this erratum, as an entry) + **N134** gains a dated `⚠ CORRECTED (C2W3)` block. |
| `claims_matrix.md` | §0.3 / §0.5 byte-ratio lines corrected in place at §0.6; the never-quote list gains the 24/28 line. |
| `HEP_primers.md` **§11.18 Record 13** | ⚠ **dated `Update (C2W3)` blockquote** — the primer is pedagogical and edits in place per its own protocol. The old closed form is shown as superseded, not deleted. |
| `chlu/experiments/memory_gym.py:321-339` + `floor_note` at `:553`; `tests/test_memory_gym.py:118-141, 294-311` | ⛔ **CODE — not the curator's.** Owner: **`harness-debt`** (charter §A14.4), landing in C2W4 with a published diff. ⭐ **If `harness-debt`'s landed numbers ever differ from this file, THEIRS ARE THE NUMBERS.** |

---

## E2 — MUNKEY is an **ICLR-2026 workshop paper (oral)**, not ICML 2026 — and the workshop's identity is **QUARANTINED**.

**Affected statement.** `PREREG-Bprime.md` **§8**, and ≥4 of our documents, cite *"MUNKEY (Laguna et al., arXiv:2603.15033, **ICML 2026**)"*.

**The evidence** (`bprime-fb1-recon` reconciliations 1–2, §3.1, §5):

- **arXiv v3 (2 Apr 2026) carries an EMPTY comments field** — the paper's own record states **no venue**.
- The **authors' own institutional (ETH) group page** lists it as **"Oral at ICLR Workshop TTU, 2026"**.
- An **independent secondary** says **"ICLR 2026 Workshop RSI"**.
- ⇒ ⛔ **Two sources disagree on WHICH workshop. The workshop's identity is QUARANTINED.**
- ⛔ **OpenReview was bot-blocked for the third consecutive wave** (`openreview.net/forum?id=gGH3Xp1lHR`), so the venue could **not** be resolved from the venue's own record. This correction rests on (a) the empty arXiv comments field and (b) the authors' institutional page. **It is a two-source correction, and it is single-sourced on the workshop name — which is exactly why the name is quarantined.**

**Approved citation form (verbatim)**

> *"MUNKEY (Laguna et al., arXiv:2603.15033), **an ICLR-2026 workshop paper (oral)**"* — ⛔ **name no workshop.**

**Second correction, same paper.** Our record describes MUNKEY as *"a ViT classifier"*, citing **v2**. **v3 self-describes as "a memory-augmented transformer"**, evaluated on *"natural image benchmarks, fine-grained recognition, and medical datasets"*. Use **"a memory-augmented transformer (image-classification benchmarks), v3, 2 Apr 2026"**.

### ⭐ What SURVIVES, and in which direction

⭐ **The narrowing itself STANDS, and it is now *weaker as a threat to us*, not stronger:** MUNKEY publishes **unlearning-by-design** evaluated with **MIA-AUROC → 0.5 by design — our own instrument** — but it is **not exact**: average gap to retraining **0.56 ± 0.21**.

⇒ Pillar 4's claim survives **materially narrowed** and phrased **only** on verified byte-exactness. ⛔ **"we alone delete" remains banned**, and ⭐ **say it before a referee does: a table deletes exactly by construction — exact deletion is a result only for a learned/superposed store.**

### Sites, and their disposition

| site | disposition |
|---|---|
| `PREREG-Bprime.md` §8 | ⛔ **NOT EDITED** (Hub ruling). This file is the correction of record. |
| charter **§A9.9** | ✅ **VERIFIED, NOT EDITED** — the Advisor has annotated it in place (*"[Add.3 correction (C2W3): MUNKEY is an ICLR-2026 workshop paper (oral), NOT ICML 2026 — the workshop's name is QUARANTINED (two sources disagree); and its v3 self-describes as 'a memory-augmented transformer', not a ViT classifier. The narrowing itself stands.]"*). Text confirmed present on disk by this pass. |
| `.claude/outputs/track2-admissibility.md` — §3.3, **never-quote item 13**, and the four other occurrences (report §1 item 2, the pillar table, §7, §9 item 3) | ⚠ **dated erratum banner appended at the head of the file**, pointing here. Body untouched. |
| `negative_results.md` **N163** | ⚠ dated `CORRECTED (C2W3)` block appended to the entry + **N168** as the erratum's own entry. |
| `claims_matrix.md` §0.5 (the *"we alone delete"* line, which cites "ICML 2026") | corrected at §0.6, with the superseded form named. |
| **C2W3 task files** (`bprime-fb4-gate.md`, `bprime-rivals.md`, `bprime-theory.md`, …) | ⛔ **NOT EDITED — the curator does not edit task files** (protocol §2). They are historical work orders; the correction is recorded here and in the registries. **Flagged for the Hub.** |
| `.claude/papers/**` | ⛔ **NOT TOUCHED** — paper-writer's. `bprime-draft` takes E2 from this file. |

### BibTeX of record (replaces every prior MUNKEY entry)

```bibtex
@article{laguna2026munkey,
  title   = {Rethinking Machine Unlearning: Models Designed to Forget via Key Deletion},
  author  = {Laguna, Sonia and da Silva Gon{\c c}alves, Jorge and Vandenhirtz, Moritz and
             Ryser, Alain and Cannistraci, Irene and Vogt, Julia E.},
  journal = {arXiv preprint arXiv:2603.15033},
  year    = {2026},
  note    = {v3, 2 Apr 2026; arXiv comments field EMPTY; ICLR-2026 workshop paper (ORAL) --
             workshop identity QUARANTINED (authors' group page: "ICLR Workshop TTU";
             an independent secondary: "ICLR 2026 Workshop RSI"). NOT ICML 2026.
             Self-described in v3 as "a memory-augmented transformer".}}
```

---

## Provenance

- **E1:** `.claude/outputs/bprime-theory.md` — reconciliation **R-BYTE**, **T1.1**, **T1.2**, §8, code requests **C1/C2**; Hub re-derivation in the `2026-07-31 (later still)` `[C2W3]` §10 entry (*"R-BYTE independently confirmed"*).
- **E2:** `.claude/outputs/bprime-fb1-recon.md` — reconciliations **1–2**, §D3.1, §5 item 1, the corrected BibTeX block; `[C2W3]` §10 reconciliation 3.
- **Rulings:** `[C2W3]` §10 reconciliation 2 (*"the PREREG is NOT edited … the correction is filed as a dated erratum"*); charter ADDENDUM 3 **§A12** (errata handled) and **§A14.4** (harness debt, one owner).
- **This file is additive and dated. It supersedes nothing by deletion; it names what is superseded.**

---

## E2a — DATED AMENDMENT to E2 (Hub ruling, 2026-08-01, `[C2W5]` wave review): the quarantine now covers the PRESENTATION TYPE as well as the workshop name.

`bprime-cite-check` (C2W5) found the only **venue-own** record reachable — the ICLR 2026 RSI workshop's
accepted-papers page — lists arXiv:2603.15033 as a **poster** (#41), while the authors' own page states
*"ICLR 2026 Workshop TTU (Oral)"*; TTU's list defers to OpenReview, which is bot-blocked (4th wave).
Dual-venue (poster at RSI + oral at TTU) and single-error readings cannot be separated from here.
**Hub ruling: E2's approved citation form drops "(oral)" — the safe form is**
> *"MUNKEY (Laguna et al., arXiv:2603.15033), an ICLR-2026 workshop paper"* — ⛔ name no workshop, state no presentation type.
The long form (both listings, both attributions) is permitted where maximal information is wanted.
E2's body above is unchanged per C-3; this amendment is the citation form of record. MUNKEY is cited
0 times in `draft-r2.md`; this is preventive.
