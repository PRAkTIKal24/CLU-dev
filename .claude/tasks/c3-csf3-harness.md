# c3-csf3-harness — the real-data harness on the merged pilot infra (Track A)

**Campaign 3, wave 1 (THE REAL-DATA WAVE). Agent:** experiment-engineer. **ONE worktree (wt1).**
Branch **`agent/experiment-engineer/c3-csf3-harness`** off **`main` @ the commit carrying the
pilot-ttt merge** (the Hub names it at spawn; it does **not** exist yet — see the precondition).
⚠ **The shared checkout currently sits on the live `pilot-ttt-nan-and-d5-wiring` branch.** Take a
worktree; cwd = the worktree on every command.
⚠ **Reuse the main venv** (w6 lesson): `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`
rather than a fresh `uv sync` inside the worktree, which can resolve a newer JAX and flip bit-level tests.
Writes `.claude/outputs/c3-csf3-harness.md` + artifacts to `.claude/outputs/c3-csf3-harness/`.
**Budget:** ≈ 2 days.

**Binding documents, read first, in this order:**
1. `.claude/advisor-head-c3-charter.md` **IN FULL** — **§2** (claim architecture: the primary claim is
   tier iii at matched params **AND** matched state-bytes vs the TTT-class system swap), **§4**
   (compute doctrine), **§5** (invariants — every one of them binds this harness).
2. `.claude/AGENT_PROTOCOL.md` **IN FULL** — §3 git discipline, §4 environment, §5 output + the
   flag-provenance rule, §7 dial declaration.
3. `.claude/outputs/pilot-ttt-nan-and-d5-wiring.md` **IN FULL** — this is your direct ancestor. Its
   §0 (the `load_journal` fingerprint repair) is *the* enabler of your resume-first ladders; its §7
   reconciliation list names three Hub rulings that land in the run-3 config, not yours to re-decide.
4. `.claude/outputs/cluformer-pilot.md` + `.claude/outputs/csf3-runbook.md` — the existing pilot and
   the cluster conventions you are extending, not replacing.

---

## ⛔⛔ MECHANICAL PRECONDITION — VERIFY ON DISK BEFORE ANY CODE, ANY WORKTREE

> Run **exactly this**, from `/Users/user/Desktop/CHLU`:
>
> ```
> git cat-file -e main:tests/test_ttt_stability_and_d5_wiring.py && echo PRECONDITION-MET
> ```
>
> **`PRECONDITION-MET` must print.** That file is added by `pilot-ttt-nan-and-d5-wiring` and was
> **verified ABSENT from `main @ c8314a8` on 2026-08-12** — so its presence in `main`'s tree *is* the
> merge, mechanically, not a promise. Then confirm the merge commit itself:
>
> ```
> git log --oneline main | head -5        # a merge commit naming pilot-ttt must be at/near the tip
> git log --oneline main..agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring   # must be EMPTY
> ```

If `PRECONDITION-MET` does not print, **or** the second command lists commits: **report `BLOCKED`,
write nothing, take no worktree, run zero cells, make zero git footprint**, name the failing check,
hand back to the Hub. ⛔ **Do not judge whether it matters, do not work around it, do not branch off
the pilot branch instead.** The pilot merge carries the `load_journal` repair without which every
banked CSF3 journal is refused and every resume ladder you build is dead on arrival.

**Non-blocking input (do NOT wait on it):** the **`d_addr` ceiling probe** (a C2W11-Hub debt, C2
charter §A43.2 / Add.16 §A45), at **`.claude/outputs/c2w11/DADDR-CEILING-PROBE.json`**. It is being
run **first or in parallel with the merge** (Advisor ruling 2026-08-12), so it is likely present by
the time you start. **If the file exists:** record the number in your report and expose it as the
config default. **If it does not:** expose `d_addr` as a plain config flag, **change no default**,
and list the probe as an owed input. ⛔ Do not invent a value, do not read a number out of a report
or a charter in place of the artifact, and do not block — it is an *in-block store* input and
nothing in this task's scope depends on it.

---

## ⚠⚠ FOLDED IN FROM `c3-benchmark-scout` (landed 2026-08-13 — read `.claude/outputs/c3-benchmark-scout.md` §1.3, §1.5)

The scout finished **before** you spawn, and four of its findings change this task. They are folded
into the sections below; this block is the index so none is missed.

### ⚠⚠ TRAP 1 — THE BYTE-LEVEL REVISIT UNIT. It will silently turn your slice into a frequency count.
At **vocab 256**, "distance to the last occurrence of the same symbol" is **a few bytes** for common
characters, and the retention bucket **degenerates**. The scout names this *"the single most likely
silent failure in `text_slices.py`"* and it is correct. ⛔ **Define the revisit unit at the
whitespace-delimited token / n-gram level even though the stream is bytes** (e.g. distance to the
previous occurrence of the *enclosing* token), and **assert the non-degeneracy** — a slice whose
distance distribution collapses to single digits for the commonest units is measuring character
frequency, not retention, and must fail the test rather than ship.

### ⚠⚠ TRAP 2 — THE RIVAL STATE-BYTE CONFIG. Inheriting a library default silently voids the byte match.
The `flash-linear-attention` defaults for GDN/GDN-2 (`head_dim=256, num_heads=6, expand_v=2`) give
**3× the state the GDN-2 paper's own numbers imply**. ⛔ **Pin every rival's state-bearing config
explicitly in our config, from the paper where the paper states it; never inherit a library default
and then claim byte-matching.** Same discipline for Mamba-2 (`d_state=128, d_conv=4, expand=2,
headdim=64`) — the scout derived ours from the official implementation because the paper's appendix
was NOT OBTAINED, so the provenance of each pinned number goes in your flag-provenance table.

### ⭐ ADOPT, DO NOT INVENT — the retention slice has a published convention (folded into §2)
### ⛔ THE MATCHED-STATE-BYTE BUDGET IS NOT YOURS TO CHOOSE (folded into §5)

---

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result

- **Dial / pillar:** **none — instrument/harness.** ⛔ **You produce NO claim, NO tier-iii number, NO
  bpc a paper could quote, and no verdict about the CLU vs anything.** You build the rig the claim
  will later be measured on, and you demonstrate it runs.
- **Laundering control:** not yours to *pass*, but yours to **make impossible to omit** — the harness
  must emit the **dynamic-evaluation substitute column** and the **per-arm byte ledger (including φ)**
  as *structural outputs of every run*, so that an arm cannot be reported without them.
- **Falsifies the task:** the smoke config not running end-to-end; a resume across a config-field
  addition being refused; a byte ledger that cannot account for φ; an eval slice that changes value
  when only the *content* (not the distance structure) is permuted.
- **Does NOT falsify the task:** the smoke numbers being bad, or the TTT arm beating the CLU on the
  smoke config. ⛔ **The smoke config is NEVER a claim venue** (charter §2) — a number off it is not
  evidence for or against anything, in either direction.

---

## 1. Track-A data plumbing

**Already built — do not rebuild:** `chlu/data/enwik8.py` (byte-level, vocab 256, canonical 90/5/5
**positional** split, concurrency-safe download-once staging) and `chlu/data/wikitext.py` (same
surface, byte + word modes; word-level vocab built from **train only** — that is a leak guard, keep it).

**Your work here is the seam, not the loaders:**

1. **A loader registry** so a stream is a **config value**, not a code path — `enwik8` | `wikitext103`
   selectable from the experiment config, both returning the identical split/iterator surface the
   trainer already consumes. ⭐ **The registry must be CORPUS-GENERIC, not a two-way switch** — a
   third stream must be *one loader module plus one config value*, with no change to the trainer, the
   iterators, the slices or the ledger. This is load-bearing and now has a named consumer: a
   **FineWeb-Edu appendix arm** is held as a **priced option** (Advisor decision 5, 2026-08-13 — it
   pre-empts the SSM-reviewer's *"why only enwik8?"*, and is decided **at the paper stage, not
   now**). ⛔ **Build no FineWeb-Edu loader.** Your obligation is only that adding one later costs a
   module and a config line — state in your report what that addition would touch, so the option
   stays genuinely cheap rather than nominally cheap.
   ⚠ **PG-19 remains DECLARED OUT OF SCOPE for this task.** The scout
   returned **GO-WITH-CAVEAT** (§1.6: GO as an *internal long-horizon retention instrument*, ⛔
   **NO-GO as an external comparison venue** — nearest published numbers are ≥5× our params on a
   different tokenizer; compute is *not* the constraint at ≈0.85–9.9 h/epoch on 2×A100). Whether the
   follow-up cell is funded is the Head's, not yours. The registry must make PG-19 **a later config
   change plus one loader module** — the pattern `wikitext.py` was written to serve. **Leave the
   seam; build no PG-19 loader.** ⭐ **Record these three scout caveats in your report so the
   follow-up cell is cheap when it comes:** (i) 11 GB across **~28,752 individual files** is an
   inode/many-small-files hazard on CSF3 — consolidate once, serially, into a single memmap-able
   uint8 stream with a sha256 contract, exactly as `chlu/data/enwik8.py` documents; (ii) PG-19's
   metric is **word-level ppl normalised by the raw word count**, computed from the text and **not**
   the tokenizer — which is why a byte-level model can be scored in the venue's own currency with no
   tokenizer confound; (iii) the validation set is **50 books / 3.0 M words**, small enough that
   per-book variance is material — report per-book spread, never the bare mean.
2. **CSF3 staging discipline, enforced not documented:** staging is **serial, once, before** a sweep;
   array tasks share one cache and must hit it, never race it. If a job can currently reach a
   download path from inside an array task, close it (fail fast with a clear message telling the
   operator to stage first).
3. **A staging smoke** that proves the cached path is used and the split boundaries are byte-exact.

## 2. ⭐ The eval slices — within-document retention / revisit on REAL text (the genuine build)

**VERIFIED as of 2026-08-12: this does not exist.** `grep -rln revisit chlu/ tests/` returns only
`chlu/experiments/memory_gym.py`, `chlu/experiments/exp_bprime_rivals.py`, `tests/test_memory_gym.py`
— all **toy/gym-side**. You are building the real-text instrument, in a new module
**`chlu/eval/text_slices.py`**.

⭐ **ADOPT, DO NOT INVENT — the field has a convention and the scout found it.**
**Sun, Krishna, Mattarella-Micke, Iyyer (2021), *Do Long-Range Language Models Actually Use
Long-Range Context?*, EMNLP 2021, arXiv:2109.09115** defines exactly our instrument — target
positions bucketed by **distance to the last occurrence in the prefix**, plus a **"never appears in
the prefix"** bucket — and it is defined **on PG-19**. Take **their definition of the slice and their
bucket structure, and cite them**; ⛔ do not rename it, and ⛔ do not derive a fresh definition and
discover later that it is theirs with different words.
**Declare as OURS, explicitly, in one line each** (the field has no convention for these): the **bin
edges**; computing it on **enwik8/WT-103 bytes** rather than PG-19 subwords; computing it **for every
arm including dyn-eval**; and the **shuffled-position control** — for which cite the precedent,
**Khandelwal, He, Qi, Jurafsky (2018), ACL, aclanthology.org/P18-1027** (the perturbation-by-distance
protocol; effective context ≈200 tokens, word order matters only within ≈50). The borrowing and the
extension must be separable by a reader.

**Spec it before you build it, in your report's §2, then build to the spec:**

- **Revisit unit — ⚠ see TRAP 1 above; this is the decision that makes or breaks the instrument.**
  Not the raw byte: the **enclosing whitespace-delimited token / n-gram**, on a byte stream. State the
  unit, define the **distance bins**, and say why. Bins must span the model's context **and beyond
  it** — the point is the *long*-horizon slice — and must include Sun et al.'s **"never appears in the
  prefix"** bucket, which is where a memory either shows up or does not.
- **Retention:** per-bin **bpc conditioned on distance-since-last-occurrence**, so a memory that
  actually retains shows up as a bpc gap that *widens* with distance rather than a scalar.
- **Document boundaries are respected** — "within-document" is load-bearing; enwik8's XML structure
  and WT-103's article structure each need an explicit, tested boundary rule.
- ⚠ **Two required controls, or the instrument is not trustworthy:**
  (a) a **shuffled-position control** — permute positions while preserving content; the slice must
      move, or it is measuring content frequency, not distance;
  (b) the slice is computed **identically for every arm including the dyn-eval substitute column** —
      the comparison is worthless if only our arm gets it.
- **Emit** slices as a JSON artifact per run with per-bin counts alongside per-bin bpc — ⛔ a bin with
  too few samples is reported with its `n`, never silently averaged away.

⛔ **No PREREG is required** (this is an instrument, not a measured ratio/exponent/slope/law — protocol
§5). ⛔ **Do not report a slice value from either real stream as a finding.** Report the *smoke* slice
to demonstrate the instrument works, and stop.

## 3. Job ladders — seeds-as-jobs, resume-first, inside the envelope

**Envelope (hard): 2×A100, 4-day per job.** Credits are effectively unconstrained — ⛔ **that changes
SCHEDULING, never CONTROLS** (charter §4).

1. **Seeds-as-jobs**: multi-seed = N independent jobs, not one long job. `scripts/csf3/job_gpu_array_seeds.sh`
   is the existing pattern; extend/parallel it for the cluformer path rather than reinventing it.
   ≥ 3 seeds is the paper bar (charter §5) — the harness must make 3 the *easy* number to launch.
   ⭐ **Size the ladder for what the scout found: EVERY modern rival's 26–47 M cell is NOT PUBLISHED**
   (Mamba-2, GDN, GDN-2, TTT and Titans publish nothing on enwik8/WT-103/PG-19 at any size), so **we
   train all five arms ourselves**. That is affordable — **≈1.5 h at 35 % MFU to ≈18 h at 3 % MFU**
   for a 40 M byte-level arm on enwik8, against a 96 h limit — but it means the ladder's natural unit
   is **≈5 arms × ≥3 seeds = 15+ jobs**, not two. Make that a one-command launch, not a loop someone
   hand-edits.
   ⚠ **The dyn-eval substitute column must be RE-MEASURED BY US at 26–47 M.** The published anchor
   (**enwik8 0.99 → 0.94 bpc**, **WT-103 18.3 → 16.4 ppl**, Krause et al. 2019, arXiv:1904.08378) is
   at **277 M / ~257 M** — ⛔ placing 0.94 beside a 40 M number is a category error. Treat dyn-eval as
   **an arm in the ladder**, not a citation.
2. **Resume-first**: a ladder is a sequence of resumes off a journal + `.eqx` checkpoint, not a
   monolith. The pilot's `load_journal` repair is what makes this legal; your job is to make it
   **routine and verified**. Include the **`.eqx` precondition check before any re-resume** — that is
   ruling (2) of the pilot's reconciliation list and it belongs in this harness.
3. **Regression-test the resume path against the failure that motivated it**: add a field to a
   `StreamMemoryConfig`-shaped fixture, confirm a journal predating it **at the field default is
   accepted**, and **non-default is still refused**. That test is the guard on a deliberately
   loosened §A20.4 provenance check — it must not silently widen further.
4. **A launch must be one literal command line per job.** ⚠ zsh does not word-split: a loop building
   arguments in a variable will submit garbage. Verify logs exist after launching anything.

## 4. The smoke config (local, minutes)

An end-to-end config — real stream, tiny shapes, minutes on the laptop — exercising: load → train →
checkpoint → **resume** → eval → **slices** → **byte ledger**. ⚠ **JAX cold start here is ~20+ min
even for `--help`; budget it and do not mistake it for a hang.** ⛔ **Never a claim venue**, and the
config file itself should say so in a comment where an operator will read it.

## 5. Byte ledgers and matched-state-bytes (structural, not optional)

The primary claim is at **matched params AND matched state-bytes** against the TTT-class swap. So:
every run emits a **byte ledger artifact** — per arm, the inference-time state in bytes, **including
φ** — computed from the config, with the arithmetic in the artifact. If an arm cannot produce its
ledger, the run should **fail loudly**, not warn. This is the invariant the whole tier-iii control
rests on; a harness that lets an unledgered arm through has quietly broken the claim.

### ⛔⛔ THE BUDGET IS NOT YOURS TO CHOOSE — expose it, do not pick it

The scout's §1.5 established, with the arithmetic, that **"matched state bytes" is a DECISION, not a
derivation**: at a fixed ≈38 M params the natural inference state spans **1.60 MB (TTT-Linear) →
100.7 MB (sliding-window @ 4 k)** — a **63× range** — and whichever number is chosen advantages some
rivals and cripples others. It is **the wave's single biggest risk**: pick it wrong and the tier-iii
control silently decides the result before any physics runs.

⇒ ⭐ **RULED (Advisor + Head, 2026-08-13): the budget is PRE-REGISTERED at ≈2 MB**, on the scout's
rationale — the CLU store at d=12 (**1,966,080 B**) and TTT-Linear (**1,597,440 B**) land there
naturally, so **the two-sided system swap is byte-honest by construction**, and every other rival
(Mamba-2 3.29× · sliding-window 6.40× · TXL-at-3800 23.7×) is **shrunk to match rather than grown** —
the defensible direction of the control.

**How you implement a ruled budget:**
- It is **one named config constant** with the ruled value and the rationale in a comment, ⛔ not a
  literal scattered across call sites — the exact byte figure is being confirmed to the last digit
  (see below) and must be changeable in one edit.
- The harness **enforces** it: every arm is ledgered against the budget and a mismatch **fails
  loudly**, same posture as an unledgered arm. Make the *shrink-to-match* direction mechanically
  possible for the over-budget rivals — that is the whole point of the ruling.
- **Report occupancy, not just compliance:** each arm's ledger states its actual bytes **and** its
  fraction of the budget, so the table shows how much of the envelope each arm truly uses.
- ⚠ **Pending one-line confirmation, and it does not block you:** "≈2 MB" is not itself
  pre-registerable. The Hub has recommended **2 MiB = 2,097,152 B as the ceiling** (CLU 0.94×,
  TTT-Linear 0.76×) over the alternative of setting the budget equal to our own store's 1,966,080 B.
  **Build against the constant; if the confirmation has not landed when you need it, use 2,097,152 B
  and say so in your report.** ⛔ Do not invent a third value.
- **Reference configs to pin or replace, stated:** *attention class* 12 L, d_model 512, 8 heads × 64,
  d_ff 2048 ⇒ **37.88 M**; *recurrent class* (SSM / linear-attn / TTT) 24 L, d_model 512 ⇒ **37.75 M**.
  Both land in the 26–47 M class. Adopt them or state your own with the param arithmetic shown.
- ⚠ **TRAP 2 binds here** — pin each rival's state-bearing hyperparameters explicitly, with the
  provenance of each number (paper table vs official implementation) in your flag-provenance table.

## 6. Declared file ownership (≤ 3 worktrees rule; you hold wt1)

**Yours to edit:**
- **new** `chlu/eval/text_slices.py` · **new** tests under `tests/` (name them for this slug)
- **new** `scripts/csf3/job_gpu_c3_seeds.sh` (or the minimal extension of `job_gpu_array_seeds.sh`)
- `chlu/data/enwik8.py`, `chlu/data/wikitext.py` — **staging/registry seam only**, ⛔ never the split
  arithmetic or the vocab-from-train-only rule
- `chlu/data/__init__.py` (registry export) · `chlu/experiments/exp_cluformer_pilot.py` (config surface)
- `chlu/training/train_cluformer.py` — ⚠ **shared with the just-merged pilot work; minimal hunks only,
  and list the exact lines in your report**
- `scripts/csf3/job_gpu_cluformer.sh` — ⚠ the pilot just wired **D5** through it; ⛔ **do not disturb
  the D5 passthrough**, and assert it still works.

**⛔ NOT yours:** `chlu/core/blocks.py` (the TTT cell — the pilot's; if a slice or ledger needs a change
there, **STOP and report**) · `chlu/eval/rollout_diag.py` (Track B, banked) · anything under the toy
experiment surface · any C2-era artifact or registry.

## 7. Kill / stop conditions (built first, per the invariant)

- Precondition unmet → **BLOCKED**, zero footprint (above).
- A byte ledger that cannot account for φ → **STOP and report**; do not ship a partial ledger.
- The shuffled-position control failing to move the slice → the slice definition is wrong; **report
  it as a finding**, do not tune the definition until it passes.
- Needing to touch `chlu/core/blocks.py` → **STOP and report**.
- The full suite failing on your branch for a reason outside your ownership list → report the exact
  failures; ⛔ do not fix foreign breakage in your worktree.

## 8. Acceptance criterion (one line, mechanically checkable)

On branch `agent/experiment-engineer/c3-csf3-harness`: the smoke config runs **load → train →
checkpoint → resume → eval → slices → byte ledger** end-to-end locally in minutes and its artifacts
are on disk; the stream is selectable by config between enwik8 and WT-103 with PG-19 left as a seam;
the resume regression test (field-added / default-accepted / non-default-refused) passes; a byte
ledger including φ is emitted for every arm and an unledgered arm fails loudly; ≥ 3 seeds is a single
documented launch inside the 2×A100/4-day envelope; **the full suite is green on your branch, with
the pass/fail counts reported against a HEAD you name and re-verify at the end** (a green against a
stale base is not a green).

## 9. Report format

Protocol §5, plus the **flag-provenance table** on every quantitative line (commit, seed(s), all
non-default flags), the **dial declaration** first, and your **git footprint** (branch, hashes, files
touched, exact shared-file line ranges). Before removing your worktree, verify from the main repo
that your branch ref shows your commits: `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/c3-csf3-harness`
(a worktree can be removed with the shared ref never having advanced — this lost 8 commits once).
If your report contains a downstream reconciliation list, say so **in the first 10 lines** (§5 corollary).
