# c3-rival-arms — PHASE 2: the three tuned rivals (Mamba-2 · GDN-2 · sliding-window attention)

**Campaign 3, PHASE 2. Agent:** experiment-engineer. ⭐⭐ **THIS FILE IS SPAWNED THREE TIMES — ONCE PER RIVAL.** Each spawn executes **the shared contract (§0–§2) plus exactly ONE venue section (§A, §B or §C)**, on its **own branch and own worktree**. The Head names the rival in the spawn line.
Branch **`agent/experiment-engineer/c3-rival-<mamba2|gdn2|swa>`** off the C3 head (the Hub names the base at spawn).
Writes `.claude/outputs/c3-rival-<name>.md`. **Budget:** ≈ 1.5 days each.

> **Why phase 2 exists.** The scout established that **every modern rival's 26–47 M cell is NOT PUBLISHED** — so every rival row in the tier-iii table is **a from-scratch run of ours**. Phase 1 (CLU + TTT swap + dyn-eval) carries the **two-sided control** but **not** the competitive-baseline half, and the genuine-win bar requires the primary claim to beat competitive baselines. ⇒ **Charter §2's tier-iii primary claim waits for you.** ⛔ No phase-1 result may be quoted as that claim.

**Binding documents:** `.claude/outputs/c3-benchmark-scout.md` **§1.1, §1.5 IN FULL** (the pinned configs and the derived state-byte arithmetic) · `.claude/outputs/c3-csf3-harness.md` **§5.4** (`RIVAL_SPECS`, the provenance test, `shrink_to_budget()`) · `.claude/outputs/c3-rival-ladder-prereg/PREREG-C3-LADDER.md` (as amended by `c3-gb-landing`) · `.claude/AGENT_PROTOCOL.md`.

---

## §0 — DIAL DECLARATION (protocol §7) — echo before your first result

- **Dial:** **none — rival implementation.** ⛔ You produce **no CLU number and no comparison.** You build a rival arm and demonstrate it trains and ledgers. **The comparison is the ladder's, not yours.**
- **Laundering control:** ⭐ **the anti-hobbling rule is your governing discipline.** Your job is to give the rival **its strongest admissible form**. A rival that loses because we implemented it badly is worthless as a control — and this program has already had **one C2W10 verdict inverted** by exactly that.
- **Falsifies your task:** an arm whose state bytes disagree with `RIVAL_SPECS`; a config inherited from a library default; an arm that cannot enter the ladder through the byte ledger.
- **Does NOT falsify:** your rival **beating the CLU**. ⭐ **That is the control doing its job** — report it plainly and without softening. ⛔ Never tune your rival down.

## §1 — The shared contract (all three spawns)

1. ⭐ **Official code is the basis.** Port from, or vendor, the authors' own implementation. ⛔ Do not reimplement from the paper's equations and hope — the C2W10 SAM-kNN port is the standard to match: **the authors' reference implementation, at published defaults, with unit-tested equivalence shims** where a C extension had to be replaced.
2. ⛔⛔ **THE `flash-linear-attention` 3× TRAP IS BINDING** (scout §1.5). FLA's defaults (`head_dim=256, num_heads=6, expand_v=2`) give **3× the state the GDN-2 paper's own numbers imply**. ⛔ **Never inherit a library default and then claim byte-matching.** Every state-bearing hyperparameter is **pinned in our config** with **per-number provenance** — `PAPER:` (with table) or `OFFICIAL IMPLEMENTATION:` (with file/symbol). A test already asserts that prefix; keep it passing.
3. ⭐ **The arm enters the ladder ONLY through the byte ledger.** It must produce a ledger row — state bytes computed from config, **`dtype_bytes` declared**, φ accounted — and it must **reproduce `RIVAL_SPECS`' pinned number to the byte** (there is already a test asserting the six rivals reproduce the scout's table; extend it, do not weaken it). An arm that cannot be ledgered **fails loudly**; it does not warn.
4. ⭐ **Shrink-to-match, never grow.** Your arm's natural state exceeds the ceiling; `shrink_to_budget()` already solves the declared knob **down**. Use its solved value, state it, and ⛔ **do not re-solve by hand**.
5. ⛔ **Convention: TOTAL state bytes, AS DEPLOYED, no dtype normalisation** (Head+Advisor). If your rival deploys at bf16 and our store at fp32, **that asymmetry is real and stays** — harder for us is the defensible direction.
6. **A smoke leg**, real stream, minutes: the arm trains, checkpoints, resumes, ledgers, and emits slices. ⛔ **The smoke config is never a claim venue** and no bpc from it is quotable.
7. ⛔ **Train no ladder arm.** The interim-budget guard blocks them and that is deliberate.

## §2 — Ownership (disjoint by construction; three of you run in parallel)

**Yours:** a new module for **your rival only** (under the rivals surface the harness established) · its registry/config entry · its tests. ⛔ **NOT yours:** the other two rivals · `chlu/core/blocks.py` and `chlu/eval/byte_ledger.py` (**`c3-gb-landing` owns both** — if you need a change there, **STOP and report**) · the CLU arm · the corpora registry · `PREREG-C3-LADDER.md`.
⚠ **Declare your file list in your report and diff your branch against its base before you finish** — three concurrent engineers is the configuration that has collided before.

---

## §A — Mamba-2  *(spawn 1)*

Pinned: `d_state 128, d_conv 4, expand 2, headdim 64, ngroups 1`, 24 L, d_model 512 ⇒ **6,475,776 B bf16** = **3.09×** ⇒ shrink knob **`d_state 128 → 39`**.
⚠ **Provenance caution, on the record:** the scout took these from the **official implementation** (`state-spaces/mamba`, `mamba_ssm/modules/mamba2.py`, `allocate_inference_cache`) because **the paper's per-size appendix was NOT OBTAINED** (PDF would not parse, ar5iv returned front matter only). ⇒ your provenance strings say `OFFICIAL IMPLEMENTATION:`, **not** `PAPER:`. ⭐ If you *can* reach the appendix table, reconcile and report any disagreement as a finding.
🔍 Known, pre-existing: `rival_reference_table()` raises below ≈197 kB because **mamba2 cannot shrink that far** — harmless at every real budget; do not "fix" it into a silent clamp.

## §B — Gated DeltaNet-2  *(spawn 2)*

**arXiv:2605.22791** (Hatamizadeh, Choi, Kautz, NVIDIA, 2026-05-21; code `NVlabs/GatedDeltaNet-2`). ⛔ **Not** GDN v1 (arXiv:2412.06464) — both exist and the ids are a known confusion; carry the **2605** id everywhere.
Pinned from the paper's own statement (H=16, d_k=d_v=128, d_model=2048 ⇒ **262,144 floats/layer** = `d_model²/H` exactly), scaled to d_model 512 at **H=4** keeping d_k=d_v=128, 24 L ⇒ **3,145,728 B bf16** = **1.50×** ⇒ shrink knob **`n_heads 4 → 6`**.
⚠⚠ **You are the venue of the FLA trap** (§1.2) — FLA's defaults give **3×** this. Pin explicitly.
⚠ **Disclosed coincidence, do not "correct" it:** GDN-2's *shrunk* value lands on **2,097,152 exactly** (`24·512²/6·2 B`, powers of two). It equals the ceiling by arithmetic accident. The prereg records it; note it in your ledger row so no one later reads it as a fitted number.

## §C — Sliding-window attention  *(spawn 3)*

Pinned: **w=512, 12 L**, caching **K and V** (hence 2×, unlike Transformer-XL which caches hidden states at 1×) ⇒ **12,582,912 B bf16** = **6.00×** ⇒ shrink knob **`window 512 → 85`**.
⚠ **The in-class published anchor is Longformer-small, 41 M → 1.00 bpc** (arXiv:2004.05150, Table 2) — ⛔ **but it is NOT protocol-comparable**: staged training over 5 phases, seq-len 2,048 → **23,040**, per-layer windows 32→512, dilation on 2 heads in layers 6–11, **evaluated at 32,256 tokens**. ⭐ Quote it as a **sanity anchor that our implementation is not broken**, ⛔ **never** as a matched baseline. Say which it is, in your report, in one line.

---

## §3 — Acceptance (per spawn, one line)

Your rival trains a smoke leg end-to-end on the real stream and emits a byte-ledger row that **reproduces `RIVAL_SPECS`' pinned value to the byte**; every state-bearing hyperparameter is pinned with a `PAPER:`/`OFFICIAL IMPLEMENTATION:` provenance string and no library default is inherited; `shrink_to_budget()`'s solved knob is used and stated; ⛔ **zero ladder arms trained**; file list declared and branch diffed against base; full suite green with counts against a **named, re-verified HEAD**; branch ref verified from the main repo before the worktree is removed.

**Report:** protocol §5 + flag-provenance + dial declaration + git footprint + a one-line statement of **which anchor (if any) your arm can be sanity-checked against, and why it is not a matched baseline.**
