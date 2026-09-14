# Task: gen-read-compose — the B path, through the READ: request → code → decode → probe

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-read-compose.md` + artefacts under `.claude/outputs/gen-read-compose/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R26 ruling 3 is the brief; R20, R23, R25 give the constraints** · `.claude/outputs/gen-dream-sampler.md` **§4 and the three-line summary** (why clamp-and-relax is retired) · `chlu/core/gen_dream.py` (`AddressBook`, `fit_address_book`, `address_requests`, `requests_from_labels`) · `chlu/core/gen_store.py` (`read_from_codes`, `address_codes`).
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 3faef27` · **Branch:** `agent/experiment-engineer/gen-read-compose` · **Worktree:** fresh (`../CHLU-gen-read`).
- **Env:** main venv, ⛔ no `uv sync`. ⛔ CSF3 unreachable — local CPU only, emit the submit line.

## ⭐ DIAL DECLARATION
- **Dial:** *isolation* — whether composing a request through the store's READ produces the requested combination.
- **Laundering control:** ⭐ **store-removed** (the read path with the store ablated) **and** a **no-store prototype
  composition** on the same requests. Both are full arms on every number.
- **Falsifies:** the read path does not beat store-removed on `shape` beyond seed spread ⇒ composition is not the
  store's. ⭐ A legitimate outcome and it returns to Head + Advisor, not to tuning.
- **Does NOT falsify:** absolute numbers below the decoder ceiling. Report against **both** ceilings (R23.5), never
  absolutely.

## The one job
Build the cell's composition path — **`request → read → code → frozen decoder → probe`** — and report A and B
accuracy on it.

## ⭐ Why this replaces what was there (R26 ruling 3)
Conditional dreaming is **dropped as the composition mechanism.** The falsifier fired and the coupling-off control
made it unambiguous: **the chain samples a mode; the clamp does not steer** (steering-attributable
`+0.0015 ± 0.0204`). And the request already specifies **4.42 of 5 wells**, so there is nothing to relax.
⭐ **Composition is the k-particle launch read over the requested modes, then decode** — the validated mechanism
(0.86 exact-set; +0.18 `shape` read margin), and the one originally named.
- ⛔ **Do not use `DreamLandscape`, `dream_rollout`, `clamp_from_posterior`, or the MALA/ULD steps.** They stay in the
  tree for within-mode diversity sampling only, off this path.
- ⛔ **Do not fit or deploy the code-coupling field.** It exists only for the retired landscape. Its **163,840 B leave
  the deployed store**, so the store is **524,288 B** again — see the byte ledger item below.
- **Partial-request completion is a different capability, banked as secondary.** ⛔ Not built, not scored here.

## What to build
1. **Request → codes.** Reuse `requests_from_labels` and `address_requests` (both shipped). ⛔ Do not reimplement the
   addressing head — it is `gen-store-grouping`'s mechanism and the thing under test.
2. **Codes → read vector** via `read_from_codes`, then **the ONE shared frozen decoder** — ⭐ **recipe
   `v1_capacity_schedule`, and assert the decoder parameter hash against `gen-decoder-lift`'s winner.**
   ⛔ `dec_recipe` still defaults to `"baseline"`, the decoder that FAILED the gate. A run through it is void and looks
   normal in the logs.
3. **Probe the decoded images.** ⭐ **`shape` reported separately from the aggregate, always** — four of six factors sit
   at ≈1.0 and carry no compositional information.
4. **Arms:** the read path · **store-removed** · **no-store prototype composition**. Same requests, same decoder, same
   probe, **≥3 seeds with spread**.
5. ⭐ **The byte ledger.** Report the deployed store at **524,288 B** with the coupling field explicitly excluded and
   the reason stated. ⛔ Do not silently inherit `688,128` from `gen-baselines`' amendment A5 — that target was correct
   only while dreaming was on the critical path.

## Metrics (R23, binding)
**Balanced accuracy is the gate statistic; AUROC is a mandatory companion and never the gate.** Shared-corner cells
(`montero`, `wide-000`) are a **two-class corner, never pooled**. Report against the decoder ceiling **and** the store
ceiling.
⚠ **Every comparison metric you use gets a pre-registered invariance check first** (R26 standing rule, third strike):
same features, different partition, same value. ⛔ If a metric cannot pass it, do not use it.

## Files you own
`chlu/core/gen_compose.py` (new) · `chlu/experiments/exp_gen_compose.py` (new) · `tests/test_gen_compose.py` (new) ·
`chlu/config.py` (**a new dataclass appended at the END of the file**, plus the three house sites) ·
`chlu/cli/experiment_cmd.py` (one appended subparser).
⛔ Nothing under `chlu/eval/rivals/`, `chlu/data/`, `scripts/csf3/`; ⛔ do not modify `gen_dream.py` or `gen_store.py`.

## Acceptance
1. The three arms run end to end, ≥3 seeds, A and B, `shape` separate from aggregate, both ceilings reported.
2. Decoder-hash assertion passes; recipe `v1_capacity_schedule` confirmed in the manifest.
3. Byte ledger at 524,288 B with the coupling exclusion stated.
4. Pre-registration filed before any reported number; outcome table including falsified predictions.
5. ⛔ **No kill-rule verdict** — that is `gen-cell-run`'s, against its own prereg.
6. Suite green (before/after, main venv); ref verified from the main repo before worktree removal.

## What would prove this spoke wrong
- ⭐ **The read path does not beat store-removed on `shape`.** Check that before anything else; it is the whole claim.
- You decoded through the baseline recipe, or inherited the 688,128 B target.
- You improved the aggregate and not `shape`.

---

# ⟲ AMENDMENT 2026-09-13 — R27: a fourth arm, `CLU-wide`, pre-registered before you run anything

## Why it exists, and why it is NOT a rescue
A byte-matched naive baseline — `knn_proto`, a per-(factor,value) **mean** composed additively — reaches **0.8884 ±
0.0088** in-corner `shape` balanced accuracy: **+0.20 above our 0.6866 store ceiling** and 94.4 % of the decoder
ceiling. It has an **unbounded read channel** (the whole 2048-float vector) against our **30 bits**. The
channel-matched kNN sits at **0.5206, near chance**.
⛔ **Kill rule (ii) still reads the byte-matched arm, as pre-registered.** ⭐ `CLU-wide` is the honest like-for-like
that sits **beside** it — ⛔ it does not replace what (ii) reads, and it is **not kinder**: *if composed modes are
just prototype means with machinery attached, this arm will tie `knn_proto` and we will say so.*

## ⛔⛔ THE TRAP — the easiest way to void the structural claim while believing you followed the ruling
`chlu/core/gen_store.py::soft_read(store, X, tau)` computes `sum_m pi_m · mode_m` — **exactly the composed per-well
mode vector this arm needs** — **but it takes `X`, the FEATURES,** and derives `pi` from them. ⛔ **That is an
encoder → decoder path. Using it as-is voids the no-bypass guarantee**, which is this programme's single strongest
result (`read_from_codes` takes *no* features, by design, and `assert_no_bypass` checks the signature).
⭐ **`CLU-wide` derives its mixture weights from the REQUEST** — `address_requests`' per-well mode log-scores,
softmaxed — **never from features.**
- **Assert the arm touches no feature tensor**, in the same spirit as `assert_no_bypass`, and report the assertion.
- Report the arm's realised **channel width** next to `knn_proto`'s, so the like-for-like is auditable rather than asserted.

## The four arms, then
`read` (30-bit, `read_from_codes`) · **`CLU-wide`** (composed mode vector from the request) · `store-removed` ·
`no-store prototype composition`. Same requests, same decoder, same probe, **≥3 seeds with spread**.
⭐ **Report `knn_proto`'s 0.8884 and the store ceiling 0.6866 as fixed reference rows on every table** so a reader
sees immediately where each arm sits.

## C reporting (R27.4), if you touch C at all
⛔ **C results are ALWAYS reported BY KIND** (`duplicate_factor`, `out_of_range`); pooled is a companion only.
Neither rival covers both kinds and ours does — **that differentiator is invisible pooled.**
