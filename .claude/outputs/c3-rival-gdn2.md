# c3-rival-gdn2 — experiment-engineer report

**Task + acceptance criterion:** build the **Gated DeltaNet-2** (arXiv:**2605.22791**) rival arm in its
strongest admissible form, from the authors' own artefacts, with every state-bearing hyperparameter pinned
and provenanced; it trains a smoke leg end-to-end on the real stream and emits a byte-ledger row that
reproduces `RIVAL_SPECS`' pinned value **to the byte**, using `shrink_to_budget()`'s solved knob; ⛔ zero
ladder arms trained.

**Status: done**, with **four STOP-and-report findings** that are not mine to resolve (§2 ownership).

> ⛳ **RECONCILIATION LIST — needs an owner (protocol §5 corollary, in the first 10 lines).**
> 1. ⛔ **`RIVAL_SPECS`' GDN-2 row omits real deployed state**: the official layer caches short-convolution
>    state (`+220,320 B` at the shrunk config). Under task §1.5's *TOTAL-as-deployed* convention the arm is
>    **1.0973× over** the 2 MiB ceiling; under the paper's own *"main recurrent state"* convention it is
>    **0.9922× under**. Both are ledgered; **choosing is `c3-gb-landing`'s** (§F1).
> 2. ⛔ **`shrink_to_budget`'s solved `n_heads = 6` has NO integer head geometry** (`512/6 = 85.33…`). The
>    pre-registered `2,097,152` is not realizable; the deployed number is **2,080,800 B**. Also: in the
>    **official** parameterization `head_dim` is *independent* of `d_model`, so `n_heads` is a shrink knob
>    **only** under `RIVAL_SPECS`' `d_k = d_model/H` tie (§F2).
> 3. ⛔ **The 24-layer pin and the 26–47 M class are jointly infeasible** with the paper's own block:
>    **95.4 M** with the MLP vs **44.5 M** MLP-free. The scout's "37.75 M at 24 L" is the MLP-free reading
>    (§F3). Moves **both** axes of the primary claim.
> 4. ⚠ **The FLA trap is narrower than recorded**: the 3× is `flash-linear-attention`'s **`GatedDeltaNet`
>    (v1)** layer; **NVlabs' own GDN-2 layer defaults reproduce the paper exactly** (§F4).
>
> ⏱ **TIME-SENSITIVE:** items 1 and 2 are owned by **`c3-gb-landing`**, which is **running right now** — I
> observed its `pytest` in `/Users/user/Desktop/CHLU-wt1` collecting a new `tests/test_c3_gb_geometry.py`
> while my suite ran. If these two do not reach it before it finishes, they become a second wave's edit to a
> file that has just been ruled on.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)

- **Dial: none — rival implementation.** ⛔ **No CLU number and no comparison appears anywhere in this
  report.** I built a rival arm and demonstrated it trains and ledgers. **The comparison is the ladder's.**
- **Laundering control:** the **anti-hobbling rule** is the governing discipline. The port keeps every
  component of the authors' block (short convolutions, channel-wise fp32 decay, L2-normalised q/k, the
  SiLU-gated RMSNorm output, the official initialisation and the negative-eigenvalue option) and refuses
  what it does not implement (grouped value attention) **loudly**.
- **Falsifies:** state bytes disagreeing with `RIVAL_SPECS`; a config inherited from a library default; an
  arm that cannot be ledgered. **None fired** — §1, §2.
- **Does NOT falsify:** GDN-2 beating the CLU. ⛔ Nothing here is tuned down; nothing here is compared.
- **Pre-registration: FILED** — `.claude/outputs/c3-rival-gdn2/PREREG-C3-RIVAL-GDN2.md`, written **before**
  the arithmetic was run. Outcomes in §3.

---

## 0. Sources — obtained THIS session, primary, none second-hand

| source | how | what it fixed |
|---|---|---|
| **PAPER** `arxiv.org/html/2605.22791v1` (665,102 B, CC BY 4.0) | `curl` | Eqs. 8–12 verbatim; §3.5 block design; App. C.1 layer parameterization; App. D.2 (q/k L2), D.5 (Xavier gain 2⁻²·⁵, RMSNorm+SiLU output gate); **§E.1 Eq. 90** the state statement |
| **OFFICIAL IMPLEMENTATION** `NVlabs/GatedDeltaNet-2` (`git clone --depth 1`) | `git` | `lit_gpt/gdn2.py` (`GatedDeltaNet2`), `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py` (**the reference recurrence**), `lit_gpt/model.py` (`Block`, `LLaMAMLP`, `_init_weights`), `lit_gpt/config.py` (`gdn2_1.3B`) |
| **TRAP CONTROL** `fla-org/flash-linear-attention` `fla/layers/gated_deltanet.py` | `curl` | the defaults task §1.2 forbids inheriting |

⚠ **NOT-OBTAINED: none.** Unlike Mamba-2 (whose appendix the scout could not parse), both the GDN-2 paper
and its code parsed, so this arm's provenance is **primary on both sides**.

⭐ **Eq. 90, verbatim** (the sentence the whole ledger row rests on): *"For fair recurrent comparisons, we
match both parameter count and **main recurrent state size**. Gated DeltaNet, KDA, and Gated DeltaNet-2 use
H=16 heads with d_k=128 and d_v=128, giving a per-layer recurrent state of H d_k d_v = 16·128·128 = 262,144
floats per batch element. Since d_model=2048, this equals 128 d_model."* — note **"main"**: the paper's own
matched quantity explicitly excludes the convolution cache. That is finding §F1.

---

## 1. What I built

| file | lines | what |
|---|---|---|
| **new** `chlu/eval/rivals/gdn2_lm.py` | 878 | the arm: `GDN2Config` (pinned) · `GDN2_PROVENANCE` (per-number) · `GDN2Layer` / `GDN2Block` / `GDN2LM` · the ledger (`gdn2_ledger_row`, `assert_reproduces_rival_specs`) · `gdn2_published_config` / `gdn2_shrunk_config` · `fla_trap_check` · `gdn2_param_class_table` · the `GDN2_ARM` registry record |
| **new** `chlu/experiments/exp_c3_rival_gdn2.py` | 298 | the smoke leg: load → train → checkpoint → **resume** → eval → slices → ledger; `ladder_ledger()` |
| **new** `scripts/smoke_c3_gdn2.sh` | 59 | the three-leg smoke + the bit-identity assertion |
| **new** `tests/test_c3_rival_gdn2.py` | 443 | 19 cases |
| `chlu/eval/rivals/__init__.py` | **+6** | ⚠ the only shared file touched — one comment block, one import, one `__all__` line. **No existing line modified or deleted.** |

**Fidelity, stated so a reviewer can audit the boundary.** Faithful: the Eq. 10 recurrence (ported from the
Triton kernel *statement for statement*, including its `1e-6` epsilon **inside** the `sqrt` and its
`d_k**-0.5` query scale); the three short convolutions with SiLU; `g = −exp(A_log)⊙softplus(f_proj x + δ)`
computed in fp32 with `A_log` per key head and `δ` per key channel; channel-wise `b` (key axis) and `w`
(value axis) via independent sigmoid projections; §3.1's negative-eigenvalue option scaling **only** the
erase gate; the RMSNorm + SiLU output gate and `o_proj`; pre-norm non-parallel residual + SwiGLU MLP;
Xavier-uniform gain 2⁻²·⁵, `A_log ~ log U(1,16)`, the `inv_dt` bias, `mamba_init` embedding std 0.02 and the
`1/√(2·n_layer)` residual rescale on the SwiGLU output matrix. **Shim:** `jax.lax.scan` instead of the
chunkwise WY kernel (§3.3) — a *throughput* device computing the same function, and the equivalence is
tested against **both** the paper's boxed Eq. 10 and a transcription of the kernel. **Refused, not faked:**
grouped value attention raises `NotImplementedError`.

---

## 2. The byte ledger — the only door into the ladder

Artifact: `.claude/outputs/c3-rival-gdn2/gdn2-ledger.json`.

| row | H | d_k=d_v | **main recurrent state** | `RIVAL_SPECS` | **Δ** | conv state | **total as deployed** | occ (main) | occ (total) | params |
|---|---|---|---|---|---|---|---|---|---|---|
| **published pin** | 4 | 128 | **3,145,728 B** | **3,145,728 B** | **0** | 221,184 | 3,366,912 | 1.500000× | 1.605469× | 95,374,944 |
| **shrink-to-match** (solved) | 6 | **85** | **2,080,800 B** | ideal 2,097,152 | **−16,352** | 220,320 | 2,301,120 | **0.992203×** | 1.097266× | 93,104,136 |

- **Reproduces `RIVAL_SPECS` to the byte at the pin: Δ = 0.** `24·4·128·128·2 B = 3,145,728 B`, and the
  paper's own `d_k = d_v = 128` survives the scaling to `d_model 512` **exactly**.
- **The shrink knob is `shrink_to_budget()`'s, not mine**: `n_heads 4 → 6`, `state_bytes_shrunk = 2,097,152`
  — the **disclosed arithmetic coincidence** (`24·512²/6·2`, powers of two), re-asserted, never re-derived.
- `dtype_bytes = 2` (**bf16 as deployed**, no normalisation — our own store's fp32 asymmetry stays).
- **φ = 0 and it is asserted, not inferred**: this is a standalone rival LM; it does not use the shared
  `StreamPhi` read-in, so there is no shell to hide budget in. `phi_accounted: true`.
- An arm that cannot reproduce the table raises `UnledgeredGDN2Error` — it **fails**, it does not warn.

---

## 3. Pre-registration outcomes (all seven, none quietly dropped)

| # | predicted | measured | verdict |
|---|---|---|---|
| **P1** | main state = `RIVAL_SPECS` = 3,145,728 B, Δ 0 | 3,145,728 B, Δ **0** | ✅ |
| **P2** | `n_heads 4 → 6`, ideal 2,097,152 | identical | ✅ |
| **P3** ⭐ | no integer geometry at H=6; `head_dim = 85`; **2,080,800 B**; −16,352 B; 0.992203× | identical | ✅ |
| **P4** ⭐ | conv state 220,320 B; total 2,301,120 B; **1.09727×** | 220,320 / 2,301,120 / **1.097266×** | ✅ |
| **P5** | FLA 786,432 = **3.000×**; official = **1.000×** | identical | ✅ |
| **P6** ⭐ | with-MLP 24 L = **95,374,944**; MLP-free = 44,478,048 | 95,374,944 ✅ ; **44,490,336** ✗ | ◐ **the MLP-free figure was wrong by 12,288 B (0.028 %)** — my prereg arithmetic omitted the per-layer pre-norm weight (`24 × 512`). Reported as a miss, not silently corrected. The conclusion (with-MLP outside the class, MLP-free inside) is unchanged. |
| **P7** | resume bit-identical; loss decreases | **0 of 47 leaves differ, max\|Δ\| = 0.0**; loss 5.6938 → 4.7684 nats | ✅ |

---

## 4. The smoke leg — how I verified (task §1.6), commands and observed output

`bash scripts/smoke_c3_gdn2.sh` on the **real** enwik8 stream (staged cache, `--no-download`),
`d_model 64, 2 layers, H=2, seq_len 256, batch 2, 6 steps, 0.6 MB, seed 0`. ⛔ **Never a claim venue** — the
banner says so in the module, the script and the artifact, and every bpc below is execution evidence only.

```
=== leg 1: fresh, 3 steps, banked ===
  step 1/3  loss 5.69376 nats … step 3/3  loss 5.03899 nats
=== leg 2: --resume to 6 steps + slices ===
[resume] lifted step 3 …
  step 4/6  loss 4.87895 … step 6/6  loss 4.76843 nats
[eval] valid 6.51072 bpc over 1,024 tokens ⛔ NOT A CLAIM
[slices] 8 bins scored ⛔ NOT A CLAIM
=== leg 3: uninterrupted 6 steps (the reference) ===
  step 1/6  5.69376 … step 6/6  4.76843 nats   (identical to legs 1+2, to 5 d.p.)
=== resume bit-identity ===
leaves 47 | differing 0
✅ resume is bit-identical
```

Every leg of the acceptance path executes: **load → train → checkpoint → RESUME → eval → slices → ledger**,
in **3.4 s**. The retention slice runs through the harness's **own** `build_revisit_index` /
`contiguous_target_positions` / `slice_bpc` / `run_controls`; only the per-token NLL is supplied by this arm
(`evaluate_slices` hard-wires the CLU's `plan_pass`, which a standalone rival has no analogue of).
⭐ The pilot's **own** `loss_fn` and `token_nll` run on this arm **unchanged** (test
`test_arm_is_drop_in_for_the_pilots_loss_and_per_token_nll`) — that is the ladder-readiness demonstration.

⛔ **ZERO ladder arms trained.** `ladder_ledger()` produces both rows from config arithmetic;
`test_the_ladder_rows_are_ARITHMETIC_and_build_no_model` monkeypatches `GDN2LM` **and** `build_gdn2_arm` to
raise and the call still succeeds.

**Other checks**

| check | result |
|---|---|
| `pytest tests/test_c3_rival_gdn2.py` | ✅ **19 passed** in 21.2 s |
| `ruff check chlu/eval/rivals/ chlu/experiments/exp_c3_rival_gdn2.py tests/test_c3_rival_gdn2.py` | ✅ All checks passed |
| `bash -n scripts/smoke_c3_gdn2.sh` | ✅ clean; executable bit set |
| recurrence vs **paper Eq. 10** (matrix form) and vs **kernel transcription** | ✅ `allclose(rtol=2e-5)` on both |
| measured params vs config arithmetic, 3 geometries | ✅ exact on all three |
| full suite | §6 |

---

## 5. Findings — four STOPs, all outside my ownership

### F1 ⛔ `RIVAL_SPECS`' GDN-2 row omits deployed state the official layer really caches

`lit_gpt/gdn2.py` ships `use_short_conv=True, conv_size=4, conv_bias=False` and `update_layer_cache` stores
`conv_state=(conv_state_q, conv_state_k, conv_state_v)` **beside** `recurrent_state`. That is
`(2·key_dim + value_dim)·(K−1)` elements per layer = **220,320 B** at the shrunk config (221,184 B at the pin).

| convention | shrunk-config bytes | occupancy of 2 MiB | verdict |
|---|---|---|---|
| paper §E.1 **"main recurrent state"** (= what `RIVAL_SPECS` implements) | 2,080,800 | 0.9922× | **fits** |
| task §1.5 **"TOTAL state bytes, AS DEPLOYED"** | **2,301,120** | **1.0973×** | **busts** |

⚠ **The two conventions give opposite verdicts and both have a good argument.** The paper's is the field's
matched quantity and is what the scout's whole §1.5 table is built on; the task's is the byte-honest one.
⭐ A relevant symmetry: **our own shell's intra-chunk causal conv is likewise un-ledgered** (`byte_ledger.phi_bytes`
returns state 0 and `StreamBlock.conv_w` never appears), so counting GDN-2's convolution while not counting
ours would be the *unfair* direction. I ledger all three columns and choose nothing.
⛔ **`chlu/eval/byte_ledger.py` is `c3-gb-landing`'s** — one line settles it.

### F2 ⛔ The solved shrink knob has no integer realisation, and it is a knob only under our tie

`RIVAL_SPECS`' formula `n_L · d_model² / H` **ties** `d_k = d_v = d_model/H`. The tie is exact at the pin
(`512/4 = 128` = the paper's own `d_k`) and **not** at the solved `H = 6` (`512/6 = 85.33…`). The floored
geometry `head_dim = 85` deploys **2,080,800 B**, i.e. **16,352 B (0.780 %) under** the ceiling — the shrink
direction is preserved and the pre-registered `2,097,152` is an *idealisation*, not a deployable number.

⚠ **The deeper point.** In the **official** parameterization `head_dim` is a **free** hyperparameter,
independent of `d_model`: `gdn2_1.3B` runs `n_embd = 2304` with the layer at its defaults `head_dim=128,
num_heads=16` ⇒ `key_dim = 2048 ≠ n_embd`. Under *that* parameterization, raising `n_heads` at fixed
`head_dim` **raises** the state (`6·128·128 = 98,304`/layer ⇒ 4,718,592 B, 2.25× the ceiling). So
`n_heads` is a *shrink* knob **only** under `RIVAL_SPECS`' tie. The tie is defensible (it holds at the pin
and at the paper's own 1.3 B setting) but it is **ours**, and the ledger should say so.

### F3 ⛔ The 24-layer pin and the 26–47 M weight class are jointly infeasible

| stack at `24 L, d_model 512, H=4` | params | in 26–47 M? |
|---|---|---|
| GDN-2 mixer **+ SwiGLU MLP** (the paper's own recurrent model, §3.5) | **95,374,944** | ❌ 2.03× the ceiling |
| GDN-2 mixer **only** (the scout's `≈6·d_model²/layer` reading) | **44,490,336** | ✅ |
| mixer + MLP at **12 L** | 47,818,800 | ❌ marginal — **and the state halves to 1,572,864 B**, so no shrink is needed at all and F2's solve changes |

The mixer alone is **7.028 · d_model²** (`q,k,v,b,w,o` = 6d², plus `f_proj`/`g_proj`/convs/`A_log`/`dt_bias`),
so the field's "≈6·d_model² per layer" convention counts the **mixer**, not the block — FLA's own docstring
says exactly that. ⇒ **The scout's 37.75 M reference is an MLP-free stack.** Whichever way the ladder goes,
it moves a claims-relevant axis: dropping the MLP is *our* modification of the rival's architecture; dropping
layers moves `n_layers`, which is **inside the state formula**. I ship `use_mlp` as a knob at the **faithful**
default and report the arithmetic. ⛔ Not mine to choose.

### F4 ⚠ The FLA trap is real but narrower than `RIVAL_SPECS`' provenance implies

Re-derived, not quoted: `fla/layers/gated_deltanet.py` defaults `hidden_size=2048, expand_v=2, head_dim=256,
num_heads=6` ⇒ `6·256·512 = 786,432` = **3.000×** the paper's 262,144 ✅ — the scout was exactly right.
⭐ **But that is FLA's `GatedDeltaNet` (v1) layer.** NVlabs' own **`GatedDeltaNet2`** defaults are
`head_dim=128, num_heads=16, expand_v=1` ⇒ `16·128·128 = 262,144` = **1.000×**, i.e. the official GDN-2 layer
**reproduces the paper exactly**. `RIVAL_SPECS`' provenance string says "the flash-linear-attention default"
without naming which layer; a one-clause edit would stop a future reader concluding the *authors'* code is
the trap. (Trap status unchanged: nothing is inherited from either.)

### The sanity anchor (task §B/§3, in one line)

⛔ **This arm has no protocol-comparable anchor.** arXiv:2605.22791's own headline (WikiText ppl **15.90**) is
zero-shot perplexity of a **1.3 B** model with a **subword** tokenizer after **100 B FineWeb-Edu** tokens at
**4 k** context — a *rival-vs-rival* number, not a venue number, differing on all four axes. The only usable
check that our implementation is not broken is the **in-class byte-level enwik8 band, ≈1.00–1.06 bpc at
39–41 M** (Longformer-small 1.00 / Adaptive-Span 1.02 / Mega 1.02 / TXL-12L 1.06) — ⭐ a **SANITY BAND, never
a matched baseline**: every one of those is quoted at an eval context larger than its train context. This is
carried in the code as `GDN2_ARM["sanity_anchor"]` and asserted by a test.

---

## 6. Full suite

```
PYTHONPATH=/Users/user/Desktop/CHLU-wt3 /Users/user/Desktop/CHLU/.venv/bin/python \
  -m pytest -q -p no:cacheprovider --no-cov
<<<SUITE_RESULT>>>
```

**Arithmetic checked, not assumed.** `--collect-only` in a clean detached worktree at **`main @ 0644c48`**
collects **1781**; my branch collects **1800 = 1781 + 19**, i.e. exactly my new test file and nothing else.

⚠ **Disclosed: the first full-suite run was KILLED and restarted, deliberately.** It was ~10 % in when I
landed `baf0166` (a six-line comment), which moved the tree under a run in progress. A comment-only change
could not plausibly have flipped a test, but *"a green against a moving HEAD is not a green"* does not have a
plausibility clause, so I killed it (`exit 144`) and re-ran from scratch at a **stable, clean** HEAD. The
result below is the second run: HEAD `baf0166` and `git status` clean **before and after**, `main` unmoved at
`0644c48` on both sides.

---

## 7. Flag-provenance table

Every number in this report comes from one of four rigs. **Commit `c7a51f7`** (branch tip), base local
**`main @ 0644c48`**, worktree `/Users/user/Desktop/CHLU-wt3`, **main venv reused, never `uv sync`d** —
**JAX 0.9.0**, equinox 0.13.4, optax 0.2.6, numpy 2.4.1, **CPU / float32**, macOS.

| | (A) byte ledger | (B) param arithmetic | (C) smoke run | (D) tests |
|---|---|---|---|---|
| seed | n/a (config arithmetic) | n/a | **0** | 0 / 1 / 3 / 4 (fixtures seeded) |
| corpus | — | — | **enwik8, REAL**, `n_bytes=600,000` ⇒ train 540,000 / valid 30,000 B | synthetic byte stream (40 kB, whitespace-delimited, `<page>` boundaries) |
| GDN-2 config | published `24 L, d_model 512, H=4, d_k=d_v=128, expand_v=1, conv 4, bf16`; shrunk `H=6, d_k=d_v=85` | same | `n_layers=2, d_model=64, n_heads=2, vocab 256` ⛔ toy | `n_layers∈{1,2}, d_model∈{6,8,16,24,30,32}` |
| non-default vs `GDN2Config()` | — | `use_mlp=False` in the MLP-free row only | `n_layers, d_model, n_heads` | as above + `use_mlp`, `use_short_conv`, `num_v_heads`, `allow_neg_eigval`, `head_dim` |
| budget | **2,097,152 B** (`MATCHED_STATE_BYTE_BUDGET`, unchanged) | — | same | same |
| optimiser | — | — | `optax.adam(lr=1e-3)`, batch 2, seq_len 256, 6 steps | `adam(1e-2)`, batch 2, seq_len 64, 4 steps |
| slice params | — | — | `edges=(1,8,32,128,512,2048,8192)`, unit `token`, `doc_boundary=b"<page>"`, `min_n=5`, 2 batches | same, `min_n=1` |
| `allow_neg_eigval` | **False** (official default) | False | **False** | False, and True in the one ablation test |
| `use_short_conv` | **True** (official default) | True | **True** | True, False in one test |

⚠ **Provenance of the externally-sourced constants:** the budget `2,097,152 B` is inherited from
`chlu.eval.byte_ledger` (Advisor+Head ruling 2026-08-13) and **not touched**; the shrink knob `n_heads = 6`
is `shrink_to_budget()`'s solve and **not re-solved**; every GDN-2 hyperparameter carries a `PAPER:` /
`OFFICIAL IMPLEMENTATION:` / `HARNESS LEDGER:` string in `GDN2_PROVENANCE` with a file/symbol or a section.

⚠ **On the third prefix.** Task §1.2 names two prefixes. `n_layers`, `d_model`, `vocab_size` and
`dtype_bytes` are **not the rival's numbers** — they are the weight class and the stream *we* chose (scout
§1.5, pinned into `RIVAL_SPECS`). Labelling them `PAPER:` would be a false citation, so they carry
`HARNESS LEDGER:` and a test asserts that **exactly** those four do and that every hyperparameter of the
rival itself carries one of the task's two. I flag the deviation rather than mislabel.

⛔ **No number in this report is a claim, and none is a comparison.** The smoke bpc values appear only as
execution evidence.

---

## 8. Git footprint

**Branch `agent/experiment-engineer/c3-rival-gdn2`**, off local **`main @ 0644c48`** (the C3 head — the
merge of `c3-csf3-harness`, which is where `RIVAL_SPECS`, `shrink_to_budget` and the slice instrument live),
in worktree `/Users/user/Desktop/CHLU-wt3`. ⚠ A worktree was **mandatory**: the shared checkout is on the
live `pilot-ttt-nan-and-d5-wiring` branch. Not pushed, no PR. Rebase onto `main`: **no-op** (base unmoved,
re-checked at finish). ⚠ Per protocol §3.5 I did **not** rebase onto `origin/main` (stale at `40c2f31`).

| commit | files | note |
|---|---|---|
| `4663c76` | **new** `chlu/eval/rivals/gdn2_lm.py` | the arm + the ledger + the pins |
| `dfcc24b` | `chlu/eval/rivals/__init__.py` (**+6, 0 modified, 0 deleted**) | ⚠ the only shared file |
| `00444fd` | **new** `chlu/experiments/exp_c3_rival_gdn2.py`, **new** `scripts/smoke_c3_gdn2.sh` | the smoke leg |
| `c7a51f7` | **new** `tests/test_c3_rival_gdn2.py` | 19 cases |
| `baf0166` | `chlu/eval/rivals/gdn2_lm.py` (**+6 comment lines**) | the output-gate shim's verified semantics, recorded beside the line |

```
$ git -C /Users/user/Desktop/CHLU-wt3 diff --stat main..HEAD
 chlu/eval/rivals/__init__.py          |   6 +
 chlu/eval/rivals/gdn2_lm.py           | 878 ++++++++++++++++++++++++++++++++++
 chlu/experiments/exp_c3_rival_gdn2.py | 298 ++++++++++++
 scripts/smoke_c3_gdn2.sh              |  59 +++
 tests/test_c3_rival_gdn2.py           | 443 +++++++++++++++++
 5 files changed, 1684 insertions(+)
```

⛔ **Nothing else touched.** `chlu/core/blocks.py` and `chlu/eval/byte_ledger.py` (`c3-gb-landing`'s) are
**bit-identical to base** — verified by the diffstat above; the other two rivals' surfaces do not exist yet;
`chlu/config.py`, the CLI, `chlu/training/*`, `chlu/data/*`, the corpora registry and
`PREREG-C3-LADDER.md` are **untouched**. Scratch: none in the repo; everything under
`.claude/outputs/c3-rival-gdn2/`. The vendored clone of the official code lives in `/tmp`, **not** in the
repo (no large binaries, no vendored third-party code committed).

**Worktree-ref verification (protocol §3.2, the lost-8-commits precedent):**
```
$ git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/c3-rival-gdn2
baf0166  c7a51f7  00444fd  dfcc24b  4663c76      (5 commits)
```
✅ All 5 commits visible on the shared ref **from the main repo**. ⚠ **The worktree
`/Users/user/Desktop/CHLU-wt3` is LEFT IN PLACE** for the Hub's review; the temporary baseline worktree
`../CHLU-wt3base` (detached at `main`, for the collect-only) was created and removed.

**Artifacts** (all under `.claude/outputs/c3-rival-gdn2/`): `PREREG-C3-RIVAL-GDN2.md` ·
`gdn2-ledger.json` (both ladder rows + shrink solution + FLA check + param table) ·
`smoke/{interrupted,uninterrupted}/gdn2_smoke.json` · `full-suite.log`.

---

## 9. Open questions / follow-ups / risks

1. ⛔ **F1–F3 are decisions, not bugs**, and each moves a published control. F1 and F2 belong to
   `c3-gb-landing` (it owns `byte_ledger.py`); **F3 belongs to the Hub** — it changes what "matched params"
   means for every rival row, not just GDN-2, and the same mixer-vs-block ambiguity will hit
   **Mamba-2** and **TTT** at the same 24 L / d_model 512 reference.
2. ⚠ **No CLI hook was added** to `chlu/cli/experiment_cmd.py`, deliberately: three rival spokes appending
   a parser to the same 700-line function is the textbook collision, and the smoke leg is reachable as
   `python -m chlu.experiments.exp_c3_rival_gdn2` and via `scripts/smoke_c3_gdn2.sh`. One `add_parser`
   block at merge time, once, is the cheaper resolution — flagged, not done.
3. ⚠ **The GVA refusal is a real capability gap**, declared: `num_v_heads > num_heads` raises. The official
   default does not use it and the paper's matched setting does not either, so no pinned config needs it;
   a ladder that later wants grouped value heads must implement the key-side repeat first.
4. ⚠ **The `lax.scan` recurrence is O(T) sequential.** It is the *same function* as the chunkwise WY kernel
   and correctness is tested, but at real seq_len × 24 layers the wall-clock is untested and is the one
   number the ladder must measure early (the same caveat the scout raised about MFU). A chunkwise
   implementation is a pure-throughput follow-up that cannot move a result.
5. 🔍 **Not done, out of scope:** no ladder training (task §1.7 forbids it), no dyn-eval column for this arm,
   no PG-19/WikiText leg, no bf16 execution (the arm *ledgers* at bf16 as deployed but *runs* in fp32 here,
   which is the harness's dtype and affects speed and precision, never the byte ledger).

---

## Proposed handover updates (for the Hub)

**§7 — new entries**

- **7.3x [NEW, DECISION] GDN-2's short-convolution state is real and `RIVAL_SPECS` does not count it.**
  `+220,320 B` at the shrunk config ⇒ **2,301,120 B = 1.0973×** the 2 MiB ceiling under a TOTAL-as-deployed
  convention, vs **0.9922×** under the paper's own "main recurrent state" (§E.1 Eq. 90). ⭐ Our own shell's
  causal conv is likewise un-ledgered, so counting theirs and not ours is the unfair direction. Owner:
  `c3-gb-landing` / Hub. The arm ledgers **all three columns** and chooses nothing.
- **7.3x [NEW] `shrink_to_budget("gated_deltanet2")`'s `n_heads = 6` has no integer head geometry.**
  `512/6 = 85.33…`; the realizable arm is **2,080,800 B (−16,352 B, 0.9922×)**, not the pre-registered
  `2,097,152`. The disclosed powers-of-two coincidence is a property of the *formula*, not of a deployable
  config. ⚠ Also: in the official parameterization `head_dim` is independent of `d_model`, so `n_heads`
  shrinks state **only** under `RIVAL_SPECS`' `d_k = d_model/H` tie — that tie is **ours** and should say so.
- **7.3x [NEW, AFFECTS EVERY RIVAL ROW] The 24 L / d_model 512 recurrent-class reference cannot hold both
  the 26–47 M param class and the paper's own block.** GDN-2: **95.4 M** with the MLP, **44.5 M** MLP-free;
  the scout's "≈6·d_model²/layer ⇒ 37.75 M" counts the **mixer**, not the block (FLA's own docstring says
  so). The same ambiguity will hit Mamba-2 and TTT. Owner: Hub.
- **7.3x [AMEND] The FLA 3× trap is `flash-linear-attention`'s `GatedDeltaNet` (v1) layer**, not NVlabs'
  GDN-2 layer, whose defaults (`head_dim=128, num_heads=16, expand_v=1`) reproduce the paper **exactly**
  (1.000×). One clause in `RIVAL_SPECS["gated_deltanet2"].provenance` prevents a future reader concluding
  the authors' code is the trap. The 3× itself is **re-verified** (786,432 floats/layer).

**§2 (architecture) — new surface**

- `chlu/eval/rivals/gdn2_lm.py` — ⭐ **the first BUILT rival**, i.e. `PREREG-C3-LADDER.md` §4.2's "⛔ No
  implementation of … GDN-2 … exists in this repository" is now **out of date for GDN-2** and should be
  amended when the Hub merges. It is a **standalone LM**, not a `MEMORY_CELLS` cell: it does not use the
  shared `StreamPhi`, so its φ row is a declared **0**. It is nonetheless drop-in for the pilot's `loss_fn`
  / `token_nll` (its `__call__` accepts and ignores `plans`).
- ⚠ **Name hazard for the curator:** `chlu.eval.rivals.deltanet.DeltaMemory(variant="gdn2")` (B′ gym memory,
  1 head, no conv) and `chlu.eval.rivals.gdn2_lm.GDN2LM` (C3 rival LM) are **different objects with the same
  paper**. The `__init__.py` comment says so; a paper table must not merge their rows.

**§3 (CLI & config) — no new global knobs.** `GDN2Config` is self-contained; `MATCHED_STATE_BYTE_BUDGET` is
consumed, never changed. ⚠ One deferred item: `chlu/cli/experiment_cmd.py` has **no** `c3-rival-gdn2` hook
(collision avoidance across three concurrent rival spokes) — one `add_parser` block at merge.
