# PREREG — the C3 Gated DeltaNet-2 arm (arXiv:2605.22791)

**Filed BEFORE the harness that measures any of these numbers was run.** Task `c3-rival-arms` §0–§2 + §B.
Agent: experiment-engineer. Branch `agent/experiment-engineer/c3-rival-gdn2`, worktree
`/Users/user/Desktop/CHLU-wt3`, base local `main @ 0644c48`.

⛔ **Dial: none — rival implementation.** No CLU number, no comparison. Every prediction below is either
(a) an arithmetic property of a pinned config, or (b) a structural property of the official implementation.

---

## Sources I will pin against (all obtained THIS session, before writing code)

| source | how obtained | what I take from it |
|---|---|---|
| **PAPER** arXiv:2605.22791v1 HTML (`https://arxiv.org/html/2605.22791v1`, 665,102 B, CC BY 4.0) | `curl` | Eqs. 8–12 verbatim, §3.5 block design, §E.1 Eq. 90 state statement, App. C.1 layer parameterization, App. D.2/D.5 |
| **OFFICIAL IMPLEMENTATION** `NVlabs/GatedDeltaNet-2` @ HEAD (`git clone --depth 1`) | `git clone` | `lit_gpt/gdn2.py` (`GatedDeltaNet2.__init__`/`forward`), `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py` (the reference recurrence), `lit_gpt/config.py` (`gdn2_1.3B`) |
| **TRAP CONTROL** `fla-org/flash-linear-attention` `fla/layers/gated_deltanet.py` | `curl` | the library defaults the task forbids inheriting |

⚠ Declared NOT-OBTAINED: none. Both the paper and the official code parsed.

---

## P1 — the pinned arm reproduces `RIVAL_SPECS` **to the byte**

At the published pin (`n_layers 24, d_model 512, n_heads 4, d_k = d_v = 128, bf16`):

**main recurrent state = 24 · 4 · 128 · 128 · 2 B = 3,145,728 B**, and
`RIVAL_SPECS["gated_deltanet2"].state_bytes()` = **3,145,728 B**, **delta = 0**.

*Falsifier:* any non-zero delta.

## P2 — `shrink_to_budget()`'s solved knob (I use it; I do not re-solve)

`shrink_to_budget("gated_deltanet2")` returns `knob="n_heads"`, `published_value=4`,
**`shrunk_value=6`**, `state_bytes_shrunk=`**2,097,152** — equal to `MATCHED_STATE_BYTE_BUDGET` exactly.
⚠ This equality is the **disclosed arithmetic coincidence** (`24·512²/6·2 = 2,097,152`, powers of two), not a
fitted number; task §B says so and the ladder prereg records it. I re-assert it, I do not re-derive it.

*Falsifier:* the call returns anything other than `n_heads 4 → 6`.

## P3 ⭐ NEW — the shrunk knob has **no integer head geometry**, and the realized arm is UNDER the ceiling

`RIVAL_SPECS`' formula is `n_L · d_model² / H`, i.e. it ties `d_k = d_v = d_model / H`. That tie is exact at
the published pin (512/4 = 128 = the paper's own `d_k`). At the **shrunk** `H = 6`, `512/6 = 85.33…` is not an
integer, so **no realizable head geometry hits 2,097,152 B**. Under the same tie with floor,
`head_dim = 512 // 6 = 85`, and I predict:

- realized main recurrent state = `24 · 6 · 85 · 85 · 2 B` = **2,080,800 B**
- occupancy = 2,080,800 / 2,097,152 = **0.992203…×**
- deficit vs the ledger's ideal = **16,352 B** (0.780 %), i.e. the arm lands **under** the ceiling.

⭐ The shrink direction (never grow) is therefore **preserved**, and the honest deployed number is
2,080,800 B, not 2,097,152 B.

*Falsifier:* the realized geometry exceeding 2,097,152 B, or `512 // 6 ≠ 85`.

## P4 ⭐ NEW — the short convolution carries state that `RIVAL_SPECS` does not count

The official layer ships `use_short_conv=True, conv_size=4, conv_bias=False` (`lit_gpt/gdn2.py`
`GatedDeltaNet2.__init__` defaults) and caches `(conv_state_q, conv_state_k, conv_state_v)` alongside
`recurrent_state` in `update_layer_cache`. Its state is `(2·key_dim + value_dim)·(conv_size−1)` per layer.

At the shrunk config (`H=6, head_dim=85, expand_v=1` ⇒ `key_dim = value_dim = 510`):

- conv state = `24 · (2·510 + 510) · 3 · 2 B` = **220,320 B**
- **TOTAL as deployed** = 2,080,800 + 220,320 = **2,301,120 B** = **1.09727×** the 2 MiB ceiling.

⇒ Under task §1.5's *"TOTAL state bytes, AS DEPLOYED"* convention the shrunk GDN-2 arm is **over budget**,
while under the paper's own convention (§E.1 says **"main recurrent state size"**, and it is that quantity
the paper matches across families) it is under. ⛔ `RIVAL_SPECS` and the ceiling are `c3-gb-landing`'s, so
this is a **STOP-and-report**, not an edit. I will ledger all three columns explicitly.

*Falsifier:* the official layer not caching conv state; or the arithmetic differing.

## P5 — the FLA 3× trap, re-verified rather than inherited

`fla/layers/gated_deltanet.py` defaults `hidden_size=2048, expand_v=2, head_dim=256, num_heads=6` ⇒ state
per layer `H·d_k·d_v = 6·256·512` = **786,432** floats = **3.000×** the paper's 262,144.
⭐ Prediction I am less sure of and therefore state: the **official GDN-2 layer's own** defaults
(`head_dim=128, num_heads=16, expand_v=1`) reproduce the paper **exactly** (16·128·128 = 262,144), i.e. the
3× trap is specifically **FLA's `GatedDeltaNet` (v1) layer**, not NVlabs' GDN-2 layer.

*Falsifier:* either default differing from the above.

## P6 ⭐ NEW — the 24-layer pin and the 26–47 M weight class are **jointly infeasible** with the paper's block

Per-layer arithmetic at `d_model 512, H=4, d_k=d_v=128, expand_v=1`:

| piece | params |
|---|---|
| `q,k,v,b,w,o` projections (6 × d²) | 1,572,864 |
| `f_proj` (d→d_v_head→key_dim) | 131,072 |
| `g_proj` (d→d_v_head→value_dim, +bias) | 131,584 |
| short convs (3 × width × 4) | 6,144 |
| `A_log` + `dt_bias` + `o_norm` | 644 |
| **GDN-2 token mixer total** | **1,842,308** ≈ **7.03 d²** |
| LLaMAMLP at the official ratio 6208/2304 ⇒ intermediate 1380 (3 matrices) | 2,119,680 |
| 2 × RMSNorm | 1,024 |
| **block total** | **3,963,012** |

Predictions:
- **with MLP, 24 L** (the paper's recurrent model): `24·3,963,012 + 131,072 + 512 + 131,072` = **95,374,944 ≈ 95.4 M** ⇒ **2.0–3.7× outside** the 26–47 M class.
- **MLP-free, 24 L** (the scout's ≈6·d_model²/layer reading): `24·1,842,308 + 262,656` = **44,478,048 ≈ 44.5 M** ⇒ **inside** the class.

⇒ The scout's "recurrent class ⇒ 37.75 M at 24 L / d_model 512" is only reachable if the stack is
**MLP-free**, which the GDN-2 paper's own recurrent model is not. This is a **ladder-level decision** (it
moves both axes of the primary claim), so I ship `use_mlp` as a config knob, default **faithful (True)**, and
report the conflict rather than resolve it.

*Falsifier:* either param count differing by more than rounding from the above.

## P7 — the smoke leg

The arm trains → checkpoints → resumes → ledgers → emits retention slices on **real enwik8** in minutes at a
declared toy geometry. ⛔ **Its bpc is not quotable and appears in no table as a result.** Only the two
mechanical facts are predictions: (a) resume produces a **bit-identical** model to the uninterrupted run at
the same seed and step count; (b) training loss at the smoke geometry decreases from step 0.

*Falsifier:* resume not bit-identical; NaN; the slice call failing on a non-`StreamModel` arm.

---

## What would falsify the TASK (task §0, echoed)

An arm whose state bytes disagree with `RIVAL_SPECS`; a config inherited from a library default; an arm that
cannot enter the ladder through the byte ledger. ⛔ Zero ladder arms trained.
