# PREREG — c3-rival-mamba2 (§A of `.claude/tasks/c3-rival-arms.md`)

Written **before** the Mamba-2 arm module existed and **before** any harness run.
Base: `main` @ `0644c48`. Branch: `agent/experiment-engineer/c3-rival-mamba2`.

Protocol §5's pre-registration rule is written for *measured* ratios/exponents/slopes.
My acceptance criterion is a **byte-exact reproduction of derived arithmetic**, which
is not a measurement — but the whole task is "commit to a number, then build the thing
that must reproduce it", so every number the arm will emit is committed here first.

## 1. Pinned config (predicted, from the scout §1.5 / `RIVAL_SPECS["mamba2"]`)

`d_model 512 · expand 2 · headdim 64 · ngroups 1 · d_conv 4 · n_layers 24 (reference)`,
`dtype_bytes = 2` (bf16 **as deployed**, per the Head+Advisor convention: no dtype
normalisation).

Derived: `d_inner = 1024`, `n_heads = d_inner/headdim = 16`,
`conv_dim = d_inner + 2·ngroups·d_state`, `conv_state = conv_dim·(d_conv−1)`.

## 2. Predicted state arithmetic

| quantity | prediction |
|---|---|
| **published** `d_state=128`: ssm/layer | 16·64·128 = **131,072** elts |
| **published**: conv_dim | 1024+256 = **1280**; conv_state = 1280·3 = **3,840** elts |
| **published**: elements/layer | **134,912** |
| **published**: 24 L total bytes (bf16) | 24·134,912·2 = **6,475,776 B** ⇒ must equal `RIVAL_SPECS["mamba2"].state_bytes()` **to the byte** |
| occupancy of the 2 MiB ceiling | **3.0879×** |
| **shrunk knob** (from `shrink_to_budget`, NOT re-solved by hand) | `d_state 128 → 39` |
| **shrunk**: ssm/layer | 16·64·39 = **39,936**; conv_dim 1102, conv_state **3,306** |
| **shrunk**: elements/layer | **43,242** ⇒ **86,484 B**/layer |
| **shrunk**: 24 L total | **2,075,616 B**, occupancy **0.98973×** ≤ budget ✓ |

## 3. Predicted AS-DEPLOYED rows (the shell is not 24 layers)

The C3 shell (`PilotConfig.n_layers = 12`, the scout's *attention-class* reference)
holds 12 cell states, not 24. Predicted deployed totals at the shrunk `d_state=39`:

| shell | total state bytes | occupancy |
|---|---|---|
| pilot, 12 L | **1,037,808 B** | **0.4948×** |
| smoke, 2 L | **172,968 B** | **0.0825×** |

⭐ **Pre-registered finding, stated before it is observed:** at `n_layers = 12` the
shrink solved on the rival's own 24-layer reference geometry leaves the arm at **~½ the
envelope**. That is *anti-hobbling-adverse* and I predict it will be visible in the
ledger. Per task §1.4 I will **not** re-solve the knob by hand; I will report it as a
STOP-and-report item for the owner of `byte_ledger.py`/the prereg.

## 4. Predicted parameter count of one deployed cell (latent seam `dim = 12`)

`d_in_proj = 2·d_inner + 2·ngroups·d_state + n_heads = 2048 + 78 + 16 = 2142`.

in_proj 12·2142 = 25,704 · conv w 1102·4 = 4,408 · conv b 1,102 · A_log 16 · dt_bias 16 ·
D 16 · gated-RMSNorm weight 1,024 · out_proj 1024·12 = 12,288 · learned-init `ssm0`
39,936 · learned-init `conv0` 3,306 ⇒ **87,816 params/layer** (learned inits are
PARAMETERS, not STATE — PREREG-Bprime §4.1, the same rule the GRU's `h0`, the TTT's
`W0` and the CLU's `V0` are held to).

## 5. What would falsify the build

* any emitted 24-L byte count ≠ 6,475,776 (published) / 2,075,616 (shrunk);
* a state-bearing hyperparameter whose provenance string is not `PAPER:`/`OFFICIAL IMPLEMENTATION:`;
* the arm failing to enter the ladder through `build_byte_ledger` (i.e. needing an
  edit to `chlu/eval/byte_ledger.py`, which is not mine);
* a NaN/divergent smoke leg.

## 6. What is NOT predicted here

⛔ **No bpc.** The smoke config is not a claim venue and no loss number from it is
quotable in either direction.
