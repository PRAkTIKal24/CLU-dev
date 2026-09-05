# PREREG — the C3 store-geometry sweep (smoke scale, real enwik8)

**Filed BEFORE the sweep ran** (protocol §5 pre-registration rule). Agent
`c3-rival-ladder-prereg`, branch `agent/experiment-engineer/c3-rival-ladder-prereg`
off `agent/experiment-engineer/c3-csf3-harness @ f98f939`. 2026-08-13.

This is the *input* prereg: it commits to what the geometry sweep will show, so
that `PREREG-C3-LADDER.md` (the deliverable) is written against measured numbers I
could not have chosen after the fact. It is deliberately short.

---

## 0. What changed the question before a single step ran (arithmetic, not opinion)

`CluSystemConfig.n_atoms` is **not** `atoms_per_item × capacity`. It is

```
n_atoms = capacity * ceil( max(atoms_per_item*K, min_atoms, round(min_atoms_base * min_atoms_c**addr_dim)) / capacity )
        = max( A*K , 384 , round(512 * sqrt(2)**d_addr) )      (rounded up to a multiple of K)
```

At the ruled **`addr_dim = 8`** the geometric term is `512 * sqrt(2)^8 = 512*16 =`
**8192**, so at the pilot's `K=32, A=256` the term `A*K = 8192` **ties** the floor:
`atoms_per_item` is *decorative* at the pilot geometry, and **no reduction of
`capacity` or `atoms_per_item` can move a single byte of the store** while the
w23 dimension-aware floor stands.

Consequence, computed not assumed (`state_floats = n_atoms*(dim+2) + K*dim`,
fp32, ×`n_layers`):

| knob moved | n_atoms | CLU total state B | occupancy of 2 MiB |
|---|---|---|---|
| pilot (`K=32,A=256`) | 8192 | 5,523,456 | 2.634× |
| `A: 256 → 64` | **8192 (unchanged)** | **5,523,456** | 2.634× |
| `K: 32 → 128, A=16` | **8192 (unchanged)** | 5,578,752 | 2.660× |

⇒ **The only levers that can move the CLU state byte count are `addr_dim`
(ruled fixed at 8), `payload_dim`/`dim`, `n_layers` (fixed by the 26–47 M param
class), and the w23 floor constants `min_atoms_base` / `min_atoms_c` themselves.**
The sweep is therefore a sweep of **`n_atoms` below the w23 floor at `d_addr=8`**,
i.e. it measures exactly the thing the floor exists to prevent: **starvation of
the write**.

The floor's own provenance (`chlu/config.py` L1518–1540) says d=8 "reaches strict
1.000 by **4096–8192**" on the w22/w23 designed-mechanism discriminator, and that
`base=512` pins the floor "with margin". So the sweep's central question is:
**is the margin real on the real stream at `d_addr=8`, and how far below 8192 can
the write still dig?**

## 1. The grid (2 seeds × the points below), real enwik8, smoke shapes

`d_model 32, n_layers 2, seq_len 256, batch 2, steps 60, addr_dim 8, payload_dim 4
(dim 12), capacity 32, data_bytes 600,000` (real enwik8 cache, `--corpus enwik8`).
`n_atoms` is set by `min_atoms_base = n_atoms/16` **with `min_atoms_c` left at
sqrt(2)**, plus `atoms_per_item = n_atoms/K`, so both terms of the max agree and
the geometry is unambiguous.

- **axis A — the byte/starvation ladder:** `n_atoms ∈ {512, 1024, 2048, 3072, 4096, 8192}` at `K=32`.
- **axis B — iso-byte spend at `n_atoms=2048`:** `K ∈ {16, 32, 64, 128}` (items vs atoms-per-item at fixed bytes).
- **axis C — dim:** `payload_dim ∈ {4, 8}` at `n_atoms=2048, K=32` (the `d ≤ 12` reach-ceiling design rule at C3's address geometry). ⛔ `addr_dim` is **held at 8** throughout (Hub R2).
- **axis D — depth linearity:** `n_layers ∈ {2, 4}` at `n_atoms=2048, K=32` (a check that per-layer cost is linear, not a claim).

## 2. Numeric predictions, with falsifiers (committed before the run)

| # | prediction | falsifier |
|---|---|---|
| **G1** | Byte arithmetic: `total_state_bytes(n_atoms=2048, K=32, dim=12, L=12) = 1,394,688 B` exactly and `= 0.6650×` the 2 MiB ceiling; the shipped ledger reproduces it to the byte. | any disagreement with `chlu/eval/byte_ledger.py`'s emitted row |
| **G2** | Compute: clu_store **s/step ratio `n_atoms 8192 : 2048` lands in [1.5, 4.0]** (sub-linear in atoms, because the per-chunk write/read cost is dominated by the integrator step count, not the dictionary size). | ratio < 1.5 (atoms are free ⇒ shrinking buys nothing) or > 4.0 (super-linear ⇒ shrinking is a *compute* win worth more than I claim) |
| **G3** ⭐ | Starvation: the write-efficacy signal at the last step (`store_health.depth_ratio_vs_untrained`, paired with `qstar_payload_spread`) at **`n_atoms=2048` is within a factor 2 of its value at `n_atoms=8192`**, on both seeds — i.e. the w23 floor's "margin" at `d_addr=8` is real and 2048 is not a starved cell. | either seed shows `2048` worse than `8192` by **more than 2×** on depth ratio, **or** `n_live` at 2048 falls below 2/3 of `n_live` at 8192 ⇒ **the shrink is NOT defensible and the §2 conflict is hard** |
| **G4** | ⛔ **The null I expect to survive:** at smoke scale the bpc dividend `bpc(none) − bpc(clu_store)` is **smaller in magnitude than the seed-to-seed spread of either arm**, so the sweep **cannot rank geometries by bpc** and I will not rank them by bpc. | a geometry ordering consistent across both seeds with `|Δbpc|` > 2× the seed spread ⇒ smoke bpc *is* informative here, and I must say so |
| **G5** | Dim: `payload_dim 8` (dim 16) costs `(16+2)/(12+2) = 1.286×` the bytes of dim 12 at equal `n_atoms`, and buys **no** measurable write-efficacy improvement at smoke (the `d ≤ 12` reach-ceiling design rule; `d=16` recorded inert). | a ≥2× write-efficacy improvement at dim 16 on both seeds |

## 3. What the sweep is NOT

⛔ Not a claim venue; no bpc from it enters any table, per `scripts/smoke_c3_local.sh`'s
banner and charter §2. It decides **one thing**: whether a store geometry below the
w23 atom floor at `d_addr = 8` can be defended on measured behaviour. If G3's
falsifier fires, the honest output of the parent task is the §2 **conflict report**,
not a frozen sub-2 MiB geometry.
