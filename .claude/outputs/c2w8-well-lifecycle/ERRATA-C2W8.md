# ERRATA-C2W8 — dated addenda to `PREREG-C2W8.md`

Filed by **experiment-engineer** (wt1, branch `c2w8-well-lifecycle`), **before the
cells each block governs**. `PREREG-C2W8.md` is not edited. Every block below is
an *instrument definition the prereg left to the implementation*, or a *declared
deviation with its measured cause* — no prediction is added, changed or re-tuned.

---

## §1 — 2026-08-06, filed BEFORE the census runs: `theta_att`, the measured capture floor

`PREREG-C2W8.md` §3.1 requires `is_attractor(i)` to use "the **measured** capture
floor `theta_att` on this rig (SC-6's 32-direction bisection ...), not a guessed
constant", but does not give the arithmetic that turns a set of bisection radii
into a depth floor. Registered here:

> For each live site `z_i`, SC-6's direction bisection
> (`chlu.core.soft_certificate.capture_radius`, `n_dirs` directions,
> `tol = sigma_q`) returns `r_cap(i)`, the largest displacement that still
> relaxes back to within `tol` of the site.
> A well **captures** iff `r_cap(i) >= sigma_q` — the read's own queries are
> drawn at that scale (`query_sigma`), so a basin narrower than the jitter
> cannot be addressed at the operating point.
> **`theta_att` := the largest fitted depth among the wells that did NOT
> capture**, and `0.0` when every well captured.
> **`is_attractor(i) := (lambda_min at the RELAXED site > 0) AND
> (depth_i > theta_att) AND (r_cap(i) >= sigma_q)`.**

`theta_att` is therefore a floor at which capture was *observed to fail on this
rig*, not a constant; it is reported with `n_capturing` / `n_non_capturing` /
`capture_radius_median` so the number is auditable. `lambda_min` is evaluated at
the **relaxed** site, never the recorded one (slot != well).

Implementation: `chlu/core/well_lifecycle.py::measure_theta_att`, asserted in
`tests/test_well_lifecycle.py::test_theta_att_is_measured_not_guessed`.

## §2 — 2026-08-06, filed BEFORE the census runs: the merge criteria's two constants

`PREREG-C2W8.md` §3.3 defines `M` as pairs with "payload distance below the
registered threshold AND center separation below the SC-1/SC-2 certificate
radius" without fixing the first. Registered here, both taken from **shipped
config**, neither tuned:

* **payload threshold := `CluSystemConfig.payload_tol` (0.1)** — the read's own
  tolerance. Two payloads closer than the read can resolve carry one value, not
  two, which is exactly what a merge asserts.
* **certificate radius := `R_cert = 2 s_max + kappa' sigma_q`**
  (`soft_certificate.cert_radius`, `kappa' = d_safe_kappa_prime = 2.576`) — the
  SC-1/SC-2 radius already computed and reported on every write.

Consequence at this wave's payload map (§4): with adjacent classes `0.111` apart
and `payload_tol = 0.1`, a pair is payload-admissible **iff it carries the same
class**. That is the near-duplicate population over-digging is supposed to
produce, and it is a mechanical criterion, not a threshold search.

## §3 — 2026-08-06, filed BEFORE the census runs: the census's address dimension, and WHY

`PREREG_CL_PHI` §7 binds `phi_dim >= 16` for the CL entry. **The census cannot
run there**, for a measured reason, and this block declares it before any census
cell runs (`.claude/scratch/c2w8/timing16.py`, main venv, `main @ d70898b`):

| `addr_dim` | `n_atoms` | fitted depth after 3 designed-site writes | self-probe strict | 3-write wall |
|---|---|---|---|---|
| 4 | 2 048 | 0.480, 0.601 | 0.812 | 7.3 s |
| 8 | 8 192 | 0.443, 0.342, 0.458 | 0.333 | 11.8 s |
| 12 | 32 768 | 0.297, 0.345, **0.000** | 0.250 | 28.8 s |
| 16 | 131 072 | **2.1e-9, 6.8e-10, 0.000** | **0.000** | 87.1 s |

⇒ **At the CL entry's binding `phi_dim = 16` the learned store is INERT** — it
digs no wells at all. Diagnosed cause (arithmetic, not conjecture): atom centers
are drawn `N(0, atom_init_scale=1)` in `dim = addr_dim + 1`, so the nearest atom
to a unit-norm site is ~`sqrt(dim)` away while the atom width stays `0.3`; the
write's gradient carries a factor `exp(-r^2 / 2 s^2)` and underflows. This is the
`cluformer-pilot` warning's surviving hypothesis (atom placement at init),
measured here on this rig.

*Supporting arithmetic (added the same day, while the census cells were running;
it changes no declaration above — it only quantifies the diagnosed cause).*
`.claude/scratch/c2w8/atomdist.py`: distance from a unit-norm site to the
**nearest of all** `n_atoms` atoms at the shipped `atom_init_scale = 1.0`, and
the Gaussian factor the write gradient carries at that distance
(`atom_width = 0.3`):

| `addr_dim` | `n_atoms` | `dim` | `min_j |c_j − z|` | `exp(−r²/2s²)` |
|---|---|---|---|---|
| 2 | 1 024 | 3 | 0.133 | 9.07e-01 |
| 4 | 2 048 | 5 | 0.294 | 6.18e-01 |
| 8 | 8 192 | 9 | 0.738 | 4.86e-02 |
| 12 | 32 768 | 13 | 1.252 | 1.65e-04 |
| 16 | 131 072 | 17 | 1.483 | **4.98e-06** |
| 32 | 33 554 432 | 33 | 2.957 | 8.13e-22 |

The `min_atoms_base · c^d` floor buys atoms geometrically but the **nearest** one
still recedes, so the factor the writer sees falls monotonically. `d = 8` is the
last row where it is O(10⁻²) rather than O(10⁻⁶).

**Declared:** the census runs at **`addr_dim = phi_dim = 8`**, the largest
dimension at which this rig demonstrably digs wells. Every reading from it is
labelled **non-promotable** as a CL benchmark entry (`flags.promotable = false`,
carried in `census.json`). ⛔ This is a *rig* limitation reported as such — it is
**not** K1's kill condition and must never be conflated with it (the honesty
clause of the task file). ⚠ `atom_local_radius` (the declared N98 lever) is
**not** usable to rescue it here: it requires per-group localization targets
known at **init**, and a `phi`-addressed stream does not know its addresses until
write time — and §7.24 records it as dead in the `LearnedVStore` path anyway.

## §4 — 2026-08-06, filed BEFORE the census runs: label -> payload

The CL stream's payload is a class label; the store's payload channel is a
bounded scalar whose magnitude is limited by the reach certificate
(`|a_i| < a_U`). Registered map: **`payload = (label - 4.5) / 9`**, i.e.
`[-0.5, +0.5]`, adjacent classes `0.111` apart. Fixed before any cell; not swept.

## §5 — 2026-08-06, filed BEFORE the census runs: B1 netting on THIS rig

`PREREG-C2W8.md` §2 B1 nets the designed decay out keyed by `last_write_chunk`,
which is the *cluformer* rig's clock. On the `CluSystem` rig the designed decay
is the controller's per-item `leak` applied to the item's own atom group, so the
netting is **exact and replayable**: `factor(i) = prod` of the `factors[i]`
recorded in the controller's own `decay` verb log since the item's write, and
`depth_netted := depth_raw / factor(i)`. Reported side by side with `depth_raw`,
per item, together with the item's own-vs-foreign atom-sum split at its site.
C2W6's residual instrument is **imported** (`exp_anti_erosion._interference_audit`)
and fed this rig's arrays; that file is not edited.

Asserted in `tests/test_well_lifecycle.py::test_decay_netting_is_exact_against_the_designed_law`
(predicted `exp(-leak*n_ticks)` vs recorded factor, `rel=1e-4`).
