# ERRATA-C2W8-PASS3 — dated addenda to `PREREG-C2W8-PASS3.md`

Filed **before** the cells each block governs. `PREREG-C2W8-PASS3.md` is **not edited** (a revised
pre-registration stops being one). Spokes append their own dated blocks below; **append only, never
rewrite another block** — ⚠ and **number your block after checking the last one present** (a §-number
collision in the pass-2 errata cost a renumber-by-banner at integration).

---

## §1 — 2026-08-09, filed by the C2W8 Hub BEFORE any pass-3 cell: the four Head rulings

The pass-3 scoping entry carried four questions. **All four are RULED, same session.** None changes a
registered prediction (Q1–Q8 stand as filed); they fix the substrate, the mapping's status, the
dimension rule, and the NO-GO semantics.

### R1 — ✅ RULED: **the spine runs on SPLIT-CIFAR-10.**
The strong-φ encoders were **built and priced** there (simclr `enc_steps = 8000`: CLU ACC
**0.16080 → 0.31912**, +0.158 paired), and a **PCA-φ reference row already exists in the same
harness** at **0.16080**. ⇒ the substrate is where the encoder is real, not where the census happens
to have history.
⛔ **CONSEQUENCE, BINDING ON EVERY PASS-3 ARM: pass-1 and pass-2 census numbers are MNIST and are NOT
the baseline for pass 3.** The comparison is **cross-dataset AND cross-encoder**, so the weak-φ
baseline must be an **INTERNAL PCA-φ CIFAR-10 census arm at the same `d`, run in the same harness,
in the same run**. ⛔ **No pass-3 number may be compared to a pass-1/2 census number.** (This is the
same cross-run/cross-checkout error the wave has caught repeatedly; here it would also be
cross-dataset.)
⚠ **Attached risk, declared now:** `CluSystem`'s learned `V_θ` has never been run on CIFAR φ. Pass 1
measured it **inert at `d ≥ 16` on MNIST** (`ERRATA-C2W8.md` §3). ⇒ **every pass-3 spoke that writes
to the store measures well depth FIRST and reports an inert store in its first 10 lines** — an inert
store makes a census vacuous for a reason that is **not** the gate's reason, and the two must never
be conflated.

### R2 — ✅ RULED: **the φ_dim → addr_dim PROJECTION IS IN SCOPE, and the LAUNDER USES IT.**
The map must be **built**, not configured — `PhiAddress` today forces `phi_dim = addr_dim` and
truncates. Confirmed in both halves:
- **(a)** building a genuine projection / read-in head is **wt2's scope**;
- **(b)** ⛔⛔ **the launder reads the PROJECTED φ, never the 256-dim φ.** A launder reading 256 dims
  while the store reads 8 is **not a launder — it is a handicap match** (fairness invariant §A4.3:
  identical φ for CLU, baselines and launder). **Assert it in code, do not merely intend it.**
- Projection parameters go on the **byte ledger of EVERY arm including the launder**;
  **`(d, atom budget)` stays ONE declared joint dial.**

### R3 — ✅ RULED: **the spine runs at the `d` wt2's geometry measurement favours; the revived rider cell runs at `d = 16`.**
Feasible band **d ∈ {8, 12, 16}** (8 192 / 32 768 / **131 072** atoms; `min_atoms = round(512·√2^d)`;
d = 256 is 1.7e41 and forbidden). The dimension is therefore **chosen by measurement, not by
default** — wt2 reports spacing / σ_q·spacing⁻¹ / `d_safe`·spacing⁻¹ at all three and the spine takes
the favoured one. ⚠ **`d = 16` is 16× pass 1's atom budget — price the rider cell before running it,
and if it will not fit, report the arithmetic rather than a truncated run** (a declared NOT-RUN,
never a null).

### R4 — ✅ RULED: **`geometry_go == false` RE-LABELS the spine; it does NOT block it.**
A NO-GO means the spine measures the physics on a substrate **known in advance not to have fixed
separability** ⇒ its null becomes **ATTRIBUTABLE rather than confounded**, which on this wave's record
has repeatedly been the more valuable product. ⛔ **The label then travels on every number the spine
produces.** ⛔ **`GATE-ADDR-VALIDATED.json` with `gate_addr_validated == true` remains a HARD block**
(§A30.1 is a binding order) — only the geometry file's *verdict* is soft; **its existence is not.**

---
