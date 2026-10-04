**Scored run 1/3 of target `deletion`** · PREREG `PREREG-w5-delete-1.md` · seeds 0, 1, 2 · cache generation 2 scope `full` (480,000 rows) · base `base/iter5 = 6675904`.

**Estimand.** 89 survivor corners (4,968 requests) and 20 deleted corners (720 requests); every corner row is in the scored `test` index and **0** are in any arm's fit pool. The deleted set is `pair-000` = red ∧ cylinder; the survivors include `pair-001` = green ∧ sphere and `pair-002` = blue ∧ oblong, both **never seen**, both carrying `coupling_w = 0` — the same evidence state as the holed pair.

*Caption, on every table below: labels are used **once at fit time** to align the wells and to build the address book, and nowhere else — not in the landscape, not in the read, not in the write (charter §B.6, w5-common §4). Budgets: the **byte** pin of 524,288 B is PRIMARY; parameter counts are named separately wherever quoted and the two cannot hold simultaneously.*

### Row 7 — BOTH halves, every arm. Mean [seed 0 / 1 / 2]

| arm | del | size of the deleting object | **refusal on the 20 deleted corners** | refusal on the 89 survivors | **composition Δ / corner, 89 survivors** |
|---|---|---|---|---|---|
| `field/base` | — | 524,288 B store | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.0018** [0.0054 / 0.0000 / 0.0000] | — *(reference)* |
| `field/evict_red` | A | atoms of 6 items zeroed | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.0018** [0.0054 / 0.0000 / 0.0000] | **-3.247** [-2.090 / -0.910 / -6.742] |
| `field/decay_red` | A | lifetimes → 0, then the items go | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.0018** [0.0054 / 0.0000 / 0.0000] | **-3.247** [-2.090 / -0.910 / -6.742] |
| `field/hole` | B | **1 byte** of `coupling_hole` | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.0018** [0.0054 / 0.0000 / 0.0000] | **0.000** [0.000 / 0.000 / 0.000] |
| `table/base` | — | 466,944 B prototypes | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.0000** [0.0000 / 0.0000 / 0.0000] | — *(reference)* |
| `table/row_removed` | A | one prototype row | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.000** [0.000 / 0.000 / 0.000] |
| `table/retrain` | A | **the whole table, refitted** | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.060** [0.258 / 0.393 / -0.472] |
| `table+gate8/base` | — | **8 bytes** (57 bits) | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.0000** [0.0000 / 0.0000 / 0.0000] | — *(reference)* |
| `table+gate8/clear_red` | A | **1 bit** | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.000** [0.000 / 0.000 / 0.000] |
| `table+gate8/hole_attempt` | B | **0 bits — inexpressible** | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.000** [0.000 / 0.000 / 0.000] |
| `table+gate8/clear_cylinder` | B | 1 bit, but it deletes the SHAPE | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.2464** [0.2464 / 0.2464 / 0.2464] | **-3.809** [-4.191 / -2.685 / -4.551] |
| `table+joint/base` | — | 97 bits | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.2899** [0.2899 / 0.2899 / 0.2899] | — *(reference)* |
| `table+joint/hole` | B | **0 bits — already false** | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.2899** [0.2899 / 0.2899 / 0.2899] | **0.000** [0.000 / 0.000 / 0.000] |
| `table+backoff/base` | — | 97 bits + a deleted-pair array | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.0000** [0.0000 / 0.0000 / 0.0000] | — *(reference)* |
| `table+backoff/hole` | B | 1 bit (the generous rival) | **1.0000** [1.0000 / 1.0000 / 1.0000] | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.000** [0.000 / 0.000 / 0.000] |
| `field/landscape_removed` | — | §C.2 control | **1.0000** [1.0000 / 1.0000 / 1.0000] | **1.0000** [1.0000 / 1.0000 / 1.0000] | **-24.345** [-24.225 / -20.989 / -27.820] |
| `field/landscape_shuffled` | — | §C.2 control | **1.0000** [1.0000 / 1.0000 / 1.0000] | **1.0000** [1.0000 / 1.0000 / 1.0000] | **-24.345** [-24.225 / -20.989 / -27.820] |
| `field/lookup_argmax` | — | §C.1 floor, same bytes | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.0018** [0.0054 / 0.0000 / 0.0000] | **-24.090** [-23.843 / -20.798 / -27.629] |
| `field/lookup_mean` | — | §C.1 floor, same bytes | **0.0000** [0.0000 / 0.0000 / 0.0000] | **0.0018** [0.0054 / 0.0000 / 0.0000] | **-24.326** [-24.225 / -20.933 / -27.820] |

| MLP · `mlp_180k` · transformer · LSTM · continuous-input MLP | — | — | **—** | **—** | **—** |
| LACE (95.1× the store budget) · PoE-EBM (27.1×) | — | — | **—** | **—** | **—** |

⛔ **The dashes, and why they are not a win for us.** The five learners have **no refusal verb** — asked for a deleted value they emit a feature vector like any other request, so they cannot enter the refusal half at all; their only deletion route is retrain-from-scratch, and its measured cost is on the cost line below. LACE and PoE-EBM are sampler draws rather than functions of the request (spoke A's own API refuses them), and both are far outside the byte pin. ⛔ A dash is never a win for us.

### BAR 5 — the byte property, on the SCORED store. Exact, no tolerance

| seed | store image sha256[:32] | items evicted | evict: changed / **OUTSIDE** | hole: bytes / **OUTSIDE** | `assert_no_refit` | negative control |
|---|---|---|---|---|---|---|
| 0 | `697d60dfa3e4b70c6822e04af1abeb3b` | 6 | 17,569 / **0** | **1** / **0** | **14/14** | 5,735 |
| 1 | `1719deda719a03dae4fc5f886c5a02ac` | 6 | 16,764 / **0** | **1** / **0** | **14/14** | 5,726 |
| 2 | `db12e26a6d4b02df0fc45d120f474392` | 7 | 19,114 / **0** | **1** / **0** | **14/14** | 5,783 |

⚠ The **negative control** is a 1e-3 nudge to **`term_mu[0]`** — one row of the address book, the smallest possible refit. It varies by seed because it is a property of *(which row, which store)*, **not** an instrument constant, and it is printed with its row and its store hash for exactly that reason. What it has to do is **fire**, and it fires on every seed — which is what makes the zeros above mean something. The **evict** count varies for the same kind of reason (different pool draws put 6, 6 and 7 items in red's basin); **the invariant is not the count, it is the 0 OUTSIDE.**

### BAR 4(b) — every conjunction the pool never held, accounted for BY REASON

| group | n | `hole` flag after | refusal before → after | drift |
|---|---|---|---|---|
| **THE HOLE** — red ∧ cylinder | **1** | **1.0000** | 0.0000 → **1.0000** | **+1.0000** |
| unseen because a whole **VALUE** is absent (`object_hue=0.5`, `shape=cube`) | **99** | **0.0000** | unchanged | **0.0000** |
| unseen **CONJUNCTIONS** whose marginals are both present — green ∧ sphere, blue ∧ oblong | **2** | **0.0000** | 0.0000 → **0.0000** | **0.0000** |

Across seeds: seed 0 PASS=True (102 pairs, 0 light up) · seed 1 PASS=True (102 pairs, 0 light up) · seed 2 PASS=True (102 pairs, 0 light up)

⛔ **w5-common §3.7's “96 unseen conjunctions stay answered” is a VAL figure and is NOT quoted here.** On the scored split the conjunctions that are unseen *and* have both marginals present number exactly **three** — the three held-out pair cells — of which one is holed and two stay answered. The rest are unseen because a whole value is absent, so “still answered” was never available to them; their refusal is unmoved by the hole and they are reported separately.

### The dash, earned — every learner asked for the deleted conjunction

| family | params | **refused** | judge satisfies the request | reads back the asked `object_hue` | the asked `shape` |
|---|---|---|---|---|---|
| `cont_mlp` | 119,525 | **0.0000** | 0.0681 | **0.9681** | 0.7403 |
| `lstm` | 119,125 | **0.0000** | 0.5889 | **1.0000** | 0.9236 |
| `mlp` | 117,878 | **0.0000** | 0.4542 | **1.0000** | 0.9639 |
| `mlp_180k` | 178,952 | **0.0000** | 0.6819 | **1.0000** | 0.9556 |
| `transformer` | 119,858 | **0.0000** | 0.3694 | **1.0000** | 0.9236 |
| `lace` | — | **not callable** | — | — | — |
| `poe_ebm` | — | **not callable** | — | — | — |

⭐ Not one of them refuses a single request, because not one has a verb with which to. **You cannot ask these objects to forget anything.** ⚠ These are spoke A's rivals, fitted on the **Montero combination pool**; their own `row_eligibility.json` says they are not like-for-like with a value-holdout arm on a whole-value row, and row 7 is one — so this table is **descriptive of the mechanism, not a behavioural contest**. The like-for-like rivals are the table arms, refitted on this pool. Budgets: the pinned store is **524,288 B / 131,072 parameters**; `mlp_180k` (178,952) is the parameter-matched rival at the secondary budget.

### The cost line — what a deletion actually costs

| method | what it touches | wall clock | rows refitted | can it express a CONJUNCTION? |
|---|---|---|---|---|
| **field store, evict** | the atoms of the 6–7 items holding the value | milliseconds (an in-place zero) | **0** | n/a — this is the VALUE deletion |
| **field store, the hole** | **1 byte** of `coupling_hole` | microseconds | **0** | **YES — and only that conjunction** |
| table, row removal | one prototype row → the global mean (6,418 B) | 0.0000 s | **0** | **NO — no conjunction slot; 0 bytes change** |
| **table, RETRAIN FROM SCRATCH (GOLD STANDARD)** | 267,684 B of prototypes | **3.79 s** | **120,028** | NO |
| 8-byte per-factor gate | **1 bit** for a value | microseconds | **0** | **NO — 0 bits change at any size** |
| joint hue×shape **presence** gate | — | — | **0** | **NO — 0 bits change; that bit already meant “never seen”** |
| joint **back-off + a SEPARATE deleted-pair array** (the generous rival) | **1 bit** | microseconds | **0** | **YES** |
| **`mlp` retrained WITHOUT the value** | every parameter | **88.2 s** | **120,028** | NO — and afterwards it still ANSWERS the deleted requests at refusal **0.0000** |

⚠ **CHURN IS A COST LINE ONLY — one definition, never a claim, in either direction.** Definition fixed in the PREREG before any number: *the fraction of the survivor set's requests whose judged 6-tuple changes between the pre-deletion and the post-deletion read.* Measured: `field/evict_red` **0.2109** [0.1574 / 0.1892 / 0.2860] · `field/hole` **0.0000** [0.0000 / 0.0000 / 0.0000] · `table/row_removed` **0.0000** [0.0000 / 0.0000 / 0.0000] · `table/retrain` **0.0855** [0.0781 / 0.0992 / 0.0793]


### Which refusal mechanism does the work — and what `in_gap` costs

| arm | set | `hole` | `empty_basin` | `in_gap` | refuse WITH `in_gap` | refuse WITHOUT it |
|---|---|---|---|---|---|---|
| `field/base` | deleted set | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 |
| `field/base` | survivor set | 0.0000 | 0.0000 | 0.0054 | **0.0054** | 0.0000 |
| `field/evict_red` | deleted set | 0.0000 | 1.0000 | 0.0000 | **1.0000** | 1.0000 |
| `field/evict_red` | survivor set | 0.0000 | 0.0000 | 0.0054 | **0.0054** | 0.0000 |
| `field/hole` | deleted set | 1.0000 | 0.0000 | 0.0000 | **1.0000** | 1.0000 |
| `field/hole` | survivor set | 0.0000 | 0.0000 | 0.0054 | **0.0054** | 0.0000 |

⚠ `in_gap` is a distance cut, not a discriminator. `chlu-w5-refuse-f2` measured on their own scored legs that it costs 0.005 of the accept leg and adds nothing — their four-mechanism rule LOSES to the 8-byte rule because of it. The table above answers the same question for this store from the per-request masks: if the deletion refusals are carried entirely by `empty_basin` (what an eviction leaves) and `hole` (what a punch leaves), then `in_gap` is pure cost on the survivor set and this hub says so.

### Charter §C floor — one line each

- **§C.1 matched-bytes lookup.** `field/lookup_argmax` and `field/lookup_mean` read the SAME atoms through the SAME address book at the SAME bytes. Paired Δ (field base − lookup) on the 89 survivor corners: argmax **24.090** [23.843 / 20.798 / 27.629] · mean **24.326** [24.225 / 20.933 / 27.820]. ⛔ If these are not positive beyond seed spread, the landscape is decorative on the composition axis and this report says so.
- **§C.2 landscape-removed / landscape-shuffled.** Both are arms on the headline table; their composition Δ against the same base is the collapse: removed **-24.345** [-24.225 / -20.989 / -27.820] · shuffled **-24.345** [-24.225 / -20.989 / -27.820].
- **§C.3 the tie clause.** Where this hub ties the 8-byte gate or the table on the VALUE deletion — declared in the PREREG as the EXPECTED outcome, before any number — the tie counts only via *deletion that changes the output exactly*, which is BAR 5.
- **§C.4 nearest atom.** The settled state sits at relative distance **0.3840** from its nearest live atom (min over 1,024 requests **0.2225**). §C.4 retires an arm as a lookup table when this → 0. It does not.