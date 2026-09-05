# c3-rival-ladder-prereg — freeze the C3 store geometry, set the ceiling digit, and pre-register the Track-A ladder

**Campaign 3, wave 1. Agent:** experiment-engineer. **ONE worktree — wt2** (⚠ `c3-run3-budget-exemption` holds **wt1**; ≤ 3 engineer worktrees stands). ⛔ **This is the artifact that gates ALL ladder training** (Head+Advisor 2026-08-13: *"comes to the Advisor before any ladder arm trains"*). Nothing on the ladder runs until the Advisor accepts it.
Branch **`agent/experiment-engineer/c3-rival-ladder-prereg`**. ⭐ **PREFERRED BASE: the exemption branch once it lands** (`agent/experiment-engineer/c3-run3-budget-exemption`) — it is a ~half-day task and it builds both the interim-budget guard you rely on and the `byte_ledger.py` changes you must not duplicate. If the Head spawns you concurrently instead, base on `agent/experiment-engineer/c3-csf3-harness` and ⛔ **treat `chlu/eval/byte_ledger.py` as read-only** (see §5). The Hub names the base at spawn.
**Deliverable: `.claude/outputs/c3-rival-ladder-prereg/PREREG-C3-LADDER.md`** + `.claude/outputs/c3-rival-ladder-prereg.md`.
**Budget:** ≈ 1.5 days. Smoke-scale compute only — ⛔ **no ladder arm trains under this task.**

**Binding documents:** `.claude/outputs/c3-csf3-harness.md` **§5 IN FULL** · `.claude/outputs/c3-benchmark-scout.md` **§1.4, §1.5** · `.claude/advisor-head-c3-charter.md` **§2, §4, §5** · `.claude/AGENT_PROTOCOL.md` **§5 (the pre-registration rule — it is the whole point of this task)**.

---

## ⭐ DIAL DECLARATION (protocol §7)

- **Dial:** **none — pre-registration + geometry freeze.** ⛔ You measure no ladder result and quote no bpc.
- **Falsifies the task:** a frozen geometry whose ceiling has no admissible value (§2); a prereg without numeric falsifiers; a job plan that does not fit the 2×A100 / 4-day envelope.
- **Does NOT falsify:** discovering the C3 geometry must differ substantially from the pilot's. **It probably must** — see §2. Pilot geometry is **not** presumed to be C3 geometry (Head+Advisor, explicit).

## 1. Freeze the C3 CLU arm's store geometry — measured, not assumed

The pilot store (`addr_dim 8, payload_dim 4, capacity 32, atoms_per_item 256`, `n_atoms 8192`/layer, 12 layers ⇒ **5,523,456 B fp32 total**) was sized in the toy era. **There is no evidence enwik8 at 26–47 M needs it**, and §2 shows it cannot stand. Run a **smoke-scale geometry sweep** on the real stream — vary `capacity`, `atoms_per_item`, `addr_dim + payload_dim`, `n_layers` — and freeze the geometry you can defend, reporting the curve you chose from. ⛔ Freeze on measured behaviour, not on hitting a byte target: state what you traded.
⚠ **`addr_dim` stays 8 unless you pre-register a change with its own falsifier** (Hub ruling R2, 2026-08-13): flipping its default refuses every banked journal and re-rolls the TTT arm's `η·n/d` stability criterion. Changing it *inside a frozen C3 geometry* is legitimate — changing it *silently* is not.

## 2. ⭐⭐ THE CEILING DIGIT — the principle is ruled; the admissible window is arithmetic

**Ruled principle (Head+Advisor 2026-08-13):** a **round** number that **both swap members fit under naturally**, with **no arm sitting exactly on it**, **all others shrink-to-match**, and **occupancy reported** per arm. Convention: **TOTAL state bytes, AS DEPLOYED** — ⛔ **no dtype normalisation**; our fp32 store pays its real 2× cost against a bf16 rival, and harder-for-us is the defensible direction.

**The Hub has already derived the admissible window from the harness's pinned rival table; verify it, do not re-derive from memory:**

| rival (natural, bf16, layer-summed) | bytes |
|---|---|
| ttt_linear ⭐ *swap member* | 1,597,440 |
| gated_deltanet2 | 3,145,728 |
| transformer_xl | 6,291,456 |
| mamba2 | 6,475,776 |
| sliding_window | 12,582,912 |
| ttt_mlp | 12,705,792 |

⇒ **ceiling ≥ max(CLU_total, TTT_matched_total)** and **ceiling < 3,145,728** (so GDN-2 *and everything above it* shrinks rather than grows). **2 MiB = 2,097,152 sits inside that window** — but **only if the frozen C3 CLU total is ≤ ~2 MiB**, i.e. a **≈2.63× shrink from pilot**. TTT-Linear at 1,597,440 fits under 2 MiB naturally (0.76×) as a swap member, not as an "other".

⛔ **If your frozen geometry cannot fit that window, do NOT quietly widen the ceiling** — a ceiling above 3,145,728 makes rivals *grow*, which reverses the direction the Advisor chose as defensible. **Report the conflict, state the trade, and hand it back.** That is a legitimate outcome of this task.

**Also pre-register:** the CLU/TTT-matched **match ratio** you expect to hold (the pilot's was **1.007×** — that property is what makes the swap two-sided and must survive the shrink), and each rival's **shrink knob and resulting value** (the harness's `shrink_to_budget()` already solves these — quote it, do not hand-roll).

## 3. The job plan

**≈5 arms × ≥3 seeds = 15+ jobs**, one arm×seed per job, inside **2×A100 / 4-day**. Use the scout's costing (a 40 M byte-level enwik8 arm ≈ **1.5 h at 35 % MFU → 18 h at 3 % MFU**) and state the MFU you actually measured at smoke scale rather than assuming one. Include: the resume-first ladder shape, the `.eqx` precondition check before any re-resume (PILOT-TTT-RULINGS ruling 2), the **dyn-eval substitute column as its own arm** (⛔ re-measured by us at 26–47 M — the published 0.94 bpc is at 277 M and is a category error beside a 40 M number), and the retention/revisit slices emitted per arm.

## 4. The prereg itself (protocol §5 — this is the deliverable)

`PREREG-C3-LADDER.md`, written **before** any ladder arm runs, stating: the **frozen geometry** and what it traded · the **ceiling digit** with its admissible-window arithmetic shown · every arm's **pinned config with provenance** (paper table vs official implementation) and its **shrink knob** · the **job plan** · and ⭐ **numeric point predictions with declared falsifiers** for the ladder's headline quantities. Commit to numbers: *a prediction that survives is evidence, one that fails is a finding, an un-pre-registered agreement is neither.* Include the **kill conditions first** — what result stops the ladder rather than extends it.

## 5. Ownership, stops, acceptance

**Yours:** the geometry/config surface, `PREREG-C3-LADDER.md`, the job-plan scripts, smoke-scale sweeps. ⛔ **NOT yours:** `chlu/core/blocks.py` · the exemption mechanism (`c3-run3-budget-exemption`, concurrent — ⚠ **both of you touch `chlu/eval/byte_ledger.py`; that task owns it, you consume it.** Coordinate by not editing it: if you need a change there, **STOP and report**) · run 3's config (frozen, exempt, not yours).

**Stops:** no admissible ceiling exists for your frozen geometry (§2) → report and hand back · the geometry sweep says the store wants to be *larger* than pilot → that is a finding, report it, do not force it.

**Acceptance (one line):** `PREREG-C3-LADDER.md` exists with a frozen geometry, a ceiling digit inside the verified admissible window (or a reported conflict), every arm's pinned config + shrink knob + expected occupancy, the CLU/TTT match ratio pre-registered, a 15+-job plan inside the envelope with a measured MFU, numeric predictions with falsifiers, kill conditions first — and **zero ladder arms trained**.

**Report:** protocol §5 + flag-provenance + dial declaration + git footprint.
