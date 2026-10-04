# track2-admissibility — the surviving Track-2 candidate list, the B′ pre-registration, and SDM positioning

**Campaign 2, wave C2W2. Agent:** web-scout. **No worktree, no code, read-only on the repo.**
Charter Addendum-1 **§A5/C2W2 task 5** ("Option C"), gated by the amendments in **§A3**.

**Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-c2-charter.md` **IN FULL incl.
ADDENDUM 1 (§A3 amendments are already in force — do not re-litigate them; §A5 C2W5 names your output's
consumer)**, `.claude/advisor-head-intervention.md` **§6 (the five admissibility criteria — all five must
hold) and §8.3/§8.4**, and your predecessor `.claude/outputs/rival-recon.md` **in full** (its
Deliverable 5 fairness checklist is banked — extend it, do not rewrite it; its §Gaps is your worklist).

⛔ **REGISTRY LAG (Head parked the curator pass, 2026-07-30).** `negative_results.md` (N122),
`claims_matrix.md` (v2.5) and `research_roadmap.md` (v0.9) are **two campaigns behind**: **C1W27's and
ALL of C2W1's results are in no registry.** Quote them **only** from `.claude/outputs/*` and the
`[C1W27]`/`[C2W1]` §10 entries. ⚠ **This bites you specifically:** the never-quote additions
`rival-recon` proposed were **never filed**, so treat its handover item 2 as in force and re-state the
full list in your own output — it is currently the only place it will live.

---

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** none — recon. No performance claim measured.
- **Laundering control:** n/a for you, **but the ex-ante substitute audit is your central instrument** —
  see §1. Every candidate task is screened *before* any harness is built.
- **Falsifies the deliverable:** any convention or metric-native verdict that cannot be traced to a
  **primary source** and is not marked **UNPINNED**.
- **Does NOT falsify:** finding that few or no candidates survive (that is a decision-grade result and
  it feeds B′); finding our positioning further occupied.

---

## 1. ⭐ Deliverable 1 — the candidate list that survives criterion 4 **AND the substitute audit, EX ANTE**
C2W1 established two things you must apply, not re-derive: **MAD/zoology/MQAR are inadmissible as the
Track-2 primary** (metric-native + Arora ICML'24 Thm 3.1's Ω(N)-bit state) — charter §A3, in force; and
the gym's real killer was not metric-nativeness but **cheap substitutability**: the audit went
**0-for-4**, with a **+0 B** substitute of the launder's own table beating CLU on two families outright
(insertion order **0.776** vs 0.302; echo **1.0000** vs −0.180).

⇒ **Screen every candidate against BOTH, before it is proposed:**
1. **Criterion 4** (not metric-native to the store) — with the *mechanism* quoted, as `rival-recon` did.
2. ⭐ **The substitute audit, ex ante:** name the strongest **cheap classical substitute** for the task
   (kNN/table + a **+0 B** read-out over what the store already holds: order, echo, aggregate, count,
   recency). If a plausible **+0 B** substitute is at or near ceiling, the task is **inadmissible**, and
   you say so *before* a harness is written rather than after 28 cells.
3. Criteria 1, 2, 3, 5 (strong baselines that do well · real headroom · memory management over time is
   the difficulty · every lever can be active).
**Deliver:** a ranked table — primary candidate, fallback, and the rejected list **with the reason per
criterion**. `rival-recon` recommended **enwik8/WikiText at 26–47M params** as primary with **MAD
`compression`** as the one admissible synthetic; test that recommendation against the substitute audit
(⚠ its own flag: at 26–47M params **local n-gram statistics may dominate**, so criterion 3 is the weak
one — a well-tuned n-gram/count baseline *is* the cheap substitute, and it must be run as a column).
⛔ **If zero candidates survive both screens, say so plainly.** That is a decision-grade finding, it is
the strongest possible input to the B′ decision, and it must not be softened.

## 2. ⭐ Deliverable 2 — the B′ pre-registration (charter §A3: *"B′ is a contribution, not a retreat"*)
B′ = **the audit paper**: *"when does test-time dynamics buy anything over a table at matched bytes?"* —
the launder/dividend protocol + the two-sided byte ledger + the substitute audit, applied to **CLU and
the TTT / Titans / Delta / Sparse-Delta-Memory rivals**. It is the standing fallback at every C2 gate,
and the **C2W2 gate may activate it at this wave's close**. File it now, before we know which branch
fires, so it is a pre-registration and not a rationalisation.

Deliver, in prereg form:
- **The claim**, in one sentence, and the **table** the paper is (rows = memory families, columns = the
  audit's instruments), with every cell marked *have / need*.
- **What must be measured** for each rival, sized: which rivals are reimplementable at this weight class
  and which are not (⚠ `rival-recon`: **Titans has no official code**, an independent group calls
  reproducibility the bottleneck, and its chunk size `b` is **UNPINNED**; SDM is a **3-week-old**
  preprint with no independent replication).
- **The byte conventions** each rival's audit needs (extend `rival-recon` §F2; the **Titans `2·|M_θ|`
  momentum accounting is our reconstruction and must be captioned as such**).
- ⭐ **The falsifier of B′ itself:** what measurement would make the audit paper *not* worth writing —
  e.g. every rival is trivially substitutable too **and everyone already knows it** (then the audit is
  not news), or the launder protocol does not transfer to a rival's state (then the comparison is not
  apples-to-apples). Register it.
- **Where B′ reuses banked evidence** rather than re-deriving it (the matched-bytes launder record, the
  byte-floor theorem `ratio = 1.4·atoms_per_item + 0.8`, the substitute audit's 0-for-4, Prop D2a's
  three independent confirmations, byte-exact deletion AUC **0.5000 ± 0.0000**).

## 3. ⭐ Deliverable 3 — the Sparse Delta Memory positioning brief
**arXiv:2607.07386 (Cabannes et al., Meta FAIR, 7 Jul 2026)** is a near-collision and it is now
**mandatory** in C2W5's baseline set. Deliver a brief that a paper-writer can lift:
- The **mechanism map**, ours ↔ theirs, at equation level: explicit slots `M[i]` ↔ our atom groups;
  PKM top-k addressing ↔ our derived addresses + admission; **Eq. 3 decay ↔ our decay law**; **Eq. 4
  top-W write selection ↔ our admission gate**; Eq. 5 top-R read ↔ our read. State plainly where they
  do the same thing and where they do not.
- **What survives as ours** after the collision (charter §A3 already ruled the phrasing): **byte-exact
  deletion (AUC 0.5000 ± 0.0000, byte-equal 3072/3072)**, **settable per-item lifetimes**, and the
  **physics-predicts-the-knobs spine**. ⛔ *"Principled forgetting"* as a novelty phrase is dead —
  Titans Eq. 13, Gated DeltaNet Eq. 8, SDM Eq. 3 own it.
- **The exact-deletion prior-art sweep** (`rival-recon` §Gaps 6, and it is cheap): does **any** rival do
  *exact* deletion, or only soft decay? One targeted pass through machine-unlearning ∩ sequence-memory.
  This either confirms or kills our **last uncontested pillar-4 claim** — the highest value-per-hour
  item in this task.
- Their **published limitations** as our fair-fight footing (state ≈ model params; MFU an order of
  magnitude below GDN; 1.49× slower training at 8B).

## 4. Riders (from `rival-recon` §Gaps — do these, they gate C2W5)
1. **RULER** — the long-context recall suite SDM and GDN report on. Characterise it: tasks, metric,
   state conventions, and whether it survives criteria 4 + the substitute audit. Scout it **before the
   Track-2 harness freezes**.
2. **Titans' chunk size and any official code release** — re-scout; nothing may be reimplemented from an
   ambiguous description without saying so in the caption.
3. **Pillar-4 preemption check** on the two 2026 preprints surfaced but unread: **arXiv:2604.07350**
   (*Fast Spatial Memory with Elastic Test-Time Training*) and **arXiv:2605.06946** (*Adaptive Memory
   Decay for Log-Linear Attention* — a direct neighbour of per-item lifetimes).
4. **Seed conventions** — one targeted OpenReview pass (reviewer threads usually force this out). Our
   ≥3-seed rule stays stricter either way; we need to know what we are comparing against.
5. **Version the "real Mamba" rule** in one line for the handover: **Mamba-2 minimum**
   (`mamba2.py`: `d_state=128, headdim=64, expand=2, chunk_size=256`), Mamba-3 named in limitations.

## 5. Method and evidence grade (non-negotiable)
Primary sources only — equations, config literals, table numbers, reference implementations; blog and
secondary for orientation only and never cited. **Every convention traces to a primary source or is
marked UNPINNED.** Keep the confidence table `rival-recon` used (pinned / pinned-but-preprint / inferred
/ unpinned / absence-of-evidence); an absence-of-evidence novelty claim is labelled as such.

## 6. Falsifiers
- ⛔ A convention or metric-native verdict that cannot be traced and is not marked UNPINNED.
- ⛔ **Zero candidates survive both screens** ⇒ Track 2 as specified has no admissible primary. Report it
  as the headline and escalate to the Hub immediately — it changes the gate's consequences.
- ⛔ **A rival is found doing exact deletion** ⇒ pillar 4's last uncontested differentiator falls; that
  is a positioning emergency and it must reach the Hub the day you find it.
- **Does NOT falsify:** finding rivals metric-native (information, not defeat); finding our framing
  occupied; recommending against the charter's own prior recommendation with evidence (your predecessor
  did exactly that and it became binding §A3).

## 7. Compute / priority
**Priority P2** — no worktree, no engineer slot, launch immediately. Estimated ~4–6 h.
⭐ **Deliverable 2 (the B′ pre-registration) must land before the wave review**, because the gate may
fire B′ at the review and the pre-registration has to predate the decision to be worth anything.

## 8. ⛔ Never-quote (inherited)
"An anytime accuracy-vs-compute curve no baseline can draw" (**falsified** — DEQ, Neural DEQ Solvers,
EBT Fig. 6a/12, Titans-Revisited; the anytime curve is a **shape** claim, §A3) · "principled forgetting"
as a novelty claim · "graceful degradation above capacity" as our discovery (Clark, PRE 2026,
arXiv:2506.05303) · any Titans state-byte number without *"our reconstruction; the paper states no
convention"* · **"Guo et al. Def. 1/Def. 2" — it does not exist** (ε-certified removal is §3 Eq. (1),
inline) · "certified"/"unlearning"/"deletion-compliant"/unqualified "exact deletion" · a margin against
an un-rescued baseline (N78: a baseline below its published range is **not rescued** and no margin
against it is quotable).

## 9. Output
`.claude/outputs/track2-admissibility.md`: the three deliverables + riders, the confidence table, a
gaps/what-to-search-next section, and BibTeX-ready refs for everything new. Reconciliation list in the
**first 10 lines** if any item changes a task file or a governing document. Report to the Hub.
