# deletion-prior-art — verify the load-bearing citations BEFORE anything is drafted

**Agent:** web-scout. **Read-only, no worktree.** Addendum-2 §B3.1 ("citation scout runs BEFORE
any drafting").

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** lifetimes/admission (R1) — this task guards the *claim wording*, not a measurement.
- **Laundering control:** N/A (literature task). The analogous discipline: **a claim that a result
  is novel must be checked against the field that owns the technique, not just the field we are
  publishing into.** The theorist already flagged that the discrete skeleton of our placement rule
  is prior art; your job is to find out exactly how much.
- **Falsifies:** any of the asserted citations does not say what we claim it says; or the
  continuous-landscape composition is *also* already owned.
- **Does NOT falsify:** the discrete skeleton being fully prior art — that is the **expected**
  finding and the paper must state it precisely, not dodge it.

## Why this is urgent
`order-independent-placement` (w25) delivered PGCP with a proof and a green permutation harness —
and stated plainly, in its own report: *"⚠ citations asserted from training knowledge, NOT
verified from primaries; web-scout must verify before any draft cites them."* The engineer landing
the code this wave (`placement-landing`) has its claim wording **gated on your answer**. Nothing
about deletion gets drafted until this lands.

## Group 1 — the history-independence / uniquely-represented data-structures literature
Verify from **primaries** (not from secondary summaries, not from memory) what each actually
claims, proves, and requires:
- **Naor & Teague, STOC 2001** — "anti-persistence: history-independent data structures". What
  exactly is strong vs weak history independence, and which do we satisfy?
- **Blelloch & Golovin, FOCS 2007** — SHI open-address hashing via key priorities. ⭐ **The
  theorist states our priority-displacement rule *is* theirs, transplanted.** Confirm or correct
  that, precisely: is our fix-up cascade the same algorithm?
- **Hartline et al.** — SHI ⟺ canonical representation (for reversible structures). Get the exact
  statement and its conditions; our Theorem 1 leans on it implicitly.
- **Micciancio 1997** — oblivious 2–3 trees.
- **Karger et al., STOC 1997** — consistent hashing, as the order-independent-assignment neighbour
  *without* spacing/geometry semantics.

**The question to answer, in one paragraph the paper can quote:** what does this literature own,
and what (if anything) is left for *"exactness in a continuous designed energy landscape, with
decay and permanence coexisting, plus a metric spacing certificate"*? Be adversarial about it.

## Group 2 — the unlearning / exact-deletion arena (the must-cites)
- **SISA** (Bourtoule et al., S&P 2021), **SILO**, **Ticketed Learning–Unlearning**, **DaRE**
  trees, **PALL** (ICLR 2025 — it sits directly in the CL ∩ forgetting cell and rehearses).
For each: the mechanism, the exact guarantee claimed, its cost model, and **what it does at
deletion time**. We need this to position "deletion cost at matched utility" (addendum-2 §B2
Candidate 2) — so pay particular attention to **per-deletion cost**: retraining a shard,
maintaining checkpoints, subnetwork surgery.

## Group 3 — is Candidate 2 already occupied?
⭐ **Does anyone already claim, or measure, "deletion cost at matched utility"** as a benchmark
axis? If a formalisation exists, we should adopt its metric rather than invent one — and if it
exists and someone already wins it, we need to know now, not after the experiment.

## Group 4 — the vocabulary check (confirm the standing ban)
Confirm from primaries what **"certified" unlearning** formally requires ((ε,δ) guarantees, the
defended DP sense). The w24 recon concluded we fail on all four counts and the word is banned
program-wide. This is a re-verification, not a re-litigation — if it holds, say so in one line and
move on. Also check whether **"exact deletion"** and **"deletion by construction"** carry
defended technical meanings we would be violating.

## Deliverable
`.claude/outputs/deletion-prior-art.md`. **Every claim carries a citation with the specific
section/theorem** — "the paper says X" without a locator is not usable and will be sent back.
Structure: per-group findings · ⭐ the one-paragraph novelty statement the paper may use verbatim
(or an explicit "the claim as currently worded is not defensible, here is what is") · a
must-cite list with one line each on why · a **do-not-claim** list · anything you could not verify
(say so plainly rather than filling the gap from model knowledge — that is exactly the failure
this task exists to fix).
Reconciliation list in the first 10 lines. No repo edits.
