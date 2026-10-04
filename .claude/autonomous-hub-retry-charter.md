# AUTO-HUB RETRY — a new hub type. Scoped by the Hub 2026-09-19 at the Head's direction.
**Binds alongside `autonomous-hub-charter.md`, which is unchanged and still applies in full.**
⭐ A retry hub is not a new kind of agent. It is a normal autonomous hub with a **shortened clock, an
inherited context, and a narrower licence.** Everything in §A–§E of the base charter still holds.

---

## 1. Why this type exists
Wave 3's four hubs finished at **1 h 19 m · 2 h 37 m · 3 h 10 m · 5 h 04 m** against a 10 h budget.
**Roughly 20 hub-hours went unused.** A retry hub spends that remainder.

## 2. ⭐⭐ WHEN IT FIRES — on TIME REMAINING, ⛔ never on the verdict
⛔ **A win is not the trigger, and using it as one gets this exactly backwards.** In wave 3 the winning
hub (`factor-wells`) **exhausted its target's scored budget**, while the hub that returned **0 of 3
runs intact** (`unseen-valid`) was a clean negative that exited at 13 % of its allowance. The negatives
free the most time and hand over the sharpest next question.

**Fire a retry iff BOTH hold at the moment of discovery:**
1. the parent hub is **finished** — final report section written, no sub-agent running, pane idle; and
2. **≥ 3 h remain in the parent's 10 h window, measured from DISCOVERY, ⛔ not from the parent's exit.**

⚠ **The floor is measured from discovery because discovery is when a retry can actually launch.**
With 2 h checks (02:00 / 04:00 / 06:00 / 08:00 / 10:00 elapsed) that yields **three** live rescope
opportunities — at 2 h, 4 h and 6 h. ⛔ The 08:00 check cannot rescope: 2 h is below the floor.
⭐ **Below the floor, bank the time and say so.** A 2 h hub produces a fragment nobody can use.

**Retry allowance = 10 h − elapsed-at-discovery.** ⛔ It does not get a fresh 10 h.

## 3. ⛔⛔ WHAT IT INHERITS — and the scored budget is usually ZERO
**Three scored runs per TARGET, across every attempt and every machine. A retry inherits the balance,
which after a winning parent is typically nothing.** `factor-wells` spent composition 2/3 **and** 3/3.
⇒ ⛔ **"Improve upon itself" in the scored sense is unavailable precisely when the parent won.**

**A retry is therefore scoped as exactly one of these three, named explicitly in its charter:**
| kind | when | scored runs |
|---|---|---|
| **R-A · robustness** | the parent won on a thin basis — seed 0, one cell, one seed of the store fit | **0** |
| **R-B · mechanism** | the parent produced an unexplained effect or an unclosed ablation | **0** |
| **R-C · adjacent target** | the parent's finding opens a target that **still has budget** | inherits that target's balance |

⭐ **R-A is the default after a win.** Wave 3's single win is seed 0 only and buys a verdict, not a paper
number; the three-seed rule is the standing blocker on every result the programme owns.

## 4. What it may and may not do
- ⭐ **Branching:** a retry may branch from its parent's tip **only as a same-target continuation.**
  ⛔ It may never branch from a *different* hub's branch (base charter §A.4), and ⛔ it never touches the
  frozen scoring harness. If it cannot continue the parent's code additively, it branches from the wave
  base instead and says so.
- ⭐ **It inherits the parent's LOG.md as reading, not as truth.** It re-verifies any number it builds on
  and reports what it could not reproduce.
- ⛔ **One retry per parent. Cap at `.1`.** If a retry also finishes early, ⭐ **bank the remainder** —
  do not spawn `.2`. Diminishing returns, and the naming stops being legible.
- ⛔ **It runs in its own worktree and its own tmux session**, with the `START_UTC.txt` refusal and the
  `has-session` guard, exactly as a parent does.

## 5. ⛔ THE HUB'S OWN BOUNDARY — where I stop and the Head decides
⭐ **Rescoping inside a target or a dial is scoping, and the Hub does it.**
⛔ **Choosing a new target is research direction, and the Hub does not.**
If the obvious next move is a new target, ⭐ **the Hub banks the time, writes the question down, and
reports it.** ⛔ It does not pick.

## 6. What a retry's charter must carry (in addition to the base charter's requirements)
1. its **kind** — R-A, R-B or R-C — and the scored balance it inherits, stated as a number;
2. **the parent, the parent's verdict, and the one finding it builds on**, quoted with its number;
3. its **allowance in hours**, and the wall-clock time it must stop by;
4. ⭐ **what the parent got wrong or left open** — a retry with nothing to correct is not worth launching;
5. the standard block: dial declaration · files owned · a mechanical precondition · acceptance ·
   and **what would prove this retry wrong**.

## 7. Naming
`<wave>-<target>` for a parent, `<wave>.1-<target>` for its retry: `4-gen`, `4.1-gen`, `4-comp`, `4-del`,
`4.1-del`. ⭐ Reports are ordered **by target with iterations nested**, never interleaved.

---

# 8. ⭐⭐ THE WAVE-CHAINING MODEL — how the Hub runs unattended
**The Head and the Advisor pre-scope waves 4, 5, … . The Hub executes them back to back without waiting
for approval.** The only cause to hold a launch is the `sshkey` device being unreachable.

**The loop, per wave:**
1. launch the wave (`AUTOHUB-SSH-RECIPE.md`, in full, including its pre-launch checklist);
2. **check every 2 h** — one SSH, cheap: sessions alive · `LOG.md` size and mtime · pane state ·
   ⚠ **an unsubmitted line sitting in an input box**, which in wave 3 made three finished hubs look busy;
3. on a finished hub: summarise it, and fire a retry if §2 permits, else bank and record;
4. when every hub in the wave is finished: **compile the wave report and the summary doc**, pull them
   back, then **immediately scope and launch the next pre-approved wave.**

## 8.1 ⛔⛔ THE PREMISE CHECK — the one thing that must happen before each launch
**A pre-approved wave was scoped before the previous wave reported. Wave 3 killed two premises outright:**
row 2 is **not scorable** with the frozen judge (a perfect arm and a perfect lookup differ by exactly
0.0000 under the harness's own metric), and **storing earlier-layer content is retired** (0 of 49
corners, p = 1.1e-09). ⛔ **A wave scoped against either premise would burn 10 h proving a dead thing.**

**So, before launching wave N+1, the Hub checks each hub's stated premise against wave N's findings:**
- ⭐ **premise survives ⇒ launch it, unchanged, immediately. ⛔ No approval, no delay.**
- ⛔ **premise is dead on measurement ⇒ do NOT launch that hub.** Launch **every other hub in the wave**,
  and report the held one with the measurement that killed it and the smallest amendment that would
  revive it. ⛔ The Hub does not redesign it — that is the Head's and the Advisor's.

⭐ **This is not a licence to hesitate.** The default is launch. Holding requires a *measurement* from the
previous wave that contradicts a premise, named in the report — ⛔ never a reservation, a preference or a
judgement call.

## 8.2 What the Hub reports at each wave boundary
The wave report (as `.claude/outputs/w<N>-advisor-report.md`), a short summary doc, the scored-run ledger
state per target, and **one line per held hub** if §8.1 held any. ⛔ Nothing waits on the Head to read it.

## 9. ⛔ What would make this operating model wrong
- **Retries produce fragments.** If two consecutive retries return nothing usable, the floor in §2 is too
  low — raise it and say so.
- **The premise check fires too often.** If it holds more than one hub per wave, the waves are being
  scoped too far ahead of the evidence, and that is worth telling the Head immediately, not at the boundary.
- **A retry's numbers get quoted as the parent's.** They are separate runs on a shorter clock; the report
  keeps them in separate rows, always.
- **Chaining outruns the scored budget.** If a wave opens with every relevant target at 0/3, the wave can
  only do selection work — ⭐ the Hub says so **before** launching, not after.
