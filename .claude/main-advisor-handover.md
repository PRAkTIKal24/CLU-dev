# MAIN ADVISOR (ICLR long) — thread handover. Written 2026-09-23 19:35Z. Replaces the 2026-09-21 handover.

## BOOT — read these, in this order, and nothing else
1. This file.
2. `.claude/outputs/advisor-wave-plan-w15-w18.md` — the LIVE plan (§0–§6: waves 15–20; §5 = w19, §6 = w20). Pre-authorised end to end.
3. The last ~60 lines of `.claude/iclr_pivot_head_advisor.md` — the decisions log. Append dated ⟲ entries after every ruling; never rewrite.
4. Whichever of `.claude/outputs/w{18,19,20}-advisor-report.md` exists when you boot (none existed at handover).
Everything else one fact at a time: `outputs/w{8..17}-advisor-report.md`, the harness records under `outputs/w15-cmaze-{t,o}-harness/`, `outputs/w16-maze-wm/`, `outputs/w17-harvest/REFEREE-v2.md`, `outputs/w13-seeds-and-tables/TABLES.md`.
⛔ Do NOT read: `outputs/advisor-wave-plan-w{4-7,8-10,12-14}.md` (superseded), the C3/C2 charters and campaign ledgers, `outputs/doc-curator-w19-w20-sync.md` (a JULY file from an old campaign whose wave numbers collide with the new w19/w20 — not ours), the shorts' PDFs (already read; findings logged 2026-09-22).

## How to talk to the Head (binding)
Simple terms; RGB shapes and maze rows (agent, key, door, tinted floor, budget) as the running examples; numbers only with their meaning in the same breath; numbered options with a recommendation; end with "Needs you:" only for decisions that are genuinely the Head's. Never cite file names, section numbers or registry codes to the Head. Do not re-derive statistics. Do not re-open rulings in the log. The Head owns timelines, venue compliance, the AI-use statement, the page trim and all prose; an abstract is already registered — never raise submission timing.

## WHAT IS LIVE RIGHT NOW (checked 19:30Z — nothing runs in THIS thread; the lead Hub runs on the box `FORGIS_14` and on the Mac `batmac` in its own sessions)
- Wave 18 (consolidation): `outputs/w18-{seeds-and-tables,registry-fold,referee-fixes,placeholder-fill,referee-4}/` exist; `referee-4` has 1 file (21:08 local) — in progress; no `w18-advisor-report.md` yet.
- Wave 19 (CMAZE-T symmetric-input reruns + task figures + success/failure panels): relayed to the Hub 2026-09-23; no `w19-*` artefacts on the Mac yet. Target `maze-t-symmetric`, val-first, ≤ 9 scored.
- Wave 20 (Shapes data-matched rivals + citation prune ≤ 55 + referee-3 fixes): relayed 2026-09-23, amended at 21:26 local to add LACE/PoE on the 131,072-row pool and retrain-without on the same pool; no `w20-*` artefacts yet. Target `shapes-matched`, 3 scored.
- Budgets after w15–w17 (from the reports): `single-design` 3/3 spent · `generation-table` 3/3 spent · `composition` exhausted · `encoder-loop` 1 left · `encoder-adapt` 2 left · `maze-wm` 9/18 spent, rest declared not-run · `read-verb`, `maze` exhausted.

## WHAT SURVIVES A THREAD CHANGE, AND WHAT DOES NOT
- Survives: everything the Hub runs (box + batmac sessions), all artefacts, the plan file, the decisions log. The Advisor thread launches nothing and watches nothing, so there is no watcher to re-arm.
- Does not survive: this thread's memory of the Head's spoken framings. All of them are in the decisions log — read the tail before answering anything.
- Boot check (Mac mirror only; the box's tree is `/c/chlu/outputs/`): `ls -dt /Users/user/Desktop/CHLU/.claude/outputs/w{18,19,20}-*/ ; ls /Users/user/Desktop/CHLU/.claude/outputs/w{18,19,20}-advisor-report.md`. If a report exists, brief the Head on it in the standard format: three points per wave, then a full simple-terms breakdown on request, then the Hub's rulings as numbered options.
- Trap: `base/iter8` = 148f3b7 on the box, fa60b15 on the Mac; the harness/arm source for CMAZE-T/O is NOT on the Mac (bundles absent) — read the harness FINDINGS/PREREG/LOG, not code.

## WHAT YOU OWE, in order
1. When `w18-advisor-report.md` lands: brief the Head; rule on the referee-4 must-fixes that are artefact-level; leave prose/venue items to the Head.
2. When `w19-advisor-report.md` lands: the CMAZE-T transition row does NOT print until the symmetric-input conditions are reported; brief every condition, win or lose; the wave-16 five-word row is the placeholder until then.
3. When `w20-advisor-report.md` lands: present the two one-to-one Shapes tables (rivals on 131,072 rows; store on 16,384/32,256) and let the Head pick the lead; the other goes to the appendix. Confirm LACE/PoE and retrain-without were re-fit on the 131,072 pool.
4. Pending Head rulings from w15 (asked, not yet answered — Advisor recommendation in brackets): deletion scoped in the ABSTRACT as store-scoped exactness + data-free adaptation, not capability erasure [yes]; report the plateau K=5…32 with the byte minimum at K=24, not a single K* [plateau — w20 already writes it]; the full-decoder (BAR D) paragraph [drop from main text, one appendix line: GPU leg mixed, w8 CPU leg unmeasured after the N359 scorer bug]; w15 harness smoke control columns [drop; w16's scored rows use the fixed derangement control].
5. Rebuttal kit (post-submission only, never in these waves): abstain output for non-sequence rivals · sharded ensemble (SISA) · rival-head unlearning (measured on the fit pool, unscored) · voting CLU ensemble.

## STATE THAT CANNOT BE RECONSTRUCTED (standing rulings — all in the log; the ones a successor trips on)
- Paper = the Naive CLU (1× single design), spine (1)–(6) as ruled 2026-09-22; the CLU is a NAIVE AI UNIT compared against naive units (MLP/LSTM/transformer/cont-MLP); no SOTA claim; systems on top of units out of scope by definition; "the intersection is empty" removed; SISA and compositional diffusion cited in one sentence each with the unit/system distinction.
- Matching = read-path parameters (~130k floats on Shapes); the ledger is memory size, printed, never matched; the inflated-rival row does not exist; rivals must now ALSO be data-matched (w20).
- Deletion = store-scoped exact (store after deletion == store fitted without the rows; frozen encoder/decoder outside the object; address book carried, not refit — the Head writes that in the methods); post-deletion adaptation is light and data-free, never "no changes"; deleting red does NOT stop red — capability removal is the hole's display.
- Maze: composition claim scoped to single transitions (rule-bearing n-way row leads); rollout validity printed once as a limitation; Task 2 (observation) = one appendix paragraph; Task 3 (route) not in the paper; maze refusal displays retired; CMAZE-1/2/2b/T/O never share a table; the shorts are non-archival ⇒ their laws enter as this paper's own theory; class name "Latent Dynamics Model".
- Seeds: 3 everywhere, never 5. Notation: n = factors composed, K = wells. No kill conditions in any task; the Hub never waits for a decision.
- Advisor errors on record: "Mexican-hat → interpolation" paraphrase (w6); repeated the 8.2e-16 maze residual from a report (true value 7.818e-16, N343); said "K* has one seed" when val K=5/32 had three and only the scored TEST row was one seed.

## WORKING PROTOCOL WITH THE HEAD (settled over w8–w17; follow it)
1. Per wave report: (a) three-point brief per wave, plain words; (b) on request, a full simple-terms breakdown per hub/spoke using the RGB-shapes and maze-rows examples — what was tested, what worked, what partially worked, what failed and why; (c) then the Head asks questions one by one; answer each from the PRIMARY RECORD (harness PREREG/FINDINGS/LOG, TABLES.md, run JSONs), not from the Hub's report — the Hub mis-paraphrased byte and scope figures five times in w12–w13 and mis-attributed numbers in w16; (d) close with the Hub's rulings as numbered options with a recommendation; (e) log every Head ruling in `.claude/iclr_pivot_head_advisor.md` the moment it is given, marking Advisor recommendations "PENDING HEAD" until answered; (f) amend the live plan file for anything the Hub must do, and hand the Head ONE copyable relay line for the Hub (everything pre-authorised; the Hub never waits).
2. Correct the Head's premises when the record disagrees (e.g. "K* has one seed" — val had three, the scored TEST row had one), and correct yourself in the log when caught.
3. Raise major issues BEFORE scoping a wave (input asymmetry, data asymmetry, double-blind, style-file persona), never leave them for the Hub.
4. Rulings the Head has repeated are settled; do not re-open them. Head decides budgets, new targets, abstract wording, prose, venue.

## WORKING PROTOCOL (Head-stated 2026-09-23 22:10 local; binding on every Advisor thread)
1. Per wave report: three-point brief per wave in plain words → on request, a full simple-terms breakdown per hub and spoke (RGB shapes / maze rows: tested, worked, partial, failed and why) → the Head's questions one at a time, answered from the PRIMARY record (harness PREREG/FINDINGS/LOG, TABLES.md, run JSONs), never from the Hub's paraphrase (the Hub has mis-stated byte, scope and attribution figures; its own spokes caught it) → the Hub's rulings as numbered options with a recommendation.
2. Log every Head ruling in `.claude/iclr_pivot_head_advisor.md` the moment it is given; Advisor recommendations are marked "PENDING HEAD" until answered. Amend the live plan file for anything the Hub must do and hand the Head ONE copyable relay line for the Hub. Everything is pre-authorised; the Hub never waits.
3. Correct the Head's premises when the record disagrees; correct yourself in the log when caught. Raise major issues before scoping a wave, never leave them for the Hub.
4. Settled rulings are not re-opened. Head decides budgets, new targets, abstract wording, prose, venue compliance. Advisor decides artefact-level referee items and claim-label placement.
5. Do not raise points already solved or inconsequential.
- Status 22:10: the four w15 items AND the three-part "forget red" definition are RULED (log entry 22:05) and folded into the plan (§0 amended rule, §6 w20 deletion re-run). Nothing is pending from the Head.

## ADDENDUM 2026-09-25 — TWO LEAD HUBS
RESULTS hub = W23+ (`outputs/advisor-wave-plan-w15-w18.md`, §5b/§5c/§6a/§7 in flight: W21 landed, W22 in flight). PAPER hub = PW-1+ (`outputs/advisor-paper-plan-pw.md`). Reports: `outputs/w{21,22}-advisor-report.md`, `outputs/pw1-advisor-report.md`. Standing Head rules since the 09-23 handover: pre-registration BANNED (everything prints, the Head decides what is quotable); labels WIN/COMPARABLE/LOSS by measured relation to the best matched rival; relay blocks only after a decision; all seeds run are printed. Ownership: the RESULTS hub holds the paper files until W22 closes; then the PAPER hub owns them.
