# FINDINGS — pw2-draft-sections (PW-2 item 2: plain headings and re-sectioning, both drafts)

Session `pw2-draft-sections`, branch `pw2/draft-sections`, batmac. Start 2026-09-25T15:36:38Z. Stage `stage-20260925T154128Z`, snapshot `snap-20260925T154128Z` (sha256 in `snap-20260925T154128Z/SHA256`).

## Status: item 2 DONE, both drafts, installed

| acceptance point | result |
|---|---|
| main text = Introduction → Methods → Experiments → Results → Discussion, then References, then Appendices | done, both drafts, identical heading sequence (style diff-check §4: 101 headings, 67 labels, identical) |
| paragraphs moved, not rewritten | v2 62 → 62, style 64 → 64; multiset of sha256 equal; 0 lost, 0 duplicated, **0 connective deletions** (no moved paragraph opened with a dangling connective); whole-file line diff = heading lines only, plus the slot line (`PROOF.txt`) |
| only new text = the five headings, plain subsection headings, `% PW2-ITEM3-SLOT` | yes; every added/removed line listed in `PROOF.txt` |
| Methods slot at the very top | `% PW2-ITEM3-SLOT` is the first line after `\section{Methods}\label{sec:object}` (line 273, both drafts) |
| AI use / ethics / reproducibility statements where the template puts them; appendix headings and order unchanged | yes: everything from the rule line above the AI-use statement to `\end{document}`, and the preamble + abstract, byte-identical before/after (`PROOF.txt`) |
| abstract untouched; maze `\TODO` byte-identical | yes (abstract bytes identical; its `\S\ref{sec:maze-wm}` now renders §4.8 instead of §4) |
| 0 undefined references / citations, 0 errors | both drafts: exit 0, ucit 0, uref 0, errors 0, multiply-defined 0, overfull 0, underfull 20 (= before), 55 bibitems, no Rerun |
| style diff-check EMPTY | ALL DIFFS EMPTY: numbers multiset, placeholders, artefact paths (and every SRC comment whole), section order; every float byte-identical and in the same order; preamble identical (`STYLE-DIFF-CHECK.md`) |
| provenance findings identical or improved | identical to before in both drafts, line numbers masked (v2: 3 unsourced lines, 9 not-found tokens; style: 3 and 10, all pre-existing) |
| `\PLACEHOLDER` / `\TODO` unchanged | unchanged: no chunk changed, and outside the main text every byte is identical |
| figure / table numbering | unchanged: 13 figures, 21 tables, 0 renumbered (`FIGTAB.md`); only pages moved |
| installed byte-identically into both live folders | `.tex` and PDF in `~/clu/pw2/live/{v2,style}`, cmp-equal to the staged build; live was cmp-equal to the snapshot just before; sha256 in `INSTALLED-SHA256.txt`; a fresh build from a copy of the live folders gives the same `\newlabel`s, 43 / 44 pages |
| bib | `refs_clu_iclr.bib` sha256 b2cf715f… at start and end, both folders; never written |

Deliverables in this directory: `MOVE-MAP.md` (every paragraph, first 8 words + sha256, old → new section, both drafts, with the proof), `HEAD-HELD.md` (+ `pw2_draft_sections/headheld.py` for the later spokes), `SECTIONS.md` (section index + page counts before → after), `FIGTAB.md` / `FIGTAB-INDEX.json` (number → label → caption lead → page → section), `REFS.md` (every section label's number before → after, with every `\ref` to it), `STYLE-DIFF-CHECK.md`, `PROOF.txt`, `LOG.md`.

## The new main text (v2; the style twin is identical in structure)

| # | heading | label(s) | paragraphs (old home) |
|---|---|---|---|
| 1 | Introduction | sec:intro | — |
| 1.1 | Positioning among naive units | sec:positioning | 5 (old 1.1) |
| 1.2 | The three properties | sec:why | 1 (old 1.2) |
| 1.3 | Contributions | sec:contrib | contribution list, "Thickness" (old 1.3) |
| 2 | Methods | sec:object | `% PW2-ITEM3-SLOT`; [style only: its Sec.-2 signpost "A claim about any of the three properties …"]; The n-CLU; The read; The memory footprint; Why a hold-out draw can remove a whole colour relation (old 4); The judge and the estimand; The retained statistic and deletion; the deletion-scope quote; the frozen-encoder scope paragraph (old 2) |
| 3 | Experiments | sec:experiments (new) | the two task figures (old 2) |
| 3.1 | The Shapes3D value hold-out | sec:shapes | the $\Kstar$ selection paragraph (old 3 lead) |
| 3.2 | Baselines and the matching rule | — | Baselines: naive against naive (old 2); A second, earlier baseline set (old 3.1) |
| 3.3 | The $n$-way ladder protocol | — | "The ladder asks whether …" (old 3.2) |
| 3.4 | The deletion protocol | — | "Three deletions are run …" (old 3.3) |
| 3.5 | The CMAZE-T transition task | sec:maze | "This section asks the n-CLU to generate …"; CMAZE-T, a minor contribution (old 4) |
| 4 | Results | sec:results (new) | — |
| 4.1 | Composition on Shapes3D | sec:shapes-t7 | Tab. t7 text, Tab. 1, baselines-matched note, "What the counts look like", Figs. 3–4, Tab. 2 footprint (old 3.1) |
| 4.2 | The $n$-way ladder | sec:ladder | Tab. 3, inference note, coverage control, Fig. 5, the retired-slot comment (old 3.2) |
| 4.3 | Deletion on Shapes3D | sec:shapes-del | Tab. 4, "After deletion …", conjunction paragraph, probe (old 3.3) |
| 4.4 | Deletion against retrained baselines | sec:del-comparators | (old 3.4) |
| 4.5 | Refusal through the hole | sec:hole | refusal paragraph, Tab. 5, comparator paragraph (old 3.5) |
| 4.6 | Two ablations | sec:ablations | (old 3.6) |
| 4.7 | Encoder adaptation and generative baselines | sec:adapt, sec:bard | encoder, LACE and PoE, generation table (old 3.7) |
| 4.8 | CMAZE-T conditions and relational coding | sec:maze-wm | symmetric inputs (six seeds), relational request coding, what is not claimed, reading the panel figures (old 4) |
| 5 | Discussion | sec:limits | [style only: its Sec.-5 signpost "We now state what the naive unit does not do …"] |
| 5.1 | Limitations | sec:limits-list | the limitations list (old 5.1) |
| 5.2 | A planned CMAZE suite | — | Toward a CMAZE suite (old 4) |
| 5.3 | The zero-dynamics clue | sec:clue | (old 5.2) |
| 5.4 | The Latent Dynamics Model class | sec:theory | (old 5.3) |
| 5.5 | Summary | sec:leaving | the leaving-note paragraph (old 5.4) |
| 5.6 | Physics follow-up | sec:followup | follow-up paragraph, Deletion future-work paragraph (old 6) |

Pages (`SECTIONS.md`): v2 42 → 43 pages, main text 17.40 → 17.67 pages; style 44 → 44, 17.71 → 17.86. The growth is the new headings (7 more numbered headings, 27 against 20, several stacked: "3 Experiments / 3.1", "4 Results / 4.1", "5 Discussion / 5.1") and float placement; no trim was made (the nine-page cut is the Head's).

## Placement decisions (each logged; open to the Head)

1. **Mixed method/result paragraphs, kept whole** (no sentence split):
   - $\Kstar$ selection (rule + plateau + footprint minimum) → Experiments 3.1: main content is the frozen selection rule.
   - "Tab. t7 is the scored result …" (hold-out sizes + control check) and the matching-protocol-plus-margins block → Results 4.1: the margins dominate (3 result sentences against 2 protocol sentences).
   - "The ladder asks whether …" (rungs, the two baseline sets, one control-collapse sentence) → Experiments 3.3.
   - Refusal paragraph (the hole mechanism with its 2-byte / 0-row cost) → Results 4.5, kept with Tab. 5 and the comparator paragraph.
   - "CMAZE-T: what is not claimed" (rollouts, two-way pairs, the asymmetric row, maze deletion numbers) → Results 4.8: it prints measurements.
   - "Reading the CMAZE-T panel figures" → Results 4.8 (it explains result figures; it also defines how the judge scores the four facts).
2. **"Why a hold-out draw can remove a whole colour relation" → Methods**, as the relational design (n-CLU-REL): the plan's mapping lists "the relational design (n-CLU-REL)" in Methods and this is the only paragraph that states what the relational columns add without results. It could equally sit in Experiments 3.5 (CMAZE-T hold-out families); in Methods it uses pocket/tint before 3.5 introduces them. The sentence that *defines* n-CLU-REL ("adding colour-equality indicators crossed with the door and the direction") sits inside the relational result paragraph, which stays whole in Results 4.8.
3. **Order inside Methods** follows the plan's list (object: n-CLU, read, footprint → relational design → judges → deletion by downdate). The footprint paragraph names "the covariance and count statistic" one paragraph before "The retained statistic and deletion" defines it.
4. **Order inside Experiments**: Shapes3D hold-out → baselines and matching rule → ladder → deletion protocol → CMAZE-T. The plan lists baselines after CMAZE-T; they were put second because the ladder paragraph uses the landscape controls and the data-matched sets, which the baselines paragraph defines.
5. **Discussion order**: Limitations → CMAZE suite → clue → theory → summary (leaving note) → physics follow-up. The follow-up stays last because the style twin's follow-up paragraph opens "The leaving note states …".
6. **"Thickness, stated in the introduction"** stays in Introduction 1.3 (it says so of itself); there is no separate seeds paragraph (see gaps).
7. **Appendix floats were not moved.** The maze conditions figure (`fig:maze-wm`, Fig. 9) and `tab:cmazet` / `tab:cmazet-sym` sit in App. N and stay there: the brief fixes appendix content and order, and main ↔ appendix placement is the Head's page decision. Results 4.8 therefore has no float of its own; it cites Tab. 20 and Fig. 9 in App. N.
8. **Headings (tell 5)**: every new heading is a noun phrase, no colon, no parenthetical. "Leaving note" became "Summary" (tell 6 names "leaving note" as a coinage). `\paragraph{}` run-in titles are part of their paragraphs and were not touched. Several still carry colons or parentheticals (e.g. "CMAZE-T, a minor contribution: a small world-model task for composition.", "Composition-only n-CLU (composes, cannot forget).", "The memory footprint, never netted."). They are item 1's to fix.

## Rendered cross-references that now point somewhere slightly different (no source change made; `REFS.md`)

- `sec:object` = Methods (2). The Introduction's first paragraph says baselines are "matched on the read path (Sec. 2)", but the matching rule now sits in Experiments 3.2, which has no label. That paragraph is Head-held, so it was not retargeted.
- `sec:limits` = Discussion (5): "incomplete (Sec. 5)" (novelty risk, now 5.1 item 10) and contribution 4's "Secs. 5–5.6" (was "Secs. 5–6").
- `sec:maze-wm` = 4.8 (was 4), `sec:shapes` = 3.1 (was 3; cited once, from App. D, for the $\Kstar$ rule, which is now there). Everything else keeps its meaning, with new numbers.

## Head-held paragraphs after this pass (`HEAD-HELD.md`)

- v2's Discussion opens with the limitations list; the style twin's Discussion opens with its own signpost sentence. As a result, the whole limitations list is Head-held in v2. The Head may want a prose opener there, as the item-3 slot provides for Methods.
- Experiments opens (in both drafts) with the two task figures, then 3.1; its first prose paragraph is the $\Kstar$ paragraph (style: "This section measures the three properties on Shapes3D. …").
- Other openers that item 1 will meet: tell 7 bans a "This section" opener. Two moved paragraphs open that way: "This section asks …" (3.5, both drafts) and, in style only, "This section measures …" (3.1, Head-held). Style's Methods signpost "we therefore fix these here, before any result" still reads correctly. Style's 3.1 opener now sits in Experiments while the measurements it announces are in Results.

## Gaps in the mapping (content the plan names that has no whole paragraph to move)

- **Seeds** (Experiments): no stand-alone seeds paragraph exists. The facts are in "Thickness" (Intro 1.3), the judge paragraph ("Seed count is a column of every table") and the limitations list (item 8).
- **Seed-5 and relational limits** (Discussion): both sit inside Results paragraphs (the symmetric-input paragraph and the relational paragraph), which were kept whole in Results 4.8. Discussion carries them only through limitation 1. Moving them would mean splitting sentences out of paragraphs, which the brief forbids.
- **n-CLU-REL definition** in Methods: see decision 2.

## Positive controls (the proof tooling catches what it must)

CTRL1: one character changed in a moved paragraph → multiset proof FAIL, line-diff offender named, "NOT FOUND AFTER" in the map. CTRL2: a dropped chunk → `restructure.py` asserts. CTRL3: a duplicated chunk → asserts. CTRL4: one added space in a Head-held paragraph → `headheld.py` FAIL (exit 1). CTRL5: two subsection headings swapped in the style twin → style diff-check exit 1, heading sequence False. The unrestructured "before" drafts also fail `headheld.py` (sections not found).

## What did NOT run

- No other PW-2 item (3, 5, 1, 4) was touched. The Methods slot is empty and is item 3's.
- No paragraph was rewritten. Heading-level tell-5 fixes to `\paragraph{}` run-in titles, the "This section" openers, and ref retargeting inside Head-held text were left alone (item 1 / the Head).
- No float moved between the main text and the appendix (decision 7). No appendix heading was changed.
- No visual page-by-page review of the PDF was done beyond the text-extracted heading sequence (`stage-20260925T154128Z/v2-after.txt`, `style-after.txt`) and the float index.
- No compute, no fits, no GPU. The bib, `outputs/w17-paper`, `outputs/pw1-*`, `outputs/w*/`, TABLES.md, the registry and the claims matrix were not written.

## Branch and bundle
Tooling commit 6b431af on `pw2/draft-sections`; bundle `pw2-draft-sections.bundle` (sha256 3a56ffa4…), `git bundle verify` okay, complete history, head refs/heads/pw2/draft-sections = 6b431af. Outputs are not committed.
