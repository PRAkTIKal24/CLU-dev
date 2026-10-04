
---

## How the style rules were applied (hand-written; not generated)

**Method.** `DRAFT-v2-style.tex` is built mechanically by `apply_patches.py`: it is `DRAFT-v2.tex` with 92 declared prose line ranges (1,049 of 2,152 source lines) replaced by paraphrases in `patches/*.tex`. Every line outside a declared range is byte-identical to `DRAFT-v2.tex`. `apply_patches.py` refuses to build if any single range loses or gains a number token, a provenance comment, a `\PLACEHOLDER{}` or a `\TODO{}`. The four checks above then run on the whole file.

**Rules from `PJ_Writing_Style_Context.pdf`** (the rules only; its thesis content and HEP persona were ignored as instructed):

| rule | where applied |
|---|---|
| ABT at section openings | Abstract (setup: three requirements; *however*: the intersection is empty; *therefore*: the Naive CLU). §1.1 (the two literatures; *however*: the screen finds the intersection empty; *therefore*: the position). §1.2 (the three properties; *however*: they pull against each other; *therefore*: one store). §2 (the claim is only as precise as its definitions; *therefore*: fix them first). §4 (§3 tests images; *however*: routes are the harder test; *therefore*: the maze). §4.2 (CMAZE-2; *but*: blind-sequential passes; *therefore*: CMAZE-2b). §4.3 (the read averages; *however*: routes compose by union; *therefore*: vary only the verb). §5.4 leaving note (*however*: what it does not buy). §6 (*therefore*: the follow-up). The ABT connectives only link content already in the draft; no opener adds a claim, number or result. |
| Macro-to-micro | Section openers state the section's scope before its first number (§2, §3, §4, §5); definitions come before measurements within §2. |
| Objective tone, no hyperbole | No magnitude word was added. The draft's own magnitude words ("sharply", "poorly", "well below 1", "essentially unchanged") are kept, each next to the numbers that support it. |
| Signposting | Section openers for §2, §3, §4, §5; "This appendix states…" (App. B); "this subsection reports what it buys" (App. T). |
| Scoping bullets when a framework is introduced | §1.2 the three properties; §2 the three deletion steps; §4 the five environments; App. Q the four status labels. Four other candidate lists (the ledger parts, interface split, deletion scopes, maze-wm protocol) were tried as bullets and returned to inline prose to keep the main text within the ≤12 pp bound (see LOG.md). |
| "We" for our actions, passive for established facts | e.g. "We declare the frozen encoder and decoder outside…", "We ran the store's exact deletion…", "We downdate the full, untruncated statistic…"; standard facts stay passive ("an external classifier … judges", "the atoms are re-eigendecomposed"). |
| LaTeX for all symbols | The source has 0 non-ASCII characters; every symbol is LaTeX. One pre-existing token, `2e$-$7` (App. P), is kept as written because the no-number-change constraint outranks restyling it as `\sci{}`. |
| Bold only for new core concepts | Run-in bold item heads in the contribution list, the limitations list, App. Q and App. V are now `\emph`; the thickness note is `\emph`. Bold is kept or added only where a core concept is defined: *Rule validity*, *Composition*, *Exact deletion*, *interface split*, *holes*, *Latent Dynamics Model*, and the environment names CMAZE-1/-2/-2b/-T/-O. |
| Succinctness | Four tightening passes (`tighten.py`, `tighten2.py`, `tighten3.py`, `tighten4.py`, 66 phrase-level edits plus one antecedent fix ("The class's laws", §5.3), all conservation-checked) brought the main text back within 12 pp. |

**Deliberately left verbatim** (content-bearing, or with no sentence structure to paraphrase): every `table` float and `\figslot` with its caption (28, byte-checked above); the preamble; all section/subsection/paragraph headings (96, order-checked above); the `\textsc{Exact is store-scoped}` definition block; the two emphasised verbatim claims (the ladder inference note and "The store has forgotten red…"), plus the emphasised "dynamics results about our own store" disclaimers; every `\PLACEHOLDER{}` and `\TODO{}`; the data-list paragraphs (K=5-ward deletion list in App. F, CMAZE-2 pins in App. K, "The leg." in App. M); the ten requirements of App. R; the placeholder index (App. W); the declared not-run items (App. X); and the corrections list (App. Y).

**Layout.** pdflatex against `iclr2027/`: 31 pp, 0 errors, 0 undefined references, 0 overfull boxes; the 31 warnings are the same 31 that DRAFT-v2 itself produces (hyperref PDF-string tokens, `h`→`ht` float specifiers). §6 ends at ruler line 646 on p. 12, which matches DRAFT-v2 (main text ≤ 12 pp).

**Provenance.** The draft-v2 provenance checker (`check_provenance.py`, unmodified copy) returns the same result on both files: 1,824 number tokens checked; the same single tokenizer artefact (the set `\{5,8,…,32\}` in the K* rule); and the same 10 unsourced lines, all template years or wave labels. So every number still sits under an SRC comment whose file contains it.
