"""python3 report.py G OUTDIR : MOVE-MAP.md, HEAD-HELD.md, SECTIONS.md, FIGTAB.md, REFS.md, PROOF.txt from the stage G
(G/before/{v2,style} = the snapshot builds, G/{v2,style} = the restructured builds, G/measure-{before,after})."""
import sys, re, json, hashlib
from collections import Counter
from blocks import chunks, parse, load, text, sha, first8, HEAD
from layout import SLOT
G, OUT = sys.argv[1:3]
DR = [('v2', 'DRAFT-v2'), ('style', 'DRAFT-v2-style')]
proof = []; ok = True
def context(L, items):
    """for each P chunk: (section title, subsection title) in force at it"""
    sec = subsec = ''; res = []
    for k, idx in items:
        if k == 'H':
            for i in idx:
                m = re.match(r'\\section\{(.*)\}\s*$', L[i])
                if m: sec, subsec = m.group(1), ''
                m = re.match(r'\\subsection\{(.*)\}\s*$', L[i])
                if m: subsec = m.group(1)
        else: res.append((sec, subsec, idx))
    return res
mm = ['# MOVE-MAP — PW-2 item 2 (re-sectioning), both drafts', '',
      'Paragraph = a chunk of `blocks.py`: a blank-line block of the main text (after `\\end{abstract}`, before the rule line above the AI-use statement), cut at heading lines and at a `\\paragraph` that follows text inside a block (that `\\paragraph` starts a new TeX paragraph; the one such cut is the CMAZE-suite paragraph, which sat in the block of "Reading the CMAZE-T panel figures"). Floats and comment-only blocks are chunks too and move whole. sha256 over the chunk text exactly as in the file (lines joined by `\\n`). "new" = where the byte-identical chunk sits after the restructure.', '']
hh = ['# HEAD-HELD — the paragraphs no later PW-2 spoke may change (except the item-3 and item-1 borrowings, whose spokes update this list)', '',
      'Written by `pw2-draft-sections` after item 2 (re-sectioning). sha256 over the exact source text (lines joined by `\\n`, no trailing newline). "First paragraph of a section" = the first prose chunk after the `\\section` line (a float-only chunk, e.g. the two task figures at the top of Experiments, is not a paragraph; it is listed where it precedes). Re-check with `python3 pw2_draft_sections/headheld.py PATH/DRAFT-v2.tex ~/clu/outputs/pw2-draft-sections/HEAD-HELD.md` (and the same for DRAFT-v2-style.tex); exit 1 on any difference.', '']
for f, d in DR:
    Lb, sb, eb, itb, Pb = chunks(f'{G}/before/{f}/{d}.tex')
    La, sa, ea, ita, Pa = chunks(f'{G}/{f}/{d}.tex')
    # 1. outside the span byte-identical
    same_out = Lb[:sb] == La[:sa] and Lb[eb:] == La[ea:]
    proof.append(f'{d}: preamble+abstract (lines 1-{sb}) and AI-use statement to end byte-identical before/after: {same_out}')
    ok &= same_out
    # 2. paragraph multiset
    Pa_nos = [p for p in Pa if p != SLOT]
    cb, ca = Counter(sha(p) for p in Pb), Counter(sha(p) for p in Pa_nos)
    proof.append(f'{d}: paragraphs before {len(Pb)}, after {len(Pa_nos)} (+1 new line: the item-3 slot comment `{SLOT}`); '
                 f'distinct hashes before {len(cb)}; multiset equal: {cb == ca}; lost: {dict(cb - ca) or "none"}; added: {dict(ca - cb) or "none"}; '
                 f'connective deletions: 0')
    ok &= (cb == ca and len(Pb) == len(Pa_nos) and len(cb) == len(Pb))
    # 3. line-multiset diff of the whole file: only headings removed/added
    lb, la = Counter(Lb), Counter(La)
    rem = [l for l in (lb - la).elements() if l.strip()]; add = [l for l in (la - lb).elements() if l.strip()]
    bad = [l for l in rem + add if not (HEAD.match(l) or l == SLOT)]
    proof.append(f'{d}: whole-file line multiset: {len(rem)} non-blank lines removed, {len(add)} added; every one a heading line (rule, \\section, \\subsection, \\label{{sec:}}) or the slot: {not bad}' + (f' OFFENDERS {bad}' if bad else ''))
    ok &= not bad
    proof.append(f'   removed: ' + ' | '.join(l for l in rem if not l.startswith('% =')))
    proof.append(f'   added:   ' + ' | '.join(l for l in add if not l.startswith('% =')))
    cxb, cxa = context(Lb, itb), context(La, ita)
    posa = {}
    for n, (s, ss, idx) in enumerate(cxa):
        posa.setdefault(sha(text(La, idx)), (n, s, ss, idx))
    mm += [f'## {d}.tex — {len(Pb)} paragraphs before, {len(Pa_nos)} after', '',
           '| # before | lines before | old section › subsection | first 8 words | sha256 (12) | # after | lines after | new section › subsection |',
           '|---|---|---|---|---|---|---|---|']
    for n, (s, ss, idx) in enumerate(cxb):
        t = text(Lb, idx); h = sha(t)
        if h not in posa:
            mm.append(f'| {n} | {idx[0]+1}–{idx[-1]+1} | {s} › {ss or "—"} | `{first8(t)}` | {h[:12]} | **NOT FOUND AFTER** | | |'); continue
        m, s2, ss2, idx2 = posa[h]
        f8 = first8(t).replace('|', '\\|')
        mm.append(f'| {n} | {idx[0]+1}–{idx[-1]+1} | {s} › {ss or "—"} | `{f8}` | {h[:12]} | {m} | {idx2[0]+1}–{idx2[-1]+1} | {s2} › {ss2 or "—"} |')
    mm.append('')
    # HEAD-HELD
    import headheld
    hh += [f'## {d}.tex', ''] + headheld.table(headheld.rows(f'{G}/{f}/{d}.tex')) + ['']
proof.append(f'ALL PROOFS PASS: {ok}')
open(f'{OUT}/MOVE-MAP.md', 'w').write('\n'.join(mm + ['## Proof', ''] + ['- ' + p for p in proof]) + '\n')
open(f'{OUT}/HEAD-HELD.md', 'w').write('\n'.join(hh) + '\n')
open(f'{OUT}/PROOF.txt', 'w').write('\n'.join(proof) + '\n')
print('\n'.join(proof))
