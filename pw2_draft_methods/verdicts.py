"""CHLU-mention table with the citation-scope verdict (brief item 2), one draft. usage: python3 verdicts.py DRAFT.tex
Rules (PLAN sec. 2 item 3; brief): jawahar2026chlu only for the unit, its symplectic Hamiltonian dynamics, the relativistic
governor; never for memory, deletion, test-time compute, retention, GMOR; the physics laws are this paper's own theory;
the shorts never cited. Verdicts: cite (added here) / cite (item 3 text) / do not cite / already fine / not the CHLU / VIOLATION.
Each verdict is assigned by the first matching rule below; a row no rule matches is printed UNCLASSIFIED (exit 1)."""
import sys, re
from chlu_scan import scan
BANNED = r'[Mm]emory|[Dd]elet|[Rr]etention|GMOR|test-time|[Dd]issipative|shorts'
RULES = [
 (r'\\citet\{jawahar2026chlu\} introduced', 'cite (item 3 text)', 'the unit; author-as-noun, verbatim Head text'),
 (r'\\citet\{jawahar2026chlu\} envision', 'cite (item 3 text)', 'the unit (their envisioned role for it)'),
 (r'special case of the CHLU', 'already fine', 'no citation; memory and deletion are attributed to "we" (this paper), not to the arXiv paper'),
 (r'limit of the CLU~\\citep\{jawahar2026chlu\}', 'cite (added here)', 'first mention of the unit and its dynamics outside Methods; the cite follows "the CLU" only, the identity is this study\'s measurement'),
 (r'settling \(physics\) CLU', 'already fine', 'the unit again; allowed but not re-cited (first mention outside Methods carries it)'),
 (r'Hamiltonian neural networks', 'not the CHLU', 'Greydanus et al. HNNs, cited to their own paper'),
 (r'zero-dynamics limit of a settling unit', 'do not cite', 'abstract: Head-owned and uncited by ICLR convention; the identity is this study\'s measurement'),
 (r'zero-dynamics identity|zero-dynamics clue\}|zero-dynamics identity: conditions', 'do not cite', 'this study\'s own measured identity (heading / signpost)'),
 (r'this paper\'s own theory', 'already fine', 'states the laws are this paper\'s own theory; App. "Prior derivations" is named, not cited'),
 (r'Prior derivations\}', 'already fine (flag)', 'shorts-derived laws (memory-level deletion, retention, GMOR, dissipative channel ...); no citation, the shorts never cited. FLAG: "prior derivation" / "before this study" may now be read as the arXiv paper cited in Methods; no sentence says so'),
 (r'GMOR', 'do not cite', 'GMOR: never jawahar2026chlu; cited to Gell-Mann et al. (1968) for the relation itself; the checkpoint result is this paper\'s theory (App. Prior derivations)'),
 (r'Dissipative channel|retention law holds', 'do not cite', 'dissipative channel / retention law: this paper\'s own theory'),
 (r'retention law|curvature--retention|field-mode', 'do not cite', 'this study\'s field-mode n-CLU / retention law: own measurement or theory'),
 (r'full settling|zero-step arm', 'do not cite', 'this study\'s measurement'),
]
def rows(path):
    out = []
    for ln, terms, s in scan(path):
        v = next(((vv, why) for rx, vv, why in RULES if re.search(rx, s)), ('UNCLASSIFIED', ''))
        if 'jawahar2026chlu' in s and (not v[0].startswith('cite') or re.search(BANNED, s)):
            v = ('VIOLATION', 'jawahar2026chlu cited in a sentence outside the allowed ones, or next to a banned topic')
        out.append((ln, terms, s, v[0], v[1]))
    return out
if __name__ == '__main__':
    r = rows(sys.argv[1]); name = sys.argv[1].split('/')[-1]
    print(f'### {name}\n\n| line | terms | sentence (first 160 chars) | verdict | reason |\n|---|---|---|---|---|')
    for ln, t, s, v, why in r:
        s2 = (s[:160] + ('…' if len(s) > 160 else '')).replace('|', '\\|')
        print(f'| {ln} | {t} | `{s2}` | **{v}** | {why} |')
    from collections import Counter
    c = Counter(x[3] for x in r); print('\ncounts:', dict(c))
    sys.exit(1 if c.get('UNCLASSIFIED') or c.get('VIOLATION') else 0)
