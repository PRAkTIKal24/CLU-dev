"""Every mention of the CHLU / its parent unit / its physics in a draft (code only: LaTeX comments stripped).
usage: python3 chlu_scan.py DRAFT.tex  -> TSV rows: line <TAB> terms <TAB> sentence (the sentence containing the hit,
joined across source lines, whitespace collapsed). Terms: CHLU; bare CLU (not n-CLU / CLU-REL-as-part-of-n-CLU-REL);
Hamiltonian; symplectic; relativistic; governor; settling; field-mode; zero-dynamics; shorts; GMOR; retention law;
dissipative channel; jawahar2026chlu."""
import re, sys
TERMS = [('CHLU', r'CHLU'), ('CLU', r'(?<![-A-Za-z])CLU(?![-A-Za-z])'), ('Hamiltonian', r'[Hh]amiltonian'),
         ('symplectic', r'[Ss]ymplectic'), ('relativistic', r'[Rr]elativistic'), ('governor', r'[Gg]overnor'),
         ('settling', r'[Ss]ettling'), ('field-mode', r'field-mode'), ('zero-dynamics', r'zero-dynamics'),
         ('shorts', r'app:shorts|\bshorts\b'), ('GMOR', r'GMOR'), ('retention law', r'[Rr]etention law'),
         ('dissipative channel', r'[Dd]issipative channel'), ('cite', r'jawahar2026chlu')]
def code(l):
    m = re.search(r'(?<!\\)%', l); return l[:m.start()] if m else l
def scan(path):
    L = open(path).read().split('\n'); C = [code(l) for l in L]
    flat = ''; pos = []  # char offset -> line
    for i, c in enumerate(C):
        pos.append(len(flat)); flat += c + '\n'
    import bisect
    def line_of(o): return bisect.bisect_right(pos, o)
    # sentence boundaries: ". " / "? " / blank line / \paragraph / \item / \section / \caption
    bounds = [0] + [m.end() for m in re.finditer(r'(?<!\b[A-Z][a-z])(?<!e\.g)(?<!i\.e)(?<!\bet al)(?<!~)[.?!]\}?\s|\n\s*\n|\\item\b|\\(sub)*section\{|\\paragraph\{|\\caption\{', flat)] + [len(flat)]
    out = []; seen = {}
    for name, rx in TERMS:
        for m in re.finditer(rx, flat):
            k = bisect.bisect_right(bounds, m.start()) - 1
            key = (bounds[k], bounds[k + 1]); seen.setdefault(key, []).append((m.start(), name))
    for (a, b), hits in sorted(seen.items()):
        s = ' '.join(flat[a:b].split())
        terms = sorted(set(h[1] for h in hits)); ln = line_of(min(h[0] for h in hits))
        out.append((ln, ','.join(terms), s))
    return out
if __name__ == '__main__':
    for ln, t, s in scan(sys.argv[1]): print(f'{ln}\t{t}\t{s}')
