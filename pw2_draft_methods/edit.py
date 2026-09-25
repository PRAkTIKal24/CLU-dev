"""PW-2 item 3 on one draft: (1) the Head's Methods opening, verbatim, at % PW2-ITEM3-SLOT (LaTeX only: each
"Jawahar \\& Pierini (2026)" -> \\citet{jawahar2026chlu}); (2) the citation-scope rule's one addition:
~\\citep{jawahar2026chlu} at the first mention of the unit outside Methods ("zero-dynamics limit of the CLU", Sec. clue).
Each anchor must occur exactly once. usage: python3 edit.py IN.tex OUT.tex"""
import sys
PLAN = ("Jawahar \\& Pierini (2026) introduced the CHLU as a latent dynamics model (defined here as an algorithm that either "
        "enforces or extracts the dynamics of the latent space itself to either design or approximately understand semantic "
        "rules created by the encoder that produced this abstract space such that downstream tasks can be appropriately dealt "
        "with or downstream design choices can be better informed). Jawahar \\& Pierini (2026) envision their unit (as the "
        "name suggests) as a step that provides a clue towards developing better latent dynamics models. We build on top of "
        "this idea and drop the complex physics design to construct our naive-CLU as a special case of the CHLU and test it "
        "as a factorized memory unit for compositional generation with additional properties such as deletion as a virtue "
        "of the design itself.")
CITE = r'\citet{jawahar2026chlu}'
def wrap(s, w=100):
    out, cur = [], ''
    for word in s.split(' '):
        if cur and len(cur) + 1 + len(word) > w: out.append(cur); cur = word
        else: cur = (cur + ' ' + word) if cur else word
    return out + [cur]
NEW = '\n'.join(['% PW2 item 3 (PLAN.md sec. 2): the Head\'s Methods opening, verbatim; LaTeX only (\\citet = "Jawahar & Pierini (2026)").']
                + wrap(PLAN.replace('Jawahar \\& Pierini (2026)', CITE)))
SLOT = '% PW2-ITEM3-SLOT'
A_OLD = 'zero-dynamics limit of the CLU:'
A_NEW = 'zero-dynamics limit of the CLU~\\citep{jawahar2026chlu}:'
if __name__ == '__main__':
    t = open(sys.argv[1]).read()
    L = t.split('\n')
    assert [l for l in L if l == SLOT] == [SLOT], 'slot not exactly once'
    assert t.count(A_OLD) == 1, ('anchor count', t.count(A_OLD))
    assert PLAN.count('Jawahar \\& Pierini (2026)') == 2
    t = t.replace('\n' + SLOT + '\n', '\n' + NEW + '\n', 1).replace(A_OLD, A_NEW, 1)
    open(sys.argv[2], 'w').write(t)
    print('ok', sys.argv[2])
