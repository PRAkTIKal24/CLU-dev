#!/usr/bin/env python3
"""PW-1 draft sweep (plan items 2, 3, 5 and "the engine's judge").

usage: sweep.py IN.tex OUT.tex COUNTS.json

Only rendered LaTeX is touched. Every LaTeX comment (from the first unescaped %), every
\\label/\\ref/\\cite/\\nocite argument, every \\figslot file stem and the abstract's CMAZE-T \\TODO
(held by the Head) are masked before any rule runs and restored byte-identically afterwards.
Rules run in order: specific phrases first (economic metaphors, one by one, in context), then the
generic token rules. Line count is preserved: when a match spans a newline, the replacement keeps
a newline at the last word gap. Every rule's hit count is written to COUNTS.json, and every hit of
a rule in the 'metaphor', 'judge' and 'special' classes is logged with its surrounding sentence.
"""
import re, sys, json
from texnum import split_comment

SRC, DST, CNT = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(SRC).read()

# ---------------------------------------------------------------- masking
masks = []
def mask(s):
    masks.append(s)
    return '\x02%d\x03' % (len(masks) - 1)

lines = text.split('\n')
out = []
for l in lines:
    code, com = split_comment(l)
    out.append(code + (mask(com) if com else ''))
text = '\n'.join(out)
# the Head's abstract TODO (CMAZE-T sentence), whole
m = re.search(r'\\TODO\{CMAZE-T sentence reserved for the Head:.*?no\s+number is written here\.\}', text, re.S)
assert m, 'abstract TODO not found'
text = text[:m.start()] + mask(m.group(0)) + text[m.end():]
text = re.sub(r'\\(label|ref|eqref|pageref|citep|citet|cite|nocite|bibliography|bibliographystyle|input|IfFileExists|includegraphics)\{[^}]*\}',
              lambda m: mask(m.group(0)), text)
text = re.sub(r'\\figslot\{[^}]*\}', lambda m: mask(m.group(0)), text)
text = re.sub(r'\\def\\input@path\{\{iclr2027/\}\{\./\}\}', lambda m: mask(m.group(0)), text)
text = re.sub(r'\\usepackage(\[[^]]*\])?\{[^}]*\}', lambda m: mask(m.group(0)), text)
text = re.sub(r'\\(newcommand|documentclass)\{[^\n]*', lambda m: mask(m.group(0)), text)

# ---------------------------------------------------------------- rules
W = r'\s+'   # word gap (may cross one newline)
counts = {}
log = []

def sentence(t, a, b):
    s = t.rfind('.', 0, a); s = 0 if s < 0 else s + 1
    e = t.find('.', b); e = len(t) if e < 0 else e + 1
    ctx = t[s:e]
    ctx = re.sub(r'\x02(\d+)\x03', lambda m: '' if masks[int(m.group(1))].startswith('%') else masks[int(m.group(1))], ctx)
    return re.sub(r'\s+', ' ', ctx).strip()

def R(rid, cls, pat, rep, flags=0, first_after=None):
    global text
    rx = re.compile(pat.replace(' ', W) if not pat.startswith('RAW:') else pat[4:], flags)
    n = 0
    start = 0
    if first_after:
        start = text.find(first_after)
        assert start >= 0, first_after
    pieces, pos = [], 0
    for m in rx.finditer(text, start):
        if callable(rep):
            r = rep(m)
        else:
            try:
                r = m.expand(rep)      # raw-string templates (\\1, \\\\emph ...)
            except re.error:
                r = rep                # plain strings holding single LaTeX backslashes
        # keep the line structure: the replacement carries exactly as many newlines as the match,
        # each placed at a word gap at the same relative position as in the match
        r = r.replace('\n', ' ')
        g = m.group(0)
        for j in [i for i, c in enumerate(g) if c == '\n']:
            target = int(round(j / max(len(g), 1) * len(r)))
            gaps = [i for i, c in enumerate(r) if c == ' ']
            if not gaps:
                r = r + '\n'
                continue
            k = min(gaps, key=lambda i: abs(i - target))
            r = r[:k] + '\n' + r[k+1:]
        if cls in ('metaphor', 'judge', 'special', 'budget', 'config', 'prereg', 'ledger', 'gate'):
            log.append({'rule': rid, 'class': cls, 'before': re.sub(r'\s+', ' ', m.group(0)), 'after': re.sub(r'\s+', ' ', r),
                        'sentence_before': sentence(text, m.start(), m.end())})
        pieces.append(text[pos:m.start()]); pieces.append(r); pos = m.end(); n += 1
        if first_after:
            break
    pieces.append(text[pos:])
    text = ''.join(pieces)
    counts[rid] = counts.get(rid, 0) + n
    return n

# ===== title and first mentions (item 2) =====
R('title', 'special', r'RAW:\\title\{Exactly Deletable Compositional Generation\\\\ on a Frozen Encoder\}',
  r'\\title{naive-CLU: Exactly Deletable Compositional Generation\\\\ on a Frozen Encoder}')
R('first-mention-abstract', 'special', r'the \\emph\{Naive CLU\}:', r'the \\emph{naive-CLU} (n-CLU):')
R('first-mention-intro', 'special', r'The Naive CLU is a', r'The naive-CLU (n-CLU) is a')
R('naive-clu-other', 'special', r'\bNaive CLU\b', 'n-CLU')
R('abstract-label-free-store', 'special', r'label-free store of', 'label-free set of')

# ===== the engine's judge (pending resolved) =====
R('judge-def-v2', 'judge', r'an output is valid if an \\emph\{exact\} judge, integer arithmetic with no tolerance, accepts it',
  r"an output is valid if the task's judge accepts it (on the maze, the engine's judge, which uses integer arithmetic with no tolerance)")
R('judge-def-style', 'judge', r'an output is valid if an \\emph\{exact\} judge, which uses integer arithmetic with no tolerance, accepts it',
  r"an output is valid if the task's judge accepts it (on the maze, the engine's judge, which uses integer arithmetic with no tolerance)")
R('judge-two-senses', 'judge', r"Two senses of ``exact'' are kept apart: an exact \\emph\{judge\} is integer arithmetic with no tolerance; exact \\emph\{deletion\} is the store-scoped definition above\.",
  '')
R('judge-two-senses-style', 'judge', r"We keep two senses of ``exact'' apart: an exact \\emph\{judge\} is integer arithmetic with no tolerance; exact \\emph\{deletion\} is the store-scoped definition above\.", '')
R('judge-audit-style', 'judge', r'The audits find the judge exact in all four parts with no tolerance', "The audits find that the engine's judge checks all four parts with no tolerance")
R('judge-shapes-fig', 'judge', r'the exact judge is the hash-pinned external classifier', 'the judge is the hash-pinned external classifier')
R('judge-abstract', 'judge', r'valid\} under an exact judge', r"valid} under the task's judge")
R('judge-colour-maze', 'judge', r'with an exact integer judge', r"with the engine's integer judge")
R('judge-route-task', 'judge', r'A route task with an exact integer judge', r"A route task under the engine's integer judge")
R('judge-maze-judge', 'judge', r"the maze's exact judge", r"the engine's judge")
R('judge-also-with', 'judge', r'also with an exact judge', r"also under the engine's judge")
R('judge-exact-judge-route', 'judge', r'(An|an) exact-judge route task', lambda m: ('An' if m.group(1) == 'An' else 'an') + ' engine-judged route task')
R('judge-engine-judges-exactly', 'judge', r'the engine judges it exactly, with no tolerance', r"the engine's judge scores it with no tolerance")
R('judge-audit', 'judge', r'the judge is exact in all four parts with no tolerance', r"the engine's judge checks all four parts with no tolerance")
R('judge-exact-placement', 'judge', r'exact placement and legality', r"placement and legality under the engine's judge")
R('judge-exact-transition', 'judge', r'exact transition judge', r"the engine's transition judge")
R('judge-marks', 'judge', r'exact-judge marks', r"the engine's judge's marks")
R('judge-mark-fig', 'judge', r"the exact judge's mark", r"the mark of the engine's judge")
R('judge-header', 'judge', r'validity \(exact judge\)', r"validity (engine's judge)")
R('judge-under-the', 'judge', r'(under|by) the exact judge', lambda m: m.group(1) + " the engine's judge")
R('judge-caption-exact-judge', 'judge', r'unseen \$n\$-way, exact judge\.', r"unseen $n$-way, the engine's judge.")
R('judge-validity-check', 'judge', r'The validity check must be exact with zero tolerance\.', 'The validity check must use zero tolerance.')
R('judge-per-step', 'judge', r'per-step exactness', 'per-step accuracy')
R('judge-is-exact-row', 'judge', r'its generated \$s_\{t\+1\}\$ is exact and one on which it is not', r'its generated $s_{t+1}$ is judged valid and one on which it is not')
R('judge-no-exact-row', 'judge', r'have no exact \$n\$-way row at seed 0', r'have no valid $n$-way row at seed 0')
R('judge-generic-exact-judge', 'judge', r'\b(an|the) exact judge\b', r"the engine's judge")

# ===== economic metaphors (item 2), one by one =====
R('met-priced-cov', 'metaphor', r'Covariance priced on the', 'Covariance counted under the')
R('met-is-priced-on', 'metaphor', r'(is|are) priced on the', r'\1 counted under the')
R('met-price-alternative', 'metaphor', r"The comparator's job is to price the alternative to a hole, not to lose\.",
  r"The comparator's job is to measure what the alternative to a hole requires, not to lose.")
R('met-price-it-does-not-pay', 'metaphor', r'which is the price it does not pay and', 'which is the requirement it does not meet and')
R('met-price-arrives', 'metaphor', r'the price arrives through the channel', 'the loss arrives through the channel')
R('met-price-law', 'metaphor', r'the price law holds', 'the retention law holds')
R('met-priced-channel', 'metaphor', r'\\textbf\{Priced channel\.\}', r'\\textbf{Dissipative channel.}')
R('met-buys-exact-deletion', 'metaphor', r'The naive unit buys exact deletion', 'The naive unit provides exact deletion')
R('met-does-not-buy', 'metaphor', r'It does not buy composition', 'It does not provide composition')
R('met-budget-buys-latents', 'metaphor', r"read budget buys more latents", 'read size holds more latents')
R('met-buys-nothing', 'metaphor', r'initialisation buys nothing', 'initialisation changes nothing')
R('met-what-it-buys', 'metaphor', r'this is what it buys\.', 'this is what it yields.')
R('met-bytes-buys', 'metaphor', r'1,001,188 B buys 200,238', '1,001,188 B provides 200,238')
R('met-statistic-buys', 'metaphor', r"``the statistic buys deletion, not\s+composition''", "``the statistic provides deletion, not composition''")
R('met-paying-bytes', 'metaphor', r'while paying 81,312 to 84,216 bytes of read-side coupling the static read does not',
  'while incurring 81,312 to 84,216 bytes of read-side coupling that the static read does not')
R('met-paying-more', 'metaphor', r'while paying 81,312 B more read-side', 'while incurring 81,312 B more read-side')
R('met-pays-for-exactness', 'metaphor', r'it pays for exactness in bytes and in the time of the book\'s refit',
  "it incurs the memory and time cost of exactness, in bytes and in the book's refit")
R('met-is-paid-before', 'metaphor', r'is paid before refusal arrives', 'is incurred before refusal arrives')
R('met-cost-is-paid', 'metaphor', r'of the cost is paid through', 'of the loss is incurred through')
R('met-t-pay', 'metaphor', r'\$t_\{\\text\{pay\}\}\(50\\%\)\$', r'$t_{1/2}$')
R('met-expensive', 'metaphor', r'Exact is expensive in bytes\.', 'Exact requires a large number of bytes.')
R('met-cheapest-ledger', 'metaphor', r'where the ledger is\s+cheapest', 'where the memory footprint is smallest')
R('met-cheapest-point', 'metaphor', r'is not the cheapest point', 'is not the point of smallest footprint')
R('met-costs-composition-evict', 'metaphor', r'and costs (\$-2\.0899\$)', r'and changes composition by \1')
R('met-costs-composition-evict-b', 'metaphor', r'(\$-6\.7416\$) composition against a declared', r'\1 against a declared')
R('met-hole-costs-exactly', 'metaphor', r'refuses at 1\.0000 and costs exactly', 'refuses at 1.0000 and changes it by exactly')
R('met-cost-free', 'metaphor', r'never cost-free erasure', 'never erasure without loss')
R('met-item-store-costs', 'metaphor', r'on what an item store\s+costs,', 'on what an item memory\nloses,')
R('met-dynamics-costs', 'metaphor', r'Switching the dynamics on costs composition', 'Switching the dynamics on lowers composition')
R('met-dynamics-cost2', 'metaphor', r'the dynamics cost composition at', 'the dynamics lower composition at')
R('met-conj-costs', 'metaphor', r'deleting a conjunction costs it what the book knew', 'deleting a conjunction removes what the book knew')
R('met-exactness-costs-content', 'metaphor', r'the exactness claim costs content', 'the exactness claim removes content')
R('met-its-costs-declared', 'metaphor', r'its costs are declared', 'its side effects are declared')
R('met-two-costs', 'metaphor', r'Two costs are declared', 'Two side effects are declared')
R('met-same-dynamics-cost', 'metaphor', r'The same dynamics cost composition\.', 'The same dynamics lower composition.')
R('met-composition-cost-differs', 'metaphor', r'its composition cost differs', 'its composition loss differs')
R('met-at-the-cost-printed', 'metaphor', r'at the cost printed in', 'at the memory cost printed in')
R('met-unspent', 'metaphor', r'two of the three scored runs were\s+returned unspent', 'two of the three declared runs were\nnot used')
R('met-spent-nine', 'metaphor', r'nine were spent and', 'nine were used and')
R('met-seeds-spent', 'metaphor', r'scored seeds were spent only after', 'seeds were used only after')
R('met-spends-no', 'metaphor', r'spends no scored run', 'uses no test run')
R('met-spending-no', 'metaphor', r'spending no scored run', 'using no test run')
# style-twin wordings of the same metaphors
R('met-price-alternative-style', 'metaphor', r'The comparator serves to price the alternative to a hole, not to lose\.', 'The comparator serves to measure what the alternative to a hole requires, not to lose.')
R('met-priced-channel-emph', 'metaphor', r'\\emph\{Priced channel\.\}', r'\\emph{Dissipative channel.}')
R('met-does-not-buy-style', 'metaphor', r'what the naive unit does not buy', 'what the naive unit does not provide')
R('met-what-it-buys-style', 'metaphor', r'reports what it buys\.', 'reports what it yields.')
R('met-paying-style', 'metaphor', r'while paying 81,312 to 84,216 bytes of read-side coupling that the static read does not', 'while incurring 81,312 to 84,216 bytes of read-side coupling that the static read does not')
R('met-paying-more-style', 'metaphor', r'while paying 81,312 B more on the read side', 'while incurring 81,312 B more on the read side')
R('met-costs-style', 'metaphor', r'but costs (\$-2\.0899\$)', r'but changes composition by \1')
R('met-at-a-cost-of-exactly', 'metaphor', r'at a cost of exactly', 'with a change of exactly')
R('met-Its-composition-cost', 'metaphor', r'Its composition cost differs', 'Its composition loss differs')
R('bud-not-a-budget-that', 'budget', r'It is not a budget that the deletable object meets\.', 'It is not a limit that the deletable object meets.')
R('bud-budget-interp-hyphen', 'budget', r'budget-interpolation', 'step-limit-interpolation')

# ===== budget (item 2): by sense =====
R('bud-lace-own-budgets-title', 'budget', r'LACE and PoE, at their own budgets\.', 'LACE and PoE, at their own settings.')
R('bud-own-declared-budgets', 'budget', r'at their own declared budgets', 'at their own declared settings')
R('bud-own-budgets', 'budget', r'at their own\s+budgets', 'at their own\nsettings')
R('bud-lace-caption', 'budget', r'LACE and PoE at their own budgets,', 'LACE and PoE at their own settings,')
R('bud-own-budget-row', 'budget', r'own budget, ', 'own settings, ')
R('bud-own-declared-budget', 'budget', r'at its own declared budget', 'at its own declared settings')
R('bud-their-budgets-own', 'budget', r'Their budgets are their own', 'Their settings are their own')
R('bud-reimpl-these', 'budget', r'as re-implemented at these budgets', 'as re-implemented at these settings')
R('bud-reimpl-own', 'budget', r're-implemented at their own budgets', 're-implemented at their own settings')
R('bud-within-budget', 'budget', r'staying within budget', 'staying within the step limit')
R('bud-solvability', 'budget', r'solvability within budget', 'solvability within the step limit')
R('bud-budget-infeasible', 'budget', r'budget-infeasible', 'step-limit-infeasible')
R('bud-budget-words', 'budget', r'(B|b)udget words', lambda m: ('Step-limit' if m.group(1) == 'B' else 'step-limit') + ' words')
R('bud-budget-interp', 'budget', r'budget interpolation', 'step-limit interpolation')
R('bud-own-budget-rule', 'budget', r"CMAZE-1's own budget rule", "CMAZE-1's own step-limit rule")
R('bud-scored-budget-condition', 'budget', r'scored budget condition', 'step-limit condition used')
R('bud-not-a-budget-meets', 'budget', r'It is not a budget the deletable object meets\.', 'It is not a limit the deletable object meets.')
R('bud-not-a-budget-met', 'budget', r'not a budget the object met', 'not a limit the object met')
R('bud-reference-line', 'budget', r'reference line, not a budget', 'reference line, not a limit')
R('bud-reference-not', 'budget', r'reference, not a budget', 'reference, not a limit')
R('bud-no-scored-budget', 'budget', r'\(a free replication,\s+no scored budget;', '(a free replication\nthat used no test run;')
R('bud-retired', 'budget', r'retired-budget selection, declared', 'selected under a retired byte limit, declared')
R('bud-envs', 'budget', r'environments,\s+budgets or recipes', 'environments,\nstep limits or recipes')
R('bud-different-budget', 'budget', r'A different environment and a different budget from', 'A different environment and a different parameter scale from')
R('bud-rival-budget-row', 'budget', r'rival budget &', 'baseline training steps &')
R('bud-either-budget', 'budget', r'at either budget', 'at either step count')
R('bud-both-training-budgets', 'budget', r'over both( training)? budgets', lambda m: 'over both step counts')
R('bud-two-training-budgets', 'budget', r'both\s+training budgets', 'both\nstep counts')
R('bud-recipe-budget-MLP', 'budget', r'recipe-budget MLP', 'recipe-steps MLP')
R('bud-recipes-budget', 'budget', r"recipe's budget", "recipe's step count")
R('bud-withdrawn-recipe', 'budget', r'the withdrawn recipe budget', 'the withdrawn recipe step count')
R('bud-budget-whose', 'budget', r'the budget whose margin is', 'the step count whose margin is')
R('bud-budget-diag', 'budget', r'budget diagnostic &', 'step diagnostic &')
R('bud-emph-budget', 'budget', r'\\emph\{Budget\}: of the eighteen', r'\\emph{Runs}: of the eighteen')
R('bud-the-budget-ladder', 'budget', r'The budget, the configuration ladder', 'The run count, the configuration ladder')
R('bud-config-ladder-and-budget', 'budget', r'Configuration ladder and budget\.', 'Configuration ladder and run count.')
R('bud-training-budget-objection', 'budget', r'training-budget objection', 'training-steps objection')
R('bud-fragment-budget', 'budget', r'against the fragment budget', 'against the fragment memory')
R('bud-mu-budget-law', 'budget', r'a \$\\mu\^\{-2\}\$ budget\s+law', 'a $\\mu^{-2}$ scaling\nlaw')
R('bud-read-budget-trilemma', 'budget', r'fixed\s+read budget', 'fixed\nread size')
R('bud-budget-at-least', 'budget', r'\(budget at least the number\s+of cells', '(capacity at least the number\nof cells')

# ===== pre-registration and scored runs (item 2) =====
R('pre-where-sentence', 'prereg', r'Where an experiment was pre-registered, its success and falsification criteria were written before its first number; we refer to each as a \\emph\{pre-registered criterion\}, labelled as in that pre-registration',
  r'Where an experiment declared its success and falsification criteria before its first number, we refer to each as a \\emph{declared criterion}, labelled as in that declaration')
R('pre-clause', 'prereg', r"the\s+pre-registration's falsification clause fires", 'the\ndeclared falsification clause fires')
R('pre-amend', 'prereg', r'Two pre-registration amendments were made', 'Two amendments to the declared protocol were made')
R('pre-declared-primary', 'prereg', r"departs from the pre-registration's declared primary", 'departs from the declared primary')
R('pre-record-cites', 'prereg', r'the pre-registration record it cites', 'the declaration record it cites')
R('pre-as-pre-registered', 'prereg', r'(A|a)s pre-registered', lambda m: m.group(1) + 's declared in advance')
R('pre-fails-as', 'prereg', r'fails as pre-registered', 'fails as declared in advance')
R('pre-scored-run', 'prereg', r'pre-registered scored run', 'declared test run')
R('pre-generic-Pre', 'prereg', r'Pre-registered', 'Declared')
R('pre-generic', 'prereg', r'pre-registered', 'declared')
R('pre-registration-generic', 'prereg', r'pre-registration', 'declared protocol')
R('scored-artefact-quote', 'prereg', r"its artefact reads ``no scored run\s+and no claim''", 'its record states that it is neither\na test run nor a claim')
R('scored-act', 'prereg', r'reading TEST would be a scored act', 'reading TEST would count as a test run')
R('scored-new-test-config', 'prereg', r'a new TEST configuration would be a scored run', 'a new TEST configuration would be a new test run')
R('scored-run-generic', 'prereg', r'\bscored run(s?)\b', r'test run\1')

# ===== configurations A1 / A10 / C-512 / C-field / B / A / C2 (item 2) =====
R('cfg-store-is-A1', 'config', r'The store is Config A1 \(latent 128', 'The n-CLU here is the base design (latent 128')
R('cfg-A1-rollout', 'config', r"Config A1's rollout validity", "The base n-CLU's rollout validity")
R('cfg-scored-A1-store', 'config', r'The store is the scored Config~A1 store:', 'The n-CLU is the base design as run:')
R('cfg-tab-A1-caption', 'config', r'TEST, Config A1, 3 seeds', 'TEST, base n-CLU, 3 seeds')
R('cfg-tab-A1-pinned', 'config', r'Config A1: latent 128', 'Base n-CLU: latent 128')
R('cfg-app-A1', 'config', r'\$\\Kstar=2\$, Config A1 \(latent 128', r'$\\Kstar=2$, the base design (latent 128')
R('cfg-REL-config', 'config', r'\(Config CLU-REL-A10\)', '(n-CLU-REL)')
R('cfg-REL-store', 'config', r'the CLU-REL-A10 store', 'n-CLU-REL')
R('cfg-REL-configuration', 'config', r'the CLU-REL-A10\s+configuration', 'n-CLU-REL')
R('cfg-REL', 'config', r'\bCLU-REL-A10\b', 'n-CLU-REL')
R('cfg-A1-paren', 'config', r'observation \(A1\)', 'observation (base n-CLU)')
R('cfg-for-A1', 'config', r'for A1 and for', 'for the base n-CLU and for')
R('cfg-A1s', 'config', r"\bA1's\b", "the base n-CLU's")
R('cfg-A1-val', 'config', r'validation, A1, every', 'validation, base n-CLU, every')
R('cfg-A1-row', 'config', r'observation, A1 \(TEST\)', 'observation, n-CLU (TEST)')
R('cfg-A1-deletion', 'config', r'& A1, words', '& base n-CLU, words')
R('cfg-A1-rollout-row', 'config', r'every arm: A1 0\.000', 'every arm: base n-CLU 0.000')
R('cfg-C512-row', 'config', r'Config C-512 \(latent 512, 8 atoms; three further', 'A wider design (latent 512, 8 atoms; three further')
R('cfg-C512-app', 'config', r'Config C-512 \(latent 512, 8 atoms, rank/width', 'a wider design (latent 512, 8 atoms, rank/width')
R('cfg-Cfield-paren', 'config', r'\(Config C-field:', '(the field design:')
R('cfg-Cfield-caption', 'config', r'TEST, Config C-field, 3', 'TEST, field design, 3')
R('cfg-Cfield-scored', 'config', r'Config C-field \(\$K=8\$', 'the field design ($K=8$')
R('cfg-on-Cfield', 'config', r'runs on C-field', 'runs on the field design')
R('cfg-Cfield-TEST', 'config', r'the C-field TEST leg', "the field design's TEST leg")
R('cfg-Cfield-against', 'config', r'C-field against converged', 'the field design against converged')
R('cfg-Cfield-gate', 'config', r'its C-field\s+store gate holds', "its field design's\ncontrol check holds")
R('cfg-B-def', 'config', r'Config B, a match-bit interface given identically to\s+every arm, failed', 'A match-bit design, an interface given identically to\nevery arm, failed')
R('cfg-B-C2', 'config', r'Configs B and C2', 'the match-bit and second latent designs')
R('cfg-B-neg', 'config', r'Config B\s+\(\\emph\{interface\}\)', 'the match-bit design\n(\\emph{interface})')
R('cfg-A-pre', 'config', r'The pre-amendment Config A was superseded', 'The pre-amendment first design was superseded')
R('cfg-O-A', 'config', r'Config A \(autoencoder latent, rank/width at most 0\.3\) failed', 'the autoencoder-latent design (rank/width at most 0.3) failed')
R('cfg-O-A-neg', 'config', r'CMAZE-O: Config A \(\\emph\{content/read\}', 'CMAZE-O: the autoencoder-latent design (\\emph{content/read}')
R('cfg-C2-def', 'config', r'Config C2 \(latent, rank/width 0\.5\) failed', 'a second latent design (rank/width 0.5) failed')
R('cfg-C2-neg', 'config', r'Config C2 \(\\emph\{read\}', 'the second latent design (\\emph{read}')
R('cfg-A-smoke', 'config', r'of the Config-A single design reads', 'of the first design reads')
R('cfg-A-deploys', 'config', r'the span fraction Config-A deploys', 'the span fraction the first design deploys')
R('cfg-A-first', 'config', r"first Config-A validation run", 'first validation run of the first design')
R('cfg-configs-header', 'config', r'& configs &', '& configurations &')

# ===== improved (item 2) =====
R('imp-paragraph', 'special', r'\\paragraph\{An improved store configuration is an input-representation result\.\}',
  r'\\paragraph{A relational request coding for the n-CLU is an input-representation result.}')
R('imp-its-improved', 'special', r'its improved configuration', 'its relational configuration')

# ===== charter / gate (item 2) =====
R('charter-controls', 'gate', r'charter-C control', 'landscape control')
R('gate-control-gate-checked', 'gate', r'The control gate is checked before any margin', 'The control check is applied before any margin')
R('gate-store-control-gate', 'gate', r"The store's control gate at \$\\Kstar\$ collapses on all 15 legs", r'The landscape controls at $\\Kstar$ collapse on all 15 legs')
R('gate-strict-control', 'gate', r'strict control gate', 'strict control check')
R('gate-strict-header', 'gate', r'& strict gate \\\\', r'& strict check \\\\')
R('gate-control-gate', 'gate', r'[Tt]he control gate', lambda m: m.group(0)[0] + 'he control check')
R('gate-paren-store-gate', 'gate', r'\(the\s+store gate\)', '(the\ncontrol check)')
R('gate-O-heading', 'gate', r'CMAZE-O: the store gate holds;', 'CMAZE-O: the control check holds;')
R('gate-what-holds', 'gate', r'What holds is the \\emph\{store gate\}', r'What holds is the \\emph{control check}')
R('gate-O-store-gate', 'gate', r'the CMAZE-O store gate', 'the CMAZE-O control check')
R('gate-textbf', 'gate', r'\\textbf\{the store gate holds\}', r'\\textbf{the control check holds}')
R('gate-caption-holds', 'gate', r'the store gate holds', 'the control check holds')
R('gate-eligible', 'gate', r'left 25/53 store-gate eligible', 'left 25/53 eligible under the control check')
R('gate-mandatory', 'gate', r'a mandatory gate', 'a mandatory check')

# ===== artefact -> record (item 2) =====
R('art-recorded', 'special', r'read from a recorded artefact of a declared or declared run', 'read from the recorded result of a declared run')
R('art-recorded2', 'special', r'recorded artefact', 'recorded result')
R('art-generic', 'special', r'\b([Aa])rte?[fi]acts?\b', lambda m: ('R' if m.group(1) == 'A' else 'r') + 'ecord' + ('s' if m.group(0).endswith('s') else ''))

# ===== ledger -> memory footprint (item 3) =====
R('led-section', 'ledger', r'\\section\{The object, the judge and the ledger\}', r'\\section{The object, the judge and the memory footprint}')
R('led-paragraph', 'ledger', r'\\paragraph\{The four-part ledger, never netted\.\}', r'\\paragraph{The memory footprint, never netted.}')
R('led-is-memory-not-matched', 'ledger', r'four-part ledger \((Table~\x02\d+\x03)\) is memory, not a matched\s+quantity', r'memory footprint (\1) is printed, never\nmatched')
R('led-is-memory-not-matched2', 'ledger', r'four-part ledger is printed in\s+(Table~\x02\d+\x03) and is memory, not a matched quantity', r'memory footprint is printed in\n\1 and is never matched')
R('led-caption', 'ledger', r'The four-part ledger of the', 'The footprint breakdown of the')
R('led-abstract', 'ledger', r'in its four-part\s+ledger', 'in its footprint\nbreakdown')
R('led-is-memory', 'ledger', r'The ledger is memory, not a read-path parameter;', 'The memory footprint is not a read-path parameter;')
R('led-store-read-ledger', 'ledger', r'the store, read and ledger of', 'the store, read and memory footprint of')
R('led-kcurve', 'ledger', r'composition and the honest ledger against', 'composition and the memory footprint against')
R('led-file', 'ledger', r"The ladder's ledger file carries", "The ladder's footprint record carries")
R('led-its-ledger-kept', 'ledger', r'Its ledger is kept for', 'Its footprint is kept for')
R('led-memory-ledger', 'ledger', r'the four-part memory ledger', 'the memory footprint')
R('led-its-four-part', 'ledger', r'Its four-part ledger:', 'Its footprint breakdown:')
R('led-Ledger', 'ledger', r'\bLedger:', 'Footprint:')
R('led-row', 'ledger', r'^ledger(:| &)', r'footprint\1', re.M)
R('led-judges', 'ledger', r'same judges, ledger and matching', 'same judges, memory footprint and matching')
R('led-estimand', 'ledger', r'estimand and ledger are', 'estimand and memory footprint are')
R('led-prior', 'ledger', r'\\section\{The prior-derivation ledger\}', r'\\section{Prior derivations}')
R('led-honest-four-part', 'ledger', r'honest four-part ledger', 'honest four-part footprint')
R('led-four-part-generic', 'ledger', r'four-part ledger', 'footprint breakdown')
R('led-byte-ledger', 'ledger', r'its byte ledger', 'its memory footprint')
R('led-is-memory-colon', 'ledger', r'The ledger is memory, not a read-path parameter:', 'The memory footprint is not a read-path parameter:')
R('led-estimand-style', 'ledger', r'estimand and ledger;', 'estimand and memory footprint;')
R('led-judges-style', 'ledger', r'judges, ledger and matching', 'judges, memory footprint and matching')
R('led-generic', 'ledger', r'\bledger\b', 'memory footprint')

# ===== store: objects that are not the n-CLU, and the content sense (item 2) =====
R('st-item-store', 'special', r'\b([Ii])tem store\b', lambda m: m.group(1) + 'tem memory')
R('st-field-store', 'special', r'\bfield\s+store\b', 'field-mode n-CLU')
R('st-learned-store', 'special', r'\blearned store\b', 'learned memory')
R('st-designed-store', 'special', r'\bdesigned store\b', 'designed memory')
R('st-store-level-designed', 'special', r'Store-level exact deletion in a designed', 'Memory-level exact deletion in a designed')
R('st-store-level-deletion', 'special', r'store-level deletion is exact', 'memory-level deletion is exact')
R('st-trilemma', 'special', r'a store cannot have all three', 'a memory cannot have all three')
R('st-laws-govern', 'special', r'these laws govern the store;', 'these laws govern the memory;')
R('st-ldm-def', 'special', r'a store whose content is read by relaxing', 'a memory whose content is read by relaxing')
R('st-static-corner', 'special', r'that this store is the static corner of', 'that this n-CLU is the static limit of')
R('st-post-delete-v2', 'special', r'the post-delete store equals the never-inserted one', 'the post-delete memory equals the never-inserted one')
R('st-post-delete-app', 'special', r'the post-delete store is byte-identical to the store built from the', 'the post-delete memory is byte-identical to the memory built from the')
R('st-falsifies-this-store', 'special', r'placement\) on this store, not the', 'placement) on this memory, not the')
R('st-retention-section', 'special', r'\\section\{The retention law on the store in full\}', r'\\section{The retention law on the field-mode n-CLU in full}')
R('st-not-stored', 'special', r'nothing it has not stored', 'nothing it has not written')
R('st-stored-activation', 'special', r'stored-activation gradients', 'saved-activation gradients')
R('st-store-removed-tt', 'special', r'\\texttt\{store\\_removed\}', 'n-CLU removed')
R('st-store-class', 'special', r'an earlier arm of this store class', 'an earlier arm of the n-CLU class')
R('st-energy-sub', 'special', r'E_\{\\text\{store\},k\}', r'E_{\\text{content},k}')
R('st-content-abstract', 'special', r'\(store mode deviation at', "(content mode deviation at")
R('st-store-content-and-book', 'special', r'the store\s+content and the address book equal', "the n-CLU's\ncontent and the address book equal")
R('st-downdate-the-store-ops', 'special', r'downdate the store; downdate the address book', "downdate the n-CLU's content; downdate the address book")
R('st-downdate-the-store-cap', 'special', r'downdate the store, downdate(\s+)the address', lambda m: "downdate the n-CLU's content, downdate" + m.group(1) + 'the address')
R('st-forgetting-caption', 'special', r'Forgetting \$c\$ is: downdate the store,', "Forgetting $c$ is: downdate the n-CLU's content,")
R('st-downdates-the-store-fig', 'special', r'downdates the store, downdates the address', "downdates the n-CLU's content, downdates the address")
R('st-a-store-equals', 'special', r'\(a\) the store equals the store refit without the rows', "(a) the n-CLU's content equals the content refit without the rows")
R('st-tabdel-a', 'special', r"store's largest absolute mode deviation from the store refit without the rows", "n-CLU content's largest absolute mode deviation from the content refit without the rows")
R('st-fig-a', 'special', r'\(a\)~The store against the store refit without the rows', "(a)~The n-CLU's content against the content refit without the rows")
R('st-store-and-head-O', 'special', r'Store and head are fitted in the observation field', "The n-CLU's content and head are fitted in the observation field")
R('st-store-head-row', 'special', r'store downdate, store \+ head: seconds', 'n-CLU downdate, content + head: seconds')
R('st-own-store-and-head', 'special', r'refitting its own\s+store and head', 'refitting its own\ncontent and head')

# ===== internal control names -> plain names, defined once (item 2) =====
R('ctl-define', 'special', r'([Ff])our controls, which we call the \\emph\{pre-declared landscape controls\} throughout \(\\texttt\{landscape\\_removed\}, \\texttt\{landscape\\_shuffled\}, \\texttt\{landscape\\_shuffled\\_across\\_wells\}, \\texttt\{random\\_atoms\}\)',
  lambda m: m.group(1) + 'our controls, which we call the \\emph{pre-declared landscape controls} throughout, \\emph{landscape removed} (atoms zeroed), \\emph{landscape shuffled} (atom values permuted within each well), \\emph{shuffled across wells} (well blocks dealt to other wells) and \\emph{random atoms} (random atoms in place of the fitted ones),')
R('ctl-lscape-hdr', 'special', r'\\texttt\{lscape\\_shuffled\}', 'landscape shuffled')
R('ctl-lscape-removed-hdr', 'special', r'\\texttt\{lscape\\_removed\}', 'landscape removed')
R('ctl-across-wells-tt', 'special', r'\\texttt\{landscape\\_shuffled\\_across\\_wells\}', 'shuffled across wells')
R('ctl-shuffled-tt', 'special', r'\\texttt\{landscape\\_shuffled\}', 'landscape shuffled')
R('ctl-removed-tt', 'special', r'\\texttt\{landscape\\_removed\}', 'landscape removed')
R('ctl-random-tt', 'special', r'\\texttt\{random\\_atoms\}', 'random atoms')
R('ctl-rival-tt', 'special', r"\\texttt\{rival\\_transformer\} and \\texttt\{rival\\_lstm\} were fitted", "transformer and LSTM were fitted")

# ===== bare CLU inside quotes =====
R('clu-quote-inference', 'special', r"the CLU's own 2-way", "the n-CLU's own 2-way")
R('clu-quote-verbatim', 'special', r'The inference note, verbatim, travels with', 'The inference note travels with')
R('clu-quote-lace', 'special', r"``the CLU out-composes LACE in general''", "``the n-CLU out-composes LACE in general''")

# ===== corner -> held-out (factor) combination (item 3) =====
R('cor-static', 'special', r'static corner', 'static limit')
R('cor-first-body', 'special', r'per corner', 'per held-out factor combination', first_after='\\section{Introduction')
R('cor-per-corner', 'corner', r'\bper corner\b', 'per held-out combination')
R('cor-per-hyphen', 'corner', r'\bper-corner\b', 'per-combination')
R('cor-held-out-corners', 'corner', r'\bheld-out corners\b', 'held-out combinations')
R('cor-corners', 'corner', r'\bcorners\b', 'combinations')
R('cor-corner', 'corner', r'\bcorner\b', 'combination')

# ===== rival -> baseline (item 2) =====
R('riv-field', 'special', r'over a rival field weakened', 'over a set of baselines weakened')
R('riv-Rivals', 'rival', r'\bRivals\b', 'Baselines')
R('riv-Rival', 'rival', r'\bRival\b', 'Baseline')
R('riv-rivals', 'rival', r'\brivals\b', 'baselines')
R('riv-rival', 'rival', r'\brival\b', 'baseline')

# ===== store -> n-CLU (item 2) =====
R('st-a-store', 'store', r'\ba store\b', 'an n-CLU')
R('st-A-store', 'store', r'\bA store\b', 'An n-CLU')
R('st-stores', 'store', r'\bstores\b', 'n-CLUs')
R('st-store', 'store', r'\b(store|Store|STORE)\b', 'n-CLU')

# ===== cross-references (item 5) =====
R('xr-S-range', 'xref', r'RAW:\\S(\x02\d+\x03)--\\S(\x02\d+\x03)', r'Secs.~\1--\2')
R('xr-S', 'xref', r'RAW:\\S(\x02\d+\x03)', r'Sec.~\1')
R('xr-Figures', 'xref', r'RAW:Figures~(\x02\d+\x03)', r'Figs.~\1')
R('xr-Figure', 'xref', r'RAW:Figure~(\x02\d+\x03)', r'Fig.~\1')
R('xr-Tables', 'xref', r'RAW:Tables~(\x02\d+\x03)', r'Tabs.~\1')
R('xr-Table', 'xref', r'RAW:Table~(\x02\d+\x03)', r'Tab.~\1')
R('xr-Appendices', 'xref', r'RAW:Appendices~(\x02\d+\x03)', r'Apps.~\1')
R('xr-Appendix', 'xref', r'RAW:Appendix~(\x02\d+\x03)', r'App.~\1')

# ---------------------------------------------------------------- unmask
for _ in range(3):
    text = re.sub(r'\x02(\d+)\x03', lambda m: masks[int(m.group(1))], text)
assert '\x02' not in text and '\x03' not in text
# pointer comments for the labels and numbers that left the rendered text (COMMON.md, draft-edit method 5)
NOTES = [
  (r'\$t_\{1/2\}\$ = 7\.992', '% pw1-draft-sweeps (item 2): the symbol was t_pay(50%), renamed t_{1/2}; values unchanged, w10-physics-section-evidence/EVIDENCE.md'),
  (r'[Aa] wider design \(latent 512', '% pw1-draft-sweeps (item 2): "a wider design" was Config C-512 in w16-maze-wm/TABLES.md and w16-maze-wm/FINDINGS.md'),
  (r'is the base design \(latent 128', '% pw1-draft-sweeps (item 2): the base design / base n-CLU was Config A1 and n-CLU-REL was Config CLU-REL-A10 in the w16/w19/w21 records'),
]
tl = text.split('\n')
for rx, note in NOTES:
    hit = [i for i, l in enumerate(tl) if re.search(rx, split_comment(l)[0])]
    for i in hit:
        if not split_comment(tl[i])[1]:
            tl[i] = tl[i] + ' ' + note
            counts['note:' + note[:40]] = counts.get('note:' + note[:40], 0) + 1
text = '\n'.join(tl)
open(DST, 'w').write(text)
assert text.count('\n') == open(SRC).read().count('\n'), 'line count changed'
json.dump({'counts': counts, 'log': log}, open(CNT, 'w'), indent=1)
