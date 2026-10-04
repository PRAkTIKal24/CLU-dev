"""Build OUT/scan-report.md from the real-package scan JSON and the positive-control JSON."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scan  # noqa: E402

real = json.load(open(sys.argv[1]))
pc = json.load(open(sys.argv[2]))
out = Path(sys.argv[3])
L = ['# scan-report — anonymity scan of `nclu-iclr-code/`', '',
     'Scanner: `tooling/scan.py` (outside the package; it holds the banned list). Target of the real scan: '
     '`%s` (%d files). Command: `python3 tooling/scan.py nclu-iclr-code --git <source clone>`.' % (Path(real['target']).name, real['files_scanned']), '',
     '## 1. Result on the real package', '',
     '**Hits: %d.** %s' % (len(real['hits']), 'The report is EMPTY.' if not real['hits'] else 'NOT EMPTY — see below.'), '']
for h in real['hits']:
    L.append('- %s %s:%s [%s] %r' % (h['where'], h['file'], h.get('line', ''), h['rule'], h['match']))
L += ['', '## 2. What is scanned, and how', '',
      '* every path component (file and directory names);',
      '* every file that decodes as text (UTF-8), whole content, line by line (code, JSON, Markdown, TOML, the licence stub);',
      '* every binary file: printable runs of >= 4 bytes (as `strings -n 4`), plus',
      '  - PNG `tEXt`, `zTXt` (zlib-decompressed) and `iTXt` chunks, decoded;',
      '  - PDF: every stream decompressed, the Info dictionary (incl. UTF-16 hex strings) and XMP;',
      '  - ZIP / NPZ: member names, archive comment, every member decompressed and scanned recursively; gzip decompressed;',
      '  - array files (`.npy`, and the equinox `.eqx` weight files, which are concatenated `.npy` blobs): every header is scanned as text; '
      'string/object arrays are scanned in full; the numeric payload is checked to be exactly the declared arrays '
      '(shape x itemsize), and any byte not accounted for is flagged (`UNACCOUNTED-BYTES`, itself a banned token) and scanned in full. '
      'Raw float payloads are not text: before this rule, `strings` on the six weight files produced random 2-4 character runs '
      "matching the short patterns (e.g. `W6/`, `M:/`, `ERc`); the positive control below shows that bytes appended to a weight file are caught.",
      '* matching is case-insensitive, except the role word `Head` (below).', '',
      '## 3. The banned list (seed list of the brief, extended; never shrunk)', '']
for cat, pats in scan.SEED.items():
    L.append('**%s**: ' % cat + ', '.join('`%s`' % p.replace('|', '\\|') for p in pats))
    L.append('')
ids = scan.git_identities('/Users/batmac/clu/worktrees/CHLU-w24-repro-package')
L += ['**people/org, extracted mechanically from `git log --all` of the source clone** (author and committer names and e-mails, '
      'co-author trailers; every full address kept, its local part and domain added as separate tokens; the ordinary words '
      '`info`, `local`, `noreply`, `github`, `claude` are not kept as separate tokens but `noreply@` and `Claude` are on the seed list): ' +
      ', '.join('`%s`' % i for i in ids), '',
      'Word boundaries: `\\bhub\\b` is widened to `(?<![a-z])hub(?![a-z])` so that `hub_x` and `NCLU_HUB` are caught while '
      '`GitHub` is not; `\\bw[0-9]{1,2}[_\\-/]` is kept and widened by `(?<![a-z0-9])w[0-9]{1,2}[a-z]?(?=[_\\-/.:]|\\b)` so that '
      '`exp_w8_`, `W16`, `w21C` are caught; `\\bN[0-9]{3}\\b` (record codes) is matched case-sensitively, because a newline escape '
      'before a number (`\\n614,400` in a tick label) otherwise matched.', '',
      '## 4. Whitelist (the only one)', '']
L += ['* ' + w for w in scan.WHITELIST_RULES]
L += ['', 'No other token is whitelisted. Legitimate collisions were rephrased instead: the licence stub reads '
      '"Licence to be added on publication."; identifiers containing `ledger` became `footprint`, `registry` became `index`, '
      '`placeholder` became `stand_in`; the paper label `ledger-kstar` is written "the K* footprint table" in the README.', '',
      '## 5. Positive control', '',
      'The package was copied to a scratch directory and one token of every category was planted in a file body, a file name, '
      'a PDF metadata field (Author) and a PNG metadata field (tEXt Author); additionally a compressed PNG zTXt chunk, bytes '
      'appended to a weight file, and an NPZ member name. For AMENDMENT-3 four body plants were added: `PW-1` in a comment, '
      '`PW1_X` as an identifier, the scar `(, 2026-09-25)` in a docstring, and a bare scar `(, )`. The scanner was run on the planted copy.', '',
      '| placement | category | file | token | caught | rule(s) |', '|---|---|---|---|---|---|']
for p in pc['plants']:
    L.append('| %s | %s | `%s` | `%s` | %s | %s |' % (p['placement'], p['category'], p['file'], p['token'],
                                                   'YES' if p['caught'] else '**NO**', ', '.join('`%s`' % r.replace('|', '\\|') for r in p['hit_rules'])))
L += ['', '**%d of %d plants caught; %d missed.**' % (pc['n_caught'], pc['n_plants'], len(pc['missed'])), '']
if len(sys.argv) > 4:
    import collections
    before = json.load(open(sys.argv[4]))
    cnt = collections.Counter((h['category'], h['rule']) for h in before['hits'])
    L += ['## 6. Hub re-scan findings and fixes (AMENDMENT-3)', '',
          "The Hub's independent re-scan of the harvested package found what the scanner of sections 1-5 missed: review-item IDs "
          "(`PW-1` x18, `PW1` x4, e.g. `PW1_BANNED`), deletion scars (`(, 2026-09-25)` x4, `(, 2026-09-14)`, `-ruled 2026-09-25`), "
          'history fields (`retired`, `retired_by`, `rival_sets_retired`) and ISO development dates (21). The old rule '
          '`\\bpw[0-9]\\b` could not match `PW-1` (hyphen) or `PW1_` (underscore is a word character).', '',
          '**Rules added** (all listed in section 3; nothing was removed): programme: `PW-?[0-9]`, `(?<![a-z])pw[0-9]`, '
          '`\\bitem [0-9]+(\\.[0-9]+)*\\b`, `\\bruled\\b`, `(?<![A-Za-z])-ruled`, ISO dates `20[0-9]{2}-[01][0-9]-[0-3][0-9]`, '
          '`retired_by`, `\\bretired\\b`, review-round references `\\bR[0-9]{1,3}(\\.[0-9]+)?\\b` (case-sensitive), `AMENDMENT-?[0-9]`; '
          'a new category **scrub scars** for the stubs a deletion leaves (a punctuation run where a word was): `\\(\\s*[,;:]`, `,\\s*,`, '
          '`\\(\\s+\\)`, a space before `,`/`;`, `(-)`, `--)`, `.:`, `on -<n>`, a doubled article (`the the`), an orphan possessive '
          "(`'s` with no word before it), `( ii)`, `PROTOCOL-§`, `ABC-)`, the stub docstring `init  .` and a dangling preposition "
          "(`spent in;`). Two of these new rules were narrowed while they were written, before any result was taken from them, "
          "because they matched working code: `[A-Za-z]-\\)` matched a regex look-behind `(?<!n-)` and became `[A-Z]{3}-\\)`; "
          "a bare `(in|at|per)[;,)]` matched variables named `at`/`per` and became the verb-plus-preposition form.", '',
          '**Before / after.** The final scanner run on the proof-C copy of the harvested package (%d files): **%d hits**. '
          'On the rebuilt package (%d files incl. MANIFEST.md): **%d hits**.' % (before['files_scanned'], len(before['hits']),
                                                                               real['files_scanned'], len(real['hits'])), '',
          '| category | rule | hits before | hits after |', '|---|---|---|---|']
    for (c, r), v in cnt.most_common():
        L.append('| %s | `%s` | %d | 0 |' % (c, r.replace('|', '\\|'), v))
    L += ['', 'The 77 review-round hits include 4 functional labels that were not review references (`"R0"` a rung label, '
          '`"R2"`/`"R3"` unused router names in a config default) and the module alias `gen_deletion as R7` (42 hits); the other 31 are review-round references in prose; '
          'they were renamed (`rung_0`, `router_2`, `router_3`, `GD`) so that the rule can stay unconditional.', '',
          '**Fixes (all in the build transform `tooling/transform.py` / `tooling/names.py`, then a full rebuild; nothing edited by hand):**', '',
          '* review-item and review-round references dropped with the preposition that introduced them '
          '("per AMENDMENT-1", "(R26.3)", "PW-1 item 10:"); a possessive reference becomes "the" ("R8\'s declared embedding" -> "the declared embedding");',
          '* identifiers renamed: `PW1_BANNED` -> `FIGURE_TEXT_BANNED`, `PW1_TEXT` -> `FIGURE_TEXT_SEEN`, `check_vocab_pw1` -> '
          '`check_figure_vocabulary`, figure stems `rc_pw1_shapes_*` -> `shapes_*`, `retired_byte_budget` -> `prior_byte_budget` '
          '(a functional quantity, 688,128 B, read by the rival-budget notes: kept, renamed);',
          '* figure data files (`figures/data/*.json`): `retired`, `retired_by` and `retired_rows` fields removed (`retired_rows` '
          'is read by no code: dropped), `_doc` fields removed (read by no code), the status value `retired` -> `not_drawn` and '
          'the output key `rival_sets_retired` -> `rival_sets_not_drawn` (read by `results_figures_b.py`, kept functional under the neutral name);',
          '* ISO development dates removed from strings, docstrings and data (a whole-string timestamp becomes "not recorded");',
          '* scar repair after every deletion (`descar`): empty or comma-led parentheses, doubled articles/spaces, `.:`, orphan '
          'possessives, while never touching the edges of an f-string fragment (an earlier draft of this step glued '
          '`{WHITENING_D_LIMIT}` to the next word; caught in review of the diff and fixed before the proof);',
          '* words that contain a banned token are replaced whole instead of cut (`spoken` -> `expressed`, `scored runs` -> `runs`, '
          '`pre-registered` -> `declared in advance`), because cutting left stubs such as `n`.', '',
          'A structural check compared every Python file of the proof-C copy and the rebuilt package with string contents masked: the '
          'ASTs are identical apart from the renames above and the two AMENDMENT-2 report fixes.', '']
out.write_text('\n'.join(L) + '\n', encoding='utf-8')
print('wrote', out)
