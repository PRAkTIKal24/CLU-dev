#!/usr/bin/env python3
"""w21 figures-redraw, draft pass 2 (Part C): the CMAZE-T engine-only figure block becomes the
per-arm panels drawn from w19's persisted words-plus-observation objects.

* the three figure slots point at figures/fig_cmazet_{families,door_then_tint,deletion}.pdf and
  their captions say what is drawn; one slot is added: figures/fig_cmazet_per_arm.pdf;
* the three \\PLACEHOLDER{...} strings are lifted VERBATIM out of the old block and re-embedded
  (asserted: ordered argument list unchanged) -- they name the w16 arms' own outputs, never persisted;
* one sentence is added to the placeholder index's figure paragraph.
usage: apply_pass2.py <in dir> <out dir>
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texnum import macro_args  # noqa: E402

HEAD = "% ---- w19 draft-figure-slots: CMAZE-T engine-only panels, beside Table tab:cmazet ----\n"


def placeholders(block):
    out, i = [], 0
    while True:
        j = block.find("\\PLACEHOLDER{", i)
        if j < 0:
            return out
        k, depth = j + len("\\PLACEHOLDER{"), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(block[k], 0) if block[k - 1] != "\\" else 0
            k += 1
        out.append(block[j:k])
        i = k


SRC = ("% SRCBLOCK: w19-maze-t-symmetric/test/BOTH_test_seed0.json; w19-maze-t-symmetric/test/BOTH_test_seed1.json; "
       "w19-maze-t-symmetric/test/BOTH_test_seed2.json; ../../outputs/w17-paper/figures/data/cmazet_items.REPRO.json; "
       "../../outputs/w17-paper/figures/data/cmazet_panels.json; w19-maze-t-symmetric/FINDINGS.md")


def new_block(P):
    fam, door, dele = P
    return f"""% ---- CMAZE-T per-arm panels (w21 figures-redraw, Part C), beside Table tab:cmazet ----
% Drawn from w19's persisted words-plus-observation (BOTH) TEST objects, hash-verified before loading;
% every arm's n-way re-read equals the scored artefact on every seed (figures/data/cmazet_items.REPRO.json).
% The three PLACEHOLDERs name the per-arm outputs of Table tab:cmazet's own arms, which w16 never persisted.
{SRC}
% SOURCE ARTEFACT: figures/fig_cmazet_families.pdf (make_figures_w21.py + cmazet_panels.py; data/cmazet_items.json)
\\figslot{{fig_cmazet_families}}{{\\textbf{{Held-out CMAZE-T transitions, one per rule family, every arm
(words plus observation, TEST).}} For each family (the first TEST $n$-way row at seed 0; the
\\texttt{{rule\\_a}} family, which has almost no TEST $n$-way rows, from seed 2): $s_t$ with the action,
the engine's $s_{{t+1}}$, and each arm's generated $s_{{t+1}}$ with the exact judge's mark: the store,
the matched rivals at 8,000 and 80,000 training steps, the four charter-C controls and the two
learned blind arms. Every panel is re-read from the condition's persisted objects, and the re-read
reproduces every arm's scored TEST score on every seed. The store is the scored Config~A1 store: its
TEST reads equal that run's on every seed (that run persisted no object to hash). The arms of
Table~\\ref{{tab:cmazet}}, whose rivals read the observation alone, were not persisted:
{fam}}}{{fig:cmazet-families}}
% SOURCE ARTEFACT: figures/fig_cmazet_per_arm.pdf
\\figslot{{fig_cmazet_per_arm}}{{\\textbf{{Each CMAZE-T arm succeeding and failing (words plus observation,
TEST, seed 0).}} For every arm, one held-out $n$-way transition on which its generated $s_{{t+1}}$ is
exact and one on which it is not, each drawn uniformly with a fixed seed among the qualifying rows.
The landscape-removed control and the blind-colour arm have no exact $n$-way row at seed 0 and show
only a failure. Columns: $s_t$ with the action, the engine's $s_{{t+1}}$, and the arm's generated
$s_{{t+1}}$ with the exact judge's mark.}}{{fig:cmazet-per-arm}}
% SOURCE ARTEFACT: figures/fig_cmazet_door_then_tint.pdf
\\figslot{{fig_cmazet_door_then_tint}}{{\\textbf{{Door-then-tint (\\texttt{{interaction}}) transitions, one
per move direction present in the seed-0 hold-out, every arm (words plus observation, TEST).}}
Columns as in Figure~\\ref{{fig:cmazet-families}}. For the arms of Table~\\ref{{tab:cmazet}}:
{door}}}{{fig:cmazet-door-tint}}
% SOURCE ARTEFACT: figures/fig_cmazet_deletion.pdf
\\figslot{{fig_cmazet_deletion}}{{\\textbf{{Deletion of colour 1 (value scope), seed 0, words plus
observation.}} On the surviving-colour TEST $n$-way rows that the deletion rows score, the first per
rule family: $s_t$ with the action, the engine's $s_{{t+1}}$, the store before deletion, after the
downdate and fitted without the deleted rows (the downdated and the fitted-without store emit the
same outcome on every surviving row, every seed), and the matched MLP as trained and retrained
without. For the arms of Table~\\ref{{tab:cmazet}}:
{dele}}}{{fig:cmazet-deletion}}
% ENDSRC
"""


INDEX_OLD = """panel per data-matched rival set as each set lands."""
INDEX_NEW = """panel per data-matched rival set as each set lands. Figures~\\ref{fig:cmazet-families},
\\ref{fig:cmazet-per-arm}, \\ref{fig:cmazet-door-tint} and~\\ref{fig:cmazet-deletion} draw every arm
of the words-plus-observation condition from its persisted objects; their three placeholders name
the per-arm outputs of the arms of Table~\\ref{tab:cmazet}, which were never persisted."""


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for name in ("DRAFT-v2.tex", "DRAFT-v2-style.tex"):
        t = open(os.path.join(src, name), encoding="utf-8").read()
        before = macro_args(t, "PLACEHOLDER")
        a = t.index(HEAD)
        assert t.count(HEAD) == 1
        b = t.index("% ENDSRC\n", a) + len("% ENDSRC\n")
        old = t[a:b]
        P = placeholders(old)
        assert len(P) == 3, len(P)
        assert "rendered/rc_w19_cmazet_families" in old and "hatched" in old
        t = t[:a] + new_block(P) + t[b:]
        assert t.count(INDEX_OLD) == 1
        t = t.replace(INDEX_OLD, INDEX_NEW)
        assert macro_args(t, "PLACEHOLDER") == before, "a PLACEHOLDER argument changed"
        orig = open(os.path.join(src, name), encoding="utf-8").read()
        assert sum(ord(c) > 127 for c in t) == sum(ord(c) > 127 for c in orig), "new non-ASCII"
        open(os.path.join(dst, name), "w", encoding="utf-8").write(t)
        print(name, "OK; placeholders", len(before))


if __name__ == "__main__":
    main()
