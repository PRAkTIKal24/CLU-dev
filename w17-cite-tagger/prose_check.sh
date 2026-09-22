#!/bin/bash
# Prose-untouched proof: strip exactly the inserted commands (~\citep{..} and \nocite{..})
# from the cited draft; the result must be byte-identical to the pre-cite draft.
# Also prints word counts (wc -w) and a word-level diff count.
set -u
pre=$1; cited=$2
strip() { perl -0pe 's/~\\citep\{[a-z0-9,]+\}//g; s/\\nocite\{[a-z0-9,]+\}//g' "$1"; }
if cmp -s <(strip "$cited") "$pre"; then echo "BYTE-IDENTICAL after stripping inserted cites: YES"; else echo "BYTE-IDENTICAL: NO"; fi
echo "pre-cite words (wc -w):              $(wc -w < "$pre")"
echo "cited words (wc -w):                 $(wc -w < "$cited")"
echo "cited, cites stripped (wc -w):       $(strip "$cited" | wc -w)"
echo "inserted \\citep: $(grep -o '~\\citep{' "$cited" | wc -l)   inserted \\nocite: $(grep -o '\\nocite{' "$cited" | wc -l)"
echo "word-level diff (pre vs cited-stripped), changed words: $(diff <(tr -s '[:space:]' '\n' < "$pre") <(strip "$cited" | tr -s '[:space:]' '\n') | grep -c '^[<>]')"
echo "line-level diff pre vs cited (lines touched): $(diff "$pre" "$cited" | grep -c '^<')"
