# AMENDMENT 3 — the Hub's independent scan found a leak your scanner missed (2026-09-26 ~01:25Z)
The Hub re-scanned the harvested package with its own separately written patterns. **The package is NOT clean.** Nothing has been copied to the Head's machine.

## What leaked (counts from the Hub's grep over the 196-file package)
- **Paper-wave IDs:** `PW-1` ×18 and `PW1` ×4. They are in `figures/results_figures_b.py`, `figures/fig_style.py` (`PW1_BANNED`, `PW1_TEXT`) and `figures/data/{ladder_rival_sets,cmazet_panels,maze_conditions}.json` (e.g. `"PW-1 item 10: ..."`, `"retired_by": "PW-1 item 10.1 (, 2026-09-25) ..."`).
  - Your rule `\bpw[0-9]\b` cannot match `PW-1` (hyphen) or `PW1_` (underscore is a word character). Your w-rule's lookbehind excludes the letter before `w`.
- **Scrub scars:** removing a word left the gap visible, and that is itself a tell. Found: `(, 2026-09-25)` ×4, `(, 2026-09-14)` ×1, `-ruled 2026-09-25` ×1 (from "Head-ruled").
- **Internal history:** `retired`, `retired_by` and `rival_sets_retired` fields record which review item removed a panel and when. Also ISO development dates (2026-09-14 … 2026-09-25, 21 occurrences).

## What to do
1. **Fix it in the build transform, not by hand in the output.**
   - Drop review-item references entirely (`PW-<n> item <x>`, `PW<n>`). Rename the `PW1_*` identifiers descriptively (e.g. `FIGURE_TEXT_BANNED`).
   - Remove retired/history fields from the figure data files. If a figure builder reads one, keep only the functional value, under a neutral name. Say which you did. `retired_byte_budget`/`retired_rows`: keep them if they are a functional quantity, rename them if not, and state which.
   - Remove development dates from comments, docstrings and data files.
   - Rebuild.
2. **Extend the scanner and never shrink it.** Add at least `PW-?[0-9]`, `(?i)(?<![a-z])pw[0-9]` (catching `PW1_`), `\bitem [0-9]+(\.[0-9]+)*\b` in comments/strings, the scar patterns `\(\s*,`, `(?<![A-Za-z])-ruled`, `\bruled\b`, and ISO dates `20[0-9]{2}-[01][0-9]-[0-3][0-9]`. Add `retired_by`. Also apply a general check for any word stub left by deletion, meaning a punctuation run directly next to where a banned word was.
   - Positive control must plant `PW-1`, `PW1_X`, and a scar `(, 2026-09-25)`, and catch all of them.
3. **Proof on the rebuilt package:**
   - Re-scan: EMPTY.
   - Fresh-venv re-smoke of every entry point whose files changed.
   - sha256 diff against the proof-C copy. Re-run a full reproduction only if a file it executes changed.
   - Regenerate MANIFEST.md.
   - Update `scan-report.md` and `w24-advisor-report.md`: add a short §9 "Hub re-scan findings and fixes" with before/after counts.
4. **When genuinely finished, write `AMEND3_END_UTC.txt`.** That is the file the Hub's watcher now reads. Leave `END_UTC.txt` alone.

Same rules as before: nothing written to any paper folder; no `tmux kill-server`; no `rm <glob>`; poll background jobs by PID captured at launch (never `pgrep -f` with a string that appears in your own command line).
