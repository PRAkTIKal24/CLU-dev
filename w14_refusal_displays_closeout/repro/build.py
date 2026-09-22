"""w14 spoke/refusal-displays-closeout — build the registry/matrix deltas as COPIES.

Additive only. Verified mechanically with difflib afterwards (see verify.py).
"""
import io, os, sys, hashlib

BASE = r"C:\chlu\evidence\outputs\w13-registry-fold"
OUT  = r"C:\chlu\outputs\w14-refusal-displays-closeout"
SP   = os.path.dirname(os.path.abspath(__file__))

def rd(p):
    with io.open(p, "r", encoding="utf-8", newline="\n") as f:
        return f.read()

def wr(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

# ---------------------------------------------------------------- negatives
neg   = rd(os.path.join(BASE, "negative_results.md"))
frag  = rd(os.path.join(SP, "frag_neg.md"))
p323  = rd(os.path.join(SP, "ptr_n323.md"))
p819  = rd(os.path.join(SP, "ptr_ni819.md"))

# --- pointer note under N323: anchor = the last line of its SCOPE paragraph
A323 = "solvability-within-budget is undecidable in general.**\n"
assert neg.count(A323) == 1, "N323 anchor not unique: %d" % neg.count(A323)
neg = neg.replace(A323, A323 + p323)

# --- pointer note under N-I8-19: anchor = its closing C.3 clause
A819 = ("**Exact deletion IS delivered; refusal is NOT**\n(`w9-maze/LOG.md:566-577`).\n")
assert neg.count(A819) == 1, "N-I8-19 anchor not unique: %d" % neg.count(A819)
neg = neg.replace(A819, A819 + p819)

# --- the fold block, appended
if not neg.endswith("\n"):
    neg += "\n"
neg += frag
wr(os.path.join(OUT, "negative_results.md"), neg)

# ------------------------------------------------------------------- matrix
mat  = rd(os.path.join(BASE, "claims_matrix.md"))
mfrag = rd(os.path.join(SP, "frag_matrix.md"))
lines = mat.split("\n")
old_title = lines[0]
assert old_title.startswith("# Claims-Consistency Matrix")

NEW_TITLE = (
    "# Claims-Consistency Matrix (M2/P4) \u2014 **v2.21 (the WAVE-14 REFUSAL-DISPLAY CLOSEOUT "
    "\u2014 `w14 spoke/refusal-displays-closeout`, `doc-curator`, 2026-09-22, fold symbol \u2299, "
    "commissioned by `tasks/w14-spoke-refusal-displays-closeout.md` on the Head ruling at "
    "`outputs/advisor-wave-plan-w12-w14.md:56`) \u2014 \u26a0 PROPOSED, PENDING HUB CONFIRMATION "
    "(the curator never self-confirms). \u2b50\u2b50 WHAT THIS VERSION ADDS: **`\u00a7I8.8`** \u2014 "
    "**`CM-W14-1`**, which carries `CM-R12-3` the rest of the way from *\"not yet informative\"* to "
    "**RETIRED BY HEAD RULING**, with the number that retires it (ridge probe **AUC 0.4822\u20130.5518 "
    "on all six seed \u00d7 tier runs at `n_test = 200`**, budget scalar **0.3840\u20130.4589**) and the "
    "**BAR TO REVIVE IT**; and **`CM-W14-2`**, which states that the **Shapes3D refusal-via-the-HOLE is "
    "UNTOUCHED** (`red \u2227 cylinder` refused at **1.0000** for **1 byte**, **0.0000** composition cost) "
    "so no reader generalises the maze's failure to the store's refusal mechanism \u2014 with its honest "
    "other half, that the Shapes3D comparator is a **declared NOT-RUN** (`N334`). \u26d4\u26d4 AND A "
    "CORRECTION THAT BINDS EVERY DRAFT: **`0.9601 / 0.9801 / 0.9781 / 0.8182` ARE 3-SEED MEANS**, printed "
    "as bare values in `N-I8-19`, in `CM-R12-3` and in `outputs/w9-advisor-report.md`; the seeds are "
    "**0.797945 / 0.855960 / 0.800633** and \u26d4 **`0.8182` is NOT seed 2**. \u2b50 The novelty control "
    "is STRENGTHENED by the per-seed read \u2014 it fires on **every** seed, not only on the mean. "
    "\u26d4 No `CM-n` number moved and no row was edited; `\u00a7I8.8` and `\u00a7I8.9` are additive, and "
    "`CM-R12-3`, `N323` and `N-I8-19` are **marked retired, never deleted**. \u26d4 ZERO scored runs, "
    "ZERO compute, ZERO GPU; the curator re-derived no statistic except four declared column means; "
    "`$CHLU_CLAUDE_ROOT` was NOT read. \u2014 PRIOR TITLE, PRESERVED VERBATIM: "
) + old_title

lines[0] = NEW_TITLE
mat = "\n".join(lines)
if not mat.endswith("\n"):
    mat += "\n"
mat += mfrag
wr(os.path.join(OUT, "claims_matrix.md"), mat)

print("WROTE")
for n in ("negative_results.md", "claims_matrix.md"):
    p = os.path.join(OUT, n)
    print("  %-22s %6d lines  md5 %s" % (n, rd(p).count("\n"), md5(p)))
print("TITLE endswith(prior) =", lines[0].endswith(old_title))
print("title chars %d -> %d" % (len(old_title), len(lines[0])))
