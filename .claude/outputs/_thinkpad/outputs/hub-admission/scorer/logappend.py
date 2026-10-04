"""Append to the running LOG safely.

⚠ WHY THIS EXISTS.  On this box ``cat >> file <<'EOF'`` from Git Bash wrote at a stale
offset twice in the first ten minutes and clobbered the head of both ``admission_lib.py``
and ``LOG.md``.  Every append from here on goes through this, which opens with mode "a"
and UTF-8 explicitly.  Usage:  python logappend.py <file> <<'EOF' ... EOF   (text on stdin)
"""
import sys

path = sys.argv[1]
text = sys.stdin.buffer.read().decode("utf-8")
with open(path, "a", encoding="utf-8", newline="\n") as fh:
    fh.write(text if text.endswith("\n") else text + "\n")
print(f"appended {len(text)} chars to {path}")
