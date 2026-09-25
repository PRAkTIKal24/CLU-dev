"""w17 spoke/rendered-compositions -- the CLU beside every rival, same requests, same decoder.

Three panel sets (CLU composes / rivals fail; ALL arms succeed; ALL arms fail) plus the
red-sphere deletion panel, rendered through THE frozen decoder the scored T7 leg judged
with, and marked with that judge's per-request verdict.

`selection` reads ONLY the scored per-corner verdicts in the evidence base and fixes which
corners each set draws from, by a stated rule.  `render` expands those corners to exact
request tuples and renders them; it refuses to run on anything but the hash-asserted
decoder.  `figure` draws the grids for both the real render and the declared placeholder.
"""
