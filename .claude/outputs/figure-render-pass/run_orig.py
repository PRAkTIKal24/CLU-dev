"""Run an ORIGINAL figure script under the data tap, redirecting its savefig."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tap
SAVE = os.path.join(HERE, "orig_renders"); os.makedirs(SAVE, exist_ok=True)
tap.redirect_saves(SAVE)
script, out = sys.argv[1], sys.argv[2]
g = {"__file__": os.path.abspath(script), "__name__": "__main__"}
exec(compile(open(script).read(), script, "exec"), g)
tap.dump(out)
