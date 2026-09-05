"""Run a figure script under the figure-render-pass data tap and dump its digest.
usage: run_tap.py <script.py> <out.json>
"""
import os, sys
FRP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure-render-pass")
sys.path.insert(0, os.path.abspath(FRP))
import tap
SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tap_renders")
os.makedirs(SAVE, exist_ok=True)
os.environ["FRP_OUT"] = SAVE
tap.redirect_saves(SAVE)
script, out = sys.argv[1], sys.argv[2]
g = {"__file__": os.path.abspath(script), "__name__": "__main__"}
exec(compile(open(script).read(), script, "exec"), g)
tap.dump(out)
