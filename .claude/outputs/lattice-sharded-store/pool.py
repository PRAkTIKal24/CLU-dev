"""Priority work-pool: keep P workers busy pulling (cell/arm/seed[/local]) units.

Units are listed one per line in a queue file, highest priority first; each is run
by `run_cells.py` in its own process (each uses ~1 core: the read is a sequential
1200-step lax.scan, latency-bound). Results append to per-unit JSONL files, so a
kill at any point leaves every completed unit on disk.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/Users/user/Desktop/CHLU/.venv/bin/python"
ENV = dict(os.environ, PYTHONPATH="/Users/user/Desktop/CHLU-shard")


def main():
    queue_file, n_workers = sys.argv[1], int(sys.argv[2])
    units = [
        ln.strip()
        for ln in open(os.path.join(HERE, queue_file))
        if ln.strip() and not ln.startswith("#")
    ]
    running, i = [], 0
    while i < len(units) or running:
        while len(running) < n_workers and i < len(units):
            unit = units[i]
            local = unit.endswith("/local")
            spec = unit[: -len("/local")] if local else unit
            tag = spec.replace(":", "_").replace("/", "-") + ("-loc" if local else "")
            cmd = [PY, os.path.join(HERE, "run_cells.py"),
                   "--out", f"u_{tag}.jsonl", "--work", spec]
            if local:
                cmd.append("--init-local")
            log = open(os.path.join(HERE, f"u_{tag}.log"), "w")
            p = subprocess.Popen(cmd, cwd=HERE, env=ENV, stdout=log, stderr=log)
            running.append((p, unit, time.time(), log))
            print(f"START {unit}", flush=True)
            i += 1
        time.sleep(10)
        for rec in list(running):
            p, unit, t0, log = rec
            if p.poll() is not None:
                log.close()
                running.remove(rec)
                print(f"DONE  {unit} rc={p.returncode} [{time.time() - t0:.0f}s]",
                      flush=True)
    print("pool complete", flush=True)


if __name__ == "__main__":
    main()
