#!/bin/sh
S="$(cat "$(dirname "$0")/START_UTC.txt")"
"/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Scripts/python.exe" - "$S" <<'PY'
import sys, datetime as dt
s = dt.datetime.strptime(sys.argv[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
n = dt.datetime.now(dt.timezone.utc)
e = (n - s).total_seconds()
print(f"start {s:%Y-%m-%dT%H:%M:%SZ}  now {n:%Y-%m-%dT%H:%M:%SZ}  elapsed {e/3600:.3f} h of 12.000 h  remaining {(12*3600-e)/3600:.3f} h  deadline {(s+dt.timedelta(hours=12)):%Y-%m-%dT%H:%M:%SZ}")
PY
