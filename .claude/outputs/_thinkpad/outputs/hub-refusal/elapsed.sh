#!/bin/sh
# True wall clock against START_UTC.txt and the real clock (COMMON section 1b).
S="$CHLU_HUB_OUTPUTS/hub-refusal/START_UTC.txt"
"$CHLU_ROOT/.venv/Scripts/python.exe" - "$S" <<'PY'
import datetime as d, sys
s = d.datetime.strptime(open(sys.argv[1]).read().strip(), '%Y-%m-%dT%H:%M:%SZ')
n = d.datetime.now(d.timezone.utc).replace(tzinfo=None)
e = (n - s).total_seconds()
print('START %s  NOW %s  ELAPSED %.2f h of 12.00 h  DEADLINE %s  REMAINING %.2f h'
      % (s.strftime('%Y-%m-%dT%H:%M:%SZ'), n.strftime('%Y-%m-%dT%H:%M:%SZ'), e/3600,
         (s + d.timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ'), 12 - e/3600))
PY
