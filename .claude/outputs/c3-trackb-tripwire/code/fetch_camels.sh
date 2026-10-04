#!/bin/bash
# Sequential, resumable fetcher for the CAMELS record (Zenodo 15529996 = DOI 10.5065/D6MW2F4D).
# One connection at a time (Zenodo throttles concurrent transfers).
set -u
D=/Users/user/Desktop/CHLU/.claude/data/c3-camels
cd "$D" || exit 1
BASE="https://zenodo.org/records/15529996/files"
get () {
  f="$1"
  for try in 1 2 3 4 5 6 7 8 9 10; do
    curl -sSL -C - --retry 5 --retry-delay 5 -o "$f" "$BASE/$f?download=1"
    rc=$?
    echo "$(date +%H:%M:%S) $f try=$try rc=$rc size=$(stat -f%z "$f" 2>/dev/null)"
    if [ $rc -eq 0 ]; then return 0; fi
    sleep 10
  done
}
# wait for any in-flight forcing download to finish first
while pgrep -f "metForcing_obsFlow.zip" > /dev/null; do sleep 20; done
get basin_timeseries_v1p2_metForcing_obsFlow.zip
get basin_timeseries_v1p2_modelOutput_daymet.zip
for f in camels_soil.txt camels_hydro.txt readme.txt camels_attributes_v2.0.pdf; do get "$f"; done
echo "ALL DONE $(date)"
