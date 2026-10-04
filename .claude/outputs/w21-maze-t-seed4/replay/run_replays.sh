source /c/chlu/outputs/w21-maze-t-seed4/env.sh
W=C:/chlu/outputs/w19-maze-t-symmetric; F=C:/chlu/outputs/w21-words-full-test/test; O=C:/chlu/outputs/w21-maze-t-seed4/replay
for s in 0 2; do python scripts/w21_pertuple_replay.py $O/BOTH_test_seed${s}_PERTUPLE.json --run_json $W/test/BOTH_test_seed$s.json --persist $W/persist/BOTH_seed$s/test; done
for s in 0 1 2; do python scripts/w21_pertuple_replay.py $O/WORDS-FULL_test_seed${s}_PERTUPLE.json --run_json $F/WORDS-FULL_test_seed$s.json --persist $F/persist/WORDS-FULL_seed$s/test; done
for s in 0 1 2; do python scripts/w21_pertuple_replay.py $O/OBS_val_seed${s}_PERTUPLE.json --run_json $W/val/OBS_val_seed$s.json --persist $W/persist/OBS_seed$s/val; done
echo REPLAYS-DONE
