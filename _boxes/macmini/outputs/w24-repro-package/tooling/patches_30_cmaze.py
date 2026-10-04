"""CMAZE-T: one runner for every seed and condition; one run-directory layout.

  runs/cmaze_t/symmetric/{val,test}/<COND>_<leg>_seed<s>.json    base n-CLU, every condition
  runs/cmaze_t/relational/{val,test}/BOTH_<leg>_seed<s>.json      n-CLU-REL (--clu_cfg)
"""


def apply(P):
    T = 'scripts/w21_tables.py'
    P.rep(T, '''OUT_W19 = "C:/chlu/outputs/w19-maze-t-symmetric"
OUT_WF = "C:/chlu/outputs/w21-words-full-test"
OUT_PEER = "C:/chlu/outputs/w21-maze-t-clu-improve"''', '''from chlu.paths import run_dir
OUT_W19 = str(run_dir("cmaze_t", "symmetric"))
OUT_WF = str(run_dir("cmaze_t", "symmetric"))
OUT_PEER = str(run_dir("cmaze_t", "relational"))''')
    P.rep(T, '''            "CLU-REL-A10-BOTH-VAL": [os.path.join(OUT_PEER if s < 3 else out,
                                                  "val_full" if s < 3 else "val_cluimprove",
                                                  "BOTH_val_seed%d.json" % s)]}''',
          '''            "CLU-REL-A10-BOTH-VAL": [os.path.join(OUT_PEER, "val", "BOTH_val_seed%d.json" % s)]}''')
    R = 'scripts/w21_relnew_tables.py'
    P.rep(R, '''OUT_PEER = "C:/chlu/outputs/w21-maze-t-clu-improve"
OUT_W19 = "C:/chlu/outputs/w19-maze-t-symmetric"
OUT_S4 = "C:/chlu/outputs/w21-maze-t-seed4"''', '''from chlu.paths import run_dir
OUT_PEER = str(run_dir("cmaze_t", "relational"))
OUT_W19 = str(run_dir("cmaze_t", "symmetric"))
OUT_S4 = str(run_dir("cmaze_t", "symmetric"))''')
    P.rep(R, '''             "REL-VAL": os.path.join(OUT_PEER if s < 3 else OUT_S4,
                                     "val_full" if s < 3 else "val_cluimprove",
                                     "BOTH_val_seed%d.json" % s)}''', '''             "REL-VAL": os.path.join(OUT_PEER, "val", "BOTH_val_seed%d.json" % s)}''')
    # seed-5 content diagnosis: the merged runner's own persisted objects and per-tuple print
    TR = 'scripts/w23_seed5_trace.py'
    P.rep(TR, '''import w21_symmetric as S                                           # noqa: E402''',
          '''import cmazet_symmetric_runner as S                                 # noqa: E402''')
    P.rep(TR, '''OUT = "C:/chlu/outputs"
PD = OUT + "/w21-maze-t-seed4/test/persist/BOTH_seed5/test"
RUNJ = OUT + "/w21-maze-t-seed4/test/BOTH_test_seed5.json"
PTJ = OUT + "/w21-maze-t-seed4/replay/check_BOTH_test_seed5_PERTUPLE.json"''', '''from chlu.paths import run_dir
PD = str(run_dir("cmaze_t", "symmetric", "test", "persist", "BOTH_seed5", "test"))
RUNJ = str(run_dir("cmaze_t", "symmetric", "test", "BOTH_test_seed5.json"))
PTJ = RUNJ''')
    P.rep('scripts/w23_seed5_content.py', '''import w21_symmetric as S                                           # noqa: E402''',
          '''import cmazet_symmetric_runner as S                                 # noqa: E402''')
