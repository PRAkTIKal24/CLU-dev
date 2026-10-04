"""Remaining Shapes3D scripts: roots, import assertions, gates."""


def apply(P):
    apply_fsq(P)
    apply_ladder_default(P)
    apply_store(P)
    P.redef('chlu/data/gen_splits.py', 'default_gen_data_root', '''def default_gen_data_root() -> Path:
    """Dataset root (``NCLU_DATA_ROOT``, default ``<package>/data``)."""
    from ..paths import data_root
    return data_root()
''')
    P.redef('chlu/experiments/exp_gen_cell.py', '_cell_root', '''def _cell_root(config: CHLUConfig) -> Path:
    r = config.gen_cell.claude_root
    if r:
        return Path(r).expanduser()
    from ..paths import out_root
    return out_root()
''')
    P.redef('chlu/experiments/exp_gen_compose.py', '_claude_root', '''def _claude_root(config: CHLUConfig) -> Path:
    r = config.gen_compose.claude_root
    if r:
        return Path(r).expanduser()
    from ..paths import out_root
    return out_root()
''')
    P.rep('chlu/config.py', '".claude/outputs/c2w10-benchmark-gate/BENCHMARK-GATE.json"', '"runs/benchmark_gate.json"')
    # import assertions keyed on old checkout names
    P.rep('scripts/w12_ladder_leg.py', '''if "CHLU-w12-composition-ladder" not in str(chlu.__file__):
    raise AssertionError("chlu resolves OUTSIDE this worktree: %s" % chlu.__file__)
''', '''from pathlib import Path as _P
if _P(__file__).resolve().parents[1] not in _P(chlu.__file__).resolve().parents:
    raise AssertionError("the package resolves outside this checkout: %s" % chlu.__file__)
''')
    for f in ('scripts/w12b_ladder_leg.py', 'scripts/w12b_ladder_1x.py'):
        P.rep(f, 'raise AssertionError("chlu resolves OUTSIDE this worktree: %s" % chlu.__file__)',
              'raise AssertionError("the package resolves outside this checkout: %s" % chlu.__file__)')
    P.rep('chlu/eval/rivals/w12_deletion_baselines.py',
          'root = Path(expected_root or os.environ.get("CHLU_ROOT") or Path.cwd()).resolve()',
          'root = Path(expected_root or Path(__file__).resolve().parents[3]).resolve()')
    # content-rows (n-CLU-content ladder set) library: same shape as the shared library
    L = 'scripts/w21L_lib.py'
    P.rep(L, '''OUT = Path("C:/chlu/outputs/w21-ladder-rivals-32k")
W20_OUT = Path("C:/chlu/outputs/w20-shapes-data-matched")   # READ-ONLY, never written
LEDGER = Path("C:/chlu/SCORED-LEDGER.md")''', '''from chlu.paths import partition_file, run_dir  # noqa: E402

OUT = run_dir("shapes3d", "content_rows")
W20_OUT = run_dir("shapes3d")''')
    P.rep(L, '''PART_VAL = "C:/chlu/outputs/hub-factor-wells/run-val/partitions.npz"
PART_TEST = "C:/chlu/outputs/hub-factor-wells/run/partitions.npz"
KSTAR_LADDER = Path("C:/chlu/outputs/w12b-nr6-and-kstar/kstar_ladder")''', '''PART_VAL = str(partition_file("val"))
PART_TEST = str(partition_file("test"))
KSTAR_LADDER = run_dir("shapes3d", "ladder_nclu")''')
    P.delete_def(L, 'ledger_has')
    S = 'scripts/w21L_score.py'
    P.rep(S, 'from w21L_lib import (OUT, assert_worktree, jdump, ledger_has, log,',
          'from w21L_lib import (OUT, PART_TEST, assert_worktree, jdump, log, rung_partition,')
    P.rep(S, '''    if a.corners == "rows34":
        if not ledger_has(int(a.run_n)):
            raise SystemExit("LEDGER ROW MISSING for run %d/3 -- refusing to score TEST "
                             "corners. The row is written BEFORE the run." % a.run_n)
        log("LEDGER GATE OPEN: run %d/3" % a.run_n)
''', '')
    P.rep(S, '''        pool0 = (np.load(str(Path("C:/chlu/outputs/w12b-nr6-and-kstar/kstar_ladder")
                             / ("%s_seed0_K32" % rung) / "partitions.npz"))["pool"]
                 if rung is not None else
                 np.load("C:/chlu/outputs/hub-factor-wells/run/partitions.npz")["pool"])''',
          '''        pool0 = (np.load(str(rung_partition(rung, 0)))["pool"]
                 if rung is not None else np.load(PART_TEST)["pool"])''')


def apply_store(P):
    P.redef('chlu/experiments/exp_gen_store.py', 'bridge_root', """def bridge_root(config: CHLUConfig) -> Path:
    \"\"\"Directory of the frozen encoder run (NCLU_ENCODER_ROOT; not shipped, see README).\"\"\"
    r = config.gen_store.bridge_root or os.environ.get("NCLU_ENCODER_ROOT")
    if r:
        return Path(r).expanduser()
    from ..paths import ROOT
    return ROOT / "weights" / "encoder"
""")


def apply_ladder_default(P):
    P.rep('scripts/w12b_ladder_kstar.py', 'ap.add_argument("--arms", default="store,controls,learned_1x")',
          'ap.add_argument("--arms", default="store,controls")')


def apply_fsq(P):
    P.rep('chlu/experiments/exp_w8_downdate.py', '''    z = dict(np.load(out_dir / "partitions.npz"))''', '''    p = Path(out_dir) / "partitions.npz"
    if not p.exists():
        import shutil
        from ..paths import partition_file
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(partition_file(str(fw.PINS["condition"])), p)
    z = dict(np.load(p))''')
