"""Figures: every input is this package's own run output (NCLU_OUT); code root = package root."""


def apply(P):
    apply_labels(P)
    apply_more(P)
    # style / evidence root
    for f, old in (('figures/make_figures.py', 'EVIDENCE = os.environ.get("CHLU_EVIDENCE", "/Users/batmac/clu/evidence/outputs")'),
                   ('figures/maze_six.py', 'EVID = os.environ.get("CHLU_EVIDENCE", "/Users/batmac/clu/evidence/outputs")'),
                   ('figures/cmazet_extract.py', 'EVID = os.environ.get("CHLU_EVIDENCE", "/Users/batmac/clu/evidence/outputs")'),
                   ('figures/cmazet_extract_rel.py', 'EVID = os.environ.get("CHLU_EVIDENCE", "/Users/batmac/clu/evidence/outputs")')):
        var = old.split(' =')[0]
        P.rep(f, old, 'from chlu.paths import out_root as _out_root\n%s = str(_out_root())' % var)
    # code root = the package root (figures/ sits in it)
    FR = 'figures/figroot.py'
    P.rep(FR, 'ENGINE_PROBE = os.path.join("chlu", "exp_w15_cmaze_t.py")', 'ENGINE_PROBE = os.path.join("chlu", "paths.py")')
    P.rep(FR, 'ENGINE_TOPS = ("chlu", "w8_envscout", "w9_harness", "w15_cmaze_t_run", "w16_cmaze_t")', 'ENGINE_TOPS = ("chlu",)')
    P.redef(FR, 'layout', '''def layout():
    return "repo"
''')
    P.redef(FR, 'code_root', '''def code_root():
    if os.path.isfile(os.path.join(PARENT, ENGINE_PROBE)):
        return PARENT
    raise SystemExit("figroot: the package root was not found above %s" % HERE)
''')
    # per-item re-read of the persisted base n-CLU objects
    CE = 'figures/cmazet_extract.py'
    P.rep(CE, 'import w19_symmetric as S                 # noqa: E402', 'import cmazet_symmetric_runner as S     # noqa: E402')
    # ... and of the persisted n-CLU-REL objects: the one runner, its own persisted tree
    CR = 'figures/cmazet_extract_rel.py'
    P.rep(CR, '''REPO = os.environ.get("CHLU_FIG_CODE") or os.path.dirname(HERE)
VEND = os.path.join(REPO, "peer_w21_clu_improve_e7fdb9b")
if not os.path.isdir(VEND):
    raise SystemExit("cmazet_extract_rel.py needs a full CHLU checkout with %s: set CHLU_FIG_CODE"
                     % os.path.basename(VEND))
for p in (VEND, os.path.join(REPO, "scripts"), REPO):''', '''REPO = os.path.dirname(HERE)
for p in (os.path.join(REPO, "scripts"), REPO):''')
    P.rep(CR, '''import w21_clu                             # noqa: E402
assert os.path.dirname(os.path.abspath(w21_clu.__file__)) == VEND, w21_clu.__file__

_spec = importlib.util.spec_from_file_location("peer_w19_symmetric_e7fdb9b",
                                               os.path.join(VEND, "w19_symmetric.py"))
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
''', '''import w21C_clu as w21_clu                  # noqa: E402
import cmazet_symmetric_runner as V        # noqa: E402
''')
    P.rep(CR, 'STAGE = "/Users/batmac/clu/stage/rel-persist"\n', '')
    P.rep(CR, '    st = Stage(os.path.join(STAGE, branch, "BOTH_seed%d" % seed, "test"))',
          '    st = Stage(os.path.join(EVID, branch, "test", "persist", "BOTH_seed%d" % seed, "test"))')
    # rendered Shapes3D panels: objects built by this package
    C = 'pw1_render_shapes/common.py'
    P.rep(C, '''OUT = Path(os.environ.get("PW1_RS_OUT", "/Users/batmac/clu/outputs/pw1-render-shapes"))
STAGE = Path(os.environ.get("W17_RC_STAGE", "/Users/batmac/clu/stage"))
W20_STAGE = STAGE / "w20-shapes-data-matched"
W20_MANIFEST = W20_STAGE / "SHA256-w20-test-objects.txt"
EVIDENCE = Path("/Users/batmac/clu/evidence/outputs")
W20_EV = EVIDENCE / "w20-shapes-data-matched"
SHAPES_MATCHED = W20_EV / "SHAPES-MATCHED.json"
WINNER_HASHES = EVIDENCE / "gen-decoder-lift" / "winner" / "winner_hashes.json"
''', '''from chlu.paths import run_dir, weights_dir
OUT = run_dir("figures", "render_shapes")
W20_STAGE = run_dir("shapes3d")
W20_EV = run_dir("shapes3d")
SHAPES_MATCHED = W20_EV / "SHAPES-MATCHED.json"
WINNER_HASHES = weights_dir() / "weights.json"
''')
    P.rep(C, '''CLU_PKL = {
    0: STAGE / "outputs/w12-single-design-crossover/scored/run1_seed0_K32/arm.pkl",
    1: STAGE / "outputs/w12b-nr6-and-kstar/scored/run3_seed1_K32/arm.pkl",
    2: STAGE / "outputs/w12b-nr6-and-kstar/scored/run3_seed2_K32/arm.pkl",
}''', '''CLU_PKL = {s: run_dir("shapes3d", "stores", "test", "seed%d" % s, "K32_n131072", "arm.pkl")
           for s in (0, 1, 2)}''')
    P.redef(C, 'setenv', '''def setenv():
    sys.path.insert(0, str(REPO.parent / "scripts"))
''')
    P.rep(C, '    tr = STAGE / "outputs" / "spoke-a-rivals" / "rivals"', '    tr = run_dir("shapes3d", "baseline_suite", "rivals")')
    P.redef(C, 'verify_manifest', '''def verify_manifest() -> dict:
    """The objects are built by this package's own commands; nothing staged is re-verified."""
    return {"n_files": 0, "all_match": None, "objects": "built by this package"}
''')
    P.redef('w17_rendered_compositions/same_decoder.py', 'expected_hash', '''def expected_hash(winner_hashes: Path, seed: int) -> str:
    wh = json.loads(Path(winner_hashes).read_text(encoding="utf-8"))
    return str(wh["decoder"]["files"][str(int(seed))]["param_hash"])
''')
    B = 'pw1_render_shapes/build.py'
    P.rep(B, '"object": ctx["meta"]["objects"][c].replace("/Users/batmac/clu/", "~/clu/")}',
          '"object": ctx["meta"]["objects"][c]}')
    P.rep(B, 'dict(g1, manifest="~/clu/stage/w20-shapes-data-matched/SHA256-w20-test-objects.txt")', 'dict(g1)')
    # task assets: the request list written by the rendered-panels build
    P.rep('w19_task_figures/extract_assets.py',
          'REQUESTS = Path("/Users/batmac/clu/outputs/w17-paper/figures/rendered/REQUESTS.json")',
          'from chlu.paths import run_dir\nREQUESTS = run_dir("figures", "render_shapes", "REQUESTS-pw1.json")')


def apply_more(P):
    CR = 'figures/cmazet_extract_rel.py'
    P.rep(CR, 'CFG = os.path.join(VEND, "CLU-REL-A10.json")', 'CFG = os.path.join(REPO, "scripts", "configs", "CLU-REL-A10.json")')
    P.rep(CR, '''               "code": {"vendored_from": "w21/rel-newseeds @ 31eeadd (bundle); runner+w21_clu = "
                                         "w21/maze-t-clu-improve @ e7fdb9b",
                        "w21_clu.py_sha256": fsha(os.path.join(VEND, "w21_clu.py")),
                        "w19_symmetric.py_sha256": fsha(os.path.join(VEND, "w19_symmetric.py"))}}''',
          '''               "code": {"relational_nclu_sha256": fsha(w21_clu.__file__),
                        "runner_sha256": fsha(V.__file__)}}''')


def apply_labels(P):
    P.rep('figures/make_figures_w21.py', """FORBIDDEN = ("PLACEHOLDER", "placeholder", "wave", "w1", "w2", "spoke", "regenerat", "pending",
             "Hub", "Head", "TODO")""", """FORBIDDEN = ("regenerat", "pending")""")
    P.rep('scripts/w20_tables.py', '    sys.path.insert(0, "C:/chlu/worktrees/CHLU-w20-shapes-data-matched")\n', '')
    P.rep('scripts/w8_single_design_dev.py', 'z = np.load(r"C:\\chlu\\data\\gen\\shapes3d\\factor_index.npz", allow_pickle=True)',
          'from chlu.paths import data_root\n    z = np.load(str(data_root() / "shapes3d" / "value_holdout" / "factor_index.npz"), allow_pickle=True)')
