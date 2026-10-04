"""Amendment 1: frozen-encoder run shipped; legacy loaders read the shipped array files."""


def apply(P):
    apply_reports(P)
    apply_lace_out(P)
    apply_sweep(P)
    apply_plan2(P)
    apply_plan(P)
    apply_gencell(P)
    apply_cache(P)
    apply_rows(P)
    apply_lace_table(P)
    apply_sites(P)
    # a checkpoint that ships as an array-only file under weights/ is rebuilt from it
    P.redef('chlu/utils/checkpoints.py', 'load_model', '''def load_model(path: Union[str, Path], model_template: Optional[Any] = None) -> Any:
    """Load a model: a pickle written by this package, or a shipped array-only file (.eqx)."""
    path = Path(path)
    if not path.exists() and path.with_suffix(".eqx").exists():
        from ..weights import load_eqx
        return load_eqx(path.with_suffix(".eqx"))[0]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    return eqx.combine(checkpoint["params"], checkpoint["static"])
''')
    # frozen-encoder run root (the old bridge root) and the feature-cache root
    R = 'chlu/experiments/exp_w4_rivals.py'
    P.redef(R, 'bridge_root', '''def bridge_root() -> Path:
    r = os.environ.get("CHLU_GEN_BRIDGE_ROOT")
    if r:
        return Path(r)
    from ..paths import weights_dir
    return weights_dir() / "encoder"
''')
    P.redef(R, 'gen2_root', '''def gen2_root() -> Path:
    from ..paths import features_root
    return features_root() / PRETEXT_DIR
''')
    # a feature cache is identified by the package's data manifest when it carries none itself
    C = 'chlu/eval/rivals/w4_common.py'
    P.rep(C, '''    d = Path(gen2_dir)
    p = d / "feature_cache.json"
    if not p.exists():
        raise FileNotFoundError(''', '''    d = Path(gen2_dir)
    p = d / "feature_cache.json"
    if not p.exists() and (d / "features.npy").exists():
        from ...paths import data_manifest
        m = data_manifest()["features"]
        seed = int(d.name.split("_")[-1])
        man = {"generation": 2, "scope": m["scope"], "features_file": "features.npy",
               "rows_file": None, "features_sha256": m["seeds"][str(seed)]["sha256"],
               "encoder_param_hash": m["seeds"][str(seed)]["encoder_param_hash"],
               "n_rows": m["n_rows"], "feat_shape": m["feat_shape"], "dtype": m["dtype"],
               "device": {"devices": ["not recorded"]}}
        if verify:
            from ...data.gen_splits import file_sha256 as _sha
            assert _sha(d / "features.npy") == man["features_sha256"]
        return (np.arange(int(m["n_rows"]), dtype=np.int64),
                np.load(d / "features.npy", mmap_mode="r"), man)
    if not p.exists():
        raise FileNotFoundError(''')
    P.rep(C, '''    claude_root = Path(config.gen_cell.claude_root or os.environ.get("CHLU_CLAUDE_ROOT"))''',
          '''    from ...paths import out_root
    claude_root = Path(config.gen_cell.claude_root or out_root())''')
    P.rep(C, '''    root = Path(cfg.classifier_root).expanduser() if cfg.classifier_root else (
        claude_root / "outputs" / "gen-external-classifier")''', '''    from ...paths import weights_dir
    root = Path(cfg.classifier_root).expanduser() if cfg.classifier_root else (weights_dir() / "classifier")''')


def apply_sites(P):
    # the pretext-run roots default to the shipped weights/ tree
    for f in ('chlu/experiments/exp_gen_cell.py',):
        P.rep(f, '''    croot = Path(cfg.classifier_root).expanduser() if cfg.classifier_root else (
        _cell_root(config) / "outputs" / "gen-external-classifier"
    )''', '''    from ..paths import weights_dir
    croot = Path(cfg.classifier_root).expanduser() if cfg.classifier_root else (weights_dir() / "classifier")''')
        P.rep(f, '''    root = Path(cfg.classifier_root).expanduser() if cfg.classifier_root else (
        _cell_root(config) / "outputs" / "gen-external-classifier"
    )''', '''    from ..paths import weights_dir
    root = Path(cfg.classifier_root).expanduser() if cfg.classifier_root else (weights_dir() / "classifier")''')
    P.rep('chlu/experiments/exp_gen_compose.py', '''        root = Path(cfg.winner_root).expanduser() if cfg.winner_root else (
            _claude_root(config) / "outputs" / "gen-decoder-lift" / "winner"
        )''', '''        from ..paths import weights_dir
        root = Path(cfg.winner_root).expanduser() if cfg.winner_root else (weights_dir() / "decoder")''')
    # a shipped array file stands in for the checkpoint an existence check looks for
    for f in ('chlu/experiments/exp_gen_cell.py', 'chlu/eval/rivals/w4_common.py'):
        P.rep(f, '    if not ckpt.exists():', '    if not (ckpt.exists() or ckpt.with_suffix(".eqx").exists()):')
    EL = 'chlu/experiments/exp_w8_encoder_loop.py'
    P.rep(EL, '    if not ckpt.exists():', '    if not (ckpt.exists() or ckpt.with_suffix(".eqx").exists()):')
    P.redef(EL, 'data_root', '''def data_root() -> Path:
    from ..paths import data_root as _root
    p = _root() / PINS["dataset"]
    if not p.is_dir():
        raise FileNotFoundError(f"no {PINS['dataset']} under {p}")
    return p
''')
    P.redef(EL, 'bridge_seed_dir', '''def bridge_seed_dir(seed: int) -> Path:
    r = os.environ.get("CHLU_GEN_BRIDGE_ROOT")
    if r:
        base = Path(r)
    else:
        from ..paths import weights_dir
        base = weights_dir() / "encoder"
    return base / f"{PINS['dataset']}_{PINS['pretext']}" / f"seed_{int(seed)}"
''')
    P.redef(EL, 'classifier_dir', '''def classifier_dir(seed: int) -> Path:
    from ..paths import weights_dir
    return weights_dir() / "classifier" / f"seed_{int(seed)}"
''')


def apply_lace_table(P):
    P.redef('scripts/w15_lace_poe_table.py', 'shipped_sources', '''def shipped_sources(seed: int) -> dict:
    """arm -> (per_corner rows, source path) on rows 3+4 at one seed, from this package's
    scoring outputs (scripts/shapes_score_arms.py --corners rows34)."""
    from chlu.paths import run_dir
    d = run_dir("shapes3d", "scores", "rows34", "seed%d" % seed)
    out = {}
    p = d / "store_K32_n131072.json"
    if p.exists():
        out["kstar_store"] = (jload(p)["per_corner"], rel(p))
    for c in KCTL_SHIPPED:
        p = d / ("store_K32_n131072__%s.json" % c)
        if p.exists():
            out[f"kstar_store__{c}"] = (jload(p)["per_corner"], rel(p))
    p = d / "table_FLOOR.json"
    if p.exists():
        out["table_FLOOR"] = (jload(p)["per_corner"], rel(p))
    return out
''')


def apply_rows(P):
    P.rep('chlu/eval/rivals/w4_common.py', '''    rows = np.load(d / man["rows_file"])''', '''    rp = d / man["rows_file"]
    if rp.exists():
        rows = np.load(rp)
    elif man.get("scope") == "full":
        rows = np.arange(int(man["n_rows"]), dtype=np.int64)
    else:
        raise FileNotFoundError(rp)''')


def apply_cache(P):
    P.rep('chlu/core/gen_bridge.py', '''    """Return ``(rows, features_memmap, manifest)`` for a cache written above."""
    out_dir = Path(out_dir)
    with open(out_dir / "feature_cache.json") as fh:''', '''    """Return ``(rows, features_memmap, manifest)`` for a cache written above.

    The shipped encoder-run directory carries no features: its seed's cache is read from
    NCLU_FEATURES_ROOT (the same layout, ``shapes3d_recon/seed_<s>``)."""
    out_dir = Path(out_dir)
    if not (out_dir / "features.npy").exists():
        from ..paths import features_root
        from ..eval.rivals.w4_common import load_gen2_cache
        return load_gen2_cache(features_root() / out_dir.parent.name / out_dir.name, verify=verify)
    with open(out_dir / "feature_cache.json") as fh:''')


def apply_gencell(P):
    P.rep('scripts/w15_lace_poe_gencell.py', '''    config = get_default_config()
    print(f"chlu.__file__ = {chlu.__file__}", flush=True)''', '''    config = get_default_config()
    # the refusal record of an unrelated experiment gates the other rows of the generation
    # table, not the K* row this launcher adds; it is not part of this package
    config.gen_cell.require_refusal_complete = False
    print(f"chlu.__file__ = {chlu.__file__}", flush=True)''')


def apply_plan(P):
    P.rep('chlu/experiments/exp_gen_compose.py', '''    return (
        _claude_root(config) / "outputs" / "gen-store-grouping" / "runs"
        / f"{dataset}_{pretext}" / "store_plan.json"
    )''', '''    return Path(__file__).resolve().parents[1] / "specs" / f"store_plan_{dataset}_{pretext}.json"''')


def apply_plan2(P):
    P.rep('chlu/experiments/exp_gen_cell.py', '''    return (_cell_root(config) / "outputs" / "gen-store-grouping" / "runs"
            / f"{dataset}_{pretext}" / "store_plan.json")''', '''    return Path(__file__).resolve().parents[1] / "specs" / f"store_plan_{dataset}_{pretext}.json"''')


def apply_sweep(P):
    P.redef('scripts/w15_crossover_seeds.py', '_assert_own_checkout', '''def _assert_own_checkout():
    """The imported package must be the one this script belongs to."""
    import chlu
    cf = pathlib.Path(chlu.__file__).resolve()
    root = pathlib.Path(__file__).resolve().parents[1]
    assert root in cf.parents, f"the package resolves outside {root}: {cf}"
    print(f"OK package={cf}", flush=True)
''')
    P.rep('scripts/w15_crossover_report.py', '''    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)''', '''    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)''')


def apply_lace_out(P):
    P.rep('scripts/w15_lace_poe_table.py', '''    with open(OUT / "GENTABLE-KSTAR.json", "w", encoding="utf-8") as fh:''', '''    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "GENTABLE-KSTAR.json", "w", encoding="utf-8") as fh:''')


def apply_reports(P):
    C = 'scripts/w15_coverage_report.py'
    P.rep(C, '''def main() -> int:
    res = {''', '''def main() -> int:
    global SEEDS
    present = tuple(s for s in SEEDS if (OUT / "val" / ("seed%d_K32" % s) / "LEG.json").exists())
    missing = [s for s in SEEDS if s not in present]
    if not present:
        print("no coverage-control leg under %s: run scripts/shapes_coverage_control.py first" % (OUT / "val"))
        return 0
    if missing:
        print("seeds without a coverage-control leg, not aggregated: %s" % missing)
    SEEDS = present
    res = {''')
    P.rep(C, '''           "n_seeds": len(SEEDS), "seeds": list(SEEDS), "val": {}, "test_context": {}}''',
          '''           "n_seeds": len(SEEDS), "seeds": list(SEEDS), "seeds_missing": missing,
           "val": {}, "test_context": {}}''')
    P.rep(C, '''            t["n%d_seed%d" % (r, s)] = {
                "store": load(f)["summary"]["distinct_valid_novel_tuples_mean"],
                "learned_1x": load(g)["summary"]["distinct_valid_novel_tuples_mean"],
                "file": str(f)}
    res["test_context"] = {''', '''            if not f.exists():
                continue
            t["n%d_seed%d" % (r, s)] = {
                "store": load(f)["summary"]["distinct_valid_novel_tuples_mean"],
                "learned_1x": (load(g)["summary"]["distinct_valid_novel_tuples_mean"]
                               if g.exists() else None),
                "file": str(f)}
    need = ["n%d_seed%d" % (r, s) for r in (3, 6) for s in SEEDS]
    if not all(k in t for k in need):
        res["test_context"] = {"not_present": [k for k in need if k not in t],
                               "note": "the K* ladder legs of rungs 3 and 6 (scripts/shapes_ladder_nclu.py) are read here"}
    else:
     res["test_context"] = {''')
    P.rep(C, '''                      "test": {k: res["test_context"][k] for k in
                               ("A_test_n3", "B_test_n6", "rise_test_B_minus_A")}}, indent=1))''',
          '''                      "test": {k: res["test_context"].get(k) for k in
                               ("A_test_n3", "B_test_n6", "rise_test_B_minus_A")}}, indent=1, default=str))''')
    R = 'scripts/w15_crossover_report.py'
    P.rep(R, "          f\"{r['drop_from_K5']:+.4f} | {t['mean']:,.0f} | \"",
          "          f\"{_opt(r['drop_from_K5'])} | {t['mean']:,.0f} | \"")
    P.rep(R, "def _sd(x):", '''def _opt(x):
    return "n/a (K=5 not present)" if x is None else "%+.4f" % x


def _sd(x):''')
