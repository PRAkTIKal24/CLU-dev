"""Shapes3D core: locations, frozen weights, feature-cache check, gates, self-contained deletion."""

FW = 'chlu/experiments/exp_gen_factor_wells.py'
D7 = 'chlu/experiments/exp_gen_delete_row7.py'


def apply(P):
    apply_more(P)
    apply_read(P)
    # ---------------- locations (factor-wells module is the hub every Shapes3D script uses)
    P.redef(FW, 'data_root', '''def data_root() -> Path:
    from ..paths import data_root as _root
    return _root() / PINS["dataset"]
''')
    P.redef(FW, 'features_root', '''def features_root() -> Path:
    """Root of the frozen-encoder feature caches (``NCLU_FEATURES_ROOT``)."""
    from ..paths import features_root as _root
    return _root()
''')
    P.redef(FW, 'feature_cache_manifest', '''def feature_cache_manifest(seed: int) -> dict:
    """The expected identity of a seed's feature cache, with a cheap on-disk check."""
    from ..paths import data_manifest
    m = data_manifest()["features"]
    s = m["seeds"].get(str(int(seed)))
    p = features_seed_dir(seed) / "features.npy"
    return {"generation": 2, "dtype": m["dtype"], "root": str(features_root()),
            "n_rows": m["n_rows"], "scope": m["scope"], "bytes": m["bytes"],
            "features_sha256": (s or {}).get("sha256"),
            "encoder_param_hash": (s or {}).get("encoder_param_hash"),
            "file": str(p), "present": p.exists(),
            "size_ok": bool(p.exists() and p.stat().st_size == m["bytes"])}
''')
    P.redef(FW, 'bridge_seed_dir', '')
    P.redef(FW, 'classifier_dir', '')
    P.rep(FW, '''        from ..core.gen_bridge import param_hash
        from ..utils.checkpoints import load_model

        wroot = Path(os.environ["CHLU_GEN_DECLIFT_DIR"])
        self.decoder = load_model(wroot / f"decoder_v1_s{int(seed)}.pkl")
        with open(wroot / "winner_hashes.json", encoding="utf-8") as fh:
            wh = json.load(fh)
        exp = wh[f"seed_{int(seed)}"]["param_hash"]
        got = param_hash(self.decoder)
        if got != exp:
            raise AssertionError(
                f"decoder hash {got[:16]} != the winner manifest's {exp[:16]} -- the "
                "decoder on disk is not the one that was gated")
        cdir = classifier_dir(seed)
        self.external = load_model(cdir / "external_classifier.pkl")
        with open(cdir / "classifier_manifest.json", encoding="utf-8") as fh:
            cman = json.load(fh)
        cexp = cman["classifier"]["param_hash"]
        cgot = param_hash(self.external)
        if cgot != cexp:
            raise AssertionError(
                f"external classifier hash {cgot[:16]} != its manifest's {cexp[:16]}")
        self.info = {
            "decoder_param_hash": got, "decoder_recipe": wh[f"seed_{int(seed)}"]["recipe"]["name"],
            "classifier_param_hash": cgot,
            "classifier_gate_balacc_on_real":
                ((cman.get("gate") or {}).get("external") or {}).get("balanced_accuracy"),
            "note": "ONE instrument for every arm, loaded once and hash-asserted",
        }
        log(f"  judge: decoder {got[:12]} ({self.info['decoder_recipe']}) + "
            f"classifier {cgot[:12]}")''', '''        from ..weights import load_classifier, load_decoder

        self.decoder, got = load_decoder(seed)
        self.external, cgot = load_classifier(seed)
        self.info = {"decoder_param_hash": got, "classifier_param_hash": cgot}
        log(f"  judge: decoder {got[:12]} + classifier {cgot[:12]}")''')

    # ---------------- the feature-cache check (replaces the cache-generation assertion)
    P.redef(D7, 'assert_generation2', '''def assert_generation2(seed: int) -> dict:
    """Refuse to run on a missing or truncated feature cache for ``seed``."""
    man = feature_cache_manifest(int(seed))
    if not man["present"]:
        raise AssertionError(
            f"seed {seed}: no feature cache at {man['file']}. Set NCLU_FEATURES_ROOT (README, Data).")
    if not man["size_ok"]:
        raise AssertionError(f"seed {seed}: feature cache {man['file']} has the wrong size.")
    return man
''')
    P.redef(D7, 'assert_out_root', '''def assert_out_root(out: Path) -> Path:
    p = Path(out).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
''')

    # ---------------- shared Shapes3D library
    L = 'scripts/w20_lib.py'
    P.rep(L, '''OUT = Path("C:/chlu/outputs/w20-shapes-data-matched")
LEDGER = Path("C:/chlu/SCORED-LEDGER.md")
HUB_TAG = "w20 hub/shapes-data-matched"
TARGET = "shapes-matched"

PART_VAL = "C:/chlu/outputs/hub-factor-wells/run-val/partitions.npz"
PART_TEST = "C:/chlu/outputs/hub-factor-wells/run/partitions.npz"
KSTAR_LADDER = Path("C:/chlu/outputs/w12b-nr6-and-kstar/kstar_ladder")''',
          '''from chlu.paths import partition_file, run_dir  # noqa: E402

OUT = run_dir("shapes3d")
PART_VAL = str(partition_file("val"))
PART_TEST = str(partition_file("test"))
KSTAR_LADDER = run_dir("shapes3d", "ladder_nclu")''')
    P.redef(L, 'assert_worktree', '''def assert_worktree():
    """The imported package must be the one this script belongs to."""
    import chlu
    wt = Path(chlu.__file__).resolve()
    if ROOT not in wt.parents:
        raise SystemExit("the package resolves to %s, outside %s." % (wt, ROOT))
    return str(wt)
''')
    P.delete_def(L, 'ledger_has')
    P.rep('scripts/w12_crossover_lib.py', '''PART_VAL = "C:/chlu/outputs/hub-factor-wells/run-val/partitions.npz"
PART_TEST = "C:/chlu/outputs/hub-factor-wells/run/partitions.npz"''',
          '''from chlu.paths import partition_file as _partition_file
PART_VAL = str(_partition_file("val"))
PART_TEST = str(_partition_file("test"))''')

    # ---------------- scorer: no gate, own partition for the judge check, no retired arms
    S = 'scripts/w20_score.py'
    P.rep(S, '''from w20_lib import (OUT, assert_worktree, jdump, ledger_has, log,  # noqa: E402
                     perturb_modes_w20)''', '''from w20_lib import (OUT, PART_TEST, assert_worktree, jdump, log,  # noqa: E402
                     perturb_modes_w20, rung_partition)''')
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
    P.rep(S, '''    ap.add_argument("--run-n", type=int, default=0)
''', '')
    P.rep(S, '''        elif kind in ("rival", "w12comp", "table"):
            from chlu.eval.rivals import w4_api
            from chlu.eval.rivals.w12_deletion_baselines import load_comparator
            if kind == "rival":
                _, tag, name = spec.split(":")
                rv = w4_api.load_rival(seed, name, root=str(OUT / "rivals" / tag / "ckpt"))
                label = "%s@%s" % (name, tag)
            elif kind == "w12comp":
                name = spec.split(":")[1]
                rv = load_comparator(seed, name, root="C:/chlu/outputs/w12-deletion-baselines/comparators")
                label = "w12comp_%s" % name
            else:
                rv = w4_api.load_rival(seed, "table")
                label = "table_FLOOR"''', '''        elif kind in ("rival", "table"):
            from chlu.eval.rivals import w4_api
            if kind == "rival":
                _, tag, name = spec.split(":")
                rv = w4_api.load_rival(seed, name, root=str(OUT / "rivals" / tag / "ckpt"))
                label = "%s@%s" % (name, tag)
            else:
                rv = w4_api.load_rival(seed, "table")
                label = "table_FLOOR"''')

    # ---------------- deletion: delete from this package's own K* object; "before" is
    # scored here instead of being read from an earlier record
    DS = 'scripts/w20_delete_store.py'
    P.rep(DS, '''T7_ARM = {0: "C:/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/arm.pkl",
          1: "C:/chlu/outputs/w12b-nr6-and-kstar/scored/run3_seed1_K32/arm.pkl",
          2: "C:/chlu/outputs/w12b-nr6-and-kstar/scored/run3_seed2_K32/arm.pkl"}
T7_DEL = {0: "C:/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json",
          1: "C:/chlu/outputs/w15-single-design-seeds/deletion/test_seed1_K32/deletion.json",
          2: "C:/chlu/outputs/w15-single-design-seeds/deletion/test_seed2_K32/deletion.json"}
''', '''

def kstar_arm(seed: int) -> Path:
    """The K*=32 object built by the build script on the test pool (131,072 rows)."""
    return OUT / "stores" / "test" / ("seed%d" % int(seed)) / "K32_n131072" / "arm.pkl"
''')
    P.rep(DS, '''    ap.add_argument("--scope", action="append", default=None)
''', '''    ap.add_argument("--scope", action="append", default=None)
    ap.add_argument("--arm", default=None, help="object to delete from (default: the K* build)")
''')
    P.rep(DS, '''    with open(T7_ARM[seed], "rb") as fh:
        saved = pickle.load(fh)''', '''    with open(a.arm or kstar_arm(seed), "rb") as fh:
        saved = pickle.load(fh)''')
    P.rep(DS, '''    t7del = json.loads(Path(T7_DEL[seed]).read_text(encoding="utf-8"))
    before_rows = {r["corner_id"]: r for r in t7del["before"]["per_corner"]}
''', '')
    P.rep(DS, '''        surv_ids = set(t7del["scopes"][scope]["survivor_corner_ids"])
        dele_ids = set(t7del["scopes"][scope]["deleted_corner_ids"])''', '''        if scope == "value":
            dele_ids = {c["corner_id"] for c in corners
                        if DR.corner_requests_value(c, grid, DR.RED_FACTOR, DR.RED_INDEX)}
        elif scope == "conjunction_live":
            dele_ids = {c["corner_id"] for c in corners
                        if DR.corner_requests_value(c, grid, DR.RED_FACTOR, DR.RED_INDEX)
                        and DR.corner_requests_value(c, grid, DR.SHAPE_FACTOR, DR.SPHERE_INDEX)}
        else:
            dele_ids = set()
        surv_ids = {c["corner_id"] for c in corners} - dele_ids''')
    P.rep(DS, '''        befo = [before_rows[r["corner_id"]] for r in perA]
        pd = fw.paired_estimand(perA, befo, "distinct_valid_novel_tuples", 1.0, "after", "before")
        pAB = fw.paired_estimand(perA, perB, "distinct_valid_novel_tuples", 1.0, "downdate", "refit_without")
        carried = {r["corner_id"]: r for r in t7del["scopes"][scope]["after"]["per_corner"]}
        pC = fw.paired_estimand(perA, [carried[r["corner_id"]] for r in perA],
                                "distinct_valid_novel_tuples", 1.0, "book_refit", "book_carried_T7")''',
          '''        obj0 = dict(objA, store=st, basis=basis0, address=addr0, arm="before")
        by0 = {r["corner_id"]: r for r in score(obj0, bitmap0, surv_ids)}
        befo = [by0[r["corner_id"]] for r in perA]
        pd = fw.paired_estimand(perA, befo, "distinct_valid_novel_tuples", 1.0, "after", "before")
        pAB = fw.paired_estimand(perA, perB, "distinct_valid_novel_tuples", 1.0, "downdate", "refit_without")''')
    P.rep(DS, '''            "book_refit_minus_T7_book_carried": {k: pC[k] for k in ("mean_paired_delta", "bootstrap_ci95")},
''', '')
    P.rep(DS, '''        rec["RETIRED_ROW"] = ("the deleted corners' composition ('c still composes after "
                              "forgetting c') is retired by Head AMENDMENT-2: not computed here, "
                              "never printed; the w15 evidence stays in its artefacts")
''', '')


def apply_more(P):
    P.redef('chlu/core/gen_bridge.py', 'dataset_module', '''def dataset_module(dataset: str):
    """Return the loader module for ``dataset`` (Shapes3D only in this package)."""
    if dataset == "shapes3d":
        from ..data import shapes3d as mod
    else:
        raise ValueError(f"unknown dataset {dataset!r} (expected shapes3d)")
    return mod
''')


def apply_read(P):
    R = 'chlu/experiments/exp_gen_read.py'
    P.redef(R, 'bind_instrument', """def bind_instrument(log: Callable = print) -> dict:
    return {"classifier": "weights/ (hash-checked by Judge)"}
""")
    P.redef(R, '_cache_manifest', """def _cache_manifest(seed: int) -> dict:
    return fw.feature_cache_manifest(int(seed))
""")
    LP = 'scripts/w20_lace_poe.py'
    P.rep(LP, "    from w20_lib import ledger_has, load_pool\n", "    from w20_lib import load_pool\n")
    P.rep(LP, """    if a.rung == "shipped" and not ledger_has(int(a.run_n)):
        raise SystemExit("LEDGER ROW MISSING for run %d/3 -- rows 3-4 are TEST corners." % a.run_n)
""", "")
    P.rep(LP, """    ap.add_argument("--run-n", type=int, default=0)
""", "")
