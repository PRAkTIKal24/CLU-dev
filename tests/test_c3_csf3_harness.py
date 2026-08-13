"""Tests for the C3 real-data harness (`c3-csf3-harness`).

Four things are guarded here, each because it is a way the harness could silently
stop being trustworthy:

1. **the corpus registry** — a stream is a config value, and staging is *enforced*
   (an array task may not download);
2. **the retention slices** — ⚠ TRAP 1: the revisit unit must be the enclosing
   token, and the degeneracy tripwire must actually FAIL on the degenerate unit;
   plus the two validity controls and the position alignment;
3. **the byte ledger** — an unledgered arm and an over-budget arm both fail
   loudly, and φ is accounted on every arm;
4. **the resume path** — the regression test for the deliberately loosened
   §A20.4 provenance check: a field added after a journal was written is accepted
   **at its default** and still refused **off it**. That guard must not widen.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


# ==========================================================================
# 1. the corpus registry
# ==========================================================================
def test_registry_exposes_both_built_streams_behind_one_surface():
    from chlu.data.corpora import available_corpora, get_corpus

    assert set(available_corpora()) >= {"enwik8", "wikitext103"}
    for name in ("enwik8", "wikitext103"):
        s = get_corpus(name)
        assert callable(s.load) and callable(s.stage)
        assert s.vocab_size == 256 and s.metric == "bpc"
        # ⛔ every corpus must declare a document boundary: "within-document"
        # retention is meaningless without one.
        assert s.doc_boundary, f"{name} declares no doc_boundary"


def test_unknown_corpus_names_the_registered_ones_and_the_seam_cost():
    from chlu.data.corpora import SEAM_COST, get_corpus

    with pytest.raises(KeyError, match="unknown corpus"):
        get_corpus("pg19")
    # the seam is costed, so "the option is cheap" is auditable rather than asserted
    assert len(SEAM_COST) == 3
    assert any("register_corpus" in s for s in SEAM_COST)


def test_pg19_and_fineweb_are_a_DECLARED_seam_and_are_NOT_built():
    """⛔ Building either was explicitly out of scope; only the notes are banked."""
    from chlu.data.corpora import SEAM_NOTES, available_corpora

    assert "pg19" not in available_corpora()
    assert "fineweb_edu" not in available_corpora()
    assert {"pg19", "fineweb_edu"} <= set(SEAM_NOTES)
    pg = " ".join(SEAM_NOTES["pg19"])
    assert "28,752" in pg and "sha256" in pg          # the inode hazard
    assert "word" in pg.lower() and "tokenizer" in pg  # the normaliser caveat
    assert "50 books" in pg or "per-book" in pg        # the variance caveat


def test_an_array_task_may_NOT_download_and_is_told_how_to_stage(monkeypatch,
                                                                 tmp_path):
    """⛔ Staging is serial-once-before-the-sweep, ENFORCED, not documented."""
    from chlu.data import corpora

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    assert corpora.in_array_task()
    with pytest.raises(corpora.NotStagedError) as e:
        corpora.stage_corpus("enwik8", tmp_path)
    msg = str(e.value)
    assert "STAGE_ONLY=1" in msg and "ARRAY TASK" in msg

    # and load_corpus refuses rather than quietly fetching into an empty cache
    with pytest.raises(corpora.NotStagedError):
        corpora.load_corpus("enwik8", tmp_path, n_bytes=1000)


def test_outside_an_array_task_downloading_is_still_allowed(monkeypatch):
    """The enforcement is scoped to array tasks — the serial stage job must work."""
    from chlu.data import corpora

    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    assert not corpora.in_array_task()


# ==========================================================================
# 2. the retention slices
# ==========================================================================
def _fake_stream(n_docs: int = 6, words_per_doc: int = 1200,
                 vocab: int = 500, seed: int = 0) -> np.ndarray:
    """A synthetic corpus with REALISTIC token statistics.

    ⚠ It has to be realistic or it cannot exercise TRAP 1: a toy string of a
    dozen repeated words has token revisit distances as short as its byte ones,
    so the token/byte contrast the trap is about would not appear. Zipf-ish word
    frequencies over a 500-word vocabulary reproduce the real separation (on
    enwik8: token median ~600 B vs byte median ~14 B).
    """
    rng = np.random.default_rng(seed)
    words = [f"w{i:04d}"[: 2 + i % 6].encode() + bytes([97 + i % 26])
             for i in range(vocab)]
    p = 1.0 / (np.arange(1, vocab + 1) ** 1.05)
    p /= p.sum()
    out = []
    for _ in range(n_docs):
        idx = rng.choice(vocab, size=words_per_doc, p=p)
        out.append(b"<page> " + b" ".join(words[i] for i in idx) + b"\n")
    return np.frombuffer(b"".join(out), dtype=np.uint8)


def test_document_starts_split_enwik8_on_page_and_wt103_on_level1_headings():
    from chlu.eval.text_slices import _document_starts

    d = _fake_stream(4)
    starts = _document_starts(d, b"<page>")
    assert len(starts) == 4, starts

    # ⚠ WT-103: ` = Title = ` opens an ARTICLE; ` = = Section = = ` does not.
    wt = b"\n = Alpha = \nbody text\n = = Sub = = \nmore body\n = Beta = \nend"
    a = _document_starts(np.frombuffer(wt, dtype=np.uint8), b"\n = ")
    # the Alpha heading IS byte 0, so the two articles are [0, 46] — and the
    # ` = = Sub = = ` SECTION heading in between must NOT open a document.
    assert list(a) == [0, 46], a
    assert 22 not in a, "a level-2 section heading was mistaken for an article"


def test_the_revisit_unit_is_the_token_and_the_BYTE_unit_is_degenerate():
    """⚠⚠ TRAP 1, the tripwire itself.

    At vocab 256 the raw-byte unit collapses to a character-frequency count. The
    instrument must (a) use the enclosing token and (b) FAIL LOUDLY if the
    distances degenerate — not quietly ship a frequency histogram.
    """
    from chlu.eval import text_slices as ts

    d = _fake_stream(8)
    tok = ts.build_revisit_index(d, doc_boundary=b"<page>", unit="token")
    byt = ts.build_revisit_index(d, doc_boundary=b"<page>", unit="byte")
    assert tok.median_distance() > byt.median_distance() * 5, (
        tok.median_distance(), byt.median_distance())
    # the degenerate unit piles into the tightest bins; the token unit does not
    bc = byt.counts()
    near = bc["[1,8)"] + bc["[8,32)"]
    assert near / max(1, sum(bc.values())) > 0.5, bc


def test_non_degeneracy_tripwire_RAISES_on_a_degenerate_stream():
    """⛔ The check must be capable of failing, or it certifies nothing."""
    from chlu.eval import text_slices as ts

    # a stream with a tiny token alphabet: every token recurs within a few bytes,
    # which is exactly the frequency-count pathology the trap describes.
    degenerate = np.frombuffer(b"<page> a b a b a b a b a b a b " * 400,
                               dtype=np.uint8)
    with pytest.raises(AssertionError, match="DEGENERATE REVISIT SLICE"):
        ts.assert_non_degenerate(degenerate, doc_boundary=b"<page>")


def test_never_bucket_is_separate_from_the_largest_distance_bin():
    """Sun et al.'s 'never appears in the prefix' bucket is not merged away."""
    from chlu.eval import text_slices as ts

    idx = ts.build_revisit_index(_fake_stream(6), doc_boundary=b"<page>")
    assert idx.labels[-1] == ts.NEVER_BIN
    assert idx.labels[-2].endswith("inf)")
    c = idx.counts()
    assert c[ts.NEVER_BIN] > 0 and idx.meta["n_units_with_prior"] > 0


def test_shuffled_position_control_moves_the_slice():
    """⚠ Control (a): permute positions, keep content — the slice MUST move."""
    from chlu.eval import text_slices as ts

    r = ts.shuffled_position_control(_fake_stream(30), doc_boundary=b"<page>",
                                     seed=0)
    assert r["slice_moved"], r


def test_content_relabelling_leaves_the_slice_invariant():
    """⚠ Control (b): permute content, keep distances — the slice must NOT move."""
    from chlu.eval import text_slices as ts

    r = ts.content_relabel_control(_fake_stream(20), doc_boundary=b"<page>")
    assert r["relabelling_injective"], r
    assert r["n_documents_original"] == r["n_documents_relabelled"]
    assert r["slice_invariant"], (r["counts_original"], r["counts_relabelled"])


def test_a_corpus_without_a_document_boundary_is_REFUSED():
    from chlu.eval import text_slices as ts

    with pytest.raises(ValueError, match="within-document"):
        ts.build_revisit_index(_fake_stream(2), doc_boundary=None)


def test_slice_positions_align_EXACTLY_with_the_real_eval_iterator():
    """⛔ The one alignment that must never drift: if the position arithmetic
    diverges from ``contiguous_batches``, every per-bin bpc is attached to the
    wrong bytes and the instrument reads plausibly while being wrong."""
    from chlu.data.enwik8 import Enwik8Split, contiguous_batches
    from chlu.eval.text_slices import contiguous_target_positions

    data = np.arange(20_000, dtype=np.int64).astype(np.uint8)
    split = Enwik8Split("test", data)
    B, T, N = 3, 16, 5
    pos = contiguous_target_positions(len(split), batch=B, seq_len=T, n_batches=N)
    got = list(contiguous_batches(split, batch=B, seq_len=T, n_batches=N))
    assert len(pos) == len(got) == N
    for (_x, y), p in zip(got, pos, strict=True):
        assert p.shape == y.shape
        np.testing.assert_array_equal(split.data[p], y)


def test_underpopulated_bins_report_their_n_and_never_a_silent_average():
    """⛔ 'a bin with too few samples is reported with its n, never averaged away'."""
    from chlu.eval.text_slices import build_revisit_index, slice_bpc

    d = _fake_stream(6)
    idx = build_revisit_index(d, doc_boundary=b"<page>")
    pos = [np.arange(len(d), dtype=np.int64)]
    nll = [np.full(len(d), 0.5, dtype=np.float64)]
    out = slice_bpc(nll, pos, idx, min_n=10**9)     # nothing can be sufficient
    for b in out["bins"].values():
        assert b["bpc"] is None and b["sufficient"] is False
        assert isinstance(b["n"], int)
    assert sum(b["n"] for b in out["bins"].values()) == out["n_scored"] > 0


# ==========================================================================
# 3. the byte ledger
# ==========================================================================
def test_the_budget_is_one_named_constant_at_the_ruled_value():
    from chlu.eval import byte_ledger as bl

    assert bl.MATCHED_STATE_BYTE_BUDGET == 2_097_152          # 2 MiB
    assert "2026-08-13" in bl.BUDGET_PROVENANCE
    # ⛔ not a literal scattered across call sites: the trainer's default is the
    # constant itself, so confirming the last digit is a one-line edit.
    from chlu.training.train_cluformer import PilotConfig
    assert PilotConfig().state_byte_budget == bl.MATCHED_STATE_BYTE_BUDGET


def test_an_UNLEDGERED_arm_fails_loudly():
    from chlu.eval.byte_ledger import UnledgeredArmError, build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    p = make_config("toy", 0)
    with pytest.raises(UnledgeredArmError, match="no byte ledger"):
        build_byte_ledger(p, {"clu_store": {"params": 1}}, ["clu_store"])


def test_a_ZERO_state_arm_is_a_LEDGER_not_a_missing_ledger():
    """`none`/`echo` genuinely hold no state; 0 must be asserted, not inferred."""
    from chlu.eval.byte_ledger import build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    p = make_config("toy", 0)
    art = build_byte_ledger(p, {"none": {"params": 0, "state_floats": 0,
                                         "state_bytes": 0}}, ["none"])
    assert art["arms"]["none"]["total_state_bytes"] == 0
    assert art["arms"]["none"]["within_budget"]


def test_phi_is_accounted_on_EVERY_arm():
    """charter §5: 'byte ledgers on every arm incl. φ'."""
    import jax

    from chlu.eval.byte_ledger import build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config
    from chlu.training.train_cluformer import solve_arms

    p = make_config("toy", 0)
    _, led = solve_arms(p, jax.random.PRNGKey(0))
    art = build_byte_ledger(p, led, p.arms)
    assert art["phi_accounted_on_every_arm"]
    for a, r in art["arms"].items():
        assert r["phi_accounted"] and r["phi_params_bytes_total"] > 0, a
        # every arm shares the SAME φ — that is what makes the swap a swap
    vals = {r["phi_params_bytes_total"] for r in art["arms"].values()}
    assert len(vals) == 1, vals


def test_an_OVER_BUDGET_arm_fails_loudly_and_can_be_declared_instead():
    from chlu.eval.byte_ledger import StateByteBudgetError, build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    p = make_config("toy", 0)
    fat = {"clu_store": {"params": 1, "state_floats": 10**7, "state_bytes": 4 * 10**7}}
    with pytest.raises(StateByteBudgetError, match="budget violated"):
        build_byte_ledger(p, fat, ["clu_store"])
    # ...and turning it off is RECORDED in the artifact, never silent
    art = build_byte_ledger(p, fat, ["clu_store"], enforce=False)
    assert art["enforced"] is False and art["over_budget"] == ["clu_store"]
    assert art["arms"]["clu_store"]["occupancy"] > 1.0


def test_occupancy_is_reported_not_just_compliance():
    import jax

    from chlu.eval.byte_ledger import build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config
    from chlu.training.train_cluformer import solve_arms

    p = make_config("toy", 0)
    _, led = solve_arms(p, jax.random.PRNGKey(0))
    art = build_byte_ledger(p, led, p.arms)
    for r in art["arms"].values():
        assert 0.0 <= r["occupancy"] <= 1.0
        assert "arithmetic" in r and "n_layers" in r["arithmetic"]


def test_rival_state_configs_are_PINNED_with_per_number_provenance():
    """⚠⚠ TRAP 2: never inherit a library default and claim byte-matching."""
    from chlu.eval.byte_ledger import RIVAL_SPECS, rival_reference_table

    gdn = RIVAL_SPECS["gated_deltanet2"]
    assert gdn.params == {"n_heads": 4, "d_k": 128, "d_v": 128}
    assert "2605.22791" in gdn.provenance
    assert "flash-linear-attention" in gdn.provenance   # the trap is named
    m2 = RIVAL_SPECS["mamba2"]
    assert m2.params == {"d_state": 128, "d_conv": 4, "expand": 2,
                         "headdim": 64, "ngroups": 1}
    # provenance distinguishes a paper table from an implementation
    assert "NOT OBTAINED" in m2.provenance and "CODE" in m2.provenance
    for s in RIVAL_SPECS.values():
        assert s.provenance.startswith(("PAPER:", "OFFICIAL IMPLEMENTATION:"))
    assert rival_reference_table()["rivals"]


def test_pinned_rivals_reproduce_the_scouts_derived_state_bytes():
    """The scout's §1.5 table is the reference; drifting off it silently would
    void the matched-bytes control."""
    from chlu.eval.byte_ledger import RIVAL_SPECS

    expected = {"ttt_linear": 1_597_440, "gated_deltanet2": 3_145_728,
                "transformer_xl": 6_291_456, "mamba2": 6_475_776,
                "sliding_window": 12_582_912, "ttt_mlp": 12_705_792}
    got = {k: RIVAL_SPECS[k].state_bytes() for k in expected}
    assert got == expected, got


def test_over_budget_rivals_are_SHRUNK_to_match_never_grown():
    """⭐ The ruling's whole point: the defensible direction of the control."""
    from chlu.eval.byte_ledger import (MATCHED_STATE_BYTE_BUDGET, RIVAL_SPECS,
                                       shrink_to_budget)

    for name, spec in RIVAL_SPECS.items():
        if spec.state_bytes() <= MATCHED_STATE_BYTE_BUDGET:
            continue
        r = shrink_to_budget(name)
        assert r["state_bytes_shrunk"] <= MATCHED_STATE_BYTE_BUDGET
        assert r["state_bytes_shrunk"] < r["state_bytes_published"]
        assert r["shrink_factor"] > 1.0
        assert r["knob"] == spec.shrink_knob


# ==========================================================================
# 4. the resume path — the guard on the loosened §A20.4 check
# ==========================================================================
def _flags_for(pcfg):
    from dataclasses import asdict as _asdict

    return {"pilot": pcfg.as_flag_table(),
            "memory": _asdict(pcfg.memory_cfg()),
            "store": pcfg.store_cfg().as_flag_table(),
            "store_dim": int(pcfg.store_cfg().dim),
            "store_n_atoms": int(pcfg.store_cfg().n_atoms)}


def _write_journal(tmp_path, flags, scale="toy", seed=0):
    import chlu.experiments.exp_cluformer_pilot as EXP

    p = EXP.partial_path(tmp_path, scale, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"flags": flags, "arms": {}, "_journal": {"trained": {}}}))
    return p


def test_a_journal_PREDATING_a_memory_field_is_accepted_AT_ITS_DEFAULT(tmp_path):
    """⭐ The regression test for the deliberately loosened provenance check.

    A field added to ``StreamMemoryConfig`` after a journal was written must not
    strand 16 h of banked A100 training — provided the current value is that
    field's OWN DEFAULT, because this repo ships every new lever gated OFF and
    bit-identical.
    """
    import chlu.experiments.exp_cluformer_pilot as EXP

    pcfg = EXP.make_config("toy", 0)
    flags = _flags_for(pcfg)
    # simulate "this field did not exist when the journal was written"
    added = "erosion_partition"
    assert added in flags["memory"]
    old = dict(flags)
    old["memory"] = {k: v for k, v in flags["memory"].items() if k != added}
    _write_journal(tmp_path, old)

    prior = EXP.load_journal(tmp_path, "toy", 0, flags)     # must NOT raise
    assert prior["flags"]["memory"].get(added, "absent") == "absent"


def test_the_same_journal_is_STILL_REFUSED_off_the_default(tmp_path):
    """⛔ The guard must not widen: only the field's own default is forgiven."""
    import chlu.experiments.exp_cluformer_pilot as EXP

    pcfg = EXP.make_config("toy", 0)
    flags = _flags_for(pcfg)
    added = "erosion_partition"
    old = dict(flags)
    old["memory"] = {k: v for k, v in flags["memory"].items() if k != added}
    _write_journal(tmp_path, old)

    now = json.loads(json.dumps(flags))
    default = now["memory"][added]
    now["memory"][added] = (not default) if isinstance(default, bool) else 123.0
    with pytest.raises(SystemExit, match="refusing to resume"):
        EXP.load_journal(tmp_path, "toy", 0, now)


def test_a_field_the_code_no_longer_has_is_still_refused(tmp_path):
    """⛔ The converse stays strict: a deleted field's value is unreconstructible."""
    import chlu.experiments.exp_cluformer_pilot as EXP

    pcfg = EXP.make_config("toy", 0)
    flags = _flags_for(pcfg)
    old = json.loads(json.dumps(flags))
    old["memory"]["a_field_that_no_longer_exists"] = 1
    _write_journal(tmp_path, old)
    with pytest.raises(SystemExit, match="refusing to resume"):
        EXP.load_journal(tmp_path, "toy", 0, flags)


def test_the_C3_config_knobs_stay_OUT_of_the_resume_fingerprint(tmp_path):
    """⭐ Every knob this task added ships at a default, so `as_flag_table` omits
    it and the five banked CSF3 journals still resume unchanged."""
    import chlu.experiments.exp_cluformer_pilot as EXP

    tbl = EXP.make_config("toy", 0).as_flag_table()
    for k in ("corpus", "corpus_level", "data_download", "state_byte_budget",
              "enforce_state_byte_budget", "slice_min_n", "slice_batches",
              "resume_require_ckpt"):
        assert k not in tbl, f"{k} entered the flag table at its default"


def test_slices_is_a_CLI_ARG_not_a_config_field_so_it_cannot_move_the_journal():
    """Same discipline as --d5: a finished leg can gain slices by re-resume."""
    from dataclasses import fields

    from chlu.training.train_cluformer import PilotConfig

    names = {f.name for f in fields(PilotConfig)}
    assert "with_slices" not in names and "slices" not in names


def test_a_banked_arm_with_a_MISSING_eqx_fails_loudly_instead_of_retraining():
    """⛔ Ruling (2) of the pilot's reconciliation list, as executable code."""
    import inspect

    import chlu.experiments.exp_cluformer_pilot as EXP

    src = inspect.getsource(EXP.run_pilot)
    assert "resume precondition failed" in src
    assert "resume_require_ckpt" in src
    from chlu.training.train_cluformer import PilotConfig
    assert PilotConfig().resume_require_ckpt is True


# ==========================================================================
# 5. the launch path
# ==========================================================================
JOB_C3 = REPO / "scripts" / "csf3" / "job_gpu_c3_seeds.sh"
JOB_PILOT = REPO / "scripts" / "csf3" / "job_gpu_cluformer.sh"
SMOKE = REPO / "scripts" / "smoke_c3_local.sh"


def test_the_ladder_script_exists_is_executable_and_parses():
    import subprocess

    assert JOB_C3.exists() and os.access(JOB_C3, os.X_OK)
    assert subprocess.run(["bash", "-n", str(JOB_C3)]).returncode == 0
    assert subprocess.run(["bash", "-n", str(SMOKE)]).returncode == 0


def test_three_seeds_is_the_documented_default_launch():
    """charter §5: >=3 seeds before any paper number — make 3 the EASY number."""
    t = JOB_C3.read_text()
    assert 'N_SEEDS="${N_SEEDS:-3}"' in t
    assert "#SBATCH -a 0-14%4" in t          # 5 arms x 3 seeds, <=4 concurrent
    assert re.search(r"sbatch -a 0-14%4", t), "no one-command ladder launch"


def test_every_ladder_flag_has_a_passthrough():
    """A pre-registered phase gated behind a flag no launch path sets is
    indistinguishable, in the artifact, from a deliberate cut (§7.33)."""
    t = JOB_C3.read_text()
    for env, flag in (("RESUME", "--resume"), ("D5", "--d5"),
                      ("SLICES", "--slices")):
        assert f'[ "${env}" = "1" ] && EXTRA="$EXTRA {flag}"' in t, env
    for env, flag in (("MEM", "--mem"), ("STORE", "--store"), ("SET", "--set")):
        assert f'[ -n "${env}" ] && EXTRA="$EXTRA {flag} ${env}"' in t, env
    assert '--corpus "$CORPUS"' in t          # the stream is a config value


def test_the_ladder_enforces_serial_staging_and_the_eqx_precondition():
    t = JOB_C3.read_text()
    assert "STAGE_ONLY must run as a SERIAL job" in t
    assert "refusing to submit" in t and "silently retrain" in t
    assert "stage_corpus" in t and "download=False" in t


def test_each_arm_gets_its_own_out_dir_because_arms_is_a_config_field():
    t = JOB_C3.read_text()
    assert 'OUT="${OUT_BASE}/${ARM}_s${SEED}"' in t
    assert "arms` IS a PilotConfig field" in t or "`arms` IS a PilotConfig" in t


def test_the_pilot_scripts_D5_passthrough_is_UNDISTURBED():
    """⛔ The pilot wave just wired D5 through this script; do not break it."""
    t = JOB_PILOT.read_text()
    assert '[ "$D5" = "1" ] && EXTRA="$EXTRA --d5"' in t
    assert '[ "$RESUME" = "1" ] && EXTRA="$EXTRA --resume"' in t
    assert 'D5="${D5:-0}"' in t


def test_the_smoke_config_says_it_is_never_a_claim_venue():
    """⛔ Where an operator will actually read it."""
    t = SMOKE.read_text()
    assert "NEVER A CLAIM VENUE" in t.upper()
    head = "\n".join(t.splitlines()[:20])
    assert "claim venue" in head.lower(), "the warning is not at the top"
    for stage in ("train", "checkpoint", "RESUME", "slices", "byte ledger"):
        assert stage in t, stage
