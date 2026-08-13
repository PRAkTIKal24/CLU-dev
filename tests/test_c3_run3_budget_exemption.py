"""⛔ ANTI-LOOPHOLE TESTS for the pre-registered-continuation exemption.

`c3-run3-budget-exemption`. This mechanism **weakens a guard** — the
matched-state-byte budget — so the deliverable is not "it works" but **the set of
things it still refuses**, evidenced by tests that actively try to break it.

Why it exists: run 3 is a *pre-registered one-flag ablation of run 2*
(``PREREG-LeakAblation``: run 2 + ``erosion_partition=True``) whose **geometry
must not change**; changing the geometry to satisfy a budget would destroy the
quantity the ablation measures. The budget was never meant to govern that leg.

What is guarded here:

1. the exemption **accepts** exactly run-2-plus-the-registered-flag, and stamps it;
2. it **refuses** a second differing key, a different key, a missing/corrupt
   journal, a fingerprint-invalid journal, a bad sha256, a non-differing flag, a
   flag list/glob, an extra spec key;
3. it **annotates, never suppresses** — true bytes and occupancy survive;
4. it does **not** reach the unledgered-arm guard, the φ accounting, or the
   interim-budget ladder guard;
5. **without** it, the original refusal is intact;
6. the two 2026-08-13 rulings are recorded in code: **no dtype normalisation**,
   and the ceiling digit is **interim** (so no ladder arm may train).
"""

from __future__ import annotations

import json

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

RUN2_FLAG = "memory.erosion_partition"     # the ONE registered flag of run 3
PREREG = ".claude/outputs/c2w6-anti-erosion/PREREG-LeakAblation.md"


# ==========================================================================
# fixtures — a "run 2" journal and a "run 3" config, at toy scale
# ==========================================================================
def _write_journal(d: Path, flags, scale="toy", seed=0) -> Path:
    import chlu.experiments.exp_cluformer_pilot as EXP

    p = EXP.partial_path(d, scale, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"flags": flags, "arms": {}, "_journal": {"trained": {}}}))
    return p


def _cfg(**memory_overrides):
    """A run config, exactly as the CLI builds one (``--mem k=v`` merges on TOY)."""
    import chlu.experiments.exp_cluformer_pilot as EXP

    ov = {"memory": dict(EXP.TOY["memory"], **memory_overrides)} if \
        memory_overrides else {}
    return EXP.make_config("toy", 0, ov)


@pytest.fixture()
def run2_journal(tmp_path):
    """A run-2 journal on disk, written from a real run-2 config.

    ⭐ Like the REAL banked CSF3 run-2 journals, it **predates the C2W6 memory
    fields**: ``erosion_partition`` is absent from its ``memory`` group entirely
    (measured on ``csf3_outs/run2/pilot_pilot_seed0_PARTIAL.json``). That is the
    case run 3 actually has to clear.
    """
    import chlu.experiments.exp_cluformer_pilot as EXP

    flags = EXP.flag_block(_cfg())
    for k in ("erosion_partition", "refresh_amp_ceiling", "refresh_max_gain",
              "refresh_monotonic"):
        flags["memory"].pop(k)
    return _write_journal(tmp_path, flags), flags


def _run3(**extra_memory):
    """run 2 + ``erosion_partition=True`` (+ any deliberate contamination)."""
    return _cfg(erosion_partition=True, **extra_memory)


def _spec(journal, flag=RUN2_FLAG, **kw):
    return dict({"journal": str(journal), "flag": flag, "prereg": PREREG}, **kw)


# ==========================================================================
# 1. ACCEPTED — exactly the registered flag differs
# ==========================================================================
def test_run3_is_ACCEPTED_and_the_exemption_is_STAMPED(run2_journal):
    import chlu.experiments.exp_cluformer_pilot as EXP

    jpath, _ = run2_journal
    ex = EXP.verify_preregistered_continuation(_spec(jpath), _run3())
    st = ex.as_stamp()
    assert st["registered_flag"] == RUN2_FLAG
    # ⭐ the journal PREDATES the field (as the real ones do), so "old" is absent
    assert (st["old_value"], st["new_value"]) == ("<absent from the journal>", True)
    assert st["journal"] == str(jpath) and len(st["journal_sha256"]) == 64
    assert st["prereg"] == PREREG
    assert "load_journal" in st["verified_by"]          # ⭐ delegated, not re-done
    assert st["exempts"] == "the state-byte BUDGET check only"
    assert "UnledgeredArmError" in st["does_not_exempt"]
    # ⚠ ONE knob, but it moves TWO fingerprint keys — the resolved config AND the
    # override dict that produced it. Both are stamped, neither is hidden.
    assert st["fingerprint_keys_moved"] == ["memory.erosion_partition",
                                            "pilot.memory"]


def test_a_journal_carrying_the_field_at_FALSE_is_also_accepted(run2_journal,
                                                                tmp_path):
    """The other run-2 shape: the field exists and is OFF."""
    import chlu.experiments.exp_cluformer_pilot as EXP

    jpath, _ = run2_journal
    full = EXP.flag_block(_cfg())                       # nothing popped
    assert full["memory"]["erosion_partition"] is False
    jpath = _write_journal(jpath.parent, full)
    ex = EXP.verify_preregistered_continuation(_spec(jpath), _run3())
    assert (ex.old_value, ex.new_value) == (False, True)


# ==========================================================================
# 2. REFUSED — every way someone would try to widen it
# ==========================================================================
def test_a_SECOND_differing_key_is_refused_and_BOTH_keys_are_named(run2_journal):
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    jpath, _ = run2_journal
    with pytest.raises(ContinuationExemptionError) as e:
        EXP.verify_preregistered_continuation(
            _spec(jpath), _run3(refresh_monotonic=True))
    msg = str(e.value)
    assert "refresh_monotonic" in msg                   # the offending second key
    assert RUN2_FLAG in msg                             # ...and the registered one
    assert "STILL differ" in msg


def test_a_DIFFERENT_single_key_than_the_registered_one_is_refused(run2_journal):
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    jpath, _ = run2_journal
    with pytest.raises(ContinuationExemptionError) as e:      # NOT the registered
        EXP.verify_preregistered_continuation(                # flag
            _spec(jpath), _cfg(refresh_monotonic=True))
    assert "does NOT differ" in str(e.value) and RUN2_FLAG in str(e.value)


def test_a_GEOMETRY_change_is_refused_even_with_the_registered_flag(run2_journal):
    """⛔ The whole point: run 3's GEOMETRY may not move to satisfy a budget."""
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    jpath, _ = run2_journal
    shrunk = EXP.make_config("toy", 0, {
        "memory": dict(EXP.TOY["memory"], erosion_partition=True),
        "atoms_per_item": EXP.TOY["atoms_per_item"] // 2})     # "shrink to fit"
    with pytest.raises(ContinuationExemptionError) as e:
        EXP.verify_preregistered_continuation(_spec(jpath), shrunk)
    msg = str(e.value)
    assert "pilot.atoms_per_item: journal=128 now=64" in msg
    assert "store.atoms_per_item: journal=128 now=64" in msg
    assert "STILL differ" in msg


def test_a_MISSING_journal_is_refused(tmp_path):
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    gone = tmp_path / "nowhere" / "pilot_toy_seed0_PARTIAL.json"
    with pytest.raises(ContinuationExemptionError, match="does not exist"):
        EXP.verify_preregistered_continuation(_spec(gone), _run3())


def test_a_CORRUPT_journal_is_refused(tmp_path):
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    bad = tmp_path / "pilot_toy_seed0_PARTIAL.json"
    bad.write_text("{not json at all")
    with pytest.raises(ContinuationExemptionError, match="unreadable/corrupt"):
        EXP.verify_preregistered_continuation(_spec(bad), _run3())


def test_a_journal_whose_OWN_FINGERPRINT_does_not_validate_is_refused(tmp_path,
                                                                      run2_journal):
    """A journal with no valid flag block certifies nothing."""
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    _, run2 = run2_journal
    for flags in ({}, {"memory": {}, "pilot": {}, "store": {}},
                  {"memory": run2["memory"], "pilot": {}, "store": {},
                   "store_dim": "twelve", "store_n_atoms": 8192}):
        p = tmp_path / "pilot_toy_seed0_PARTIAL.json"
        p.write_text(json.dumps({"flags": flags}))
        with pytest.raises(ContinuationExemptionError, match="no VALID flag"):
            EXP.verify_preregistered_continuation(_spec(p), _run3())


def test_a_PINNED_sha256_that_does_not_match_is_refused(run2_journal):
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    jpath, _ = run2_journal
    ok = EXP.verify_preregistered_continuation(
        _spec(jpath, sha256=_sha(jpath)), _run3())
    assert ok.journal_sha256 == _sha(jpath)
    with pytest.raises(ContinuationExemptionError, match="sha256 mismatch"):
        EXP.verify_preregistered_continuation(_spec(jpath, sha256="00" * 32),
                                              _run3())


def _sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_a_final_artifact_is_not_a_journal(tmp_path, run2_journal):
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    _, run2 = run2_journal
    p = tmp_path / "pilot_toy_seed0_S3.json"
    p.write_text(json.dumps({"flags": run2}))
    with pytest.raises(ContinuationExemptionError, match="not a resume journal"):
        EXP.verify_preregistered_continuation(_spec(p), _run3())


@pytest.mark.parametrize("flag", [
    "memory.erosion_partition,memory.refresh_monotonic",   # a list
    "memory.*",                                            # a glob
    "memory.refresh_?",                                    # a wildcard
    "erosion_partition",                                   # unqualified
    "store_dim",                                           # a DERIVED quantity
    "pilot.arms memory.erosion_partition",                 # two, space-separated
    "pilot.memory",                                        # ⛔ the whole dict
    "pilot.store",                                         # ⛔ ditto
    "pilot.not_a_field",                                   # not a knob at all
])
def test_only_ONE_fully_qualified_flag_is_ever_accepted(run2_journal, flag):
    """⛔ No list, no wildcard, no glob, no 'allow these N keys'."""
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    jpath, _ = run2_journal
    with pytest.raises(ContinuationExemptionError,
                       match="ONE fully-qualified|names no PilotConfig field"):
        EXP.verify_preregistered_continuation(_spec(jpath, flag=flag), _run3())


@pytest.mark.parametrize("spec", [
    "journal=x flag=y",                                   # not a mapping
    {},                                                   # empty
    {"journal": "x", "flag": RUN2_FLAG},                  # no prereg
    {"journal": "x", "flag": RUN2_FLAG, "prereg": ""},    # empty prereg
])
def test_a_malformed_spec_is_refused_before_anything_else(spec):
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    with pytest.raises(ContinuationExemptionError):
        EXP.verify_preregistered_continuation(spec, _run3())


def test_an_EXTRA_spec_key_is_refused_so_nothing_can_be_smuggled(run2_journal):
    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.eval.byte_ledger import ContinuationExemptionError

    jpath, _ = run2_journal
    with pytest.raises(ContinuationExemptionError, match="unknown key"):
        EXP.verify_preregistered_continuation(
            _spec(jpath, also_allow="memory.refresh_monotonic"), _run3())


# ==========================================================================
# 3. the ledger ANNOTATES, never suppresses
# ==========================================================================
def _over_budget_ledger():
    return {"clu_store": {"params": 1, "state_floats": 10 ** 7,
                          "state_bytes": 4 * 10 ** 7}}


def _exemption(run2_journal, **kw):
    import chlu.experiments.exp_cluformer_pilot as EXP

    jpath, _ = run2_journal
    return EXP.verify_preregistered_continuation(_spec(jpath, **kw), _run3())


def test_over_budget_UNDER_the_exemption_still_reports_TRUE_bytes(run2_journal,
                                                                  capsys):
    """⛔ Not zeroed, not omitted, not 'within budget' — annotated."""
    from chlu.eval.byte_ledger import build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    p = make_config("toy", 0)
    art = build_byte_ledger(p, _over_budget_ledger(), ["clu_store"],
                            exemption=_exemption(run2_journal))
    row = art["arms"]["clu_store"]
    assert row["total_state_bytes"] == p.n_layers * 4 * 10 ** 7
    assert row["occupancy"] == pytest.approx(
        row["total_state_bytes"] / art["budget_bytes"])
    assert row["within_budget"] is False                 # ⛔ NOT relabelled
    assert art["over_budget"] == ["clu_store"]           # ⛔ still listed
    assert art["enforced"] is True                       # ⛔ the check still RAN
    assert art["budget_exempted"] is True
    st = art["preregistered_continuation"]
    assert st["registered_flag"] == RUN2_FLAG and len(st["journal_sha256"]) == 64
    # ...and an auditor reading stdout alone sees it too
    out = capsys.readouterr().out
    assert "PRE-REGISTERED CONTINUATION" in out and "TRUE bytes" in out
    assert f"{row['total_state_bytes']:,}" in out


def test_the_ledger_is_PRINTED_in_full_either_way(run2_journal, capsys):
    from chlu.eval.byte_ledger import build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    p = make_config("toy", 0)
    build_byte_ledger(p, {"none": {"params": 0, "state_floats": 0,
                                   "state_bytes": 0}}, ["none"])
    out = capsys.readouterr().out
    assert "[byte-ledger]" in out and "occupancy" in out
    assert "INTERIM" in out and "dtype normalisation: NONE" in out


def test_an_unverified_dict_cannot_pose_as_an_exemption():
    """⛔ The TYPE is the proof that the fingerprint check ran."""
    from chlu.eval.byte_ledger import ContinuationExemptionError, build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    p = make_config("toy", 0)
    for forged in ({"verified": True}, "yes", True, 1):
        with pytest.raises(ContinuationExemptionError, match="VERIFIED"):
            build_byte_ledger(p, _over_budget_ledger(), ["clu_store"],
                              exemption=forged)


# ==========================================================================
# 4. what the exemption does NOT reach
# ==========================================================================
def test_the_UNLEDGERED_ARM_guard_is_NOT_reachable_through_the_exemption(
        run2_journal):
    from chlu.eval.byte_ledger import UnledgeredArmError, build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    p = make_config("toy", 0)
    with pytest.raises(UnledgeredArmError, match="no byte ledger"):
        build_byte_ledger(p, {"clu_store": {"params": 1}}, ["clu_store"],
                          exemption=_exemption(run2_journal))


def test_PHI_is_still_accounted_on_every_arm_under_the_exemption(run2_journal):
    import jax

    from chlu.eval.byte_ledger import build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config
    from chlu.training.train_cluformer import solve_arms

    p = make_config("toy", 0)
    _, led = solve_arms(p, jax.random.PRNGKey(0))
    art = build_byte_ledger(p, led, p.arms, exemption=_exemption(run2_journal))
    assert art["phi_accounted_on_every_arm"]
    assert all(r["phi_accounted"] and r["phi_params_bytes_total"] > 0
               for r in art["arms"].values())


def test_the_LADDER_guard_refuses_a_rival_arm_while_the_ceiling_is_INTERIM():
    """⭐ Ruling: the digit is set in the rival-ladder prereg, not by the pilot."""
    from chlu.eval import byte_ledger as bl

    with pytest.raises(bl.InterimBudgetError) as e:
        bl.assert_ladder_arms_admissible(["clu_store", "mamba2", "ttt_linear"])
    msg = str(e.value)
    assert "MISSING PREREG" in msg and "rival-ladder" in msg
    assert "mamba2" in msg and "ttt_linear" in msg
    assert bl.BUDGET_IS_INTERIM is True and bl.BUDGET_CEILING_PREREG is None
    # ⭐ run 3's arms are NOT ladder arms: the guard is inert for it
    ours = ("clu_store", "gru_matched", "ttt_matched", "none", "echo")
    g = bl.assert_ladder_arms_admissible(ours)
    assert g["ladder_arms_requested"] == [] and g["budget_is_interim"] is True
    # ...and it is not reachable through the exemption either: it takes no
    # exemption argument at all, and run_pilot calls it BEFORE building one.
    import inspect

    assert "exemption" not in inspect.signature(
        bl.assert_ladder_arms_admissible).parameters
    import chlu.experiments.exp_cluformer_pilot as EXP

    body = inspect.getsource(EXP.run_pilot).split('"""')[2]      # drop the docstring
    assert body.index("assert_ladder_arms_admissible(") < body.index(
        "verify_preregistered_continuation(")


# ==========================================================================
# 5. the ORIGINAL behaviour, intact
# ==========================================================================
def test_NO_exemption_plus_over_budget_is_STILL_REFUSED():
    from chlu.eval.byte_ledger import StateByteBudgetError, build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    p = make_config("toy", 0)
    with pytest.raises(StateByteBudgetError, match="budget violated"):
        build_byte_ledger(p, _over_budget_ledger(), ["clu_store"])


def test_the_refusal_message_names_the_exemption_without_offering_a_shortcut():
    """A hostile reader WILL find this flag; the message must not sell it."""
    from chlu.eval.byte_ledger import StateByteBudgetError, build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    p = make_config("toy", 0)
    with pytest.raises(StateByteBudgetError) as e:
        build_byte_ledger(p, _over_budget_ledger(), ["clu_store"])
    msg = str(e.value)
    assert "not a way round this check" in msg
    assert "bit-identical" in msg and "ONE registered flag" in msg


def test_the_exemption_is_a_RUN_ARGUMENT_and_NOT_a_PilotConfig_field():
    """⛔⛔ The trap this design avoids, and it is not hypothetical.

    As a ``PilotConfig`` field the declaration would enter ``as_flag_table()`` and
    become a **SECOND** key differing from the run-2 journal — so the exemption
    would fail its own identity check, and run 3's journal would differ from run
    2's by two tokens, violating ``PREREG-LeakAblation`` §4 ("run 3 changes
    exactly one token"). It is a run-scoped argument, like ``--d5``/``--slices``.
    """
    import inspect
    from dataclasses import fields

    import chlu.experiments.exp_cluformer_pilot as EXP
    from chlu.training.train_cluformer import PilotConfig

    names = {f.name for f in fields(PilotConfig)}
    assert "preregistered_continuation" not in names
    assert "prereg_continuation" not in names
    sig = inspect.signature(EXP.run_pilot).parameters
    assert "prereg_continuation" in sig and sig["prereg_continuation"].default is None
    # ⛔ ...and the CLI must never route it into the config overrides
    src = inspect.getsource(EXP.main)
    assert "--prereg-continuation" in src
    assert 'ov["preregistered_continuation"]' not in src


def test_run3s_journal_differs_from_run2s_by_EXACTLY_the_one_token(run2_journal):
    """⭐ The prereg's confound declaration, checked mechanically."""
    import chlu.experiments.exp_cluformer_pilot as EXP

    _, run2 = run2_journal
    d_old = EXP._flag_dict(run2)
    d_new = EXP._flag_dict(EXP.flag_block(_run3()))
    defaults = EXP._flag_defaults()
    predated = {k for k in set(d_new) - set(d_old)
                if k in defaults and d_new[k] == defaults[k]}
    moved = sorted(k for k in set(d_old) | set(d_new)
                   if d_old.get(k) != d_new.get(k) and k not in predated)
    # the ONE knob, in its two representations — and nothing else
    assert moved == ["memory.erosion_partition", "pilot.memory"]
    # ⚠ the three OTHER C2W6 fields the journal predates are forgiven (§A20.4)
    assert predated == {"memory.refresh_amp_ceiling", "memory.refresh_max_gain",
                        "memory.refresh_monotonic"}
    assert (json.loads(d_new["pilot.memory"]).keys()
            ^ json.loads(d_old["pilot.memory"]).keys()) == {"erosion_partition"}


def test_the_LADDER_SCRIPT_can_actually_set_it(monkeypatch):
    """⛔ §7.33: a pre-registered leg behind a flag NO LAUNCH PATH SETS is
    indistinguishable, in the artifact, from a deliberate cut — and run 3 must
    carry the exemption on every (re-)submission, including every re-resume."""
    import subprocess

    job = REPO / "scripts" / "csf3" / "job_gpu_c3_seeds.sh"
    t = job.read_text()
    assert 'PREREG_CONT="${PREREG_CONT:-}"' in t
    assert '[ -n "$PREREG_CONT" ] && EXTRA="$EXTRA --prereg-continuation ' \
           '$PREREG_CONT"' in t
    assert subprocess.run(["bash", "-n", str(job)]).returncode == 0
    # ⚠ and it must word-split into the three KEY=VALUE argv entries argparse
    # wants — the zsh trap, checked rather than assumed
    out = subprocess.run(
        ["bash", "-c", 'EXTRA=""; [ -n "$PREREG_CONT" ] && '
                       'EXTRA="$EXTRA --prereg-continuation $PREREG_CONT"; '
                       'for w in $EXTRA; do echo "$w"; done'],
        capture_output=True, text=True,
        env={"PREREG_CONT": "journal=/tmp/j.json flag=memory.erosion_partition "
                            "prereg=P.md", "PATH": "/usr/bin:/bin"})
    assert out.stdout.split() == ["--prereg-continuation", "journal=/tmp/j.json",
                                  "flag=memory.erosion_partition", "prereg=P.md"]


# ==========================================================================
# 6. the two rulings, recorded in code
# ==========================================================================
def test_the_NO_DTYPE_NORMALISATION_ruling_is_recorded_where_it_is_used():
    from chlu.eval import byte_ledger as bl

    r = bl.DTYPE_NORMALISATION_RULING
    assert r.startswith("NONE")
    assert "2026-08-13" in r and "AS DEPLOYED" in r
    assert "fp32" in r and "bf16" in r
    assert "NOT a bug" in r                      # ⛔ nobody may "fix" it later
    assert "NO DTYPE NORMALISATION" in bl.BUDGET_PROVENANCE
    # ...and in the module docstring, where the next reader lands first
    assert "NO dtype normalisation" in bl.__doc__
    assert "as deployed" in bl.__doc__.lower()


def test_the_budget_constant_is_named_INTERIM_at_the_point_of_use():
    from chlu.eval import byte_ledger as bl
    from chlu.training.train_cluformer import PilotConfig

    assert bl.INTERIM_MATCHED_STATE_BYTE_BUDGET == 2_097_152
    # the old name is an ALIAS, not a second literal — one edit still moves all
    assert bl.MATCHED_STATE_BYTE_BUDGET == bl.INTERIM_MATCHED_STATE_BYTE_BUDGET
    assert PilotConfig().state_byte_budget == bl.INTERIM_MATCHED_STATE_BYTE_BUDGET
    assert "INTERIM AND BINDS NOTHING YET" in bl.BUDGET_PROVENANCE
    assert "rival-ladder" in bl.BUDGET_PROVENANCE.lower()


def test_every_artifact_carries_both_rulings():
    from chlu.eval.byte_ledger import build_byte_ledger
    from chlu.experiments.exp_cluformer_pilot import make_config

    art = build_byte_ledger(make_config("toy", 0),
                            {"none": {"params": 0, "state_floats": 0,
                                      "state_bytes": 0}}, ["none"])
    assert art["budget_is_interim"] is True
    assert art["budget_ceiling_prereg"] is None
    assert art["dtype_normalisation"].startswith("NONE")
    assert art["budget_exempted"] is False
    assert art["preregistered_continuation"] is None
