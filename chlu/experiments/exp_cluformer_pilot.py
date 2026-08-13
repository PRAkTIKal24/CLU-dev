"""⭐ TIER-III PILOT — the full C2W1 CLU as a streaming block's memory, on enwik8.

⛔ **What is being claimed and what is not.** The memory in the block is the
**C2W1 full store** (learned ``V_theta``, admission, per-item lifetimes, masked
local write, permitted basin interaction under the soft certificate, learned
``phi`` in, learned **trajectory** ``psi`` out, two-phase relaxation, mass and
friction as trainable selectors, confidence-gated retry, the controller's verb
set, all 13 monitors). It is **not** ``CLUBlock``, the w20/w21 driven-Hamiltonian
recurrence with no store — that object is ruled out as a tier-iii arm.

**The control is the SYSTEM-LEVEL SWAP**: the same block with the CLU replaced by
a matched-state GRU cell and by a matched-state TTT-class cell — matched
parameters AND matched state-bytes, same embedding, same depth, same norms, same
residual, same optimiser, same data order, same seeds, same chunk granularity.
⛔ The tier-i settle-deleted / matched-bytes launder is **not** this task's
control and is not run here.

**The acceptance criterion is inherited verbatim from ``full-clu-harness``:** the
system runs the stream **without tripping a silent collapse mode**. *"Does not
collapse", not "wins".* Every monitor's trip-state is a reported artifact.

Staged acceptance (task §3) — the runner reports which stage it reached:

* **S1** the block exists and does not collapse (``--stage s1``);
* **S2** the training path is real — ``||dL/dphi||`` end to end through the
  trajectory read vs the settled-point arm's 0.0 (``--stage s2``);
* **S3** the swap control is defined and matched, ledgers published, all arms
  trained on the same data order and seeds (``--stage s3``);
* **S4** the 26-47 M CSF3 run with the dynamic-evaluation substitute column and
  the pre-registered directional falsifier adjudicated (``--stage s4``).

Run::

    PYTHONPATH=. python -m chlu.experiments.exp_cluformer_pilot --stage s3 --seed 0

⚠ **No CLI hook.** ``chlu/cli/experiment_cmd.py`` is read-only to this task
(owned by ``bprime-c6`` this wave), so the runner is invoked as a module. Noted
in the report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import equinox as eqx
import jax
import numpy as np

from chlu.core.blocks import _count
from chlu.data.corpora import get_corpus, load_corpus
from chlu.data.enwik8 import contiguous_batches, random_batches
from chlu.eval.byte_ledger import (
    ContinuationExemption,
    ContinuationExemptionError,
    assert_ladder_arms_admissible,
    build_byte_ledger,
)
from chlu.training.train_cluformer import (
    PilotConfig,
    anytime_curve,
    allocation_liveness,
    assert_shared_shell_identical,
    build_arm,
    calibrate_phi_gain,
    dynamic_eval,
    evaluate,
    gradient_probe,
    host_rss,
    monitor_pass,
    release_host_memory,
    save_json,
    solve_arms,
    train_arm,
)

#: Toy scale — the LOCAL configuration. ⛔ Not a 26-47 M number and never
#: reported as one (task §0.2: no 26-47 M run happens on the laptop).
TOY = dict(d_model=64, n_layers=2, seq_len=512, batch=4,
           addr_dim=2, payload_dim=1, capacity=8, atoms_per_item=128,
           steps=60, warmup=10, eval_batches=4, dyneval_batches=4,
           data_bytes=4_000_000,
           memory=dict(chunk=32, address_steps=24, read_steps=24, traj_stride=8,
                       psi_hidden=32, write_inner_steps=4, write_n_perturb=8,
                       retry_rounds=1, conv_kernel=4, mlp_mult=4))

#: ⭐ The PILOT scale — 26-47 M, **CSF3 only** (Head ruling §0.2). Declared here
#: so the job script and the report quote the same object.
PILOT = dict(d_model=512, n_layers=12, seq_len=1024, batch=8,
             addr_dim=8, payload_dim=4, capacity=32, atoms_per_item=256,
             steps=4000, warmup=200, eval_batches=40, dyneval_batches=40,
             data_bytes=None,
             memory=dict(chunk=64, address_steps=64, read_steps=64, traj_stride=8,
                         psi_hidden=128, write_inner_steps=4, write_n_perturb=8,
                         retry_rounds=1, conv_kernel=4, mlp_mult=4))


#: ⭐ **The C2W11 `d_addr` ceiling probe, as a CONFIG INPUT** (charter §6.3: C3W1
#: only *consumes* this). Read from the artifact itself —
#: ``.claude/outputs/c2w11/DADDR-CEILING-PROBE.json`` — never from a report or a
#: charter paraphrase.
#:
#: **What it says:** the optimistic exact-set bound first clears V1's bar
#: (``v1_bar = 0.0504``) at **``addr_dim = 12``** (``optimistic_exact_set_at_ceiling``
#: 0.1375 at d=12 vs 0.0340 at d=8). Its regression anchor reproduced the banked
#: ``assignment_ceiling`` at d=4 exactly (``abs_delta = 0.0``). The artifact's own
#: ``reading.statement`` calls this **NECESSARY, not sufficient**, and
#: ``placement_headroom_ceiling_minus_identity`` **falls to 0.0 at d=12** (0.085 at d=4).
#:
#: ⛔⛔ **DELIBERATELY NOT APPLIED AS `PilotConfig.addr_dim`'s DEFAULT — a Hub
#: ruling, for three measured reasons** (`c3-csf3-harness` §7.1):
#:   1. :data:`PILOT` sets ``addr_dim=8`` explicitly, so moving the *default* to 12
#:      would make it a NON-default key in ``as_flag_table()``, change the resume
#:      fingerprint, and **refuse all five banked CSF3 journals** — the exact
#:      failure the pilot merge repaired;
#:   2. ``dim = addr_dim + payload_dim`` feeds ``solve_matched_ttt``, and the TTT
#:      arm's divergence criterion is ``eta*n/d`` — a pure function of the solved
#:      geometry — so this silently re-rolls a published rival's stability;
#:   3. it moves every state-byte number in the matched-bytes control.
#: ⇒ ``addr_dim`` stays a plain config flag: ``--set addr_dim=12``.
DADDR_CEILING_PROBE = {
    "artifact": ".claude/outputs/c2w11/DADDR-CEILING-PROBE.json",
    "d_addr_clearing_v1_bar": 12,
    "v1_bar": 0.0504,
    "optimistic_exact_set_at_ceiling": {4: 0.011714, 6: 0.019338,
                                        8: 0.034001, 12: 0.137536},
    "placement_headroom": {4: 0.08506, 6: 0.037461, 8: 0.010926, 12: 0.0},
    "reading": "NECESSARY, not sufficient (the K6 leak squeeze still applies).",
    "applied_as_default": False,
    "why_not": "claims-relevant: refuses every banked journal, moves eta*n/d, "
               "and moves the matched-state-byte control. Hub ruling owed.",
}


def make_config(scale: str, seed: int, overrides: Optional[dict] = None) -> PilotConfig:
    base = dict(TOY if scale == "toy" else PILOT)
    base["seed"] = int(seed)
    base.update(dict(overrides or {}))
    return PilotConfig.from_mapping(base)


def _data(pcfg: PilotConfig):
    """⭐ The stream is selected by CONFIG (``pcfg.corpus``), not by import.

    Routed through :func:`chlu.data.corpora.load_corpus`, which (a) presents every
    corpus behind one surface so the trainer is corpus-agnostic and (b) refuses to
    download from inside a Slurm array task — staging is serial, once, before a
    sweep. ⛔ Default ``corpus="enwik8"`` reproduces the previous call exactly.
    """
    kw = {}
    spec = get_corpus(pcfg.corpus)
    if spec.name == "wikitext103":
        kw["level"] = pcfg.corpus_level
    return load_corpus(pcfg.corpus, pcfg.data_root, n_bytes=pcfg.data_bytes,
                       download=bool(getattr(pcfg, "data_download", True)), **kw)


def _train_batches(split, pcfg: PilotConfig) -> List:
    """⭐ Materialised ONCE and reused by every arm — identical data order."""
    return list(random_batches(split, batch=pcfg.batch, seq_len=pcfg.seq_len,
                               n_batches=pcfg.steps, seed=pcfg.seed))


def _eval_batches(split, pcfg: PilotConfig, n: int) -> List:
    return list(contiguous_batches(split, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                   n_batches=n))


# ==========================================================================
# ⭐ the crash-resume JOURNAL (`pilot-checkpoint-resume`)
# ==========================================================================
#: Keys the journal carries that the FINAL artifact must NOT gain. The final
#: JSON's content-shape is consumed downstream (``--plot-only``/``aggregate``
#: and the analyst), so the instrumentation is additive **to the PARTIAL only**.
_JOURNAL_ONLY_KEYS = ("host_rss", "_journal")
#: Config fields exempt from the resume flag-equality check — operational
#: hooks, not physics. ``stop_after_arms`` is exactly the flag an interrupted
#: run carries and its resumption does not.
_RESUME_FLAG_EXEMPT = ("stop_after_arms",)


def partial_path(out: Path, scale: str, seed: int) -> Path:
    return Path(out) / f"pilot_{scale}_seed{seed}_PARTIAL.json"


def ckpt_path(out: Path, arm: str, seed: int) -> Path:
    return Path(out) / f"ckpt_{arm}_seed{seed}.eqx"


def save_arm_checkpoint(out: Path, arm: str, seed: int, model) -> Path:
    """Serialise one trained arm's leaves; ``tmp`` + ``os.replace`` (atomic)."""
    p = ckpt_path(out, arm, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    eqx.tree_serialise_leaves(tmp, model)
    os.replace(tmp, p)
    return p


def load_arm_checkpoint(out: Path, arm: str, seed: int, like):
    """Deserialise onto ``like`` — the freshly-built arm of the SAME geometry.

    ``like`` is rebuilt from ``seed`` alone (``build_arm`` splits a key derived
    from ``PRNGKey(1000 + seed)``), so the template is reproducible without any
    reference to the interrupted process.
    """
    return eqx.tree_deserialise_leaves(ckpt_path(out, arm, seed), like)


def _flag_dict(flags: Dict[str, Any]) -> Dict[str, Any]:
    """The config identity a resume must match, flattened to ``group.key``.

    ⚠ ``memory.phi_gain`` is EXCLUDED — it is a calibrated *output* written back
    into the flag block after the fact, and on resume it is lifted verbatim from
    the journal rather than recomputed, so comparing it here would compare the
    journal to itself. ``stop_after_arms`` is excluded because the interrupted
    run carries it and its resumption does not.
    """
    out: Dict[str, Any] = {}
    for grp, drop in (("pilot", _RESUME_FLAG_EXEMPT), ("memory", ("phi_gain",)),
                      ("store", ())):
        for k, v in dict(flags.get(grp) or {}).items():
            if k not in drop:
                out[f"{grp}.{k}"] = json.dumps(v, sort_keys=True, default=str)
    for k in ("store_dim", "store_n_atoms"):
        out[k] = json.dumps(flags.get(k), default=str)
    return out


def _flag_fingerprint(flags: Dict[str, Any]) -> str:
    return json.dumps(_flag_dict(flags), sort_keys=True)


def _flag_defaults() -> Dict[str, Any]:
    """The fingerprint value each ``memory.*`` key takes at its FIELD DEFAULT.

    ⭐ Why this exists (`pilot-ttt-nan-and-d5-wiring`). The ``pilot`` and
    ``store`` groups are ``as_flag_table()`` outputs — **non-default keys only**
    — so a config field added after a journal was written simply does not appear
    and the fingerprint is unchanged. The ``memory`` group is not: it is a full
    ``asdict(StreamMemoryConfig())``, so **every field added to that dataclass
    retro-invalidates every journal on disk.** That already happened: the four
    C2W6 fields (``erosion_partition``, ``refresh_amp_ceiling``,
    ``refresh_max_gain``, ``refresh_monotonic``) make the banked CSF3 tier-iii
    journals unresumable on current ``main`` — 4 x 16 h of A100 training that the
    checkpoint machinery was built to protect.
    """
    from chlu.core.blocks import StreamMemoryConfig
    return {f"memory.{k}": json.dumps(v, sort_keys=True, default=str)
            for k, v in asdict(StreamMemoryConfig()).items() if k != "phi_gain"}


def load_journal(out: Path, scale: str, seed: int, flags: Dict[str, Any],
                 ) -> Dict[str, Any]:
    """Read the PARTIAL record, or ``{}``; refuse a config-mismatched resume."""
    p = partial_path(out, scale, seed)
    if not p.exists():
        print(f"[resume] no journal at {p} — starting from scratch", flush=True)
        return {}
    prior = json.loads(p.read_text())
    old, new = _flag_dict(prior.get("flags", {})), _flag_dict(flags)
    # ⭐ A key the journal PREDATES is not a config change when the current value
    # is that field's own DEFAULT: the journal was necessarily written by code
    # running that default, because this repo ships every new lever gated OFF and
    # bit-identical (the toy bit-identity gate is what licenses saying so). Set
    # the same field to anything else and the keys differ, so the leg is still
    # refused — this loosens NOTHING except "the dataclass grew a field".
    # ⛔ The converse (a key the journal HAS and the current code does not) stays
    # a refusal: a deleted field's old value is not reconstructible.
    if set(new) - set(old):
        defaults = _flag_defaults()
        for k in sorted(set(new) - set(old)):
            if k in defaults and new[k] == defaults[k]:
                old[k] = new[k]
                print(f"[resume] '{k}' post-dates this journal and is at its "
                      f"default ({new[k]}) — same leg, accepted", flush=True)
    if old != new:
        bad = sorted(k for k in set(old) | set(new)
                     if old.get(k) != new.get(k))
        raise SystemExit(
            f"⛔ refusing to resume: {p} was written under a DIFFERENT config.\n"
            + "".join(f"    {k}: journal={old.get(k, '<absent>')} "
                      f"now={new.get(k, '<absent>')}\n" for k in bad)
            + "  (§A20.4: a resumed leg must be the SAME leg. If two ablation "
              "legs are sharing one --out, give each its own; otherwise delete "
              "the journal to start over.)")
    done = list((prior.get("arms") or {}).keys())
    trained = list((prior.get("_journal") or {}).get("trained", {}).keys())
    print(f"[resume] journal {p.name}: arms complete {done or '[]'} | "
          f"trained-not-evaluated {[a for a in trained if a not in done] or '[]'}",
          flush=True)
    return prior


# ==========================================================================
# ⭐⭐ THE PRE-REGISTERED-CONTINUATION EXEMPTION (`c3-run3-budget-exemption`)
# ==========================================================================
#: The journal file name the resume machinery owns. Anything else is refused:
#: :func:`load_journal` derives its path from ``(out, scale, seed)``, so the
#: exemption can only be verified against a file of this shape.
_JOURNAL_NAME_RE = re.compile(r"^pilot_(?P<scale>[A-Za-z0-9]+)_seed(?P<seed>\d+)"
                              r"_PARTIAL\.json$")
#: ⛔ EXACTLY these keys. ``sha256`` is the only optional one, and adding a key
#: here is a code change — which is precisely the friction that is wanted.
_CONTINUATION_REQUIRED = ("journal", "flag", "prereg")
_CONTINUATION_OPTIONAL = ("sha256",)
#: The fingerprint groups a registered flag may name. ``store_dim``/``store_n_atoms``
#: are DERIVED (they are functions of the store config), so they are not nameable:
#: a "flag" that is an output cannot be the one registered difference.
_CONTINUATION_GROUPS = ("pilot", "memory", "store")
#: ⛔ ...and these ``pilot.*`` names are the two override DICTS, not knobs. Naming
#: one would "register" every memory/store key at once — the wildcard by the back
#: door. Refused by name.
_CONTINUATION_NOT_KNOBS = ("pilot.memory", "pilot.store")
_ABSENT = "<absent from the journal>"


def flag_block(pcfg) -> Dict[str, Any]:
    """The config-identity block a journal is fingerprinted from.

    ⛔ Extracted verbatim from :func:`run_pilot` (it built this dict inline) so
    that the continuation check and the run agree **by construction** on what the
    flag block *is*. ``jax_version``/``jax_backend`` are added by ``run_pilot``
    and are not part of it — :func:`_flag_dict` never reads them.
    """
    return {
        "pilot": pcfg.as_flag_table(),
        "memory": asdict(pcfg.memory_cfg()),
        "store": pcfg.store_cfg().as_flag_table(),
        "store_dim": int(pcfg.store_cfg().dim),
        "store_n_atoms": int(pcfg.store_cfg().n_atoms),
    }


def _revert_knob(pcfg, group: str, key: str, jflags: Dict[str, Any]):
    """A copy of ``pcfg`` with ONE knob put back to the journal's setting.

    ⭐⭐ **Why the revert is at the KNOB and not at the fingerprint key.** One knob
    can move several fingerprint keys: ``memory.erosion_partition`` moves BOTH
    ``memory.erosion_partition`` (the resolved :class:`StreamMemoryConfig`) **and**
    ``pilot.memory`` (the override dict that produced it) — measured on the real
    run-2 journal. Reverting a single *fingerprint key* would therefore leave the
    other one differing and refuse the very leg this exists for, while reverting
    the *knob* regenerates both consistently. ⛔ The unit stays **one registered
    flag**; its derived consequences travel with it, and they are stamped into the
    ledger (``fingerprint_keys_moved``) so an auditor sees exactly what moved.

    ⛔ The journal's ``pilot`` group is an ``as_flag_table()`` output — the
    OVERRIDE table — so "revert" means *restore the override*, i.e. delete it when
    the journal did not carry it. If this revert is wrong in any way, the
    consequence is a **refusal** (``load_journal`` sees a difference), never a
    false acceptance: the failure direction is safe by construction.
    """
    cf = copy.deepcopy(pcfg)
    jpilot = dict(jflags.get("pilot") or {})
    if group in ("memory", "store"):
        d = dict(getattr(cf, group) or {})
        jd = dict(jpilot.get(group) or {})
        if key in jd:
            d[key] = jd[key]
        else:
            d.pop(key, None)
        setattr(cf, group, d)
    else:                       # a top-level PilotConfig field
        v = jpilot[key] if key in jpilot else getattr(PilotConfig(), key)
        if isinstance(getattr(cf, key, None), tuple) and isinstance(v, list):
            v = tuple(v)        # JSON has no tuples; `arms` is one
        setattr(cf, key, v)
    return cf


def _refuse_continuation(msg: str) -> "ContinuationExemptionError":
    return ContinuationExemptionError(
        "⛔ PRE-REGISTERED-CONTINUATION EXEMPTION REFUSED.\n" + msg
        + "\n  ⛔ This exemption is deliberately narrow: it accepts a config that "
          "is bit-identical to the named journal EXCEPT ONE registered flag, and "
          "nothing else. If a second flag must move, that is a NEW "
          "pre-registration and a code change — not a wider exemption.")


def verify_preregistered_continuation(
        spec: Any, pcfg, flags: Optional[Dict[str, Any]] = None, *,
        journal_loader: Optional[Callable[..., Dict[str, Any]]] = None,
) -> ContinuationExemption:
    """⭐ Verify that this config is **the registered continuation** it claims to be.

    ⛔ **The one and only constructor of a**
    :class:`~chlu.eval.byte_ledger.ContinuationExemption`. The identity check is
    **not reimplemented here**: it is delegated to :func:`load_journal`, the
    existing resume-fingerprint machinery, by asking it a counterfactual
    question —

        *"put the ONE registered knob back to the journal's setting; is this
        still the SAME LEG?"*

    If ``load_journal`` accepts the counterfactual, the two configs are identical
    except that knob; if it refuses, it refuses **by name**, printing every
    differing key, and that refusal is propagated verbatim. ⭐ Delegating matters
    more than it looks: the pilot merge repaired that machinery precisely so there
    is **one** source of truth about "is this the same leg", and a second parallel
    comparison is how the two silently drift apart — including the ``§A20.4``
    loosening (a key the journal *predates*, at its default, is the same leg),
    which run 3 needs, because the real run-2 journals predate
    ``erosion_partition`` entirely.

    Refuses, loudly and by name: a malformed/extra-keyed spec · a flag that is not
    exactly one fully-qualified ``group.key`` knob · a missing, unreadable or
    fingerprint-invalid journal · a sha256 that does not match · a registered flag
    that does not actually differ · **any second differing knob** · a differing
    knob that is not the registered one.
    """
    load = journal_loader or load_journal
    flags = flag_block(pcfg) if flags is None else flags
    # -- 1. the spec's shape ------------------------------------------------
    if not isinstance(spec, dict) or not spec:
        raise _refuse_continuation(
            f"    the config value is {type(spec).__name__} {spec!r}, not a "
            "mapping. Expected keys: journal=<PARTIAL.json> flag=<group.key> "
            "prereg=<doc> [sha256=<hex>]  (CLI: --prereg-continuation "
            "journal=... flag=... prereg=...)")
    unknown = sorted(set(spec) - set(_CONTINUATION_REQUIRED)
                     - set(_CONTINUATION_OPTIONAL))
    missing = [k for k in _CONTINUATION_REQUIRED if not str(spec.get(k, "")).strip()]
    if unknown or missing:
        raise _refuse_continuation(
            f"    unknown key(s): {unknown or '[]'}; missing/empty required "
            f"key(s): {missing or '[]'}. Required {list(_CONTINUATION_REQUIRED)}, "
            f"optional {list(_CONTINUATION_OPTIONAL)}.")
    flag = str(spec["flag"]).strip()
    prereg = str(spec["prereg"]).strip()
    # -- 2. EXACTLY ONE flag: no list, no wildcard, no glob -----------------
    if (not re.fullmatch(r"(?:%s)\.[A-Za-z_][A-Za-z0-9_]*"
                         % "|".join(_CONTINUATION_GROUPS), flag)
            or flag in _CONTINUATION_NOT_KNOBS):
        raise _refuse_continuation(
            f"    flag={flag!r} is not ONE fully-qualified knob. It must match "
            f"'<group>.<key>' with group in {list(_CONTINUATION_GROUPS)} — ⛔ a "
            "comma-separated list, a glob, a bare key, a derived quantity "
            f"(store_dim/store_n_atoms) and the override dicts "
            f"{list(_CONTINUATION_NOT_KNOBS)} (which would register EVERY key "
            "under them at once) are all refused. Exactly one knob, by name.")
    group, key = flag.split(".", 1)
    if group == "pilot" and key not in {f.name for f in fields(PilotConfig)}:
        raise _refuse_continuation(
            f"    flag={flag!r} names no PilotConfig field, so it is not a knob "
            "of this run.")
    # -- 3. the journal: present, readable, and its own fingerprint valid ---
    jpath = Path(str(spec["journal"]).strip())
    if not jpath.is_file():
        raise _refuse_continuation(
            f"    journal {jpath} does not exist (or is not a file). The prior "
            "leg's journal is the evidence that this IS a continuation; without "
            "it there is nothing to be identical to.")
    raw = jpath.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    pinned = str(spec.get("sha256", "")).strip().lower()
    if pinned and pinned != sha:
        raise _refuse_continuation(
            f"    journal sha256 mismatch: pinned {pinned}, on disk {sha} "
            f"({jpath}).")
    m = _JOURNAL_NAME_RE.match(jpath.name)
    if m is None:
        raise _refuse_continuation(
            f"    journal {jpath.name!r} is not a resume journal: the name must "
            "be 'pilot_<scale>_seed<N>_PARTIAL.json' (that is the file the "
            "fingerprint machinery owns — a final S*.json artifact is not it).")
    try:
        prior = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise _refuse_continuation(
            f"    journal {jpath} is unreadable/corrupt ({type(e).__name__}: "
            f"{e}).") from None
    jflags = prior.get("flags") if isinstance(prior, dict) else None
    ok_shape = (isinstance(jflags, dict)
                and all(isinstance(jflags.get(g), dict) for g in _CONTINUATION_GROUPS)
                and bool(jflags.get("memory"))
                and isinstance(jflags.get("store_dim"), int)
                and isinstance(jflags.get("store_n_atoms"), int))
    if not ok_shape:
        raise _refuse_continuation(
            f"    journal {jpath} carries no VALID flag fingerprint: it needs a "
            f"'flags' block with {list(_CONTINUATION_GROUPS)} groups, a non-empty "
            "'memory' group and integer store_dim/store_n_atoms. A journal whose "
            "own fingerprint does not validate cannot certify anything.")
    # -- 4. the registered knob must ACTUALLY differ -------------------------
    #     ⛔ read through _flag_dict — the SAME function load_journal uses — so
    #     "differs" means differs in the fingerprint, not to the eye.
    d_old, d_new = _flag_dict(jflags), _flag_dict(flags)
    #     ⭐ ...and forgiven exactly as load_journal forgives: a key the journal
    #     PREDATES, sitting at its own field default, is not a difference (§A20.4).
    #     ⛔ Same helper (`_flag_defaults`), so the two can never disagree.
    _defaults = _flag_defaults()
    predated = sorted(k for k in set(d_new) - set(d_old)
                      if k in _defaults and d_new[k] == _defaults[k])
    moved = sorted(k for k in set(d_old) | set(d_new)
                   if d_old.get(k) != d_new.get(k) and k not in predated)
    old_raw = jflags.get(group, {}).get(key, _ABSENT)
    new_raw = (flags.get(group) or {}).get(key, _ABSENT)
    if old_raw is _ABSENT and new_raw is _ABSENT:
        raise _refuse_continuation(
            f"    the registered flag {flag!r} is present in NEITHER the journal "
            "nor this config — it is not a flag of this run.")
    if flag not in moved:
        raise _refuse_continuation(
            f"    the registered flag {flag!r} does NOT differ from the journal "
            f"(journal={old_raw!r}, now={new_raw!r}). An exemption is for a "
            "REGISTERED CHANGE; a leg identical to the journal is the prior run "
            "itself and gets no exemption.\n"
            f"    Fingerprint keys that DO differ: {moved or '[]'}.")
    # -- 5. THE DELEGATED IDENTITY CHECK ------------------------------------
    #     Counterfactual: put the registered KNOB back to the journal's setting
    #     and ask the resume machinery whether this is still the same leg. ⛔ Any
    #     OTHER difference is refused there, by name, in its own words.
    cf_flags = flag_block(_revert_knob(pcfg, group, key, jflags))
    try:
        load(jpath.parent, m.group("scale"), int(m.group("seed")), cf_flags)
    except SystemExit as e:
        raise _refuse_continuation(
            f"    with the knob {flag!r} put back to the journal's setting "
            f"({old_raw!r}), the configs STILL differ — so this is not "
            "run-N-plus-ONE-registered-flag.\n"
            "    The resume fingerprint machinery names the offending key(s):\n"
            f"{e}") from None
    return ContinuationExemption(
        journal=str(jpath), journal_sha256=sha, prereg=prereg, flag=flag,
        old_value=old_raw, new_value=new_raw,
        fingerprint_keys_moved=tuple(moved),
        journal_predates_at_default=tuple(predated),
        verified_by=("load_journal(_flag_dict) counterfactual: with "
                     f"{flag} reverted, identical to {jpath.name}"),
    )


class _Phases:
    """Phase bookkeeping: RSS marks, cache hygiene, and journal banking.

    One object per run. ``step(key, fn)`` is the whole contract: it returns the
    banked value if the journal already has it (that is the resume), otherwise
    it computes it, marks the host RSS on both sides of the call, hands the
    memory back, and rewrites the journal atomically.
    """

    def __init__(self, rec: Dict[str, Any], prior: Dict[str, Any], pcfg,
                 out: Path, scale: str, seed: int, t0: float):
        self.rec, self.prior, self.pcfg = rec, prior, pcfg
        self.path = partial_path(out, scale, seed)
        self.t0 = t0
        rec.setdefault("host_rss", list(prior.get("host_rss") or []))
        rec.setdefault("_journal", {"trained": dict(
            (prior.get("_journal") or {}).get("trained", {}))})

    # -- instrumentation ----------------------------------------------------
    def mark(self, phase: str) -> Dict[str, float]:
        """One host-RSS reading, to stdout **and** the journal.

        ⛔ stdout is the load-bearing channel, not the JSON: a job killed by the
        kernel writes no artifact at all, so the ``[rss]`` line of the phase
        that was running is the only evidence that will exist.
        """
        if not bool(getattr(self.pcfg, "rss_log", True)):
            return {}
        r = host_rss()
        r["phase"] = phase
        r["t_s"] = round(time.time() - self.t0, 1)
        self.rec["host_rss"].append(r)
        print(f"[rss] {phase:<34s} rss {r.get('rss_gb', float('nan')):7.2f} GB | "
              f"peak {r.get('hwm_gb', float('nan')):7.2f} GB | "
              f"children {r.get('children_rss_gb', 0.0):6.2f} GB "
              f"(n={int(r.get('n_children', 0.0))}) | "
              f"t {r['t_s']:.0f}s", flush=True)
        return r

    def hygiene(self, phase: str) -> None:
        if bool(getattr(self.pcfg, "eval_cache_hygiene", True)):
            release_host_memory()
            self.mark(f"{phase}/released")

    def bank(self) -> None:
        save_json(self.path, self.rec, atomic=True)

    # -- the resume primitive ------------------------------------------------
    def step(self, key: str, fn: Callable[[], Any], *,
             into: Optional[Dict[str, Any]] = None,
             prior_into: Optional[Dict[str, Any]] = None,
             label: Optional[str] = None) -> Any:
        """``rec[key] = fn()`` — or lift it verbatim from the journal."""
        dst = self.rec if into is None else into
        src = self.prior if prior_into is None else prior_into
        name = label or key
        if key in (src or {}):
            dst[key] = src[key]
            print(f"[resume] phase '{name}': lifted from the journal", flush=True)
            return dst[key]
        self.mark(f"{name}/enter")
        dst[key] = fn()
        self.mark(f"{name}/exit")
        self.hygiene(name)
        self.bank()
        return dst[key]


def run_pilot(scale: str = "toy", seed: int = 0, stage: str = "s3",
              out_dir: str = ".claude/outputs/cluformer-pilot",
              overrides: Optional[dict] = None,
              with_d5: bool = False, resume: bool = False,
              with_slices: bool = False,
              prereg_continuation: Optional[Dict[str, str]] = None,
              ) -> Dict[str, Any]:
    """Run the pilot to ``stage``; write one JSON artifact; return the record.

    ⭐ ``prereg_continuation`` declares this leg a **pre-registered continuation**
    of a prior one — ``{journal, flag, prereg[, sha256]}``, verified by
    :func:`verify_preregistered_continuation` before anything is loaded or
    trained. ⛔ It is a **run-scoped argument and NOT a** ``PilotConfig`` **field**,
    exactly like ``with_d5``/``with_slices``: a config field would enter
    ``as_flag_table()`` and *be* a second key differing from the journal, breaking
    the identity check it is verified by (and the prereg's own "run 3 changes
    exactly one token"). It exempts the state-byte **budget check only**.

    ⭐ **Crash-resumable** (``resume=True``). Every phase is banked to
    ``pilot_{scale}_seed{N}_PARTIAL.json`` as it completes and every arm's
    trained weights to ``ckpt_{arm}_seed{N}.eqx``, both atomically; a resumed
    run lifts the banked phases verbatim and recomputes only what is missing.
    CSF3 attempt 1 lost 22 h of training to a host-RAM ``oom_kill`` **45 min
    into the post-training eval block** — so the weight checkpoint is written
    the instant training returns, before any evaluation, which is what makes
    that particular crash cost minutes instead of a day.

    ⭐ **The resume is EXACT for every arm, and here is why.** ``_train_batches``
    materialises the whole training stream ONCE from ``(seed, steps)`` and each
    arm consumes ``iter(batches)`` — a fresh iterator over the same list — so
    arm *k*'s data stream depends on nothing whatsoever carried out of arms
    ``0..k-1``. The eval iterators are ``contiguous_batches`` (deterministic, no
    seed). No fast-forwarding is therefore required and none is performed: a
    resumed arm is handed the identical batches in the identical order. The one
    piece of state that genuinely crosses a phase boundary — the persistent
    monitor registry, whose ``(write_loss, acq)`` window monitor #6 needs — is
    handled by computing ``monitors_final`` inside the training segment, while
    that registry is still alive.

    ⛔ The FINAL artifact is unchanged in content-shape; the journal keys
    (``host_rss``, ``_journal``) are stripped by :func:`_finish` and live only
    in the PARTIAL.
    """
    t_all = time.time()
    pcfg = make_config(scale, seed, overrides)
    out = Path(out_dir)
    rec: Dict[str, Any] = {
        "scale": scale, "seed": seed, "stage_requested": stage,
        # ⛔ `flag_block` IS this dict — extracted so the continuation check and
        # the run cannot drift on what the config identity is. Key order (and so
        # the artifact's bytes) is unchanged.
        "flags": {**flag_block(pcfg),
                  "jax_version": jax.__version__,
                  "jax_backend": jax.default_backend()},
        "stages_reached": [],
        "not_run": [],
    }
    # ⛔⛔ BOTH BUDGET GUARDS FIRE HERE, BEFORE THE DATA IS EVEN LOADED.
    # (i) the interim-ceiling guard: no rival-ladder rung may train while the
    #     matched-state-byte ceiling is a placeholder (it is set in the
    #     rival-ladder prereg). ⭐ Inert for run 3, whose arms are ours.
    _ladder_guard = assert_ladder_arms_admissible(pcfg.arms)
    # (ii) the pre-registered continuation, if declared. ⚠ Verified HERE and not
    #     at the ledger, because `phi_gain` is written back into `pcfg.memory`
    #     below — after which `flag_block(pcfg)` would carry a calibrated OUTPUT
    #     inside `pilot.memory` and the counterfactual would no longer match the
    #     journal. Verifying on the pristine config also means a bad exemption
    #     costs seconds, not a staged corpus.
    exemption = (verify_preregistered_continuation(
        prereg_continuation, pcfg, rec["flags"]) if prereg_continuation else None)
    prior = load_journal(out, scale, seed, rec["flags"]) if resume else {}
    ph = _Phases(rec, prior, pcfg, out, scale, seed, t_all)
    ph.mark("run_pilot/enter")
    tr, va, te = _data(pcfg)
    rec["data"] = {"train_B": len(tr), "valid_B": len(va), "test_B": len(te),
                   "n_bytes_staged": pcfg.data_bytes or 100_000_000}

    key = jax.random.PRNGKey(1000 + seed)
    k_cal, k_solve, k_model = jax.random.split(key, 3)
    calib_x = next(iter(random_batches(tr, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                       n_batches=1, seed=seed)))[0]
    # ⭐ the gain is a pure function of (seed, config); lifting it from the
    # journal on resume is exact AND skips a full-model probe forward.
    gain = ph.step("phi_gain_calibrated",
                   lambda: calibrate_phi_gain(pcfg, calib_x, key=k_cal),
                   label="phi_gain")
    pcfg.memory = dict(pcfg.memory)
    pcfg.memory["phi_gain"] = gain
    rec["flags"]["memory"]["phi_gain"] = gain

    specs, ledger = solve_arms(pcfg, k_solve)
    rec["swap_ledger"] = ledger

    arms = [a for a in pcfg.arms]
    # ⭐⭐ THE BYTE LEDGER IS STRUCTURAL, NOT OPTIONAL, AND IT IS TAKEN BEFORE ANY
    # TRAINING. The primary claim is at matched params AND matched state-bytes
    # (charter §2), so an arm that cannot state its inference-time state in bytes
    # — including φ — must not be reportable at all. An unledgered arm raises
    # UnledgeredArmError and an over-budget arm raises StateByteBudgetError; both
    # fail the run LOUDLY rather than warning, and both fire here, before a single
    # A100-hour is spent on a config whose control is not byte-honest.
    # ⭐ Ledger EVERY solved cell, not just the arms this job happens to run: the
    # seeds-as-jobs ladder submits one arm per job, and a per-job artifact that
    # carried only its own arm's bytes could not be assembled into the
    # matched-state-byte table the claim is read off.
    _ledger_arms = [k for k in ledger
                    if not k.startswith("_") and not k.endswith("_ARITHMETIC_ONLY")]
    rec["byte_ledger"] = build_byte_ledger(
        pcfg, ledger, sorted(set(_ledger_arms) | set(arms)),
        budget=int(pcfg.state_byte_budget),
        enforce=bool(pcfg.enforce_state_byte_budget),
        exemption=exemption)
    rec["byte_ledger"]["ladder_guard"] = _ladder_guard
    ph.bank()
    if with_slices:
        # ⚠ The slice's VALIDITY CONTROLS run once per run, before any arm is
        # scored: the non-degeneracy tripwire (TRAP 1) RAISES rather than
        # shipping a character-frequency count, and the shuffled-position control
        # must show the slice actually moves. Cheap (numpy on the test split) and
        # a hard precondition on every per-arm number that follows.
        from chlu.eval.text_slices import run_controls
        ph.step("slice_controls",
                lambda: run_controls(te.data,
                                     doc_boundary=get_corpus(pcfg.corpus).doc_boundary,
                                     seed=seed))
    models = {a: build_arm(a, pcfg, specs, key=k_model) for a in arms}
    rec["shell"] = assert_shared_shell_identical(models)
    rec["total_params"] = {a: _count(m) for a, m in models.items()}

    # ---------------- S1: the block runs the stream, monitors reported --------
    x0, y0 = _eval_batches(va, pcfg, 1)[0]
    t = time.time()

    def _monitors_init():
        r = monitor_pass(models["clu_store"], pcfg, x0)
        r["wall_s"] = time.time() - t
        return r

    ph.step("monitors_init", _monitors_init)
    ph.step("allocation_liveness_init",
            lambda: allocation_liveness(models["clu_store"], pcfg, x0, y0))
    rec["stages_reached"].append("S1")
    if stage == "s1":
        return _finish(rec, out, t_all, pcfg, ph)

    # ---------------- S2: the training path is real ---------------------------
    def _probe_init():
        t2 = time.time()
        r = gradient_probe(models["clu_store"], pcfg, x0, y0)
        r["wall_s_total"] = time.time() - t2
        return r

    ph.step("gradient_probe_init", _probe_init)
    rec["stages_reached"].append("S2")
    if stage == "s2":
        return _finish(rec, out, t_all, pcfg, ph)

    # ---------------- S3: the swap control, trained on identical data ---------
    # ⭐ ONE materialised stream, consumed by every arm through a FRESH iterator.
    # That is the whole resume guarantee: arm k's batches are a function of
    # (seed, steps) alone and carry nothing out of arms 0..k-1.
    batches = _train_batches(tr, pcfg)
    ev = _eval_batches(te, pcfg, pcfg.eval_batches)
    dv = _eval_batches(te, pcfg, pcfg.dyneval_batches)
    rec["train_log"] = []
    rec["arms"] = {}
    prior_arms = dict(prior.get("arms") or {})
    prior_log = list(prior.get("train_log") or [])
    trained_bank: Dict[str, Any] = rec["_journal"]["trained"]
    from chlu.core.monitors import default_registry
    for ai, a in enumerate(arms):
        t = time.time()
        banked = trained_bank.get(a)
        ck = ckpt_path(out, a, seed)
        # ⛔ RULING (2) of `pilot-ttt-nan-and-d5-wiring`'s reconciliation list —
        # THE .eqx PRECONDITION CHECK, BEFORE ANY RE-RESUME. A journal that banks
        # an arm as trained, whose checkpoint has since gone missing, silently
        # RETRAINS that arm (~16 h at pilot scale on CSF3) and the operator learns
        # only from the wall clock. Make it loud.
        if banked is not None and not ck.exists() and bool(
                getattr(pcfg, "resume_require_ckpt", True)):
            raise SystemExit(
                f"⛔ resume precondition failed for arm '{a}': the journal banks "
                f"it as TRAINED but {ck} is MISSING, so resuming would silently "
                f"RETRAIN it (~16 h at pilot scale).\n"
                f"    Check before submitting:  ls {out}/ckpt_*_seed{seed}.eqx | wc -l\n"
                f"    (expected one per arm in {list(arms)})\n"
                f"    To retrain deliberately, pass --set resume_require_ckpt=false.")
        # ⛔ Banked EVAL phases are only valid against banked WEIGHTS. If the
        # checkpoint is gone the arm is retrained, and its stale eval rows go
        # with it — a resumed static bpc must never describe a different model.
        parts: Dict[str, Any] = (dict(prior_arms.get(a, {}))
                                 if (banked is not None and ck.exists()) else {})
        rec["arms"][a] = parts
        wall_prior = float(parts.get("wall_s", 0.0))

        def _bank_arm(_p=parts, _t=t, _w=wall_prior):
            _p["wall_s"] = _w + (time.time() - _t)
            ph.bank()

        # -- (i) TRAIN, or lift the banked weights ---------------------------
        if banked is not None and ck.exists():
            m = load_arm_checkpoint(out, a, seed, models[a])
            parts.setdefault("train", banked["train"])
            if a == "clu_store":
                parts.setdefault("monitors_during", banked["monitors_during"])
                parts.setdefault("monitors_final", banked["monitors_final"])
            rec["train_log"].extend([e for e in prior_log if e.get("arm") == a])
            print(f"[resume] arm '{a}': {ck.name} loaded, training SKIPPED "
                  f"({len(parts['train']['loss_history'])} banked steps)", flush=True)
        else:
            ph.mark(f"{a}/train/enter")
            # ⭐ ONE persistent registry per arm, so monitor #6's (loss, acq)
            # window accumulates across the run instead of restarting at every
            # observation. ⚠ It cannot cross a process boundary, which is why
            # `monitors_final` is taken HERE, inside the training segment, while
            # the registry is still alive — a pure function of (m, reg), so
            # taking it before the eval phases instead of after is bitwise inert.
            reg = default_registry(loud=False) if a == "clu_store" else None
            during: List[Dict[str, Any]] = []
            m, hist = train_arm(a, models[a], pcfg, iter(batches),
                                log=rec["train_log"], monitor_registry=reg,
                                monitor_tokens=(x0 if reg is not None else None),
                                monitor_out=during)
            parts["train"] = hist
            bank_entry: Dict[str, Any] = {"train": hist}
            if a == "clu_store":
                parts["monitors_during"] = [
                    {k: v for k, v in d.items() if k != "readings"} for d in during]
                parts["monitors_final"] = monitor_pass(
                    m, pcfg, x0, registry=reg,
                    write_loss_now=float(hist["loss_history"][-1]))
                bank_entry["monitors_during"] = parts["monitors_during"]
                bank_entry["monitors_final"] = parts["monitors_final"]
            ph.mark(f"{a}/train/exit")
            # ⛔ the weights hit the disk BEFORE any evaluation — the eval block
            # is where attempt 1 died and 22 h of training died with it.
            save_arm_checkpoint(out, a, seed, m)
            trained_bank[a] = bank_entry
            _bank_arm()
            ph.hygiene(f"{a}/train")

        # -- (ii) EVALUATE, phase by phase, each one banked ------------------
        ph.step("static", lambda m=m: evaluate(m, pcfg, iter(ev)),
                into=parts, prior_into=parts, label=f"{a}/static")
        _bank_arm()
        ph.step("dyneval", lambda m=m: dynamic_eval(m, pcfg, iter(dv)),
                into=parts, prior_into=parts, label=f"{a}/dyneval")
        _bank_arm()
        # ⭐ THE RETENTION SLICE — computed for EVERY arm, and for each arm's
        # dyn-eval substitute column, from the same index on the same split.
        # ⛔ Never only for clu_store: a slice one arm gets is not a comparison.
        if with_slices:
            ph.step("slices",
                    lambda m=m, _p=parts: _arm_slices(m, pcfg, te, _p),
                    into=parts, prior_into=parts, label=f"{a}/slices")
            _bank_arm()
        if a == "clu_store":
            ph.step("blank_store",
                    lambda m=m: evaluate(m, pcfg, iter(ev), blank=True),
                    into=parts, prior_into=parts, label=f"{a}/blank_store")
            _bank_arm()
            if with_d5:
                mc = pcfg.memory_cfg()
                base = (mc.address_steps, mc.read_steps)
                ph.step("anytime_curve", lambda m=m, b=base: anytime_curve(
                    m, pcfg, ev, [(max(2, b[0] // f), max(2, b[1] // f))
                                  for f in (8, 4, 2, 1)] + [(b[0] * 2, b[1] * 2)]),
                    into=parts, prior_into=parts, label=f"{a}/anytime_curve")
                _bank_arm()
            ph.step("gradient_probe_final",
                    lambda m=m: gradient_probe(m, pcfg, x0, y0),
                    into=parts, prior_into=parts, label=f"{a}/gradient_probe_final")
            ph.step("selectors_final", lambda m=m: _selectors(m),
                    into=parts, prior_into=parts, label=f"{a}/selectors_final")
        parts["wall_s"] = wall_prior + (time.time() - t)
        rec["arms"][a] = _arm_row(parts, a, with_d5)
        models[a] = m
        ph.bank()
        print(f"[{a}] static bpc {rec['arms'][a]['static']['bpc']:.4f} | "
              f"dyneval bpc {rec['arms'][a]['dyneval']['bpc']:.4f} | "
              f"{rec['arms'][a]['wall_s']:.0f}s", flush=True)
        ph.hygiene(f"{a}/done")
        if 0 < int(getattr(pcfg, "stop_after_arms", 0)) <= ai + 1:
            print(f"⛔ stop_after_arms={pcfg.stop_after_arms}: hard-exiting after "
                  f"'{a}' with the journal on disk (no finalisers, as an "
                  f"oom_kill has none)", flush=True)
            os._exit(137)
    rec["swap_table"] = _swap_table(rec)
    rec["stages_reached"].append("S3")
    if stage in ("s3",):
        rec["not_run"].append(
            "S4 (26-47 M on CSF3): NOT RUN at this scale. See report.")
        return _finish(rec, out, t_all, pcfg, ph)

    rec["stages_reached"].append("S4")
    return _finish(rec, out, t_all, pcfg, ph)


def _slice_index(pcfg: PilotConfig, split):
    """Build (once per run) the revisit index for the scored split."""
    from chlu.eval.text_slices import build_revisit_index

    spec = get_corpus(pcfg.corpus)
    return build_revisit_index(split.data, doc_boundary=spec.doc_boundary,
                               unit="token")


def _arm_slices(model, pcfg: PilotConfig, split, parts: Dict[str, Any]
                ) -> Dict[str, Any]:
    """One arm's retention slices: the static column AND the dyn-eval column."""
    from chlu.eval.text_slices import (evaluate_slices, evaluate_slices_dyneval,
                                       slice_gap)

    spec = get_corpus(pcfg.corpus)
    idx = _slice_index(pcfg, split)
    nb = int(pcfg.slice_batches) or int(pcfg.eval_batches)
    kw = dict(corpus=pcfg.corpus, doc_boundary=spec.doc_boundary,
              n_batches=nb, min_n=int(pcfg.slice_min_n), index=idx)
    static = evaluate_slices(model, pcfg, split, **kw)
    # the substitute column gets the IDENTICAL slice, at its own best LR
    lr = float((parts.get("dyneval") or {}).get("best_lr", pcfg.dyneval_lr))
    dyn = evaluate_slices_dyneval(model, pcfg, split, lr=lr, **kw)
    return {"static": static, "dyneval": dyn,
            "dyneval_minus_static": slice_gap(static, dyn)}


def _arm_row(parts: Dict[str, Any], arm: str, with_d5: bool) -> Dict[str, Any]:
    """Re-key one arm's banked pieces into the artifact's CANONICAL order.

    ⛔ The journal accumulates phases in whatever order a (possibly resumed) run
    produced them; the final artifact's key order is part of its content-shape
    and is fixed here, so an interrupted+resumed run and an uninterrupted one
    emit the same object.
    """
    order = ["train", "static", "dyneval", "slices"]
    if arm == "clu_store":
        order += ["blank_store"]
        if with_d5:
            order += ["anytime_curve"]
        order += ["monitors_during", "monitors_final", "gradient_probe_final",
                  "selectors_final"]
    order += ["wall_s"]
    return {k: parts[k] for k in order if k in parts}


def _selectors(model) -> Dict[str, Any]:
    """The trainable friction/mass selectors after training (§A13 rule 3, P8)."""
    import jax.numpy as jnp
    ga = [float(jnp.exp(b.cell.log_gamma_addr)) for b in model.blocks]
    gr = [float(jnp.exp(b.cell.log_gamma_read)) for b in model.blocks]
    mm = [float(jnp.mean(jax.nn.softplus(b.cell.clu.log_mass))) for b in model.blocks]
    return {"gamma_address": ga, "gamma_read": gr, "mean_mass": mm}


def _swap_table(rec: Dict[str, Any]) -> Dict[str, Any]:
    """⭐ The swap table with the dynamic-evaluation column **in it**, not a footnote."""
    ref = rec["arms"].get("clu_store")
    rows = {}
    for a, r in rec["arms"].items():
        row = {"bpc_static": r["static"]["bpc"], "bpc_dyneval": r["dyneval"]["bpc"],
               "params_total": rec["total_params"][a],
               "cell_params": rec["swap_ledger"].get(a, {}).get("params"),
               "cell_state_bytes": rec["swap_ledger"].get(a, {}).get("state_bytes"),
               "wall_s": r["wall_s"]}
        if ref is not None:
            row["margin_vs_clu_static"] = ref["static"]["bpc"] - r["static"]["bpc"]
            row["margin_vs_clu_dyneval"] = ref["dyneval"]["bpc"] - r["dyneval"]["bpc"]
        rows[a] = row
    return rows


def _finish(rec, out: Path, t0: float, pcfg: PilotConfig,
            ph: Optional["_Phases"] = None) -> Dict[str, Any]:
    """Write the FINAL artifact.

    ⛔ ``_JOURNAL_ONLY_KEYS`` are stripped here: the crash journal's RSS series
    and per-arm weight bookkeeping are additive artifacts of the PARTIAL file,
    and the final artifact's content-shape is exactly what it was before
    `pilot-checkpoint-resume` (``--plot-only``/:func:`aggregate` and the analyst
    consume it).
    """
    rec["wall_s_total"] = time.time() - t0
    if ph is not None:
        ph.mark("run_pilot/finish")
        ph.bank()
    body = {k: v for k, v in rec.items() if k not in _JOURNAL_ONLY_KEYS}
    # ⭐ The retention slices also land as their OWN artifact, with the citation
    # block and the validity controls attached, so the instrument's output can be
    # consumed (and audited) without parsing the whole pilot record.
    if rec.get("slice_controls") or any(
            "slices" in v for v in (rec.get("arms") or {}).values()):
        from chlu.eval.text_slices import SliceReport
        SliceReport(
            corpus=str(pcfg.corpus), split="test",
            arms={a: v["slices"] for a, v in (rec.get("arms") or {}).items()
                  if "slices" in v},
            controls=dict(rec.get("slice_controls") or {}),
            meta={"scale": rec["scale"], "seed": rec["seed"],
                  "eval_batches": int(pcfg.slice_batches or pcfg.eval_batches),
                  "min_n": int(pcfg.slice_min_n)},
        ).save(out / f"slices_{rec['scale']}_seed{rec['seed']}.json")
    p = save_json(out / f"pilot_{rec['scale']}_seed{rec['seed']}_"
                        f"{rec['stages_reached'][-1]}.json", body)
    rec["artifact"] = str(p)
    print(f"wrote {p} ({rec['wall_s_total']:.0f}s, stages "
          f"{'+'.join(rec['stages_reached'])})", flush=True)
    _ = pcfg
    return rec


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Seed-mean +- s.e. of every arm's bpc, and the falsifier's adjudication.

    ⛔ **DF1 as pre-registered**: tier iii is alive only if the CLU's seed-mean
    bpc is at least 0.02 below BOTH matched swap arms', with the gap exceeding
    the sum of the two +-1 s.e. bars. ⛔ **DF3**: if the CLU's advantage does not
    survive dynamic evaluation, the primary is dead.
    """
    arms = sorted({a for r in records for a in r.get("arms", {})})
    out: Dict[str, Any] = {"n_seeds": len(records), "seeds": [r["seed"] for r in records],
                           "arms": {}}
    for a in arms:
        for col in ("static", "dyneval"):
            v = [r["arms"][a][col]["bpc"] for r in records if a in r.get("arms", {})]
            out["arms"].setdefault(a, {})[f"bpc_{col}_mean"] = float(np.mean(v))
            out["arms"][a][f"bpc_{col}_se"] = (
                float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))
            out["arms"][a][f"bpc_{col}_per_seed"] = [float(x) for x in v]
    if "clu_store" in out["arms"]:
        c = out["arms"]["clu_store"]
        verdict = {}
        for opp in ("gru_matched", "ttt_matched"):
            if opp not in out["arms"]:
                continue
            o = out["arms"][opp]
            for col in ("static", "dyneval"):
                m = o[f"bpc_{col}_mean"] - c[f"bpc_{col}_mean"]   # >0 => CLU better
                se = c[f"bpc_{col}_se"] + o[f"bpc_{col}_se"]
                verdict[f"{opp}_{col}"] = {
                    "clu_advantage_bpc": m, "se_sum": se,
                    "passes_0.02_and_se": bool(m >= 0.02 and m > se),
                }
        out["DF1_alive"] = all(v["passes_0.02_and_se"]
                               for k, v in verdict.items() if k.endswith("static"))
        out["DF3_primary_dead"] = any(
            verdict.get(f"{o}_dyneval", {}).get("clu_advantage_bpc", -1e9)
            < verdict.get(f"{o}_static", {}).get("clu_advantage_bpc", 1e9) - 0.02
            for o in ("gru_matched", "ttt_matched") if f"{o}_static" in verdict)
        out["verdict"] = verdict
    return out


# ==========================================================================
# plots (artifacts live under .claude/, never in the repo)
# ==========================================================================
def plot_pilot(records: List[Dict[str, Any]], out_dir: str) -> Optional[str]:
    """Four panels: training curves - the swap table - the anytime curve - monitors.

    Self-contained rather than routed through ``chlu/utils/plotting.py``: that
    module is shared across every campaign and this task owns none of it, so a
    new helper there is a merge-conflict surface for no benefit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [r for r in records if r.get("arms")]
    if not recs:
        return None
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"clu_store": "tab:red", "gru_matched": "tab:blue",
              "ttt_matched": "tab:green", "none": "0.5", "echo": "tab:orange"}

    # (a) training curves, seed 0
    r0 = recs[0]
    for arm in r0["arms"]:
        h = r0["arms"][arm]["train"]["loss_history"]
        b = [v / np.log(2.0) for v in h]
        k = max(1, len(b) // 60)
        sm = [float(np.mean(b[max(0, i - k):i + 1])) for i in range(len(b))]
        ax[0, 0].plot(sm, label=arm, color=colors.get(arm))
    ax[0, 0].set(xlabel="optimisation step", ylabel="train bpc",
                 title=f"(a) training, seed {r0['seed']} ({r0['scale']} scale)")
    ax[0, 0].legend(fontsize=8)

    # (b) the swap table: static vs dyn-eval, seed-mean +- se
    arms = list(r0["arms"])
    xs = np.arange(len(arms))
    lo, hi = np.inf, -np.inf
    for j, col in enumerate(("static", "dyneval")):
        mu = [np.mean([r["arms"][a][col]["bpc"] for r in recs]) for a in arms]
        se = [(np.std([r["arms"][a][col]["bpc"] for r in recs], ddof=1)
               / np.sqrt(len(recs))) if len(recs) > 1 else 0.0 for a in arms]
        ax[0, 1].bar(xs + 0.36 * j - 0.18, mu, 0.34, yerr=se, capsize=3,
                     label=("static" if col == "static" else "dynamic eval"))
        lo = min(lo, min(np.array(mu) - np.array(se)))
        hi = max(hi, max(np.array(mu) + np.array(se)))
    pad = 0.25 * (hi - lo) + 1e-6
    ax[0, 1].set_ylim(lo - pad, hi + pad)     # zoomed: a 0-based axis hides 0.01 bpc
    ax[0, 1].set_xticks(xs)
    ax[0, 1].set_xticklabels(arms, rotation=20, fontsize=8)
    ax[0, 1].set(ylabel="held-out bpc",
                 title=f"(b) system-level swap, {len(recs)} seeds "
                       "(dyn-eval IN the table)")
    ax[0, 1].legend(fontsize=8)

    # (c) the anytime curve — SHAPE only
    any_c = [r["arms"]["clu_store"].get("anytime_curve") for r in recs
             if "clu_store" in r["arms"]]
    if any(any_c):
        for r, c in zip(recs, any_c, strict=False):
            if not c:
                continue
            ax[1, 0].plot([p["verlet_per_read"] for p in c],
                          [p["bpc"] for p in c], "o-", label=f"seed {r['seed']}")
        ax[1, 0].set(xscale="log", xlabel="Verlet steps per read",
                     ylabel="held-out bpc",
                     title="(c) anytime shape (SHAPE claim only, §A3)")
        ax[1, 0].legend(fontsize=8)
    else:
        ax[1, 0].text(0.5, 0.5, "D5 NOT RUN", ha="center", transform=ax[1, 0].transAxes)
        ax[1, 0].set_axis_off()

    # (d) monitors
    m = r0["arms"].get("clu_store", {}).get("monitors_final") or r0["monitors_init"]
    # ⚠ THREE states, not two: `inapplicable` is NOT `clear`, and colouring it
    # green would be exactly the silent pass the acceptance criterion forbids.
    names = [x["name"] for x in m["readings"]]
    state = ["TRIPPED" if x.get("tripped") else
             ("clear" if x.get("applicable", True) else "inapplicable")
             for x in m["readings"]]
    cmap = {"TRIPPED": "tab:red", "clear": "tab:green", "inapplicable": "0.75"}
    ax[1, 1].barh(names, [1.0] * len(names), color=[cmap[s_] for s_ in state])
    for yi, s_ in enumerate(state):
        ax[1, 1].text(0.5, yi, s_, ha="center", va="center", fontsize=7,
                      color="white" if s_ != "inapplicable" else "black")
    ax[1, 1].set(xlim=(0, 1), xticks=[],
                 title=f"(d) 13 monitors + M14, seed {r0['seed']} — "
                       f"{m['n_tripped']} TRIPPED, "
                       f"{len(names) - m['n_applicable']} inapplicable")
    ax[1, 1].tick_params(labelsize=7)
    fig.suptitle("Tier-iii pilot: the FULL C2W1 CLU store as a streaming block's "
                 "memory (enwik8)  —  acceptance = DOES NOT COLLAPSE, not WINS")
    fig.tight_layout()
    p = Path(out_dir) / f"pilot_{r0['scale']}_panels.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def _parse_kv(pair: str):
    """``key=value`` with int/float/bool/None inference (strings pass through).

    Exists so the CSF3 job script can carry a recommended config without the
    module being edited on the cluster — an edited module is a provenance hole
    (the artifact would not say which config produced it), whereas a flag is
    recorded verbatim in ``rec['flags']``.
    """
    if "=" not in pair:
        raise SystemExit(f"expected KEY=VALUE, got {pair!r}")
    k, v = pair.split("=", 1)
    low = v.strip().lower()
    if low in ("true", "false"):
        return k.strip(), low == "true"
    if low in ("none", "null"):
        return k.strip(), None
    for cast in (int, float):
        try:
            return k.strip(), cast(v)
        except ValueError:
            pass
    return k.strip(), v


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", choices=("toy", "pilot"), default="toy")
    ap.add_argument("--stage", choices=("s1", "s2", "s3", "s4"), default="s3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="run several seeds and aggregate (>=3 for any reported number)")
    ap.add_argument("--out", default=".claude/outputs/cluformer-pilot")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-aggregate + re-plot from artifacts already on disk")
    ap.add_argument("--d5", action="store_true",
                    help="also run the anytime shape curve (secondary; §A3 shape only)")
    ap.add_argument("--corpus", default=None,
                    help="the real stream, by config value (enwik8 | wikitext103). "
                         "⭐ A third stream is one loader module + one "
                         "register_corpus() call — see chlu/data/corpora.py.")
    ap.add_argument("--slices", action="store_true",
                    help="⭐ also compute the WITHIN-DOCUMENT RETENTION/REVISIT "
                         "slices (Sun et al., EMNLP 2021, arXiv:2109.09115) for "
                         "EVERY arm and for every arm's dyn-eval substitute "
                         "column, plus the validity controls. Like --d5 this is a "
                         "CLI argument and NOT a PilotConfig field, so it cannot "
                         "move the resume fingerprint: a finished leg can be "
                         "re-resumed with --resume --slices to add the slice "
                         "phases and nothing else.")
    ap.add_argument("--resume", action="store_true",
                    help="⭐ resume from `pilot_<scale>_seed<N>_PARTIAL.json` + the "
                         "per-arm `ckpt_<arm>_seed<N>.eqx` in --out: banked phases "
                         "are lifted verbatim and completed arms are skipped "
                         "entirely. Refuses to resume a journal written under a "
                         "different config. Safe on a fresh --out (no journal => "
                         "a normal run).")
    ap.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE",
                    help="top-level PilotConfig overrides, e.g. monitor_every=25 "
                         "plan_workers=8 accum_steps=2 liveness_lanes=1 "
                         "(⭐ the last two are `csf3-memory-fit`'s out-of-model "
                         "memory levers; both default to the shipped behaviour). "
                         "⭐ `pilot-checkpoint-resume` adds the HOST-memory ones: "
                         "eval_cache_hygiene=false (keep the eval block's "
                         "one-shot executables — the attempt-1 behaviour), "
                         "rss_log=false (silence the per-phase [rss] lines), "
                         "stop_after_arms=N (hard-exit after N arms; test hook)")
    ap.add_argument("--mem", nargs="*", default=None, metavar="KEY=VALUE",
                    help="StreamMemoryConfig overrides, e.g. atom_place_radius=0.3 "
                         "write_inner_steps=40 remat_chunks=true (⭐ the "
                         "`pilot-placement-probe` recommendation block and "
                         "`csf3-memory-fit`'s remat levers both set the submitted "
                         "config here, so the scale run never needs the module "
                         "edited)")
    ap.add_argument("--store", nargs="*", default=None, metavar="KEY=VALUE",
                    help="CluSystemConfig overrides")
    ap.add_argument("--prereg-continuation", nargs="*", default=None,
                    metavar="KEY=VALUE",
                    help="⭐ declare this leg a PRE-REGISTERED CONTINUATION of a "
                         "prior one: journal=<pilot_<scale>_seed<N>_PARTIAL.json> "
                         "flag=<group.key> prereg=<doc> [sha256=<hex>]. ⛔ Exempts "
                         "the state-byte BUDGET check ONLY, and only after the "
                         "resume-fingerprint machinery has confirmed this config "
                         "is identical to that journal EXCEPT the ONE registered "
                         "flag (run 3 = run 2 + memory.erosion_partition). The "
                         "ledger is still computed and printed in full, stamped "
                         "with the journal, its sha256 and the flag's old->new "
                         "value. A second differing key is refused BY NAME.")
    a = ap.parse_args(argv)
    ov: Dict[str, Any] = {}
    for flag, key in (("set", None), ("mem", "memory"), ("store", "store")):
        pairs = getattr(a, flag)
        if not pairs:
            continue
        parsed = dict(_parse_kv(p) for p in pairs)
        if key is None:
            ov.update(parsed)
        else:
            ov[key] = dict((TOY if a.scale == "toy" else PILOT).get(key, {}),
                           **parsed)
    # ⛔ str values only — every field of the exemption is a name (a path, a flag,
    # a doc, a hex digest), never a number `_parse_kv` would cast. ⭐ It goes to
    # `run_pilot` as an ARGUMENT and never into `ov`: it must not reach
    # `PilotConfig`, or it would itself become a second key differing from the
    # journal it is being verified against.
    pc: Optional[Dict[str, str]] = None
    if a.prereg_continuation:
        pc = {}
        for p in a.prereg_continuation:
            if "=" not in p:
                raise SystemExit(f"--prereg-continuation expects KEY=VALUE, "
                                 f"got {p!r}")
            k, v = p.split("=", 1)
            pc[k.strip()] = v.strip()
    if a.steps is not None:
        ov["steps"] = a.steps
        ov["warmup"] = max(1, a.steps // 10)
    if a.arms:
        ov["arms"] = tuple(a.arms)
    if a.corpus:
        ov["corpus"] = a.corpus
    if a.quick:
        ov.update(steps=6, warmup=2, eval_batches=2, dyneval_batches=2,
                  data_bytes=1_000_000)
    if a.plot_only:
        recs = [json.loads(p.read_text())
                for p in sorted(Path(a.out).glob(f"pilot_{a.scale}_seed*_S*.json"))]
        agg = aggregate(recs)
        save_json(Path(a.out) / f"pilot_{a.scale}_aggregate.json", agg)
        print("plot:", plot_pilot(recs, a.out), flush=True)
        print(json.dumps(agg, indent=2, default=float), flush=True)
        return 0
    seeds = a.seeds if a.seeds else [a.seed]
    recs = [run_pilot(a.scale, s, a.stage, a.out, ov, a.d5, a.resume, a.slices,
                      prereg_continuation=pc)
            for s in seeds]
    if len(recs) > 1:
        agg = aggregate(recs)
        save_json(Path(a.out) / f"pilot_{a.scale}_aggregate.json", agg)
        print("plot:", plot_pilot(recs, a.out), flush=True)
        print(json.dumps(agg, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
