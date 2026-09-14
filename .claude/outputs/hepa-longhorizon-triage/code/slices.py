"""The information-age slice (task hepa-lh-verdict §5).

Splits ft_test EVENTS by the age of the **decisive precursor** and lifts that
split onto the evaluation WINDOWS, so any (N, K) probability surface -- HEPA's
from `preds/`, or the tripwire's -- can be scored on each slice separately.

Definitions (NOT invented here: D4/D5 of `.claude/scratch/hepa-lh-audit/
DEFINITIONS.md`, frozen by the audit spoke before its harness existed; the
arithmetic below is a line-for-line mirror of `hepa-lh-audit/audit.py` and is
cross-checked against `audit.json`'s aggregate median / p90 / censored_frac by
`verify_against_audit`).

  decisive precursor age of onset e
      = time-to-last-similar-regime (D5) if it exists,
      = time-to-previous-event      (D4) if D5 is CENSORED,
      = UNDEFINED if neither exists (no prior similar window AND no prior event).

  slice(e) = "old"   if age  > BOUNDARY (512, the reference rung)
             "young" if age <= BOUNDARY
             "undefined" otherwise -- reported, never silently merged.

Window lifting. Every ft_test cut t with a finite time-to-event is predicting
the run that contains index t + tte; that run's onset is its event, so the cut
inherits that event's slice. Cuts with tte = inf are NEGATIVES: they belong to
no event and are therefore included in BOTH slices, unchanged. Without them a
slice has no negative class and h-AUROC is undefined. This is declared here, in
advance, because it is the one modelling choice in the whole section.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import hepa_iface as HI
from hepa_iface import load_dataset  # HEPA's own loader, re-exported

W: int = 32                    # audit D5: 2 HEPA patches at P=16
SEED: int = 42                 # audit D5: tau sample seed
MAX_WINDOWS_FOR_TAU: int = 5000
BOUNDARY: int = 512            # PREREG §3 reference rung
UNDERPOWERED_N: int = 10       # task §5


def onsets(labels: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels).astype(np.int8)
    if len(lab) == 0:
        return np.zeros(0, dtype=np.int64)
    d = np.diff(lab, prepend=0)
    return np.flatnonzero(d == 1).astype(np.int64)


def _win_feats(seg: np.ndarray):
    n = len(seg) // W
    if n == 0:
        return np.zeros((0, seg.shape[1]), np.float64), np.zeros(0, np.int64)
    f = seg[: n * W].reshape(n, W, seg.shape[1]).mean(axis=1)
    ends = (np.arange(1, n + 1) * W).astype(np.int64)      # exclusive end index
    return f.astype(np.float64), ends


def precursor_table(dataset: str) -> Dict[str, object]:
    """Per-event decisive-precursor age, for every event in every split.

    Returns the whole dataset (all three splits) so the aggregates can be
    checked against `audit.json`; the slice only ever uses ft_test.
    """
    b = load_dataset(dataset)
    segs = []
    for sp in ("ft_train", "ft_val", "ft_test"):
        for e in b[sp]:
            segs.append((str(e["entity_id"]), sp,
                         np.asarray(e["test"], np.float32),
                         np.asarray(e["labels"], np.int32)))

    allF, seg_windows = [], []
    for _, _, x, _ in segs:
        f, ends = _win_feats(x)
        seg_windows.append((f, ends))
        if len(f):
            allF.append(f)
    tau = mu = sd = None
    if allF:
        F = np.concatenate(allF, 0)
        mu, sd = F.mean(0), F.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(F), size=min(MAX_WINDOWS_FOR_TAU, len(F)), replace=False)
        S = (F[idx] - mu) / sd
        D = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
        iu = np.triu_indices(len(S), k=1)
        tau = float(np.percentile(D[iu], 10.0)) if len(iu[0]) else None

    events: List[Dict[str, object]] = []
    for (eid, sp, x, lab), (f, ends) in zip(segs, seg_windows):
        on = onsets(lab)
        Fz = (f - mu) / sd if (tau is not None and len(f)) else None
        for i, e in enumerate(on):
            ttprev = int(e - on[i - 1]) if i > 0 else None
            ttlsr: Optional[int] = None
            status = "EXCLUDED_e_lt_2W"
            if tau is not None and Fz is not None:
                if e < 2 * W:
                    status = "EXCLUDED_e_lt_2W"
                else:
                    anchor = ((x[e - W:e].mean(axis=0) - mu) / sd).astype(np.float64)
                    cand = np.flatnonzero(ends <= e - W)
                    if len(cand) == 0:
                        status = "CENSORED"
                    else:
                        d = np.linalg.norm(Fz[cand] - anchor[None, :], axis=1)
                        ok = np.flatnonzero(d <= tau)
                        if len(ok) == 0:
                            status = "CENSORED"
                        else:
                            ttlsr = int(e - ends[cand[ok[-1]]])
                            status = "MATCHED"
            if ttlsr is not None:
                age, src = ttlsr, "time_to_last_similar_regime"
            elif ttprev is not None:
                age, src = ttprev, "time_to_previous_event(fallback: D5 %s)" % status
            else:
                age, src = None, "UNDEFINED(D5 %s, no previous event)" % status
            events.append({
                "entity_id": eid, "split": sp, "onset": int(e),
                "ttlsr": ttlsr, "ttlsr_status": status, "ttprev": ttprev,
                "decisive_precursor_age": age, "age_source": src,
                "slice": ("undefined" if age is None
                          else ("old" if age > BOUNDARY else "young")),
            })
    return {"dataset": dataset, "tau_10pct": tau, "W": W, "boundary": BOUNDARY,
            "events": events}


def verify_against_audit(dataset: str, table: Dict[str, object],
                         audit_row: Dict[str, object]) -> Dict[str, object]:
    """Reproduce audit.json's D5 aggregates from the per-event table."""
    ev = table["events"]
    t = np.array([e["ttlsr"] for e in ev if e["ttlsr"] is not None], dtype=float)
    n_cens = sum(1 for e in ev if e["ttlsr_status"] == "CENSORED")
    n_excl = sum(1 for e in ev if e["ttlsr_status"] == "EXCLUDED_e_lt_2W")
    a = audit_row["time_to_last_similar_regime"]
    got = {"tau_10pct": table["tau_10pct"], "n": int(len(t)),
           "median": float(np.median(t)) if len(t) else None,
           "p90": float(np.percentile(t, 90)) if len(t) else None,
           # audit.py's denominator is n_events INCLUDING the e<2W exclusions
           "censored_frac": float(n_cens / max(1, len(t) + n_cens + n_excl)),
           "n_excluded_e_lt_2W": int(n_excl)}
    want = {k: a.get(k) for k in got}
    ok = all(
        (got[k] is None and want[k] is None)
        or (isinstance(got[k], (int, float)) and isinstance(want[k], (int, float))
            and abs(got[k] - want[k]) < 1e-9)
        for k in got
    )
    return {"reproduces_audit_json": bool(ok), "recomputed": got, "audit_json": want}


def window_slices(dataset: str, context: int,
                  table: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Boolean masks over the ft_test windows at this rung, plus their counts."""
    if table is None:
        table = precursor_table(dataset)
    bundle = HI.load_bundle(dataset, context)
    te = bundle.test

    # onset index of the run containing each positive step, per test entity
    b = load_dataset(dataset)
    run_onset: Dict[str, np.ndarray] = {}
    ev_slice: Dict[str, Dict[int, str]] = {}
    ev_age: Dict[str, Dict[int, Optional[int]]] = {}
    for e in b["ft_test"]:
        eid = str(e["entity_id"])
        lab = np.asarray(e["labels"], np.int32)
        cur, ro = -1, np.full(len(lab), -1, dtype=np.int64)
        for i, v in enumerate(lab):
            if v == 1 and (i == 0 or lab[i - 1] == 0):
                cur = i
            ro[i] = cur if v == 1 else -1
        run_onset[eid] = ro
        ev_slice[eid] = {}
        ev_age[eid] = {}
    for rec in table["events"]:
        if rec["split"] == "ft_test":
            ev_slice[rec["entity_id"]][int(rec["onset"])] = rec["slice"]
            ev_age[rec["entity_id"]][int(rec["onset"])] = rec["decisive_precursor_age"]

    n = len(te)
    is_neg = ~np.isfinite(te.tte)
    win_slice = np.array(["negative"] * n, dtype=object)
    for i in range(n):
        if is_neg[i]:
            continue
        eid = str(te.entity[i])
        p = int(te.t_index[i] + te.tte[i])
        ro = run_onset[eid]
        o = int(ro[p]) if 0 <= p < len(ro) else -1
        win_slice[i] = ev_slice.get(eid, {}).get(o, "undefined")

    masks = {
        "old": (win_slice == "old") | is_neg,
        "young": (win_slice == "young") | is_neg,
        "undefined": (win_slice == "undefined") | is_neg,
    }
    ev_test = [r for r in table["events"] if r["split"] == "ft_test"]
    counts = {
        s: {
            "n_events": sum(1 for r in ev_test if r["slice"] == s),
            "n_positive_windows": int((win_slice == s).sum()),
            "n_windows_scored": int(masks[s].sum()),
            "underpowered": bool(sum(1 for r in ev_test if r["slice"] == s)
                                 < UNDERPOWERED_N),
        }
        for s in ("old", "young", "undefined")
    }
    counts["negative_windows_shared"] = int(is_neg.sum())
    counts["n_events_ft_test"] = len(ev_test)
    counts["n_windows_ft_test"] = n
    return {"dataset": dataset, "context": context, "masks": masks,
            "win_slice": win_slice, "counts": counts, "bundle": bundle,
            "ages_ft_test": {k: v for k, v in ev_age.items()}}
