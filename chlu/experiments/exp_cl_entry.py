"""⭐ Experiment CL-ENTRY (w25): the continual-learning entry — one build, three results.

**What this is.** Rehearsal-free **Class-Incremental Learning** (van de Ven & Tolias's
hardest scenario: new classes each task, task identity **not** given at test time) on
Split-MNIST and Split-CIFAR-10, from scratch, entered with a **designed CLU store**:

* the address is ``φ(x)`` — a read-in fit **once, on task-1 classes only**, then frozen
  for the whole stream (`PREREG_CL_PHI.md`, ``phi_dim ≥ 16`` is binding);
* the store is a designed landscape of Gaussian wells over those addresses, one well
  per admitted item, **payload = the class label** (no raw exemplar is ever stored);
* the write policy is the hand-coded **MVC-0 controller** (`chlu.core.controller`):
  admission (spacing gate), placement, class-balanced eviction under a fixed item
  budget, and per-item **scheduled decay**;
* the read is a damped-Verlet settle in that landscape; the prediction is the label of
  the well the particle lands in.

**Three results share the build** (they are three views of one run, not three systems):

1. **R4 — the entry.** ACC + forgetting/BWT (GEM formulas) against the mandatory
   baseline table: tuned ER · iCaRL · **GDumb at matched memory** (the pathology check)
   · EWC/SI/LwF (the **known nulls** — never presented as a CLU win) · and the
   **kNN-in-φ laundering control** at matched memory (N89/CM-22(i)).
2. **R3-native — retry in its home regime.** The accuracy-vs-compute retry ladder on
   **crowded-store retrieval of past-task items**, mid-stream and end-of-stream, per
   task-age, with the RUD-C mechanism controls (kick · ensemble · ungated · matched-
   compute feedforward). ⭐ **No oracle exists here**: the corruption is applied in
   *pixel* space and the store's metric is ``φ``, so the ML-optimal "masked-NN" oracle
   that capped w24's measurement (`headroom-retry-benchmark`) **cannot be constructed**
   — there is no coordinate subset of the store's space that is known-erased. The
   ambiguity lives in the landscape (class-clustered wells, packing pressure), not in a
   query mask. The honest floor is kNN-in-φ, which is also the laundering control.
3. **R1-survivor — scheduled per-item retention on the live stream.** A cohort written
   **permanent** (``leak = 0``) rides through every task at retention 1.0 while cohorts
   with scheduled half-lives decay as ``exp(−leak·t)`` and self-evict. ⛔ **Naming is
   fixed (CM-22 m/n/o): scheduled per-item retention / scheduled forgetting. Never
   "certified", never "unlearning", never "deletion by construction", never "exact
   deletion".** This is a capability demonstration inside a benchmarked system, not a
   privacy claim.

**Scope, stated up front (the Head's filing rule CM-23(n)).** Class-IL at this scale is
*solved* by replay (DGR 90.8 / iCaRL 94.6) and by a dumb balanced buffer (GDumb). The
entry's claim is **"best rehearsal-free"** + the two dial results — **NOT** "beats
replay", which is never claimed under any outcome.

Runnable: ``uv run python -m chlu.experiments.exp_cl_entry --quick`` or via the CLI
``chlu exp-cl-entry [--project N] [--seed I] [--quick] [--dataset mnist|cifar10]
[--items entry,retry,retention]``.
"""

import copy
import json
import os
from typing import Optional

import jax
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.controller import Controller
from chlu.core.memory_potentials import AtomStorePotential, GaussianMemoryPotential
from chlu.experiments.cl_baselines import (
    REHEARSAL_FREE,
    REPLAY,
    cl_metrics,
    run_baseline_stream,
)
from chlu.experiments.exp_hopfield_capacity import (
    _median_nn_distance,
    _settle_read,
    dropout_query,
)
from chlu.experiments.exp_phi_read_in import build_read_in
from chlu.experiments.exp_phi_stream import load_labeled_images
from chlu.experiments.exp_retry_compute import (
    _confidence_and_nn,
    _ensemble_ladder,
    _feedforward_ladder,
    _retry_ladder,
)
from chlu.experiments.goldstone_harness import clu_with_potential

#: the φ regimes this entry may run (PREREG_CL_PHI §1). ``online`` is out of protocol.
PHI_PRIMARY = "task1_only"
PHI_REFERENCE = "generic_frozen"

#: wording guard for the R1 result (CM-22 m/n/o) — imported by the tests
FORBIDDEN_R1_WORDS = (
    "certified",
    "unlearning",
    "deletion by construction",
    "exact deletion",
)
R1_NAME = "scheduled per-item retention"


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


def task_classes(cfg, t: int):
    c = cfg.classes_per_task
    return list(range(t * c, (t + 1) * c))


def build_cl_stream(cfg, seed: int, data=None):
    """Build the Class-IL stream with the φ-fairness regions of `PREREG_CL_PHI` §2.

    The TRAIN split is permuted once and cut into a **fit region** (the only source of
    φ fit pools) and a **stream region** (the only source of items the store and the
    baselines ever see) — so no φ, in either regime, is fit on a stored item. The TEST
    split is held out entirely and provides the per-task evaluation sets.

    ``data`` may be an explicit ``((Xtr, ytr), (Xte, yte))`` pair — used by the tests to
    drive the whole pipeline on tiny synthetic labelled data.
    """
    if data is None:
        Xtr, ytr = load_labeled_images(cfg.dataset, "train")
        Xte, yte = load_labeled_images(cfg.dataset, "test")
    else:
        (Xtr, ytr), (Xte, yte) = data
    Xtr = np.asarray(Xtr, np.float32)
    ytr = np.asarray(ytr).astype(int)
    Xte = np.asarray(Xte, np.float32)
    yte = np.asarray(yte).astype(int)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    n_fit = min(cfg.n_fit_region, len(Xtr) // 2)
    fit_region, stream_region = perm[:n_fit], perm[n_fit:]

    train_X, train_y, test_X, test_y, classes = [], [], [], [], []
    for t in range(cfg.n_tasks):
        cls = task_classes(cfg, t)
        pool = stream_region[np.isin(ytr[stream_region], cls)][: cfg.n_train_per_task]
        train_X.append(Xtr[pool])
        train_y.append(ytr[pool])
        te = np.flatnonzero(np.isin(yte, cls))[: cfg.n_test_per_task]
        test_X.append(Xte[te])
        test_y.append(yte[te])
        classes.append(cls)

    n_classes = cfg.n_tasks * cfg.classes_per_task
    seen_upto = []
    m = np.zeros(n_classes, bool)
    for cls in classes:
        m = m.copy()
        m[cls] = True
        seen_upto.append(m)

    fit_generic = Xtr[fit_region[: cfg.n_fit_pool]]
    t1 = fit_region[np.isin(ytr[fit_region], classes[0])][: cfg.n_fit_pool]
    return {
        "train_X": train_X, "train_y": train_y,
        "test_X": test_X, "test_y": test_y,
        "task_classes": classes, "seen_mask_upto": seen_upto,
        "fit_pool_task1_only": Xtr[t1],
        "fit_pool_generic_frozen": fit_generic,
        "dim": int(Xtr.shape[1]),
    }


def build_phi(regime: str, stream, cfg, seed: int):
    """The frozen read-in for one regime (PREREG_CL_PHI §1/§3)."""
    if regime not in (PHI_PRIMARY, PHI_REFERENCE):
        raise ValueError(
            f"φ regime {regime!r} is out of protocol (PREREG_CL_PHI §1); "
            f"expected {PHI_PRIMARY!r} or {PHI_REFERENCE!r}"
        )
    pool = stream[f"fit_pool_{regime}"]
    phi, prov = build_read_in(cfg.phi_arm, cfg.dataset, stream["train_X"][0], pool, cfg, seed)
    prov = {
        **prov,
        "regime": regime,
        "role": ("PRIMARY — every headline number"
                 if regime == PHI_PRIMARY
                 else "REFERENCE — declared upper bound, LEAKS future tasks"),
        "may_see": (f"task-1 classes only {stream['task_classes'][0]}"
                    if regime == PHI_PRIMARY else "all stream classes"),
        "n_fit_pool": int(len(pool)),
        "frozen_from": "end of task 1 (never refit, never re-keyed)",
        "phi_dim": int(cfg.phi_dim),
    }
    return phi, prov


# ---------------------------------------------------------------------------
# The designed store, driven by the MVC-0 controller
# ---------------------------------------------------------------------------


class PhiStore:
    """The CL entry's memory: MVC-0 controller over a designed store addressed by φ.

    * **address** ``= φ(x) ∈ R^{phi_dim}`` (the store is *content-addressed*, so the
      controller runs with ``allow_relocation=False``: an item may be refused, never
      moved — a relocated address is an address no query can reach);
    * **payload** ``=`` the class label (an integer, not an exemplar);
    * **budget** = a fixed number of live items, enforced by a **class-balanced LRU**
      eviction policy (GDumb's balancer, applied to a landscape);
    * **width** ``s = clu_s_frac · median-NN(keys already stored)``, recomputed at each
      stream position under ``s_policy="refit"`` — never from unseen data.
    """

    def __init__(self, cfg, phi_dim: int, seed: int):
        self.cfg = cfg
        self.phi_dim = int(phi_dim)
        self.budget = int(cfg.memory_items)
        self.s = None
        store = AtomStorePotential(
            dim=self.phi_dim + 1, capacity=self.budget + 1, alpha=cfg.store_alpha,
            s=1.0, kappa=1.0, addr_dim=self.phi_dim,
        )
        self.ctrl = Controller(
            store, d_safe=0.0, budget=self.budget + 1, amp=1.0,
            leak=0.0, amp_floor=cfg.amp_floor, evict_policy="staleness",
            allow_relocation=False,
        )
        self.rng = np.random.default_rng(seed + 31337)
        self.labels = {}   # item_id -> label
        self.item_task = {}  # item_id -> task index at write
        self.cohort = {}   # item_id -> retention cohort name
        self.next_id = 0
        self.per_task_stats = []

    # -- geometry ---------------------------------------------------------
    def set_width(self, keys):
        """Set ``s`` (hence ``d_safe``) from the given keys — the fixed rule."""
        keys = np.asarray(keys, float)
        if len(keys) < 3:
            return
        med = float(_median_nn_distance(keys))
        self.s = self.cfg.clu_s_frac * med
        self.ctrl.d_safe = self.cfg.d_safe_mult * self.s

    # -- write ------------------------------------------------------------
    def _victim(self):
        """Class-balanced LRU: the least-recently-used item of the most-represented
        class (GDumb's balancer). Permanent items are exempt (never evicted)."""
        recs = [r for r in self.ctrl.records.values() if not r.permanent]
        if not recs:
            return None
        labels = np.array([self.labels[r.item_id] for r in recs])
        vals, counts = np.unique(labels, return_counts=True)
        big = vals[np.argmax(counts)]
        cand = [r for r, lab in zip(recs, labels, strict=True) if lab == big]
        return min(cand, key=lambda r: (r.last_used, r.born)).item_id

    def offer(self, key_vec, label: int, task: int, leak: float = 0.0,
              permanent: bool = False, cohort: str = "no_decay"):
        """Offer one item. Returns the controller's decision row."""
        item_id = self.next_id
        self.next_id += 1
        if self.ctrl.n_live >= self.budget:
            vid = self._victim()
            if vid is None:
                # every live item is permanent: a **capacity alarm**, never a
                # silent overwrite. Permanent writes consume budget forever —
                # that is the price of the ``leak = 0`` cohort, and it is reported.
                self.ctrl.stats["refused_full"] += 1
                return {"decision": "refuse_full", "item_id": item_id, "slot": None}
            self.ctrl.evict_item(vid)
            self.labels.pop(vid, None)
        row = self.ctrl.offer(
            item_id, np.asarray(key_vec, float), float(label),
            permanent=permanent, leak=leak,
        )
        if row["decision"] in ("admit", "relocate"):
            self.labels[item_id] = int(label)
            self.item_task[item_id] = int(task)
            self.cohort[item_id] = cohort
        return row

    def tick(self):
        """One scheduled-decay tick (per-item retention)."""
        self.ctrl.tick()  # eviction bookkeeping lives in the controller's stats

    # -- read -------------------------------------------------------------
    def live(self):
        ids, centers, payloads = self.ctrl.live_items()
        return np.asarray(ids), np.asarray(centers, float), np.asarray(payloads, float)

    def read_model(self):
        """The read landscape: per-well amplitudes (so decay is physical) over the
        live addresses. ``None`` when the store is empty."""
        _, centers, _ = self.live()
        if len(centers) == 0:
            return None, None
        amps = self.ctrl.live_amps()
        V = GaussianMemoryPotential(
            centers, s=self.s, b=self.cfg.clu_b, alpha=self.cfg.clu_alpha, amps=amps
        )
        model = clu_with_potential(
            V, dim=self.phi_dim, kinetic_mode=self.cfg.clu_kinetic_mode
        )
        dt = self.cfg.clu_dt if self.cfg.clu_dt > 0 else 0.5 * self.s / np.sqrt(self.cfg.clu_b)
        return model, float(dt)

    def settle(self, feat_q):
        model, dt = self.read_model()
        if model is None:
            return None
        tail = int(max(1, self.cfg.clu_tail_frac * self.cfg.clu_steps))
        return _settle_read(
            model, np.asarray(feat_q, np.float32), self.cfg.clu_steps, dt,
            self.cfg.clu_gamma, tail, self.cfg.rollout_chunk,
        )

    def predict(self, feat_q):
        """Class-IL read-out: settle, then take the label of the well landed in."""
        ids, centers, payloads = self.live()
        if len(centers) == 0:
            return np.full(len(feat_q), -1), np.full(len(feat_q), np.inf), None
        reads = np.asarray(self.settle(feat_q))
        d2 = ((reads[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        idx = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(idx)), idx])
        return payloads[idx].round().astype(int), dist, reads

    def knn_predict(self, feat_q):
        """The laundering control on the SAME keys: skip the settle, take the
        nearest stored address directly. Isolates what the physics contributes."""
        _, centers, payloads = self.live()
        if len(centers) == 0:
            return np.full(len(feat_q), -1)
        d2 = ((np.asarray(feat_q, float)[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        return payloads[d2.argmin(1)].round().astype(int)


# ---------------------------------------------------------------------------
# The trivial-buffer laundering line (no admission gate, no landscape)
# ---------------------------------------------------------------------------


class RingBufferKNN:
    """kNN-in-φ over a **class-balanced ring buffer** at the same item budget — the
    N89 launder in its strongest form (a balanced buffer beats FIFO, so this is the
    adversarial version). If it matches the entry, the win is φ's and the buffer's."""

    def __init__(self, budget: int):
        self.budget = int(budget)
        self.keys, self.labels, self.age = [], [], []
        self.t = 0

    def offer(self, key_vec, label: int):
        self.t += 1
        self.keys.append(np.asarray(key_vec, float))
        self.labels.append(int(label))
        self.age.append(self.t)
        if len(self.keys) > self.budget:
            lab = np.array(self.labels)
            vals, counts = np.unique(lab, return_counts=True)
            big = vals[np.argmax(counts)]
            cand = np.flatnonzero(lab == big)
            drop = int(cand[np.argmin(np.array(self.age)[cand])])
            for lst in (self.keys, self.labels, self.age):
                lst.pop(drop)

    def predict(self, feat_q):
        if not self.keys:
            return np.full(len(feat_q), -1)
        K = np.stack(self.keys)
        d2 = ((np.asarray(feat_q, float)[:, None, :] - K[None, :, :]) ** 2).sum(-1)
        return np.asarray(self.labels)[d2.argmin(1)]


# ---------------------------------------------------------------------------
# Item 1 — the CLU entry over the stream
# ---------------------------------------------------------------------------


def _schedule_for(cfg, n_offered_this_task: int, decay_on: bool):
    """Cohort assignment for the R1 demo, **rate-limited by design**.

    The first ``permanent_per_task`` items offered in each task are written
    ``leak = 0`` (permanent); the rest alternate between the slow and fast
    half-lives. The cap matters: permanent items are exempt from eviction, so an
    unlimited permanent cohort would consume the whole budget in task 1 and the
    store would (correctly) start refusing writes. That trade — *permanence costs
    budget forever* — is the honest content of the dial, and it is reported.
    """
    if not decay_on:
        return 0.0, False, "no_decay"
    if n_offered_this_task < cfg.permanent_per_task:
        return 0.0, True, "permanent"
    return (
        (float(cfg.leak_slow), False, "slow")
        if n_offered_this_task % 2 == 0
        else (float(cfg.leak_fast), False, "fast")
    )


def run_clu_entry(cfg, stream, regime: str, seed: int, decay_on: bool = False,
                  collect_store: bool = False):
    """Walk the stream once with the designed store; return the Class-IL accuracy
    matrix, the controller's per-task statistics and (optionally) the live store."""
    phi, prov = build_phi(regime, stream, cfg, seed)
    T = cfg.n_tasks
    A = np.zeros((T, T))
    A_knn_same = np.zeros((T, T))
    A_ring = np.zeros((T, T))
    store = PhiStore(cfg, cfg.phi_dim, seed)
    out_mid = None
    ring = RingBufferKNN(cfg.memory_items)
    per_task, retention_rows = [], []
    key = jax.random.PRNGKey(seed + 5150)
    ticks = 0

    for t in range(T):
        Xt = stream["train_X"][t]
        yt = stream["train_y"][t]
        keys_t = np.asarray(phi(Xt), float)

        # width policy: task 1 sets s from its own (already-seen) keys; later tasks
        # refit from the keys ALREADY IN THE STORE (never from unseen data)
        if t == 0:
            store.set_width(keys_t[: cfg.s_init_items])
            s_task1 = store.s
        elif cfg.s_policy == "refit":
            _, centers, _ = store.live()
            store.set_width(centers)
        else:
            store.s = s_task1
            store.ctrl.d_safe = cfg.d_safe_mult * store.s

        n_offered = len(keys_t)
        tick_every = max(1, n_offered // max(1, cfg.ticks_per_task))
        stats0 = dict(store.ctrl.stats)
        for i in range(n_offered):
            leak, perm, cohort = _schedule_for(cfg, i, decay_on)
            store.offer(keys_t[i], int(yt[i]), t, leak=leak, permanent=perm,
                        cohort=cohort)
            ring.offer(keys_t[i], int(yt[i]))
            if decay_on and (i + 1) % tick_every == 0:
                store.tick()
                ticks += 1
                retention_rows.append(_retention_snapshot(cfg, store, stream, phi, ticks, key))

        stats = {k: store.ctrl.stats[k] - stats0.get(k, 0) for k in store.ctrl.stats}
        stats.update({
            "task": t, "offered": n_offered, "n_live": store.ctrl.n_live,
            "well_width_s": float(store.s), "d_safe": float(store.ctrl.d_safe),
            "admitted_fraction": stats["admitted"] / max(1, n_offered),
            "intervention_rate": 1.0 - stats["admitted"] / max(1, n_offered),
        })
        per_task.append(stats)

        # mid-stream snapshot for the R3-native measurement (Item 3 asks for BOTH
        # mid-stream and end-of-stream crowded-store retrieval)
        if collect_store and t == int(cfg.retry_mid_task):
            out_mid = copy.deepcopy(store)

        # --- evaluate every task seen so far (Class-IL: no task id) --------
        for i in range(t + 1):
            fq = np.asarray(phi(stream["test_X"][i]), float)
            ye = stream["test_y"][i]
            pred, _dist, _ = store.predict(fq)
            A[t, i] = float(np.mean(pred == ye))
            A_knn_same[t, i] = float(np.mean(store.knn_predict(fq) == ye))
            A_ring[t, i] = float(np.mean(ring.predict(fq) == ye))

    # end-of-stream geometry diagnostics (corrected packing slack, w24 A1)
    _, centers, _ = store.live()
    fq_all = np.concatenate(
        [np.asarray(phi(stream["test_X"][i]), float) for i in range(T)]
    )
    diag = _geometry_report(centers, fq_all, store.s)
    out = {
        "regime": regime, "seed": int(seed), "phi_provenance": prov,
        "decay_on": bool(decay_on),
        "A_clu": A.tolist(), "A_knn_same_keys": A_knn_same.tolist(),
        "A_knn_ringbuffer": A_ring.tolist(),
        "metrics_clu": cl_metrics(A),
        "metrics_knn_same_keys": cl_metrics(A_knn_same),
        "metrics_knn_ringbuffer": cl_metrics(A_ring),
        "per_task": per_task, "geometry": diag,
        "memory_items": int(len(centers)),
        "memory_floats": int(len(centers) * cfg.phi_dim),
        "retention_rows": retention_rows,
    }
    if collect_store:
        out["_store"] = store
        out["_store_mid"] = out_mid
        out["_phi"] = phi
    return out


def _geometry_report(centers, feat_q, s):
    """The store's geometry at the end of the stream, with the **corrected** packing
    slack (w24 A1: σ_q must be a displacement NORM, not a per-element RMS — the
    per-element form pinned the slack at the tautology ``1/(3.1·clu_s_frac)``).

    Here σ_q is the RMS distance from a test query's φ to its nearest stored address:
    in a *classification* stream the query is a different image, not a corrupted copy
    of a stored one, so this is the honest query displacement.
    """
    centers = np.asarray(centers, float)
    if len(centers) < 3:
        return {"n_stored": int(len(centers))}
    F = np.asarray(feat_q, float)
    d2 = ((F[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    nn_d = np.sqrt(d2.min(1))
    sigma_q = float(np.sqrt(np.mean(nn_d**2)))
    med = float(_median_nn_distance(centers))
    slack = med / (3.1 * max(float(s), sigma_q) + 1e-12)
    return {
        "n_stored": int(len(centers)),
        "median_nn_addresses": med,
        "well_width_s": float(s),
        "sigma_q_norm": sigma_q,
        "packing_slack_corrected": float(slack),
        "packing_slack_note": (
            "slack = median_NN / (3.1·max(s, σ_q)), σ_q = RMS‖φ(test query) − nearest "
            "address‖ (a NORM, per the w24 unit correction). slack < 1 ⇒ the store "
            "runs past the packing bound — intrinsic to classification, and the "
            "source of the geometric ambiguity Item 3 measures."
        ),
    }


# ---------------------------------------------------------------------------
# Item 3 — the R3-native retry ladder (crowded-store recall of past-task items)
# ---------------------------------------------------------------------------


def _auroc(scores, labels):
    """AUROC of ``scores`` for the binary ``labels`` (ties handled by rank means)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    if y.all() or (~y).any() is False or y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def retry_native(cfg, store, stream, phi, seed: int, label: str):
    """⭐ Item 3 — the retry ladder on **stored past-task items** in the crowded store.

    Queries are pixel-space dropout corruptions of the stored images, embedded through
    the frozen φ; the target is the item's **own well**. ⭐ **No mask oracle exists**:
    the erasure is applied in PIXEL space while the store is addressed in φ, so no
    baseline can be told which store coordinates were destroyed. The honest floor is
    kNN-in-φ at matched compute (which is also the laundering control).

    Runs one cell per corruption level in ``cfg.retry_mask_levels`` and, on the first
    level, a τ-sweep of the confidence gate (the auto-stop is measured, not assumed).
    """
    ids, centers, _ = store.live()
    if len(centers) < 8:
        return {"label": label, "skipped": "store too small"}
    model, dt = store.read_model()
    key = jax.random.PRNGKey(seed + 606)
    tail = int(max(1, cfg.clu_tail_frac * cfg.clu_steps))
    true_idx = np.arange(len(centers))
    # item_id is the global offer index ⇒ it indexes the concatenated stream
    stored_raw = np.concatenate(stream["train_X"])[np.asarray(ids, int)]
    age = np.array([store.item_task[int(i)] for i in ids])

    cells = []
    for li, p in enumerate(cfg.retry_mask_levels):
        key, kq = jax.random.split(key)
        Q = np.asarray(phi(np.asarray(dropout_query(stored_raw, p, kq))), float)
        reads0 = _settle_read(
            model, Q.astype(np.float32), cfg.clu_steps, dt, cfg.clu_gamma, tail,
            cfg.rollout_chunk,
        )
        cos0, nn0 = _confidence_and_nn(reads0, centers)
        correct0 = nn0 == true_idx
        d_well = np.linalg.norm(reads0 - centers[nn0], axis=1)

        lines = {}
        for mode in ("gated", "kick", "ungated"):
            lines[mode] = _retry_ladder(
                model, Q, reads0, centers, true_idx, cfg, dt, cfg.retry_tau, mode,
                np.random.default_rng(seed + 111),
            )
        lines["ensemble"] = _ensemble_ladder(
            model, Q, reads0, centers, true_idx, cfg, dt,
            np.random.default_rng(seed + 909),
        )
        lines["feedforward_knn_phi"] = _feedforward_ladder(
            Q, centers, true_idx, cfg, np.random.default_rng(seed + 313)
        )

        tau_sweep = {}
        if li == 0:
            for tau in cfg.retry_tau_grid:
                tau_sweep[str(tau)] = _retry_ladder(
                    model, Q, reads0, centers, true_idx, cfg, dt, tau, "gated",
                    np.random.default_rng(seed + 111),
                )

        by_age = {}
        for a in sorted(set(age.tolist())):
            m = age == a
            if m.sum() < 3:
                continue
            by_age[str(int(a))] = {
                "n": int(m.sum()),
                "first_pass_acc": float(np.mean(correct0[m])),
                "gated": _retry_ladder(
                    model, Q[m], reads0[m], centers, true_idx[m], cfg, dt,
                    cfg.retry_tau, "gated", np.random.default_rng(seed + 222 + a),
                ),
                "knn_phi": float(
                    np.mean(
                        ((Q[m][:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
                        == true_idx[m]
                    )
                ),
            }

        cells.append({
            "label": label,
            "mask_p": float(p),
            "n_items": int(len(centers)),
            "first_pass_acc": float(np.mean(correct0)),
            "mean_confidence": float(np.mean(cos0)),
            "confidence_auroc_cosine": _auroc(cos0, correct0),
            "confidence_auroc_neg_distance": _auroc(-d_well, correct0),
            "knn_phi_floor": float(
                np.mean(((Q[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
                        == true_idx)
            ),
            "ladders": {
                k: {str(kk): list(vv) for kk, vv in v.items()} for k, v in lines.items()
            },
            "tau_sweep": {
                t: {str(kk): list(vv) for kk, vv in v.items()}
                for t, v in tau_sweep.items()
            },
            "by_task_age": by_age,
        })

    return {
        "label": label,
        "cells": cells,
        "no_oracle_note": (
            "No mask oracle exists in this regime: the corruption is applied in PIXEL "
            "space and the store is addressed in φ, so there is no known-erased "
            "coordinate subset of the store's space to hand a baseline (this is exactly "
            "the 'space the store is not metric-native to' that w24's "
            "headroom-retry-benchmark named as the only untested route). The honest "
            "floor is kNN-in-φ (= the laundering control), reported at matched compute."
        ),
    }


# ---------------------------------------------------------------------------
# Item 4 — scheduled per-item retention (the R1 survivor)
# ---------------------------------------------------------------------------


def _retention_snapshot(cfg, store, stream, phi, tick: int, key):
    """One retention measurement on the live stream: per-cohort well amplitude
    (against the scheduled law) and per-cohort **retrieval** retention — can the item
    still be recalled from a corrupted query? ⛔ *scheduled per-item retention* — not
    deletion, not unlearning, not a privacy claim."""
    ids, centers, _ = store.live()
    if len(centers) < 3:
        return {"tick": int(tick), "n_live": int(len(centers))}
    amps = store.ctrl.live_amps()
    slots = store.ctrl.live_slots()
    recs = sorted((store.ctrl.records[s] for s in slots), key=lambda r: r.item_id)
    born = np.array([r.born for r in recs], float)
    leaks = np.array([r.leak for r in recs], float)
    cohort_of = np.array([store.cohort.get(int(i), "?") for i in ids])

    raw = np.concatenate(stream["train_X"])
    stored_raw = raw[np.asarray(ids, int)]
    kq = jax.random.fold_in(key, tick)
    Q = np.asarray(
        phi(np.asarray(dropout_query(stored_raw, cfg.retry_mask_p, kq))), float
    )
    reads = np.asarray(store.settle(Q.astype(np.float32)))
    d2 = ((reads[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    hit = d2.argmin(1) == np.arange(len(centers))
    age = tick - born
    pred_amp = np.exp(-leaks * age)

    cohorts = {}
    for cohort in ("permanent", "slow", "fast"):
        m = cohort_of == cohort
        if m.sum() == 0:
            cohorts[cohort] = {"n": 0}
            continue
        cohorts[cohort] = {
            "n": int(m.sum()),
            "mean_amp": float(np.mean(amps[m])),
            "mean_predicted_amp": float(np.mean(pred_amp[m])),
            "max_abs_amp_error": float(np.max(np.abs(amps[m] - pred_amp[m]))),
            "retrieval_retention": float(np.mean(hit[m])),
            "mean_age_ticks": float(np.mean(age[m])),
        }
    return {
        "tick": int(tick), "n_live": int(len(centers)),
        "decayed_out_cum": int(store.ctrl.stats["decayed_out"]),
        "refused_full_cum": int(store.ctrl.stats["refused_full"]),
        "cohorts": cohorts,
    }


def retention_law_check(rows, cfg):
    """Does the measured amplitude follow the scheduled law ``exp(−leak·t)``?"""
    out = {}
    for cohort, leak in (("permanent", 0.0), ("slow", cfg.leak_slow),
                         ("fast", cfg.leak_fast)):
        sel = [
            r for r in rows
            if "cohorts" in r and r["cohorts"].get(cohort, {}).get("n", 0) > 0
        ]
        if not sel:
            out[cohort] = {"n_points": 0}
            continue
        out[cohort] = {
            "leak": float(leak),
            "half_life_ticks": (float(np.log(2) / leak) if leak > 0 else None),
            "n_points": len(sel),
            "ticks": [int(r["tick"]) for r in sel],
            "n_live": [int(r["cohorts"][cohort]["n"]) for r in sel],
            "mean_age_ticks": [r["cohorts"][cohort]["mean_age_ticks"] for r in sel],
            "measured_amp": [r["cohorts"][cohort]["mean_amp"] for r in sel],
            "predicted_amp_exp_minus_leak_t": [
                r["cohorts"][cohort]["mean_predicted_amp"] for r in sel
            ],
            "max_abs_error": float(
                max(r["cohorts"][cohort]["max_abs_amp_error"] for r in sel)
            ),
            "retrieval_retention": [
                r["cohorts"][cohort]["retrieval_retention"] for r in sel
            ],
        }
    out["evictions"] = {
        "decayed_out_total": int(rows[-1].get("decayed_out_cum", 0)) if rows else 0,
        "refused_full_total": int(rows[-1].get("refused_full_cum", 0)) if rows else 0,
    }
    out["naming"] = (
        f"{R1_NAME} / scheduled forgetting — a capability of the store, NOT "
        "unlearning, NOT deletion, NOT a privacy claim (CM-22 m/n/o)"
    )
    return out


# ---------------------------------------------------------------------------
# Aggregation across seeds + the mandatory table
# ---------------------------------------------------------------------------


def _mean_std(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=0))


#: the ONE defining hyper-parameter of each rehearsal-free baseline (N78 discipline)
TUNED_HYPER = {"ewc": "ewc_lambda", "si": "si_c", "lwf": "lwf_alpha"}


def tune_baselines(cfg, stream, seed: int):
    """N78 rescued-baseline discipline: sweep each rehearsal-free baseline's single
    defining hyper-parameter on ONE seed, keep the best ACC, and use it for every
    seed. The grid, every value's ACC and the winner are reported — a baseline that
    was never tuned is not a baseline (and its collapse would not be evidence)."""
    chosen, log = {}, []
    for method, name in TUNED_HYPER.items():
        if method not in cfg.baselines:
            continue
        grid = list(getattr(cfg, f"{name}_grid"))
        accs = []
        for val in grid:
            A, _ = run_baseline_stream(method, stream, cfg, seed, hyper={name: val})
            accs.append(cl_metrics(A)["ACC"])
        best = int(np.argmax(accs))
        chosen[method] = {name: grid[best]}
        log.append({
            "method": method, "hyper": name, "grid": grid,
            "ACC_per_value": accs, "chosen": grid[best],
            "tuning_seed": int(seed),
        })
    return chosen, log


def baseline_table(rows, cfg):
    """The mandatory table: every method, ACC/BWT/forgetting (mean ± sd over seeds),
    memory in items AND floats, and the class each method belongs to."""
    table = []
    for name in sorted({r["method"] for r in rows}):
        sel = [r for r in rows if r["method"] == name]
        acc = _mean_std([r["metrics"]["ACC"] for r in sel])
        bwt = _mean_std([r["metrics"]["BWT"] for r in sel])
        fgt = _mean_std([r["metrics"]["forgetting"] for r in sel])
        cls = sel[0].get("class", "")
        table.append({
            "method": name, "class": cls,
            "ACC": acc[0], "ACC_sd": acc[1],
            "BWT": bwt[0], "BWT_sd": bwt[1],
            "forgetting": fgt[0], "forgetting_sd": fgt[1],
            "memory_items": sel[0].get("memory_items", 0),
            "memory_floats": sel[0].get("memory_floats", 0),
            "n_seeds": len(sel),
        })
    return sorted(table, key=lambda r: -r["ACC"])


def entry_verdict(table, cfg):
    """The three sentences the entry is allowed to say, computed from the table."""
    by = {r["method"]: r for r in table}
    clu = by.get("clu_entry_task1_only")
    if clu is None:
        return {"verdict": "no CLU entry row"}
    free = [r for r in table if r["class"] == "rehearsal-free" and r["method"] != "clu_entry_task1_only"]
    replay = [r for r in table if r["class"] == "replay"]
    upper = [r for r in table if r["class"] == "upper-bound"]
    launder = [r for r in table if r["class"] == "launder"]
    best_free = max(free, key=lambda r: r["ACC"]) if free else None
    best_replay = max(replay, key=lambda r: r["ACC"]) if replay else None
    best_launder = max(launder, key=lambda r: r["ACC"]) if launder else None
    return {
        "clu_ACC": clu["ACC"],
        "best_rehearsal_free_baseline": (best_free["method"] if best_free else None),
        "margin_over_rehearsal_free_class": (
            clu["ACC"] - best_free["ACC"] if best_free else None
        ),
        "wins_rehearsal_free_class": bool(best_free and clu["ACC"] > best_free["ACC"]),
        "best_replay_method": (best_replay["method"] if best_replay else None),
        "deficit_vs_replay": (clu["ACC"] - best_replay["ACC"] if best_replay else None),
        "offline_upper_bound": (upper[0]["ACC"] if upper else None),
        "best_launder": (best_launder["method"] if best_launder else None),
        "clu_minus_launder": (clu["ACC"] - best_launder["ACC"] if best_launder else None),
        "laundered": bool(
            best_launder and (clu["ACC"] - best_launder["ACC"]) <= cfg.laundering_tie_band
        ),
        "filing_rule": (
            "CM-23(n): winning the rehearsal-free class while sitting below replay IS "
            "a publishable success. 'Beats replay' is NEVER claimed."
        ),
        "laundering_rule": (
            "N89/CM-22(i): if kNN-in-φ at matched memory ties or beats the entry, the "
            "win is φ's and the buffer's, not the store's — and the report says so."
        ),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plots(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    paths = []
    ds = results["config"]["dataset"]

    tab = results.get("baseline_table", [])
    if tab:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        colors = {"rehearsal-free": "tab:blue", "replay": "tab:orange",
                  "launder": "tab:red", "upper-bound": "tab:green",
                  "clu": "black"}
        names = [r["method"] for r in tab]
        accs = [r["ACC"] for r in tab]
        sds = [r["ACC_sd"] for r in tab]
        cols = [colors.get("clu" if "clu_entry" in n else r["class"], "grey")
                for n, r in zip(names, tab, strict=True)]
        ax.bar(range(len(tab)), accs, yerr=sds, color=cols, capsize=3)
        ax.set_xticks(range(len(tab)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Class-IL ACC (end of stream)")
        ax.set_title(f"Split-{ds.upper()} Class-IL — rehearsal-free entry vs the field")
        fig.tight_layout()
        p = os.path.join(save_dir, f"cl_entry_table_{ds}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    for rt in [c for r in results.get("retry_native", []) for c in r.get("cells", [])]:
        if "ladders" not in rt:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for line, dd in rt["ladders"].items():
            ks = sorted(int(k) for k in dd)
            xs = [dd[str(k)][1] for k in ks]
            ys = [dd[str(k)][0] for k in ks]
            axes[0].plot(xs, ys, "o-", label=line)
        axes[0].axhline(rt["knn_phi_floor"], ls=":", c="k", label="kNN-in-φ floor")
        axes[0].set_xlabel("compute multiplier")
        axes[0].set_ylabel("stored-item recall")
        axes[0].set_title(f"retry, crowded store ({rt['label']}, p={rt['mask_p']})")
        axes[0].legend(fontsize=7)
        for a, dd in sorted(rt.get("by_task_age", {}).items()):
            g = dd["gated"]
            ks = sorted(g)
            axes[1].plot([g[k][1] for k in ks], [g[k][0] for k in ks], "o-",
                         label=f"task-age {a} (n={dd['n']})")
        axes[1].set_xlabel("compute multiplier")
        axes[1].set_ylabel("stored-item recall")
        axes[1].set_title("per task-age (gated)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(
            save_dir, f"cl_entry_retry_{ds}_{rt['label']}_p{rt['mask_p']}.png"
        )
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    ret = results.get("retention", {}).get("law", {})
    if ret:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for cohort in ("permanent", "slow", "fast"):
            d = ret.get(cohort, {})
            if not d.get("n_points"):
                continue
            axes[0].plot(d["ticks"], d["measured_amp"], "o-", label=f"{cohort} measured")
            axes[0].plot(d["ticks"], d["predicted_amp_exp_minus_leak_t"], "--",
                         label=f"{cohort} exp(−leak·t)")
            axes[1].plot(d["ticks"], d["retrieval_retention"], "o-", label=cohort)
        axes[0].set_xlabel("stream tick")
        axes[0].set_ylabel("mean well amplitude")
        axes[0].set_title("scheduled per-item retention (amplitude)")
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel("stream tick")
        axes[1].set_ylabel("retrieval retention")
        axes[1].set_title("retrieval consequence")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(save_dir, f"cl_entry_retention_{ds}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_cl_entry(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    items: Optional[list] = None,
    data=None,
):
    config = config or get_default_config()
    cfg = config.experiment_cl_entry
    seeds = [seed] if seed is not None else list(cfg.seeds)
    items = items or list(cfg.items)
    os.makedirs(save_dir, exist_ok=True)

    method_class = {m: "rehearsal-free" for m in REHEARSAL_FREE}
    method_class.update({m: "replay" for m in REPLAY})
    method_class["joint"] = "upper-bound"

    rows, retry_out, retention = [], [], {}
    entry_runs = []
    tuned, tuning_log = {}, []

    for sd in seeds:
        stream = build_cl_stream(cfg, sd, data=data)

        if "entry" in items and cfg.tune_baselines and not tuning_log and any(
            m in cfg.baselines for m in TUNED_HYPER
        ):
            tuned, tuning_log = tune_baselines(cfg, stream, sd)

        if "entry" in items:
            for regime in cfg.phi_regimes:
                res = run_clu_entry(cfg, stream, regime, sd,
                                    collect_store=(regime == PHI_PRIMARY))
                entry_runs.append({k: v for k, v in res.items() if not k.startswith("_")})
                name = f"clu_entry_{regime}"
                rows.append({
                    "method": name, "class": "rehearsal-free", "seed": sd,
                    "metrics": res["metrics_clu"],
                    "memory_items": res["memory_items"],
                    "memory_floats": res["memory_floats"],
                })
                rows.append({
                    "method": f"knn_phi_same_keys_{regime}", "class": "launder",
                    "seed": sd, "metrics": res["metrics_knn_same_keys"],
                    "memory_items": res["memory_items"],
                    "memory_floats": res["memory_floats"],
                })
                rows.append({
                    "method": f"knn_phi_ringbuffer_{regime}", "class": "launder",
                    "seed": sd, "metrics": res["metrics_knn_ringbuffer"],
                    "memory_items": int(cfg.memory_items),
                    "memory_floats": int(cfg.memory_items * cfg.phi_dim),
                })
                if regime == PHI_PRIMARY and "retry" in items:
                    if res.get("_store_mid") is not None:
                        retry_out.append(
                            retry_native(cfg, res["_store_mid"], stream, res["_phi"],
                                         sd, label=f"mid_stream_seed{sd}")
                        )
                    retry_out.append(
                        retry_native(cfg, res["_store"], stream, res["_phi"], sd,
                                     label=f"end_of_stream_seed{sd}")
                    )

            for method in [m for m in cfg.baselines if m and m != "none"]:
                A, diag = run_baseline_stream(
                    method, stream, cfg, sd, hyper=tuned.get(method)
                )
                rows.append({
                    "method": method, "class": method_class.get(method, ""),
                    "seed": sd, "metrics": cl_metrics(A),
                    "memory_items": diag["memory_items"],
                    "memory_floats": diag["memory_floats"],
                    "A": A.tolist(),
                })

        if "retention" in items:
            res_d = run_clu_entry(cfg, stream, PHI_PRIMARY, sd, decay_on=True)
            retention.setdefault("runs", []).append({
                "seed": sd,
                "metrics_clu_with_decay": res_d["metrics_clu"],
                "rows": res_d["retention_rows"],
                "law": retention_law_check(res_d["retention_rows"], cfg),
                "decayed_out": int(res_d["per_task"][-1].get("decayed_out", 0)),
            })
            retention["law"] = retention_law_check(res_d["retention_rows"], cfg)

    table = baseline_table(rows, cfg) if rows else []
    results = {
        "seeds": [int(s) for s in seeds],
        "items": items,
        "protocol": {
            "scenario": "Class-IL (van de Ven & Tolias): new classes per task, task "
                        "identity NOT given at test time",
            "stream": f"Split-{cfg.dataset.upper()}: {cfg.n_tasks} tasks × "
                      f"{cfg.classes_per_task} classes, from scratch",
            "metrics": "GEM formulas (Lopez-Paz & Ranzato 2017): ACC, BWT, forgetting",
            "phi": f"{PHI_PRIMARY} = PRIMARY (frozen after task 1); {PHI_REFERENCE} = "
                   f"declared upper bound that LEAKS future tasks; phi_dim="
                   f"{cfg.phi_dim} (≥16 binding, PREREG_CL_PHI §7)",
            "store": "designed Gaussian wells over φ(x), payload = class label, "
                     "MVC-0 controller (admission + placement + class-balanced "
                     "eviction + scheduled decay). No exemplar is stored.",
            "memory": f"matched at {cfg.memory_items} items for the store, ER, iCaRL, "
                      f"GDumb and the kNN-in-φ launders; float counts reported too",
            "filing_rule": "CM-23(n) — best rehearsal-free is the claim; 'beats "
                           "replay' is never claimed",
            "known_null": "EWC/SI/LwF collapse to ≈chance in Class-IL by construction "
                          "(van de Ven & Tolias 2019); their collapse is NOT a CLU win",
            "r1_naming": f"{R1_NAME}; forbidden: {', '.join(FORBIDDEN_R1_WORDS)}",
        },
        "config": {
            k: getattr(cfg, k)
            for k in (
                "dataset", "n_tasks", "classes_per_task", "seeds", "phi_regimes",
                "phi_arm", "phi_dim", "n_train_per_task", "n_test_per_task",
                "n_fit_pool", "n_fit_region", "memory_items", "clu_s_frac",
                "d_safe_mult", "s_policy", "s_init_items", "store_alpha", "clu_b",
                "clu_alpha", "clu_gamma", "clu_steps", "clu_dt", "clu_tail_frac",
                "clu_kinetic_mode", "rollout_chunk", "baselines", "backbone",
                "mlp_width", "mlp_depth", "baseline_iters", "baseline_batch",
                "baseline_lr", "ewc_lambda", "si_c", "si_xi", "lwf_alpha",
                "lwf_temp", "fisher_samples", "tune_baselines", "retry_ladder",
                "retry_tau", "retry_tau_grid", "retry_boost", "retry_step_frac",
                "retry_mask_p", "retry_mask_levels", "ticks_per_task",
                "leak_slow", "leak_fast", "amp_floor", "permanent_per_task",
                "laundering_tie_band",
            )
        },
        "entry_runs": entry_runs,
        "baseline_tuning": tuning_log,
        "rows": rows,
        "baseline_table": table,
        "verdict": entry_verdict(table, cfg) if table else {},
        "retry_native": retry_out,
        "retention": retention,
    }
    results["figures"] = _plots(results, save_dir)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"exp_cl_entry_{cfg.dataset}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — the same code path, tiny everything."""
    cfg = config.experiment_cl_entry
    cfg.seeds = [0]
    cfg.n_train_per_task = 60
    cfg.n_test_per_task = 40
    cfg.n_fit_pool = 300
    cfg.n_fit_region = 2000
    cfg.memory_items = 24
    cfg.phi_dim = 16
    cfg.clu_steps = 40
    cfg.baseline_iters = 30
    cfg.baselines = ["finetune", "ewc", "si", "lwf", "er", "icarl", "gdumb", "joint"]
    cfg.tune_baselines = False
    cfg.retry_ladder = [0, 1, 2]
    cfg.retry_mask_levels = [0.5]
    cfg.retry_tau_grid = [0.99, 1.0]
    cfg.ticks_per_task = 2
    cfg.permanent_per_task = 4
    cfg.fisher_samples = 40
    cfg.rollout_chunk = 64


def apply_cifar10(config: CHLUConfig) -> None:
    """⚠ The **reduced from-scratch Split-CIFAR-10 protocol** (Item 5).

    CIFAR-10 Class-IL from scratch is normally run with a ResNet-18 for tens of
    epochs per task; that is out of compute budget on this machine and would not
    change the *ordering* of the methods, which is what the entry is measured on.
    This preset states the reduction explicitly instead of hiding it: a small 3-layer
    CNN (``cnn_channels``), 1000 stream items per task and ``baseline_iters`` steps
    per task, identical for **every** method including the CLU entry's own budget. All
    CIFAR numbers must be reported as *this protocol*, never as literature-comparable
    Split-CIFAR-10 numbers.
    """
    cfg = config.experiment_cl_entry
    cfg.dataset = "cifar10"
    cfg.backbone = "cnn"
    cfg.n_train_per_task = 1000
    cfg.n_test_per_task = 500
    cfg.n_fit_region = 10000
    cfg.n_fit_pool = 3000
    cfg.baseline_iters = 150
    cfg.mlp_width = 128
    cfg.fisher_samples = 100
    cfg.tune_baselines = False  # the grid would triple an already reduced budget
    cfg.clu_steps = 150


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="The continual-learning entry: rehearsal-free Class-IL with a "
                    "designed CLU store (+ R3-native retry, + scheduled retention)"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Single seed (overrides cfg.seeds)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], help="Override dataset")
    parser.add_argument("--items", help="Comma-separated: entry,retry,retention")
    parser.add_argument("--baselines", help="Comma-separated baseline override")
    args = parser.parse_args()

    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        paths = pm.get_paths(args.project)
        save_dir, models_dir = str(paths["plots"]), str(paths["models"])
    else:
        config = get_default_config()
        save_dir, models_dir = "results", None
        os.makedirs(save_dir, exist_ok=True)

    if args.quick:
        apply_quick(config)
    if args.dataset:
        config.experiment_cl_entry.dataset = args.dataset
        if args.dataset == "cifar10":
            apply_cifar10(config)
    if args.baselines:
        config.experiment_cl_entry.baselines = args.baselines.split(",")

    res = run_experiment_cl_entry(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed,
        items=(args.items.split(",") if args.items else None),
    )
    print(json.dumps({
        "baseline_table": res["baseline_table"],
        "verdict": res["verdict"],
        "metrics_path": res["metrics_path"],
    }, indent=2, default=float)[:8000])


if __name__ == "__main__":
    main()
