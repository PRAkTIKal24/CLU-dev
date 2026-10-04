"""⭐ **The C3 Gated DeltaNet-2 arm's SMOKE LEG** (`c3-rival-arms` §B, task §1.6).

############################################################################
⛔⛔ **THIS IS NEVER A CLAIM VENUE.** ⛔⛔

The default shape below (``d_model 64``, 2 layers, a handful of optimisation
steps, ~0.6 MB of enwik8) exists to make every **seam** execute in minutes, not
to make any number mean anything. **No bpc produced by this module is quotable**,
in either direction, including the case where GDN-2 beats anything. Charter §2:
claims live at 26-47 M with >=3 seeds and the tier's own control.
############################################################################

What it proves, in order (task §1.6): the arm **trains -> checkpoints ->
RESUMES -> ledgers -> emits retention slices** on the **real** stream.

⛔ **Zero ladder arms are trained here** (task §1.7). The ladder configs
(:func:`~chlu.eval.rivals.gdn2_lm.gdn2_published_config` /
:func:`~chlu.eval.rivals.gdn2_lm.gdn2_shrunk_config`) are **ledgered, never
built**: the ledger is arithmetic on a config and needs no A100-second.

⭐ The loss, the per-token NLL and the eval iterator are the pilot's own
(:func:`chlu.training.train_cluformer.loss_fn` /
:func:`~chlu.training.train_cluformer.eval_token_nll`,
:func:`chlu.data.enwik8.contiguous_batches`) — reused **unchanged**, which is
itself the demonstration that the arm is drop-in for the ladder.

Usage::

    python -m chlu.experiments.exp_c3_rival_gdn2 --out DIR [--resume] [--slices]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.data.corpora import get_corpus, load_corpus
from chlu.data.enwik8 import bits_per_character, contiguous_batches, random_batches
from chlu.eval.byte_ledger import MATCHED_STATE_BYTE_BUDGET
from chlu.eval.rivals.gdn2_lm import (
    GDN2_ARM,
    GDN2Config,
    build_gdn2_arm,
    fla_trap_check,
    gdn2_ledger_row,
    gdn2_param_class_table,
    gdn2_published_config,
    gdn2_shrunk_config,
)
from chlu.eval.text_slices import (
    DEFAULT_BIN_EDGES,
    build_revisit_index,
    contiguous_target_positions,
    run_controls,
    slice_bpc,
)
from chlu.training.train_cluformer import loss_fn
from chlu.utils.checkpoints import load_model, save_model

SMOKE_BANNER = (
    "############################################################\n"
    "⛔ SMOKE CONFIG — NEVER A CLAIM VENUE. No bpc below is quotable.\n"
    "############################################################"
)


# ==========================================================================
# config
# ==========================================================================
def default_smoke_config() -> GDN2Config:
    """A **declared toy geometry**. ⛔ Not the ladder's, and never reported as it."""
    return GDN2Config(n_layers=2, d_model=64, n_heads=2, vocab_size=256)


@eqx.filter_jit
def _train_step(model, opt_state, tokens, targets, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tokens, targets, [])
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


@eqx.filter_jit
def _eval_token_nll(model, tokens, targets):
    logits = jax.vmap(lambda t: model(t, [], None, None))(tokens)
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(lp, targets[..., None].astype(jnp.int32),
                                axis=-1)[..., 0]


# ==========================================================================
# the legs
# ==========================================================================
def train(model, split, *, steps: int, batch: int, seq_len: int, lr: float,
          seed: int, start_step: int = 0, opt_state=None,
          optimizer=None) -> Dict[str, Any]:
    """Train ``steps - start_step`` steps on the REAL stream, resumably.

    ⭐ The data order is a pure function of ``seed`` (``random_batches`` uses a
    seeded ``np.random.Generator``), so resuming = re-creating the generator and
    skipping ``start_step`` batches. That is what makes the resume bit-identical
    rather than merely close.
    """
    optimizer = optimizer or optax.adam(lr)
    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    losses: List[float] = []
    it = random_batches(split, batch=batch, seq_len=seq_len, n_batches=steps,
                        seed=seed)
    for i, (x, y) in enumerate(it):
        if i < start_step:
            continue
        tk = jnp.asarray(x, dtype=jnp.int32)
        tg = jnp.asarray(y, dtype=jnp.int32)
        model, opt_state, loss = _train_step(model, opt_state, tk, tg, optimizer)
        losses.append(float(loss))
        print(f"  step {i + 1}/{steps}  loss {float(loss):.5f} nats  "
              f"({bits_per_character(float(loss)):.5f} bpc)", flush=True)
    return {"model": model, "opt_state": opt_state, "optimizer": optimizer,
            "losses": losses, "step": steps}


def evaluate(model, split, *, batch: int, seq_len: int, n_batches: int
             ) -> Dict[str, Any]:
    tot, n = 0.0, 0
    for x, y in contiguous_batches(split, batch=batch, seq_len=seq_len,
                                   n_batches=n_batches):
        v = _eval_token_nll(model, jnp.asarray(x, jnp.int32),
                            jnp.asarray(y, jnp.int32))
        tot += float(jnp.sum(v))
        n += int(v.size)
    mean = tot / max(n, 1)
    return {"mean_nll_nats": mean, "bpc": bits_per_character(mean),
            "n_tokens": n,
            "NOT_A_CLAIM": "smoke config; see the module banner"}


def slices(model, split, *, corpus: str, batch: int, seq_len: int,
           n_batches: int, min_n: int = 5,
           edges=DEFAULT_BIN_EDGES) -> Dict[str, Any]:
    """The within-document retention slice, on the arm, with the controls.

    ⭐ Reuses :mod:`chlu.eval.text_slices`' own index, position arithmetic,
    binning and validity controls verbatim; only the per-token NLL is supplied
    by this arm (``chlu.eval.text_slices.evaluate_slices`` hard-wires the CLU's
    ``plan_pass``, which a standalone rival has no analogue of).
    """
    doc = get_corpus(corpus).doc_boundary
    idx = build_revisit_index(split.data, doc_boundary=doc, unit="token",
                              edges=edges)
    pos = contiguous_target_positions(len(split), batch=batch, seq_len=seq_len,
                                      n_batches=n_batches)
    nlls, used = [], []
    for (x, y), p in zip(contiguous_batches(split, batch=batch, seq_len=seq_len,
                                            n_batches=n_batches), pos,
                         strict=False):
        nlls.append(np.asarray(_eval_token_nll(model, jnp.asarray(x, jnp.int32),
                                               jnp.asarray(y, jnp.int32))))
        used.append(p)
    out = slice_bpc(nlls, used, idx, min_n=min_n)
    out["corpus"] = corpus
    out["split"] = split.name
    out["controls"] = run_controls(split.data, doc_boundary=doc, edges=edges,
                                   sample_bytes=min(len(split), 200_000))
    return out


def ladder_ledger(budget: int = MATCHED_STATE_BYTE_BUDGET) -> Dict[str, Any]:
    """⭐ The rows the ladder consumes. ⛔ **Ledgered from config; NOT trained.**"""
    pub = gdn2_published_config()
    shrunk_cfg, solved = gdn2_shrunk_config(budget)
    return {
        "budget_bytes": int(budget),
        "published": gdn2_ledger_row(pub, budget=budget,
                                     label="gdn2_published"),
        "shrunk": gdn2_ledger_row(shrunk_cfg, budget=budget,
                                  label="gdn2_shrunk"),
        "shrink_solution": solved,
        "fla_trap_check": fla_trap_check(),
        "param_class": gdn2_param_class_table(),
        "sanity_anchor": GDN2_ARM["sanity_anchor"],
        "trained_by_this_spoke": GDN2_ARM["trained_by_this_spoke"],
    }


# ==========================================================================
# the driver
# ==========================================================================
def _ckpt_paths(out: Path):
    return out / "ckpt" / "gdn2_model.pkl", out / "ckpt" / "gdn2_opt.pkl", \
        out / "ckpt" / "gdn2_journal.json"


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".claude/outputs/c3-rival-gdn2/smoke")
    ap.add_argument("--corpus", default="enwik8")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-bytes", type=int, default=600_000)
    ap.add_argument("--eval-batches", type=int, default=2)
    ap.add_argument("--slice-batches", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--slices", action="store_true")
    a = ap.parse_args(argv)

    print(SMOKE_BANNER, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mp, op, jp = _ckpt_paths(out)
    t0 = time.time()

    train_split, valid_split, _ = load_corpus(
        a.corpus, download=not a.no_download, n_bytes=a.data_bytes)
    print(f"[data] {a.corpus}: train {len(train_split):,} B / "
          f"valid {len(valid_split):,} B", flush=True)

    cfg = GDN2Config(n_layers=a.layers, d_model=a.d_model, n_heads=a.heads,
                     vocab_size=get_corpus(a.corpus).vocab_size)
    start = 0
    opt_state = None
    optimizer = optax.adam(a.lr)
    if a.resume and mp.exists() and jp.exists():
        journal = json.loads(jp.read_text())
        if journal["config"] != cfg.as_flag_table():
            raise ValueError(
                "⛔ resume refused: the banked config differs from this one.\n"
                f"  banked: {journal['config']}\n  now:    {cfg.as_flag_table()}")
        model = load_model(mp)
        opt_state = load_model(op)
        start = int(journal["step"])
        print(f"[resume] lifted step {start} from {mp}", flush=True)
    else:
        model = build_gdn2_arm(cfg, key=jax.random.PRNGKey(a.seed),
                               check_ledger=False)

    res = train(model, train_split, steps=a.steps, batch=a.batch,
                seq_len=a.seq_len, lr=a.lr, seed=a.seed, start_step=start,
                opt_state=opt_state, optimizer=optimizer)
    model = res["model"]
    mp.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, mp, metadata={"step": res["step"]})
    save_model(res["opt_state"], op)
    jp.write_text(json.dumps({"step": res["step"], "seed": a.seed,
                              "config": cfg.as_flag_table()}, indent=2))
    print(f"[ckpt] banked step {res['step']} -> {mp}", flush=True)

    ev = evaluate(model, valid_split, batch=a.batch, seq_len=a.seq_len,
                  n_batches=a.eval_batches)
    print(f"[eval] valid {ev['bpc']:.5f} bpc over {ev['n_tokens']:,} tokens "
          f"⛔ NOT A CLAIM", flush=True)

    art: Dict[str, Any] = {
        "banner": SMOKE_BANNER,
        "arm": "gdn2",
        "arxiv": GDN2_ARM["arxiv"],
        "citation": GDN2_ARM["citation"],
        "corpus": a.corpus,
        "cli": vars(a),
        "smoke_config": cfg.as_flag_table(),
        "smoke_ledger": gdn2_ledger_row(cfg, label="gdn2_smoke"),
        "train_losses_nats": res["losses"],
        "resumed_from_step": start,
        "eval": ev,
        "ladder_ledger": ladder_ledger(),
        "wall_s": round(time.time() - t0, 2),
    }
    if a.slices:
        art["slices"] = slices(model, valid_split, corpus=a.corpus,
                               batch=a.batch, seq_len=a.seq_len,
                               n_batches=a.slice_batches)
        nb = art["slices"].get("bins", [])
        print(f"[slices] {len(nb)} bins scored ⛔ NOT A CLAIM", flush=True)

    p = out / "gdn2_smoke.json"
    p.write_text(json.dumps(art, indent=2, default=float))
    print(f"[artifact] {p}  ({art['wall_s']} s)", flush=True)
    return art


if __name__ == "__main__":       # pragma: no cover
    main()
