"""Figures for the primitive harness (w20).

The headline figure the task asks for: **recall accuracy vs number of
distractors**, one line per primitive, plus the capacity axis (accuracy vs
number of KV pairs) and the compute-cost panel.

Kept out of ``chlu/utils/plotting.py`` deliberately: that module is shared with
concurrent agents, and the harness's figures are self-contained (the
`exp_paid_access` precedent).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "mlp": ("#999999", "o", "MLP (control, no mixing)"),
    "gru": ("#1b9e77", "s", "GRU"),
    "ssm": ("#7570b3", "^", "SSM (S4D/Mamba-lite)"),
    "attention": ("#d95f02", "D", "Attention"),
    "clu": ("#e7298a", "*", "CLU"),
}


def _by(results, prefix, key_fn):
    """Group results whose family name starts with `prefix` by primitive."""
    out = {}
    for r in results:
        if not r["family"].startswith(prefix):
            continue
        k = key_fn(r["family"])
        if k is None:
            continue
        out.setdefault(r["primitive"], []).append((k, r))
    for p in out:
        out[p].sort(key=lambda t: t[0])
    return out


def plot_harness(json_path, out_dir):
    with open(json_path) as f:
        summary = json.load(f)
    results = summary["results"]
    os.makedirs(out_dir, exist_ok=True)

    def seqlen_key(name):
        # mqar_T{T}_kv{kv}: distractor axis = the cells at the fixed kv
        parts = name.split("_")
        T, kv = int(parts[1][1:]), int(parts[2][2:])
        return T if kv == 4 else None

    def kv_key(name):
        parts = name.split("_")
        T, kv = int(parts[1][1:]), int(parts[2][2:])
        return kv if T == 128 else None

    # ---------------- Figure 1: the headline ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    for primitive, rows in _by(results, "mqar", seqlen_key).items():
        color, marker, label = STYLE[primitive]
        xs = [T - 2 * 4 for T, _ in rows]  # distractor slots = T - kv block
        ys = [r["metric_mean"] for _, r in rows]
        es = [r["metric_std"] for _, r in rows]
        ax.errorbar(xs, ys, yerr=es, color=color, marker=marker, label=label,
                    capsize=3, lw=1.8, ms=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of distractor positions  (seq_len $-$ 2$\\cdot$kv)")
    ax.set_ylabel("MQAR recall accuracy")
    ax.set_title("(a) Recall vs distractors\n(kv = 4 pairs, matched params)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    for primitive, rows in _by(results, "mqar", kv_key).items():
        color, marker, label = STYLE[primitive]
        xs = [kv for kv, _ in rows]
        ys = [r["metric_mean"] for _, r in rows]
        es = [r["metric_std"] for _, r in rows]
        ax.errorbar(xs, ys, yerr=es, color=color, marker=marker, label=label,
                    capsize=3, lw=1.8, ms=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of key-value pairs stored")
    ax.set_ylabel("MQAR recall accuracy")
    ax.set_title("(b) Recall vs stored items\n(seq_len = 128, matched params)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "MQAR associative recall — CLU vs standard primitives in one drop-in slot",
        fontsize=12,
    )
    fig.tight_layout()
    p1 = os.path.join(out_dir, "harness_fig1_recall.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # ---------------- Figure 2: per-family + compute cost ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, (fam_prefix, title, ylabel) in zip(
        axes[:2],
        [("adding", "(a) Adding problem (T=128)\nlower is better", "MSE"),
         ("parity", "(b) Parity (T=64)\nhigher is better", "accuracy")],
        strict=False,
    ):
        rows = [r for r in results if r["family"].startswith(fam_prefix)]
        names = [r["primitive"] for r in rows]
        vals = [r["metric_mean"] for r in rows]
        errs = [r["metric_std"] for r in rows]
        colors = [STYLE[n][0] for n in names]
        ax.bar(names, vals, yerr=errs, color=colors, capsize=4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    rows = [r for r in results if r["family"] == "mqar_T128_kv4"]
    names = [r["primitive"] for r in rows]
    wall = [r["wallclock_s_per_step"] * 1e3 for r in rows]
    ax.bar(names, wall, color=[STYLE[n][0] for n in names])
    ax.set_title("(c) Training cost, stated not hidden\n(MQAR T=128, matched params)")
    ax.set_ylabel("wall-clock ms / training step")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    for i, r in enumerate(rows):
        ax.text(i, wall[i], f"{r['fwd_flops'] / 1e6:.1f}\nMFLOP",
                ha="center", va="bottom", fontsize=7)

    fig.suptitle("Per-family results (never averaged) and compute cost", fontsize=12)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "harness_fig2_families_cost.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    return [p1, p2]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot the primitive harness")
    parser.add_argument("json_path")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    for p in plot_harness(args.json_path, args.out_dir):
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
