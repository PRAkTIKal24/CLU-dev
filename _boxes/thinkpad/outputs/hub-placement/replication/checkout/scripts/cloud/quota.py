#!/usr/bin/env python3
"""Quota preflight: decide whether a card can actually be launched, before we try.

The launch script refuses rather than submits when quota is short, and the
refusal names the card and the measured limit. That matters because GCE's own
failure for this is a `ZONE_RESOURCE_POOL_EXHAUSTED`/`QUOTA_EXCEEDED` string
arriving minutes later, from which you cannot tell "we have no quota" apart
from "the zone is momentarily full" -- the first needs a quota request from the
Head, the second needs a retry.

Two metrics are checked per card, never one:

  * the GPU metric        (e.g. PREEMPTIBLE_NVIDIA_A100_GPUS)
  * the machine-family CPU metric (e.g. A2_CPUS)

In us-central1 this project holds 16 preemptible A100 GPUs but 12 A2_CPUS.
One a2-highgpu-1g is 12 vCPU, so the real concurrency is ONE, not sixteen. A
GPU-only check would have reported sixteen and been wrong every time after the
first.

Stdlib only: this runs on the laptop before anything is created, and under
pytest with a recorded fixture instead of a live API call.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS_JSON = os.path.join(HERE, "cards.json")


def load_cards(path: str = CARDS_JSON) -> dict:
    with open(path) as fh:
        return json.load(fh)


def fetch_region(project: str, region: str) -> dict:
    """Live `gcloud compute regions describe` payload."""
    out = subprocess.run(
        [os.environ.get("GCLOUD_BIN", "gcloud"), "compute", "regions", "describe", region,
         f"--project={project}", "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def quota_map(region_payload: dict) -> dict[str, tuple[float, float]]:
    """metric -> (limit, usage)."""
    return {q["metric"]: (float(q["limit"]), float(q.get("usage", 0.0)))
            for q in region_payload.get("quotas", [])}


class QuotaRefusal(RuntimeError):
    pass


def check(card: str, spot: bool, gpus: int, quotas: dict[str, tuple[float, float]],
          cards: dict | None = None) -> list[dict]:
    """Return one row per checked metric. Raises QuotaRefusal if the job cannot fit.

    Raising (rather than returning a bool) is deliberate: a caller that forgets
    to look at the bool still gets stopped.
    """
    cards = cards or load_cards()
    if card not in cards["cards"]:
        raise QuotaRefusal(
            f"unknown card {card!r}. Known: {', '.join(sorted(cards['cards']))}. "
            f"Not available to this project at all: "
            f"{', '.join(f'{k} ({v})' for k, v in cards['unavailable'].items())}"
        )
    spec = cards["cards"][card]
    gpu_metric = spec["gpu_metric_spot"] if spot else spec["gpu_metric"]
    need_cpu = spec["vcpu"] * gpus
    rows: list[dict] = []
    problems: list[str] = []

    # --- GPU metric ---
    limit, usage = quotas.get(gpu_metric, (0.0, 0.0))
    free = limit - usage
    rows.append({"metric": gpu_metric, "need": gpus, "limit": limit, "usage": usage, "free": free,
                 "ok": free >= gpus})
    if limit <= 0:
        problems.append(
            f"card '{card}' ({spec['accelerator']}) has NO QUOTA in this project/region: "
            f"{gpu_metric} limit = {limit:g}. "
            + ("Spot quota is separate from on-demand -- check the other one too. " if spot else "")
            + spec.get("note", "")
        )
    elif free < gpus:
        problems.append(
            f"card '{card}' ({spec['accelerator']}): need {gpus}, but {gpu_metric} "
            f"limit = {limit:g} with {usage:g} IN USE by existing instances => {free:g} free."
        )

    # --- machine-family CPU metric (and the preemptible-CPU special case) ---
    cpu_metric = spec["cpu_metric"]
    if spot and "PREEMPTIBLE_" + cpu_metric in quotas:
        p_limit, _ = quotas["PREEMPTIBLE_" + cpu_metric]
        # GCE: a preemptible-CPU quota of 0 is not a block -- Spot VMs then draw
        # on the regular CPU quota. A non-zero one is the binding limit.
        if p_limit > 0:
            cpu_metric = "PREEMPTIBLE_" + cpu_metric
    limit, usage = quotas.get(cpu_metric, (0.0, 0.0))
    free = limit - usage
    rows.append({"metric": cpu_metric, "need": need_cpu, "limit": limit, "usage": usage, "free": free,
                 "ok": free >= need_cpu})
    if free < need_cpu:
        problems.append(
            f"card '{card}': {gpus}x {spec['machine_type']} needs {need_cpu} vCPU, but "
            f"{cpu_metric} limit = {limit:g} with {usage:g} in use => {free:g} free. "
            f"This caps concurrency at {int(max(0, free) // spec['vcpu'])} such VM(s) "
            f"regardless of GPU quota."
        )

    if problems:
        raise QuotaRefusal("REFUSING TO LAUNCH.\n  - " + "\n  - ".join(problems))
    return rows


def format_rows(rows: list[dict]) -> str:
    head = f"{'metric':38s} {'need':>6s} {'limit':>8s} {'used':>8s} {'free':>8s}  ok"
    body = [f"{r['metric']:38s} {r['need']:>6g} {r['limit']:>8g} {r['usage']:>8g} "
            f"{r['free']:>8g}  {'yes' if r['ok'] else 'NO'}" for r in rows]
    return "\n".join([head, *body])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--project", default=os.environ.get("CHLU_PROJECT", "forgisprova"))
    ap.add_argument("--region", default=os.environ.get("CHLU_REGION", "us-central1"))
    ap.add_argument("--from-json", help="read the region payload from a file instead of the API")
    args = ap.parse_args(argv)

    if args.from_json:
        with open(args.from_json) as fh:
            payload = json.load(fh)
    else:
        payload = fetch_region(args.project, args.region)

    try:
        rows = check(args.card, args.spot, args.gpus, quota_map(payload))
    except QuotaRefusal as exc:
        print(f"[quota] {args.region}: {exc}", file=sys.stderr)
        return 1
    mode = "spot" if args.spot else "on-demand"
    print(f"[quota] {args.region} {args.card} x{args.gpus} ({mode}) OK")
    print(format_rows(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
