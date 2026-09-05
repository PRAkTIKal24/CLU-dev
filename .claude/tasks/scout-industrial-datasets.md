# Task: scout-industrial-datasets — the open-data anchor for the ICLR evaluation

- **Agent:** `web-scout` · **Output:** `.claude/outputs/scout-industrial-datasets.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, `.claude/handover_context.md` (§1 — what the primitive is built for), `.claude/brainstorm_log.md` (P1–P3, Thread 4 — decisions D1/D2), `.claude/research_roadmap.md`.

## Why
Decision D1/D2: performance anchors on **open datasets** (Forgis FactoryNet/FactoryBench is a WIP bonus), **anomaly/fault-first** framing under principle P2 (CLU = general primitive with physical levers; anomaly is the *first frontier*, not the identity). The Head explicitly asked for a deep search for **the datasets that truly test what the CLU is built for** — not just the most-cited ones.

## What the CLU is built for (selection criteria — rank against these)
1. **Physically-generated dynamics** (mechanical/electrical/thermal systems with real conservation/dissipation structure — rotating machinery is ideal: quasi-periodic ≈ our attractor experiments).
2. **Anomalies/faults as energy excursions** — labeled faults whose physical signature is a departure from normal dynamics (not purely statistical/pointwise outliers).
3. **Long-horizon structure** — degradation trends, run-to-failure records (future RUL extension), long sequences where drift-free rollout matters.
4. **Sane scale** — trainable on 2×A100; sequence data, not massive multimodal.
5. **Established baselines** — published SOTA numbers we can compare against without reproducing everything (note the best-known results + their eval protocol).
6. **Clean licensing** for academic publication.

## Sub-tasks
1. **Vet the candidate pool** (and expand it): CWRU / FEMTO-PRONOSTIA / Paderborn bearings; NASA C-MAPSS (+ N-CMAPSS) turbofan; SKAB; SWaT/WADI; NASA MSL/SMAP; SMD (Server Machine); Tennessee Eastman; UCI hydraulic systems; MIMII / DCASE machine-sound anomaly; tool-wear (PHM 2010, NUAA milling); robot-actuator/proprioception datasets (e.g., anomaly datasets from robot arms); anything newer (2024–2026) purpose-built for industrial anomaly/PHM benchmarking (e.g., successors to the above; check TSB-AD/UCR-style curated suites for the *time-series anomaly* side and their known criticisms).
2. **For each serious candidate:** modality & channels, sampling rate, size, label quality (known label-quality controversies! e.g., criticisms of SMAP/MSL/SWaT labels in the anomaly-benchmark literature — surface these), physical system description, train/test protocol conventions, 3–5 strongest published baselines with numbers, license, download path.
3. **Rank top 3 for the anomaly-first frontier** against criteria 1–6, with a one-paragraph justification each: which would a physics-prior model *most plausibly shine on and why*. Separately note the best **RUL extension** choice (for the later ICLR chapter) and the best **"honest negative control"** (a dataset where CLU's priors should NOT help — per P1's falsifiability discipline).
4. **Benchmark-hygiene scan:** the time-series-anomaly-detection literature has well-known evaluation controversies (point-adjust inflation, flawed benchmarks). Summarize the current (2025–2026) consensus on *credible* evaluation protocol/metrics so our results are review-proof (this feeds the F2 eval harness).

## Output format
(1) Ranked shortlist table (dataset | system | why-CLU-fit | scale | baselines+numbers | license | red flags); (2) top-3 justifications + RUL pick + negative-control pick; (3) evaluation-protocol recommendations (metrics to use/avoid, with citations); (4) bibtex; (5) `## Proposed handover updates`.
