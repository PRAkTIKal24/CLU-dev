# Vendored TSB-AD evaluation code (pinned)

**Upstream:** TSB-AD v1.5 — Qinghua Liu & John Paparrizos, *"The Elephant in the
Room: Towards A Reliable Time-Series Anomaly Detection Benchmark"*, NeurIPS 2024
Datasets & Benchmarks. https://github.com/TheDatumOrg/TSB-AD — **Apache-2.0**
(license text retained in `LICENSE` next to this file).

**Exact source:** PyPI sdist
`https://files.pythonhosted.org/packages/51/c8/b1656c4fba492904c79fec0891c952c9a5cbf4bc3f6e13714cf7741c183c/tsb_ad-1.5.tar.gz`
sha256 `52e474cda6aeb3c2f8f6b3a45e58b11b5b7b55a1510bb5c6f6a15b9053f7b0da`,
subtree `TSB_AD/evaluation/`.

## Why vendored instead of `pip install TSB-AD`

The `TSB-AD` distribution pins `numpy<2.0` and requires torch/transformers/
tslearn/stumpy. This repo's uv lock resolves numpy 2.x for the JAX stack; adding
TSB-AD to `pyproject.toml` (even as an optional extra — uv builds one universal
resolution) would force a project-wide numpy downgrade. The evaluation subtree
itself only needs numpy + scikit-learn + stdlib, so we pin and vendor exactly
that subtree. This complies with the F2 binding rule "wrap the TSB-AD harness —
do not reimplement VUS-PR": the VUS/AUC/range-metric code below is TSB-AD's own,
byte-identical except for the modifications listed here.

## Modifications relative to upstream (complete list)

1. `basic_metrics.py`: removed `basic_metricor._adjust_predicts`,
   `basic_metricor.metric_new`, `basic_metricor.metric_PointF1PA`
   (the point-adjust members).
2. `metrics.py`: removed the `PA-F1` computation/entries in `get_metrics` and
   `get_metrics_pred`, and one commented-out reference.
3. A provenance header comment prepended to `metrics.py`, `basic_metrics.py`,
   `__init__.py`.

**Rationale:** point-adjust F1 is forbidden everywhere in the CHLU evaluation
harness (it can make a random score look SOTA — Kim et al., AAAI 2022,
arXiv:2109.05257). It is excised at the source so it cannot be quietly used.
Note: `metric_EventF1PA` is *retained* — despite the upstream name it computes
event-wise recall (one credit per ground-truth event) with point-wise precision,
i.e. the legitimate "Event-based-F1" from the TSB-AD paper, not point-adjust.

The `affiliation/` subpackage (Huet et al., KDD 2022 affiliation metrics, as
redistributed by TSB-AD) is byte-verbatim.

## Regenerating

Do not edit these files by hand. Re-run
`.claude/scratch/f2-eval-harness/vendor_tsb.py` against a checksum-verified
sdist and record any version bump here.
