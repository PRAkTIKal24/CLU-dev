# Working notes (engineer scratch) — cl-entry-build

## Run inventory
| run | file | code state | notes |
|---|---|---|---|
| MNIST run 1 | `/tmp/mnist_run1.json` | pre-fix (no mid-stream, pixel-TTA ff line, buggy verdict grouping) | superseded; used to find the three fixes in `e7f9e73` |
| CIFAR run 1 | `/tmp/cifar_run1.json` | same as MNIST run 1 + mid-stream | superseded by the final CIFAR run if it completes |
| MNIST final | `results/exp_cl_entry_mnist_metrics.json` | `e7f9e73` (final) | the reported MNIST numbers |
| MNIST replication seeds 3,4,5 | `/tmp/mnist_replic_seed{3,4,5}.json` | `e7f9e73` | PREREG §5 replication of the retry cell |
| CIFAR final | `results/exp_cl_entry_cifar10_metrics.json` | `e7f9e73` | the reported CIFAR numbers |

## Asymmetries to state in the report (a referee will find them)
- The baselines get **parametric learning on the whole stream** (2000 items/task on MNIST,
  1000 on CIFAR) *plus* their 200-item buffer. The CLU entry runs **no gradient descent on
  the stream at all** — it writes ≤200 wells and reads them. The comparison is therefore
  generous to the baselines on information and unfavourable to them on compute.
- The CLU entry's `φ` is fit (unsupervised, reconstruction only) on a **disjoint** pool of
  task-1-class images. The baselines never get a pretraining phase — but they see every
  stream item **with its label**, which is far more information.
- Memory is matched in **items** (200 everywhere). In **floats** the store is 6 400 vs
  156 800 (MNIST) / 614 400 (CIFAR) for a raw-exemplar buffer — 24.5× / 96× cheaper. At
  matched *bytes* the replay baselines would hold 8 (MNIST) / 2 (CIFAR) exemplars. We do
  NOT run that comparison as a headline: matched-items is the generous choice and the
  float column is reported next to it.
- `n_live` saturates at the 200-item budget from task 0 onward, so the entry is always
  **budget-bound, not gate-bound**: the spacing gate refuses 3–50 % of offers per task and
  the class-balanced policy evicts the rest.
