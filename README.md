# Joint control of batch size, learning rate, and momentum

CMU 11-785 / 11-685, Team 26. A reproducible midterm implementation of the
original proposal's priced progress-score controller. This repository contains
experiments; the report is separately maintained in the connected Overleaf
project under `mid-report/main.tex`.

## Setup

Python 3.11+ is recommended (reported execution: Python 3.13.5, Windows, NumPy
2.1.3). No GPU, PyTorch, checkpoint, API key, or paid service is required.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

On macOS/Linux, use `.venv/bin/python`. Activate that environment, or substitute
its interpreter for `python` in the commands below. `requirements.txt` lists
compatible version ranges; `requirements-lock.txt` pins the direct dependencies
used for the saved results. We executed the experiment commands with existing
Anaconda dependencies; the fresh-venv installation command is provided for
setup and was not itself used for the reported measurements.

## Reproduce the evidence

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
python scripts/run.py --suite all
python scripts/figures.py --report-dir "../IDL_Project/mid-report"
python scripts/validate_results.py
```

These commands were actually executed. To run the baseline and proposed neural
methods without synthetic experiments, use `python scripts/run.py --suite neural`.
That command runs validation-only tuning followed by fixed SGDM, cosine SGDM,
adaptive B+LR, joint control, fixed-B ablation, and pilot-32 iteration. For only
the reference/oracle experiment, use `python scripts/run.py --suite synthetic`.
Those suite entry points are components of the executed `--suite all` command.
Use `--output results/reproduction` to preserve the delivered logs during a
fresh run. To plot such a run, pass `--results results/reproduction` to the
figure script. Do not mix raw files from different configurations.

`configs/midterm.json` specifies seeds, budgets, candidate sets, neural model,
data split, development tuning grid, and synthetic noise. `src/joint_control/`
separates model/data, controller, neural experiment, and exact synthetic oracle.
The test set is never used for choosing optimizer settings. Tests exercise
analytic gradient correctness, loss/accuracy, sample-variance scaling, score
protection, stale history, Armijo bracketing, and pilot-inclusive accounting.

## Data and checkpoints

The loader downloads the official Fashion-MNIST gzip IDX files from the
[authors' repository](https://github.com/zalandoresearch/fashion-mnist), validates
their published MD5 checksums, and saves SHA-256 hashes and exact split indices
in `results/raw/data_manifest.json`. The source dataset is MIT licensed; see
its repository for the license and data description. Downloaded files are
excluded from Git; no checkpoint is used or required. Training uses 12,000
stratified images, validation 2,000 disjoint images (split seed 2026), and
the entire official 10,000-image test set only for final evaluation. The other
46,000 official training images are unused. Pixels are scaled by 1/255 with
no augmentation or fitted preprocessing.

## Implemented comparisons

The normalized momentum update is `v = beta*v + (1-beta)*gradient`. This differs
from the unnormalized convention in many libraries; learning rates should not
be transferred between conventions without conversion.

- Fixed and cosine SGDM: B=128, validation-selected LR and momentum.
- Pairwise control: adaptive B+LR with momentum fixed at the selected 0.9.
- Proposed control: candidate B in {32,64,128}, beta in {0,.5,.9,.99}, pilot 64,
  estimated progress score and pilot-objective dyadic Armijo.
- Fixed-B ablation: same joint controller with update B=64.
- Pilot iteration: joint controller with pilot 32 instead of 64.

Each final neural run spends exactly 38,400 gradient examples, including pilots,
with seeds 0, 1, 2. Update batch and pilot batch use independent streams. A tiny
unusable residual receives a logged fixed-SGDM tail update. Four pilot
microbatches estimate trace variance. The dimensionless sample-price heuristic
is documented in `docs/RESEARCH_AUDIT.md` and the report. No stochastic safety
guarantee is asserted. Line-search forward examples are extra work; timings
include them and exclude periodic monitor evaluation. BLAS uses two threads.

Synthetic runs retain the fixed 200-update horizon and a 12,800-update-example
cap; exact analytic gradient/objective information is free oracle access.
This is not a fair wall-clock/sample-efficiency model of a neural network.
The exhaustive 60-state oracle uses a deterministic optimal step for the exact
quadratic expectation, while the score optimizes a smoothness lower model.
Baseline settings are chosen on a separate development seed; candidates whose
development objective exceeds ten times its initial value are rejected before
ranking by final loss. This rejects explosive cosine transients that otherwise
recover late and misleadingly win a final-loss-only search.

## Actual preliminary findings

Mean official-test accuracy +/- sample SD (three seeds):

| Method | Accuracy (%) |
|---|---:|
| Fixed SGDM | 81.71 +/- 0.97 |
| Cosine SGDM | 83.39 +/- 0.17 |
| Adaptive B+LR | 76.18 +/- 0.61 |
| Joint, pilot 64 | 73.02 +/- 1.53 |
| Joint, fixed B | 73.84 +/- 1.24 |
| Joint, pilot 32 | 61.11 +/- 13.60 |

The current practical controller **does not improve** on the tuned baseline.
The smaller pilot is unstable across seeds. Exact quadratic score/oracle
agreement is 57/60 (95%), mean relative regret 0.312%, and the 720 Monte Carlo
candidate checks have no certificate violation at the stated 3-SE diagnostic.
Strong idealized results do not establish practical gains.

## Artifacts and traceability

- `results/raw/neural_*.json`: every step, curve, search status, counted examples,
  seed, configuration, validation/test metrics, and timing.
- `results/raw/tuning_*.json`: all 12 neural tuning trials, with test results null.
- `results/raw/synthetic_*.json`, `results/raw/oracle.json`: 30 trajectories,
  exhaustive candidates, and Monte Carlo outcomes.
- `results/aggregated/`: tuning choices and means/SDs derived from raw logs.
- `results/figures/`: vector PDF and PNG plots; `results/tables/neural.tex` is
  generated from the same aggregates. `scripts/figures.py` exports identical
  files and SHA-256 evidence to the Overleaf report.
- `results/source_manifest.json`: SHA-256 of source/config files at run start;
  validation checks these exact bytes. `results/environment.json` records
  the actual Python/NumPy/platform/hardware information and installed packages.
- `docs/EXECUTION_NOTES.md`: actual debugging and revisions.

Random seeds are fixed. On this environment reruns gave identical losses and
accuracies. Different NumPy/BLAS/CPU versions may perturb floating-point sums
and discrete controller choices; timing is necessarily nondeterministic and
includes ordinary host scheduling noise. SDs from three seeds are descriptive,
not confidence intervals. No full sensitivity study is claimed.

## Scope and next work

Only the small CPU MLP and controlled quadratics were run. CIFAR-10/ResNet-18,
more seeds, estimator calibration, independent verification batches, and a
confidence-aware momentum guard remain future experiments. The exact-objective
theory does not automatically cover pilot losses or estimated variance.
See `docs/TEAM.md` for user-confirmed team roles and Codex assistance disclosure.
