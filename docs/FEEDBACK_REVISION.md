# Feedback revision, 2026-10-23

## Independent local repositories

`IDL_Project/Experiment` is the GitHub working tree; `IDL_Project/Overleaf` is
the Overleaf working tree. The workspace parent is not itself a repository.
Original histories, datasets, caches, and local files were moved intact. No
submodule or shared Git directory is used. Each repository has only its proper
origin. Export from Experiment to `../Overleaf/mid-report`.

## 1. Residual learning-rate artifact

The original fallback formed a fixed-SGDM direction and reset eta to 0.3 even
after using much smaller Armijo rates. The revised fallback uses
`min(fixed_lr, last_positive_accepted_armijo_lr)`; if no rate was accepted, it
uses zero. A non-descent skip does not erase the last accepted rate. This is
a rate cap, not a proof that any new residual direction will decrease loss.

`scripts/feedback.py` crosses strict/tolerant ties with legacy/capped fallback
for all three Pilot-32 seeds. It executes nine extra runs and reuses the three
revised final runs. Each policy pair verifies identical pre-tail parameter and
direction hashes (or identical final parameters if there is no tail).

Final **validation** accuracy (%), not test accuracy:

| Tie rule / residual | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Strict / reset to 0.3 | 53.80 | 76.80 | 54.70 |
| Strict / capped | 78.05 | 76.80 | 74.35 |
| Tolerant / reset to 0.3 | 53.80 | 31.20 | 54.70 |
| Tolerant / capped | 78.05 | 77.50 | 74.35 |

For tolerant seeds 0/1, pre-tail validation is respectively 76.75/77.65%.
Only replacing the last rate 0.3 with 0.015625/0.0078125 gives 78.05/77.50%.
These matched-prefix interventions identify the immediate cause of these
collapses. They do not imply that all seed variability is explained. Seed 2
still drops from 75.60% before its capped step to 74.35% afterward. The earlier
report's pilot-size-only interpretation of the large spread is withdrawn.

Evidence: `results/feedback/raw/`, `summary.json`, and `residual_ablation.pdf`.
Main results use tolerant ties and capped residuals for every adaptive method;
baselines are retested with unchanged validation-selected settings.

## 2. Pilot judgments versus the retained-training objective

For revised Pilot-64 runs, predetermined steps 0,20,40,... plus any final
controller step exhausting the budget are audited. There are 48 total audit
steps. The observer computes the full loss/gradient on all **12,000 retained
training examples**, not the unused 46,000 or a population objective. It uses
neither validation nor test gradients and never influences decisions.

- 40 sampled updates were accepted; 6 increase full retained-training loss.
- 8 were skipped by pilot alignment; 7 have positive full-training gradient
  alignment with the proposed direction.
- Loss-increase tolerance: `1e-12*(1+abs(loss_before))`.
- Alignment tolerance: `1e-12*max(1,norm(full_gradient)*norm(direction))`.

These are sampled phenomena, not a representative failure-rate estimate or
an explanation of the entire final-accuracy gap. The observer costs 576,000
gradient and 480,000 forward-only example evaluations. These are logged as
diagnostic overhead and excluded from optimizer budgets and reported training
times. Noninterference is checked using identical final parameter hashes with
the observer enabled/disabled. Raw before/after losses and alignments are in
`results/feedback/pilot_audit.json` and each final joint run's `audits` field.

## 3. Actual controller selections

Across all 894 revised Pilot-64 decisions (no residual steps):

| Control | Value | Count | Percentage |
|---|---:|---:|---:|
| Batch | 32 | 34 | 3.80 |
| Batch | 64 | 831 | 92.95 |
| Batch | 128 | 29 | 3.24 |
| Momentum | 0 | 779 | 87.14 |
| Momentum | 0.5 | 114 | 12.75 |
| Momentum | 0.9 | 1 | 0.11 |
| Momentum | 0.99 | 0 | 0.00 |

`controller_diagnostics.pdf` shows frequencies and per-seed mean learning rates
in 20 equal gradient-budget bins, including zero/skipped rates. Raw decisions
and four budget-phase summaries remain available. The published reference
commit had 92.95% batch 64 and 86.58% zero momentum; revised percentages are
recomputed rather than copied from old logs.

## 4. Floating-point ties

Let U be score minus sample price. Compare every candidate with the global
maximum, using tolerance `1e-15 + 1e-12*max(abs(best),abs(candidate))`, then
prefer smaller B, then smaller beta. This avoids nontransitive chained ties.
With zero stored velocity, all beta candidates at fixed B have identical
mathematical score; float rounding previously selected beta 0.9 in first
Pilot-64 steps at seeds 0/2. Tolerant selection chooses beta 0 for all three
seeds. Actual first-pilot candidate scores are saved in `summary.json`.

This is approximate maximization within the documented tolerance; exact-score
protection is correspondingly relaxed by that tolerance. It improves this
selection ambiguity, not all sources of cross-platform nondeterminism. Unit
tests cover several gradient scales, permuted candidate order, and genuinely
different candidates. Strict ties remain accessible only for controlled replays
through zero tolerance; reset-to-fixed fallback is similarly explicit legacy
comparison code, not the default.

## Reproduction

Run from Experiment: unit tests, `scripts/run.py --suite all`,
`scripts/feedback.py`, `scripts/figures.py --report-dir ../Overleaf/mid-report`,
and `scripts/validate_results.py`. These commands were executed. There are 18
main neural runs, 30 synthetic trajectories, 12/24 neural/synthetic tuning
trials, and nine additional residual replays. Every numerical claim above is
derived from saved outputs. Original historical results remain in Git history.
