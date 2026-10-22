# Research audit (before implementation)

Source: complete `Proposal.tex` and `Research Source.tex` from Overleaf commit
757ba1d594bb2a93c62c4b5bde72e08b0fd6f0fe. No implementation was found in the
project or sibling course directories. GitHub was empty on inspection.

## Scientific specification

- Question: can online joint choice of batch size B, normalized momentum beta,
  and learning rate eta improve optimization under a fixed example budget?
- Update: v = beta*v_old + (1-beta)*g_batch; theta -= eta*v.
- Score: m = beta*v_old+(1-beta)*g; A = dot(g,m);
  Q = dot(m,m)+(1-beta)^2*sigma_squared/B; S = max(A,0)^2/Q (zero at Q=0).
- Choice: maximize S-lambda*B over feasible finite candidates, lambda>=0.
- Inputs: current parameters, stored momentum, gradient/variance information,
  remaining budget, finite candidate sets. Outputs: B, beta, eta, new parameters.
- Reference: exact gradient/noise, fresh update noise, exact-objective
  bidirectional Armijo (coefficient 1/2), no runtime curvature estimate.
- Practical approximation: independent pilot gradient/variance, independent fresh
  update batch, pilot-objective Armijo; exact-objective theorems do not transfer.
- Data: noisy quadratic first; Fashion-MNIST development benchmark allowed by
  proposal. CIFAR-10/ResNet-18 remains the intended larger final experiment.
- Losses: positive-definite quadratic and classification cross entropy.
- Metrics: objective/loss versus samples, final held-out classification accuracy,
  elapsed time, sample/forward overhead, score versus realized progress,
  exhaustive one-step oracle agreement and regret.
- Baselines: validation-selected fixed normalized SGDM, cosine SGDM, adaptive
  B+eta with fixed beta; ablation fixes B. Pairwise baseline is our restriction
  of the controller, not a claimed reproduction of CABS or YellowFin.
- Contribution: executable reference/practical controller plus a controlled
  preliminary test of the coupling, not a claim of globally optimal scheduling.

## Choices and corrections

1. CPU only: Intel Core Ultra 5 226V (8 cores), about 16 GB RAM, Intel Arc 130V;
   no NVIDIA/CUDA. Use a NumPy 784-64-10 ReLU MLP (analytic backprop, tested by
   finite differences) instead of installing a large framework. No pretrained
   checkpoint is needed. Original ResNet experiment remains future work.
2. Count all pilot and update gradient examples in the neural budget, including
   repeated examples. Record line-search forward-only examples separately;
   gradient-example equality is not equality of FLOPs or elapsed time.
3. Neural runs use a fixed total budget and variable update horizon; reference
   synthetic trajectories retain the proposal's fixed horizon. A final small
   residual neural batch uses the selected fixed-SGDM rule when a pilot plus
   update cannot fit, and is explicitly logged. No data-dependent test tuning.
4. Pilot variance uses independent equal-sized microbatches: microbatch size
   times the unbiased trace sample covariance of their gradients. It estimates
   per-example variance; it is noisy and is not a certified upper bound.
5. Stochastic Armijo uses the same pilot loss and pilot gradient, separate from
   the update batch. Bidirectional search has finite safety bounds. Caps,
   skips, and forward work are logged, not represented as theoretical guarantees.
6. Price is dimensionless and rescaled each step by ||g_pilot||^2/B_min, with a
   projected dual update around target update batch 64. This documented heuristic
   avoids a fixed absolute price becoming meaningless as the loss scale changes.
7. The supplied YellowFin bibliography disagrees between proposal and research
   source. Verified arXiv:1706.03471 lists Jian Zhang and Ioannis Mitliagkas.
8. Known team roster is preserved. User subsequently confirmed roles: Doyoung
   theory/experiments; Seongbeom experiments; Chaeyeon and Ivor writing/verification.
   These are attributed responsibilities rather than Git-verified authorship.

## Evaluation controls

Fixed stratified train/validation indices, independent official test set,
validation-only baseline selection, identical initialization and seed across
methods, independent pilot/update random streams, all raw logs retained. Neural
results span three seeds; this is preliminary evidence, not a broad benchmark.
Synthetic score ranking is compared against the *exact expected quadratic*
one-step optimum, which is curvature-aware and different from a smoothness lower
certificate. Imperfect agreement is scientifically meaningful, not a code bug.
