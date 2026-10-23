# Execution record and resolved issues

1. Read both complete source documents, original style, Git remotes, parent
   instructions, sibling directories, and hardware. No existing usable code;
   GitHub was empty. Created an independent sibling GitHub working tree so
   experimental code cannot accidentally enter the Overleaf history.
2. Implemented NumPy analytic MLP/backprop and exact/practical controllers.
   Seven scientific tests passed, including finite differences and budgets.
3. First oracle run exposed NumPy int64 JSON serialization. Corrected conversion
   of NumPy scalars in the JSON writer; nonfinite results still fail strictly.
4. Initial oracle used absolute noise variances 1/100 with randomly scaled
   gradients. This was too easy (100% ranking agreement). Before analysis,
   specified relative-noise coverage and perturbed correlated histories; the
   final 60-state protocol is explicit in source, configuration, and logs.
   Revised agreement is 95%, and this is the result reported.
5. Initial synthetic development selection minimized only final loss and chose
   a cosine run with an enormous transient objective (~1e35) before recovery.
   Added a training-only development stability screen (max <= 10*initial),
   retained every tuning trial and the rejection flag, reran all final runs,
   and regenerated figures. No final evaluation seed was used for that screen.
6. Initial neural control underperformed SGDM and showed large Pilot-32 spread.
   The 2026-10-23 feedback revision identifies the residual-rate artifact behind
   specific collapses and replaces the pilot-size-only interpretation. Original
   results remain available at commit 6836c47f346c182fa236f7069e608ffec79a2dde.
7. Original Proposal.tex has a pre-existing natbib author-year/numeric bibitem
   mismatch. Do not modify the original file. New report explicitly sets numeric
   natbib mode, reuses the same article/final NeurIPS style, and compiles.
8. First report compile exposed a nonexpandable IfFileExists filename inside
   input; corrected the path helper to invoke input inside its branches.
9. Rendered PDF pages, inspected figures and mathematical typesetting, then
   increased plot labels to report-scale sizes. Final QA is recorded in the
   Overleaf report's validation files, not inferred from a successful TeX run.
10. Final code audit found that a first-1000 training monitor sampled just the
    first class of the class-ordered split. Replaced it with fixed indices across
    the complete training split and reran the final suite. The monitor does not
    drive selection, and final test/validation results were unchanged.
11. Source manifests now use relative paths for the actually executed modules
    and CLI/config, while the figure script has its own plotting manifest.
    Git attributes preserve exact bytes to keep hashes valid after cloning.

The delivered `results/` contains the final executed protocol. Earlier diagnostic
outputs in `results/oracle_revision/` are excluded from delivery to avoid mixing
protocols. All retained final metrics are produced by the documented run command.

## 2026-10-23 feedback revision

Moved the complete checkout to `IDL_Project/Experiment`, and the independent
Overleaf checkout to `IDL_Project/Overleaf`. Both retained history/data and their
respective origin. Updated export/build documentation for sibling directories.

Implemented a last-accepted-rate cap on residual updates and a scale-aware
near-tie policy with deterministic lexicographic choice. Added tests and ran the
complete suite, then nine extra replays to cross tie rule and tail policy.
First-step beta is now 0 for every final Pilot-64 seed; legacy rounding selected
0.9 at seeds 0/2. Added full-training audits without feeding them into decisions.
During QA, separated audit gradient cost from the optimization-budget position
in log fields, reran the full protocol, and validated 576,000 diagnostic gradient
examples. Twelve tests pass, including audit accounting/noninterference and the
last-accepted-rate behavior after a skipped step. Report red text is replaced
with measured evidence and a qualified interpretation. Full details are in
`FEEDBACK_REVISION.md`.
