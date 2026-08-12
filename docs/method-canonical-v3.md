# SpecDistill canonical formulation v3 (2026-08-12, post PI story session)

## Story (one paragraph)
Deployments want a small agent that does exactly one job, specified by a few
dozen example queries. Existing distillation asks "which data helps the
target" (LESS) and never "where the task's boundary is / what else the
student absorbs" — students with equal in-task scores carry 40-77% out-of-
scope capability, paid for by the deployer's own budget. Task boundary and
capability supply/demand both live in gradient space and can be read from
the k queries directly, then book-kept end to end.

## Method (5 steps, LESS-style)
1. Fingerprints: 5% LoRA warmup w/ checkpoints; one backward pass per query
   and per candidate trace -> Adam-preconditioned gradient fingerprints (JL).
2. Atomization: one sparse dictionary over fingerprints -> capability atoms;
   task DEMAND vector (spec atom mass) + per-trace SUPPLY vector (per token).
3. Boundary admission: conformal p-value gate (p >= alpha/2) learned from
   the query calibration distribution; no manual switches.
4. Market-clearing selection: among admitted traces repeatedly take the one
   satisfying most remaining demand per token, deducting demand on pick;
   budget flows to still-hungry atoms until spent or cleared.
5. Behavioral timing + certificates: SFT with behavioral probe every 8 steps
   (keep best ckpt); post-hoc two-sided behavioral eval -> Clopper-Pearson
   bounds; deployment conformal gate routes with same fingerprints.

## Highlights
- One atom ledger spans boundary/selection/stopping/certificates.
- Constructive realization of "geometry selects, behavior judges".
- Budget-overshoot immunity: surplus budget becomes out-of-scope capability
  for baselines, is harmless for ours.

## Insight-2 pedagogy (atoms)
Learning-demand vectors of traces are combinations of few recurring
directions (recipes from basic operations). Dictionary = the basis set;
sparse = each trace uses few entries; c_a(x) = how much learning signal
trace x supplies to skill a. Validation: JOIN atom interpretability;
fine-grained AUROC 0.906 vs dense 0.615; behavior flip on 5/128 atoms.

## Insight-3 resolution
Three falsifications (gate ledger, closed-loop reselection, geometric stop)
show geometry cannot judge/time behavior. The method complies by
architecture: geometry only selects; all judging/timing is behavioral.
