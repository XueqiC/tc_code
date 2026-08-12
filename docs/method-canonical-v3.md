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

## Dictionary-learning deep-dive (for method section, 2026-08-12)
- Objective: min_{D,C} ||X - C D||^2 + lambda ||C||_1 (K=64 atoms, lambda=.05),
  alternating lasso coding / dictionary update (MiniBatch).
- WHY SPARSITY: without L1 this is PCA/SVD — dense, rotation-unidentifiable
  directions with no skill correspondence. L1 kills the rotation freedom:
  the only way to make ALL traces sparse is to align atoms with recurring
  self-contained patterns = skills (same mechanism as NMF parts).
- WHY GRADIENTS: an atom is a recurring direction of parameter change;
  granularity is decided by the data (always-co-occurring skills merge;
  skills appearing in varied mixtures must be isolated for sparsity).
  Identifiability is forced by optimization, not annotation.
- Reading codes: c_a(x) = learning signal trace x supplies to skill a;
  spec mean code = demand distribution; 90% cumulative mass = support.
- Four validations: JOIN interpretability; 0.906 vs 0.615 granularity;
  absorption order ~ spectral order (rho=.73); behavior flip 85.5% mass on
  5/128 atoms.
- One-liner: dictionary learning is a TOKENIZER FOR GRADIENT SPACE — it
  segments continuous learning demand into a reusable, nameable, countable
  skill vocabulary, and the whole pipeline does its bookkeeping in that
  vocabulary.

## Writing positioning notes (PI QA, 2026-08-12 evening)
- JOIN atom = interpretability CASE STUDY on the SQL slice only; never a
  general claim. General claims rest on task-agnostic metrics: granularity
  AUROC (both sql & pandas), flip concentration 5/128, absorption-spectral
  rho=0.73.
- Demand extraction = dictionary TRANSFORM: fixed dictionary, per-query
  lasso for sparse coefficients; |c_a| averaged over k queries.
- p-gate = rank test against the calibration score distribution of genuine
  spec members ("where would this trace rank among real members"); the
  distribution itself is the threshold.
- Diminishing returns in clearing = min(ledger remainder, supply) + deduction;
  no decay constants — supply beyond remaining demand counts zero.
- Warmup: utility side only (Adam moments); boundary works at theta_0
  (AUROC 1.000) and can skip warmup in gate-only deployments.

## Restocking design (PI QA 2026-08-12 late)
- New TRACES: only when the ledger has residual demand after clearing; the
  residual IS the purchase order (targeted teacher re-sampling on the
  spec queries loading on starved atoms). This is a DATA-ACQUISITION-level
  closed loop (static, pre-training quantities) — does not violate the
  separation principle (unlike the falsified weight-level loop).
  Evidence: AppWorld thickening (59->128 eps) was untargeted restocking.
  Planned experiment: targeted vs uniform restocking on AppWorld.
- New QUERIES: never required; optional as (a) elicitation instruments for
  restocking (teacher paraphrases of high-loading spec queries; traces
  still pass admission), (b) tighter certificates (bounds ~ 1/sqrt(n)).
  INVARIANT: demand ledger and boundary derive ONLY from the deployer's
  original k queries — the specification is never silently expanded.

## Positioning: teacher-agnostic core, agent-anchored problem (PI QA 2026-08-12)
The clearing core is deliberately teacher-agnostic (general spec-matched
data selection). The agent setting enters substantively at four points:
(1) pool is GENERATED + execution-verified, enabling restocking (residual
ledger -> targeted teacher sampling) which has no analogue on static
corpora; (2) transfer unit = acting episodes, evaluation behavioral;
(3) out-of-scope capability ACTS (gate/certificates are demanded, not
optional); (4) budget = real teacher-API spend. Paper stance: own both
identities — general mechanism (breadth claim in conclusion), agent
distillation as the anchoring problem where boundary+certificates+
restocking+behavioral verification are simultaneously forced.

## Active distillation loop (PI directive 2026-08-12 evening)
Upgrade from data selection to true distillation: after (and during)
clearing, ASK the teacher. Design (all endogenous):
1. WHAT to ask: residual ledger W_res IS the purchase order (which atom
   accounts unfilled, by how many tokens).
2. HOW to build prompts: no new spec — for each under-filled atom, sample
   loading-weighted from (pool prompts with highest atom loading) +
   (spec queries with heaviest demand on that atom), teacher resamples
   at higher temperature. Supply-side only; k queries stay sole boundary.
3. WHEN (mid-method): each clearing round compare marginal gain of best
   remaining pool trace vs expected gain of a restocked trace (estimated
   from atom composition). Buy-vs-ask is computed, not set.
4. Restocked traces re-enter the pool; loop until ledger cleared or
   budget spent.
Evidence hook: D_atom underperforms exactly at supply-constrained cells
(0.8B x 10k: 0.56 vs 0.77) — the regime where you should ask, not settle
for surplus. Narrative: passive one-shot market -> teacher-in-the-loop
closed market; teacher becomes on-demand supplier, answering the
"teacher seems irrelevant" challenge constructively.
Validation: (a) gsm/sqljoin simulated restocking (hard-clear vs targeted
restock, expect small-budget parity/reversal); (b) AppWorld targeted vs
uniform thickening.

## Query-quality robustness via atoms (PI directive 2026-08-12 eve, #2)
Few-shot query quality strongly determines output quality; goal is to
minimize this dependence USING atoms:
1. Sample-complexity claim: atoms compress boundary estimation from
   ambient-dim mean to ~#active-atom weights -> inherently lower k
   sensitivity (quantifiable/provable).
2. Fragility diagnosis (endogenous): per-atom demand variance across k
   (bootstrap / leave-one-out); single-anchor atoms (supported by 1-2
   queries) are the risk points (consistent with lambda single-anchor
   backfire finding). Ledger becomes confidence intervals, not points.
3. Teacher-probe calibration: for uncertain atoms, synthesize probes
   (loading-weighted pool prompts / query variants), teacher answers,
   re-encode answer fingerprints through dictionary -> lands on same
   atom = confirm; scatters = query noise, down-weight/drop. Teacher as
   falsification oracle; k queries remain sole spec source.
4. Experiment: degrade query quality (halve k, paraphrase noise, 1-2
   off-task contaminants) -> sensitivity curve student-perf vs quality,
   three lines: embedding/LESS (expect steep), atom demand (flatter),
   +probe calibration (flattest). Selling point: spec-quality
   sensitivity becomes a controlled quantity.
Shares teacher-in-the-loop infra with restocking experiments.

## QA: where does "PLOT is missing" knowledge come from (2026-08-12)
Not from the k queries — from the dictionary corpus. After learning
atoms, compute co-occurrence stats across corpus trace codes (e.g.
P(PLOT|GROUPBY)=0.6): world knowledge about how skills travel together,
available before seeing any query. Inference = market-basket completion
(Amazon analogy: this customer's basket vs all customers' baskets).
Two guards: LOO first validates whether co-occurrence inference works
on this task at all (strength -> 0 if not); teacher probes falsify each
hypothesized account. Honest scope: skills that NEVER co-occur with any
observed atom in corpus are information-theoretically unrecoverable
without new spec — state this limitation in the paper.
