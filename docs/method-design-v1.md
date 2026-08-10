# Distill-to-Specification: Aligning the Capability Boundary of a Distilled Code Agent to a Few-Shot Task Specification

**Method design document v1 — 2026-08-09.** Target venue: ICLR 2027.
Status: for PI review. Pilot evidence: `results/pilot/`, `results/pilot_distill/`
(all figures regenerable from `src/`).

---

## 1 Problem setting

### 1.1 Setup

A **teacher agent** consists of a large LM `π_T` that acts by emitting executable
code (tool calls, data transformations, control flow) in an environment `E`; a
rollout on query `q` produces a trajectory `τ(q) = (q, a_1, o_1, …, a_H, o_H)`
whose actions `a_h` are code blocks. A **task domain** is a distribution `P_𝒯`
over queries, given to us *only* through `k` i.i.d. few-shot examples
`Q_k = {q_1, …, q_k} ~ P_𝒯`, with `k` ranging from 5 to a few hundred.

Given `(π_T, Q_k)` and a small base LM `θ_0`, we must produce:

1. a **boundary estimator** `B̂: q ↦ {in, out}` for membership in the support
   of `P_𝒯`, with a distribution-free coverage guarantee;
2. a **distilled student** `θ_S` such that the agent built on `θ_S` satisfies
   the *two-sided alignment objective* of §1.2.

### 1.2 Capability sets and the alignment objective

For tolerance `ε ∈ (0,1)`, define the **capability set** of parameters `θ`:

```
C_ε(θ) = { q : P[ success(θ, q) ] ≥ 1 − ε },
```

where `success` is the environment's task-level verifier (exec-based; §4.5).
The deliverable is a student whose capability set matches the task domain:

- **Coverage risk** (under-capability): `R_cov(θ) = P_{q~P_𝒯}[ q ∉ C_ε(θ) ]`
- **Leakage** (over-capability): `L(θ) = P_{q~P_out}[ q ∈ C_ε(θ) ]`
  for a declared out-of-domain reference distribution `P_out`.

We report the `(R_cov, L)` **frontier** across method configurations and
student sizes rather than a single scalar, and additionally the **minimal
capacity** `min{ |θ| : R_cov(θ) ≤ δ }` as a function of `δ`.

### 1.3 Scope and honest boundaries of the claims

- **Capability vs. behavior.** Distillation *adds* competence; it does not
  erase what the base model already knows. Our leakage reductions are
  *behavioral* (the agent no longer performs/attempts out-of-domain requests)
  unless explicitly combined with unlearning, which we treat as an optional
  module, not a claim. The pilot already shows the two dissociate sharply
  (§5, finding 3): action-space adoption saturates in tens of steps while
  competence lags. We measure both channels separately throughout.
- **No per-request certification.** Per-request correctness prediction is
  impossible in general (Barber, 2020-style impossibility); our guarantee is
  *marginal, domain-level* conformal coverage, conditional per declared topic
  (Gibbs et al., 2025), not a certificate for individual outputs.
- **One-sided reporting discipline.** Where a comparison is confounded (e.g.,
  feature dimensionality, trajectory availability), we pre-register which
  direction of the outcome is claimable and hold the other side to a
  matched-control standard (§4.7).

---

## 2 Method

The obstacle to "aligning C to 𝒯" is that 𝒯 lives in query space and C lives
in parameter space. Our central move is to embed both in the **same** space:
the tangent (gradient) space of the student base model.

### 2.1 Trajectory gradient features

For query `q` with teacher trajectory `τ(q)`, define the feature

```
g(q) = normalize( P · ∇_θ ℓ(τ(q); θ_ref) ),
```

where `ℓ` is the token-level NLL of the trajectory's action tokens under the
reference model `θ_ref` (default: the student base `θ_0`), and `P` is a fixed
random projection. Implementation: attach zero-initialized LoRA adapters; at
init `B = 0`, so `∂ℓ/∂B = G Aᵀ` is already a random sketch of the full
per-module gradient `G` (the LESS observation); we further JL-project each
module's sketch to 64 dims with per-module fixed seeds and concatenate
(≈12.5k dims for a 1.5B model), then L2-normalize.

Interpretation: `g(q)` is the *direction the student would have to move to
learn this trajectory*. Two queries belong to the same task exactly when
learning them requires correlated parameter movement — a definition of "task"
that references the learner, not the surface text.

### 2.2 Task specification: spectral subspace + conformal radius

From `Q_k`, split `k = n_fit + n_cal`. Compute features of the fit split, take
the SVD, and keep the top-`r` right singular vectors `U_𝒯` covering a fixed
energy fraction (90%). The **membership score** of a query is the projected
energy ratio

```
s(q) = ‖U_𝒯ᵀ g(q)‖ / ‖g(q)‖ ∈ [0,1].
```

Calibrate a threshold `t_α` as the `⌊α(n_cal+1)⌋`-th order statistic of the
calibration scores. Split-conformal validity gives, for exchangeable data,

```
P_{q~P_𝒯}[ s(q) ≥ t_α ] ≥ 1 − α        (finite-sample, distribution-free).
```

`B̂(q) = 1[s(q) ≥ t_α]` is the boundary estimator. Note the guarantee is on
*coverage of true in-domain queries*; out-of-domain rejection is an empirical
quantity we measure, not a theorem.

**Hierarchical refinement ("stacking").** The flat subspace has three natural
refinements, evaluated as ablations: (i) *per-layer* subspaces `U_𝒯^(l)`
(which depths carry task identity?); (ii) *per-step* subspaces from
trajectory segments (skill composition; replaces the hand-designed structural
coordinates of the earlier draft); (iii) *multi-cluster* unions for
multi-modal domains (cluster features, one subspace per cluster,
`s = max_j s_j`).

### 2.3 Reading the student's capability in the same space

For the current student `θ_S`, the **residual gradient**
`ρ(q) = ‖∇_{θ_S} ℓ(τ(q); θ_S)‖` measures how far `θ_S` is from having
absorbed `τ(q)`: small residual + verified execution ⇒ learned. We use:

- `ρ(q)` distributions over in-/out-domain probes as the *geometric* read of C;
- exec-verified success as the *behavioral* read of C;
- their agreement (E2, §4.3) as a validity check that the geometry tracks the
  behavior — a prerequisite for using geometric C in the training loop.

Alignment between 𝒯 and C is then measurable as: coverage gap
`E_{q∈𝒯}[ρ(q)]`, leakage drop `Δρ` out-of-domain, and principal angles
between `U_𝒯` and the top subspace of realized parameter updates.

### 2.4 Boundary-guided distillation

Given a heterogeneous pool of teacher trajectories `D` (self-generated from
`Q_k` seeds plus distractors — the realistic case where the teacher's rollout
pool is broader than the target domain):

1. **Selection (primary).** Keep/weight `x ∈ D` by `s(x)`; train on the
   selected set under a matched token budget. This is LESS-style data
   selection with the target changed from "helps a validation set" to "lies
   in the task subspace".
2. **Projected updates (secondary, ablation).** During SFT, project each
   optimizer step onto `U_𝒯` (or penalize the orthogonal component,
   `λ‖(I − U_𝒯U_𝒯ᵀ) Δθ‖²`). Tests whether *update-space* constraint tightens
   leakage beyond *data-space* selection without hurting coverage.
3. **Boundary-conditioned refusal (optional module).** Augment with refusals
   on `B̂(q)=out` queries — the behavioral half of shaping, reported
   separately from capability metrics per §1.3.
4. **Capacity sweep.** Repeat across student sizes to trace minimal-capacity
   curves; the pilot's "sharpening" effect (§5, finding 2) predicts smaller
   students shape faster.

---

## 3 Theoretical grounding (what we can and cannot say)

- **Why gradients should encode task identity.** In the lazy/NTK regime, SFT
  moves outputs along `K(·, x)` for trained points `x`; the span of trajectory
  gradients is exactly the span of realizable functional updates at first
  order. Hence a subspace capturing `{g(q_i)}` captures *what fine-tuning on
  the domain can change* — the right first-order notion of a task for a given
  learner. We state this as motivation with the standard caveat that 33-step
  LoRA SFT is near-lazy but full training is not; E2 tests the assumption
  empirically rather than assuming it.
- **What is guaranteed.** Only the conformal coverage statement of §2.2, plus
  its topic-conditional variant when multiple declared topics are calibrated
  separately. Everything else (rejection power, leakage, capacity curves) is
  empirical.
- **Failure mode we watch for.** Gradient features can be dominated by
  low-level statistics (length, format). Pilot controls already rule out the
  two most obvious confounds (dimensionality, template tokens; §5); the full
  protocol adds length-stratified analysis and loss-position ablations.

---

## 4 Pre-registered experimental protocol

### 4.1 Environments and domains

Scale beyond the pilot's four static domains to agentic settings where actions
are code and success is executable:

- **Tier 1 (static trajectories, cheap, rai):** pilot domains (GSM8K-code /
  GSM8K-CoT hard pair, DS-1000-pandas, Spider-SQL, Alpaca pool) + *intra-domain
  splits* (e.g., pandas-filtering vs pandas-groupby; SQL single-table vs join)
  to probe boundary resolution.
- **Tier 2 (interactive code agents, hpg):** CodeAct-style environments —
  proposed panels: BFCL-v3 (tool composition), StableToolBench subset (API
  closure), and a data-analysis agent suite; teacher = frontier API model
  producing multi-step rollouts. Final panel choice confirmed with PI before
  GPU spend.

### 4.2 E1 — Boundary quality (extends pilot)

Feature spaces: gradient (ours), BGE embedding (query / query+trajectory),
last-hidden-state, perplexity-based OOD, activation-probe (SCOPE-style).
Same subspace+conformal pipeline for all. Metrics: AUROC, conformal coverage
and false-accept at `α ∈ {0.05, 0.1}`, `k ∈ {5,10,25,50,100}` × 5 seeds.
Hard-pair axes: (a) same surface / different capability (code vs CoT — pilot:
grad 1.000 vs emb 0.687); (b) different surface / same capability
(paraphrase + tool-renaming probes); (c) intra-domain resolution.
Robustness: `θ_ref ∈` {student base, other 1–2B base, 7B base}; feature
stability under teacher re-rolls (trajectory sampling noise).

### 4.3 E2 — Does geometric C track behavioral C?

Across SFT checkpoints (and across student sizes): correlate `ρ(q)` with
exec success per query. Pre-registered success criterion: rank correlation
ρ_spearman ≤ −0.5 in-domain (lower residual ⇔ higher success). This gates the
use of geometric C anywhere in the training loop.

### 4.4 E3 — Main comparison: boundary-guided vs standard distillation

Pool `D`: teacher rollouts seeded from `Q_k` plus near-domain and off-domain
distractors (composition to be logged exactly). Conditions, all at **matched
token budget** and identical hyperparameters:

| id | condition |
|----|-----------|
| A | all data (standard distillation) |
| B | random subset |
| C | embedding-filtered subset |
| D | `U_𝒯`-filtered subset (ours) |
| E | D + projected updates (ours, full) |
| F | D + refusal augmentation (behavioral module) |

Outcomes: `(R_cov, L)` frontier, per-task success patterns (IRT-matched
comparison as in the earlier draft's protocol), minimal-capacity curve over
student sizes {0.6B, 1.7B, 4B} (Qwen3 line; final selection with PI).
Claim structure: D ≥ A/B/C on the frontier is the primary hypothesis;
E > D and F's behavioral gains are secondary.

### 4.5 Verifiers

Success = exec-based checks per environment (unit tests / answer match /
API-call validation). All generated code runs in a sandboxed subprocess with
resource limits; no network. Verifier code is fixed before E3 runs.

### 4.6 Statistics

Per-task paired comparisons across seeds (≥3 training seeds for E3);
cluster-robust CIs at the task level; all hypothesis directions and
thresholds in this section are frozen before hpg-scale runs; deviations get
logged in the repo as amendments.

### 4.7 Pre-registered confound controls

Dimensionality matching (done in pilot, kept), template stripping (done,
kept), response-length stratification, trajectory-availability parity between
gradient and embedding baselines (both get query+trajectory), stable-hash
splits (pilot's salted-hash bug is fixed in the framework), and a
format-permutation null for any spectral claim (inherited from the earlier
draft's instrumentation).

### 4.8 Compute plan

- rai (1 GPU): E1 complete, E2 at 1.5B, all feature extraction.
- hpg B200: E3/E4 sweeps — ≤4 concurrent jobs, one B200 per 1.7B-and-below
  student, 2–4 per 4B; budget estimate before submission, logged per run.

---

## 5 Pilot evidence (2026-08-09, all single-GPU, full details in repo)

1. **Boundary (T side).** Hard pair (identical questions, code vs CoT
   solutions): grad AUROC **1.000**, emb-traj 0.687, emb-query 0.404
   (≈ chance by construction); cross-domain grad AUROC 1.0, conformal
   coverage 93–97% at nominal 90%, false-accept 0. Saturates at `k = 5`,
   where emb-traj *degrades* with k (0.80 → 0.60). Controls: dim-matched
   768-d projection 0.999; template-stripped 1.000.
2. **Capability (C side).** After 33-step SFT on 90 in-domain trajectories:
   in-domain residual gradient median 5.1 → 2.6 with loss 1.67 → 0.17;
   *all four* out-domains rise in both loss and residual — narrow
   distillation sharpens the student into 𝒯 without any explicit
   suppression.
3. **Behavior ≠ capability.** The same 33 steps flip the action space
   completely (0% → 100% code format, CoT output vanishes) while competence
   only partially transfers (exec 0% → 37% vs 67% prior CoT accuracy) —
   direct evidence for keeping the two channels separate, and a headline
   figure candidate.

## 6 Positioning (one line per nearest cluster; full scan in novelty doc)

Agent distillation (SmartAD ACL'26; NeurIPS'25 code-tool distillation)
maximizes benchmark scores — none estimates a task boundary or shapes C.
Gradient methods (LESS line; GaLore/SubTrack) use subspaces for selection or
memory, not as a *task specification with guarantees*. Conformal OOD (SCOPE)
filters at serving time, decoupled from training and capability. Learnware
*retrieves* models matching a spec; we *synthesize* one — the generative
dual. Unlearning removes capability from big models; we shape during
distillation and borrow its evaluation protocols for the behavioral module.

## 7 Risks and mitigations

| risk | mitigation |
|---|---|
| Perfect pilot numbers reflect easy domains | intra-domain splits + paraphrase/renaming hard pairs (E1) |
| Geometric C fails to track behavior (E2 gate fails) | fall back: boundary used for data selection only; C measured behaviorally |
| Teacher rollout noise destabilizes `U_𝒯` | multi-roll feature averaging; stability reported |
| Reviewer: "yet another distillation paper" | lead with problem formulation + guarantee + two-sided frontier; method numbers second |
| Sept deadline compute crunch | Tier-1 results complete on rai by design; hpg only for E3/E4 |

## 8 Timeline to ICLR 2027 (submission ~late Sept 2026)

- **W1 (now–Aug 16):** framework hardening (stable splits, verifier sandbox),
  intra-domain + robustness E1; E2 at 1.5B. PI sign-off on env panel + models.
- **W2–3:** teacher rollout pool generation; E3 tier-1 on rai; first hpg E3.
- **W4–5:** full E3/E4 sweeps on hpg; ablations (§2.2 stacking, §2.4 E/F).
- **W6:** writing (intro/method from this doc), figures frozen, internal red-team
  pass on claims vs evidence.
