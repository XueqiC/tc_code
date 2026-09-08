# CRCD Next-Stage Research & Engineering Prompt

## Goal: turn CRCD into an ICLR-level distillation method, not merely a localized-DPO recipe

You are working on **CRCD: Capability-Residual Correction Distillation** for small agent models (current student: Qwen3.5-4B), primarily on BFCL v4, with ALFWorld/AppWorld/τ² as cross-benchmark tests.

Your job is **not** simply to implement the next experiment list.

Your job has two simultaneous objectives:

1. **Strengthen and falsify the current CRCD mechanism**, especially the parts that could constitute real ICLR-level novelty.
2. **Actively search for adjacent or stronger mechanism-level innovations** that remain centered on distillation and can be tested within our current infrastructure.

Do not optimize only for benchmark score.
The main objective is to determine whether we have a genuinely new distillation principle and, if so, make the algorithm express that principle as cleanly as possible.

---

# 0. Current empirical state

The current BFCL base is approximately:

$$
46.06
$$

The most important observations so far are:

### Absolute corrective CE catastrophically fails

Local correction CE on \(a^+\):

* B2 CE: 10.95
* CE → preference: 9.13
* r2 CE: 18.38

Failure is structural, not merely lack of OOS negatives.

For example, early CE caused Irrelevance to collapse from ~82.7 to ~3.6 and MT/Memory nearly to zero. Adding explicit abstention examples repaired much of the invoke/abstain bias but did **not** repair MT/Memory collapse.

Therefore:

> local correct labels are not sufficient; the geometry of corrective learning matters.

### Relative corrective preference works

Base-centered preference correction:

* r1: 46.09
* r2: 47.34
* r3 cumulative: ~47.82–47.85
* balanced-anchor C3: 48.02

This is the first mechanism family that exceeds base under the official BFCL protocol; previous SFT/curriculum/static weighting/retention/repair variants were below base.

### Training dose has a narrow useful regime

Roughly:

* 27 steps: 47.34
* ~41 steps: 47.82 / 48.02 depending on pool/anchors
* ~81 steps: 47.61–47.85
* > 120 steps: degradation to ~44–45

Target capabilities can continue improving while preservation degrades.

Thus the optimization variable is likely not “epochs” but:

$$
\text{marginal corrective gain}
\quad \text{vs} \quad
\text{marginal policy drift}.
$$

### Anchors behave like decision-boundary supports

Different anchor types protect opposite sides of the invoke/abstain boundary.

* abstention/flip anchors recover Irrelevance but can hurt Relevance/MT;
* self-correct anchors preserve Relevance/MT but do not sufficiently protect Irrelevance;
* balanced C3 currently performs best.

Therefore anchors should probably not be treated as generic replay or fixed KL regularization.

### Residual self-iteration is promising

Residual event count contracted substantially between rounds, approximately:

$$
322 \rightarrow 115.
$$

Training each round **from base on the accumulated correction pool** is better than continuing training from the previous adapted checkpoint.

This suggests:

$$
\theta_t \rightarrow \text{discover residual data}
$$

but

$$
\theta_{t+1}
=
Train(\theta_0,\cup_{i \leq t}E_i)
$$

rather than repeatedly applying parameter updates along the same path.

### Capability fingerprints are not yet validated

Current differential-gradient representation A3/A4 is conceptually attractive, but the first cross-cluster transfer validity experiment was inconclusive:

$$
Spearman(P,G)\approx0.08.
$$

The first experiment also contained a serious ID collision bug and high measurement noise.

A stronger gate v2 is running / planned.

Until cross-event transfer prediction is demonstrated, **do not assume gradient clusters are real capability atoms**.

---

# 1. Central research question

The main scientific question is:

> **What should count as supervision in black-box agent distillation?**

The current candidate answer is:

> A teacher/expert correction should not become supervision merely because it differs from the student.
> It becomes supervision only if intervening on that local decision demonstrably improves the student's downstream outcome.

This leads to the working principle:

# Distillation as Counterfactual Consequential Policy Repair

For a student-visited state \(h\):

* \(a^-\): action actually chosen by the student;
* \(a^+\): candidate corrected action from GT / verified demo / teacher;
* evaluate both interventions by continuing the student's policy:

$$
\Delta U(h,a^+,a^-)
=
\mathbb{E}[R \mid h,do(a^+)]
-
\mathbb{E}[R \mid h,do(a^-)].
$$

Only a correction with positive intervention value is presumptively valid supervision.

Then perform a minimal policy repair rather than absolute imitation.

The key distinction from nearby work must be preserved:

> first-divergence localization + DPO is **not sufficient novelty**.

Prior work already covers student rollouts, first divergence / first critical error, corrected-vs-error preference learning, sparse step supervision, and reference-based DPO.

The research novelty must therefore come primarily from one or more of:

1. **Outcome-validated consequential supervision**
2. **Utility-weighted corrective distillation**
3. **Residual-aware stopping / supervision allocation**
4. **Risk-triggered preservation rather than universal preservation**
5. **Validated capability-level sharing of corrective evidence**
6. A stronger related mechanism you discover that is not already subsumed by nearby work

---

# 2. Formal event abstraction

Use the following abstraction everywhere possible:

$$
e=(h,a^-,a^+,\widehat{\Delta U},m)
$$

where:

* \(h\): student-visited history/state;
* \(a^-\): student's actual response/action;
* \(a^+\): candidate local correction;
* \(\widehat{\Delta U}\): estimated downstream intervention effect;
* \(m\): metadata including benchmark/task type, rollout round, source, confidence, current residual estimates, anchor type, etc.

Important:

A **disagreement is not automatically a correction**.

Categorize events as:

### Harmful candidate

$$
\widehat{\Delta U}<-\epsilon
$$

Teacher/demo candidate is worse than student's behavior.

Do not distill it.

Potentially retain this as anti-supervision / negative teacher evidence.

### Behaviorally equivalent

$$
|\widehat{\Delta U}| \leq \epsilon
$$

Teacher/student behaviors differ but downstream utility is effectively unchanged.

Default action: preserve the student's original policy; do not imitate the expert merely for stylistic/trajectory matching.

### Consequential correction

$$
\widehat{\Delta U}>\epsilon
$$

Only these are primary CRCD training events.

---

# 3. Main CRCD objective to implement and study

The baseline relative correction objective is reference-centered pairwise preference learning:

$$
\mathcal{L}_{pair}(e)
=
-
\log\sigma\left(
\beta
[
r_\theta(h,a^+)-r_\theta(h,a^-)
]
\right)
$$

with

$$
r_\theta(h,a)
=
\log
\frac{\pi_\theta(a|h)}
{\pi_0(a|h)}.
$$

Here \(\pi_0\) is the frozen base student.

Do not treat base-centered DPO itself as the novelty.

Instead investigate:

$$
\mathcal L_{\mathrm{CRCD}}
=
\mathbb E_e
[
w_e
\mathcal L_{pair}(e)
]
+
\mathcal L_{\mathrm{preserve}}
$$

where candidate event weight is progressively enriched from:

### Version 0

$$
w_e = 1
$$

for all disagreements.

### Version 1: consequential filtering

$$
w_e
=
\mathbf 1[
\widehat{\Delta U}_e>\epsilon
].
$$

### Version 2: utility weighting

$$
w_e
=
f(\widehat{\Delta U}_e).
$$

Candidate choices:

* raw clipped \(\Delta U\);
* rank-normalized \(\Delta U\);
* confidence-adjusted lower bound;
* posterior probability \(P(\Delta U>0)\);
* nonlinear saturation, e.g.

  $$
  w=\tanh(\Delta U/\tau).
  $$

### Version 3: utility × residual

$$
w_e^{(t)}
=
f(\widehat{\Delta U}_e)
\cdot
\rho_e^{(t)}
$$

or, if validated capability atoms become available:

$$
w_e^{(t)}
=
f(\widehat{\Delta U}_e)
\left(
\alpha_e^\top \delta^{(t)}
\right).
$$

The distinction should be:

* \(\Delta U\): **should this correction ever be learned?**
* residual \(\rho/\delta\): **does the student still need to learn it now?**

---

# 4. Highest-priority novelty experiment: consequentiality ablation

This experiment is mandatory.

Use the **same candidate disagreement pool** wherever possible.

Compare:

## A. All disagreement preference

Every student/expert disagreement becomes a DPO pair.

## B. First-divergence preference

Only first disagreement / first divergence is used.

This is the most important nearby-method baseline.

It approximates S-AA / first-error localized preference methods.

## C. Consequential-only CRCD

Only events with:

$$
\widehat{\Delta U}>\epsilon
$$

enter training.

## D. Utility-weighted CRCD

Same consequential events, but:

$$
w=f(\widehat{\Delta U}).
$$

Ideal hypothesis:

$$
D > C > B \ge A.
$$

But do not force this conclusion.

Interpret honestly:

* If B ≈ C ≈ D, consequentiality is probably not the main mechanism.
* If C > B, causal filtering is validated.
* If D > C, intervention magnitude carries useful learning-value information.
* If A/B outperform C/D, re-examine \(\Delta U\) estimator quality and the core hypothesis.

Run this first on ALFWorld if BFCL lacks sufficiently rich neutral disagreements.

ALFWorld is especially diagnostic because many student/demo disagreements such as early `look` appear to have:

$$
\Delta U\approx0.
$$

These are ideal examples where ordinary imitation/localized-DPO says “correct,” while CRCD says “leave the student's behavior alone.”

---

# 5. Validate \(\widehat{\Delta U}\) as a distillation-value estimator

Do not merely use \(\Delta U\) as a filtering heuristic.

Test whether it is calibrated to **actual learning utility**.

Bucket events by estimated intervention utility:

1. negative;
2. approximately zero;
3. small positive;
4. medium positive;
5. large positive.

For each bucket, train controlled equal-size/equal-step models or otherwise estimate post-training effect.

Measure:

$$
\widehat{\Delta U}
\rightarrow
\text{actual downstream gain}.
$$

Useful metrics:

* Spearman rank correlation;
* isotonic / calibration curve;
* positive-vs-nonpositive enrichment;
* top-decile intervention efficiency;
* gain per optimizer step;
* gain per teacher/output token where relevant.

This is one of the highest-value figures for an ICLR paper.

The desired claim is:

> Counterfactual intervention value predicts which teacher disagreements are actually worth turning into parameter updates.

---

# 6. Separate the mechanisms inside pairwise correction

Current evidence proves:

$$
CE \ll base\text{-centered preference}.
$$

But this confounds several ingredients.

Run:

### CE

$$
-\log \pi(a^+|h)
$$

### Reference-free pairwise

For example logistic pairwise objective using:

$$
\log\pi(a^+|h)-\log\pi(a^-|h).
$$

### Base-centered DPO

Current CRCD objective.

Optionally:

### Minimal-KL policy repair

Explicitly solve / approximate:

$$
\max_{\pi}
\Delta U \cdot
[
\log\pi(a^+|h)-\log\pi(a^-|h)
]
-
\lambda
D_{KL}(\pi||\pi_0).
$$

Questions:

1. Is pairwise contrast the important part?
2. Is explicit base reference important?
3. Is ordinary DPO merely one optimizer for a more general local policy-repair problem?

If reference-free preference nearly matches base-DPO, do **not** claim base centering as a major contribution.

If base-DPO significantly improves preservation, formalize it as a trust-region component.

---

# 7. Develop the “corrective overwrite” mechanism

Current CE collapse is scientifically interesting.

Investigate why a tiny local CE pool can catastrophically overwrite unrelated agent behavior.

Instrument CE vs pairwise training.

Measure:

### Parameter/update geometry

* gradient norm;
* Adam-preconditioned norm;
* cosine to base-task anchor gradients;
* LoRA singular spectrum;
* layer-wise update distribution;
* update effective rank.

### Distributional drift

On base-correct states:

$$
KL(
\pi_\theta(\cdot|h)
||
\pi_0(\cdot|h)
).
$$

Also measure:

* action entropy;
* tool-call prior;
* abstention probability;
* average response length;
* response-format shifts;
* MT state-transition behavior.

Test the hypothesis:

> Absolute correction CE pushes globally on the corrected action manifold, whereas pairwise repair mostly changes the local decision margin.

If this can be empirically shown, formalize the phenomenon as **Corrective Overwrite**.

A strong result would be:

$$
\text{equal local correction success}
$$

but

$$
KL_{\mathrm{CE}}
\gg
KL_{\mathrm{pairwise}}
$$

on unrelated mastered states.

This would significantly strengthen the paper.

---

# 8. Adaptive distillation dose

Current evidence suggests a 40–80 optimizer-step useful region and degradation beyond ~120 steps.

Do not merely tune this as a hyperparameter.

Test whether training should stop according to **marginal repair efficiency**.

Candidate stopping metrics:

## Residual exhaustion

Stop when:

$$
\frac{\Delta \text{residual errors}}{\Delta steps}
<\tau.
$$

## Utility saturation

Stop when expected remaining:

$$
\sum_e \rho_e \widehat{\Delta U}_e
$$

stops decreasing.

## Repair-to-drift ratio

Maintain probes:

$$
R_t=
\frac{
\Delta J_{\mathrm{target}}
}{
-\Delta J_{\mathrm{mastered}}+\epsilon
}.
$$

Stop when \(R_t\) falls below a threshold.

## Policy-drift trust region

Stop / reduce LR when:

$$
\mathbb E_{h\sim anchor}
KL(\pi_t||\pi_0)
>
\epsilon.
$$

The ideal conclusion is not:

> “3 epochs is best.”

It is:

> **CRCD should continue only while marginal corrective value exceeds marginal policy-drift cost.**

---

# 9. Replace fixed anchor mixing with risk-triggered preservation

Current C3 balanced anchors work, but fixed replay is unlikely to be the final mechanism.

Implement adaptive preservation.

For each protected dimension \(k\):

$$
D_k(t)
=
J_k(\theta_t)-J_k(\theta_0).
$$

Maintain tolerance:

$$
D_k(t)\ge-\epsilon_k.
$$

Only when violated, activate an associated preservation term.

Candidate primal-dual update:

$$
\lambda_k^{t+1}
=
[
\lambda_k^t
+
\eta(
-\epsilon_k-D_k(t)
)
]_+.
$$

Then:

$$
\mathcal L
=
\mathcal L_{\rm CRCD}
+
\sum_k
\lambda_k
\mathcal L_{\rm anchor,k}.
$$

Compare:

1. no anchor;
2. fixed C3 balanced anchors;
3. adaptive category-triggered anchors;
4. eventually atom-risk-triggered anchors if atom validity succeeds.

Important:

anchors should be treated as **boundary constraints**, not generic replay.

---

# 10. Improve anchor construction: local counterfactual boundary anchors

The current global anchor pool may overcorrect the invoke/abstain boundary.

Explore constructing anchors locally around each repair event.

For a correction event \(e\), retrieve/generate:

### Positive-side anchor

A nearby context where the corrected behavior \(a^+\) is appropriate.

### Negative-side anchor

A minimally changed context where that same behavior would be inappropriate.

Example:

* invoke tool here;
* nearly identical query but should abstain.

Train:

$$
\text{repair pair}
+
\text{local positive anchor}
+
\text{local negative anchor}.
$$

Interpret this as preserving the **local decision boundary** rather than globally forcing an action prior.

Especially test on:

* BFCL Relevance vs Irrelevance;
* tool invocation vs abstention;
* clarification vs action;
* continue vs terminate.

This may become a novel “boundary-preserving corrective distillation” extension.

---

# 11. Residual self-iteration

Continue the current iterative process, but formalize it.

At round \(t\):

1. evaluate current round model \(\theta_t\);
2. discover remaining consequential events:

   $$
   E_t;
   $$
3. add them to accumulated pool:

   $$
   E_{\le t};
   $$
4. retrain from frozen base:

   $$
   \theta_{t+1}=Train(\theta_0,E_{\le t});
   $$
5. remeasure residuals.

Compare against:

### Continual update

$$
\theta_{t+1}=Train(\theta_t,E_{t+1}).
$$

Test:

* final success;
* KL drift from base;
* residual contraction;
* forgetting;
* optimizer/path dependence.

Try to establish:

> current adapted model should determine **what data remains useful**, but need not serve as the initialization for the next optimization problem.

If reproducible, this is a meaningful secondary mechanism.

---

# 12. Capability atom redesign

Do not assume gradient clusters are atoms.

Use the following hierarchy:

## Corrective event

Observable unit:

$$
e=(h,a^-,a^+,\Delta U).
$$

## Candidate capability representation

Differential fingerprint:

$$
\Delta g_e
=
g(h,a^+)-
g(h,a^-).
$$

Preferred versions to compare:

### A1 full/prompt gradient

Negative control likely dominated by task/template identity.

### A2 positive response gradient

$$
g(a^+|h)
$$

### A3 normalized differential gradient

$$
\frac{
g(a^+|h)-g(a^-|h)
}{
\|
g(a^+|h)-g(a^-|h)
\|
}.
$$

### A4 utility-bearing differential fingerprint

Do **not** normalize away \(\Delta U\):

$$
z_e
=
\widehat{\Delta U}_e
\cdot
\frac{\Delta g_e}{\|\Delta g_e\|}.
$$

### Additional representations to explore

* per-layer differential gradient profile;
* normalized Adam-preconditioned differential gradients;
* sign-only sketches;
* Fisher-whitened sketches;
* low-rank layer spectra;
* gradient × activation interaction;
* pairwise influence-function approximations;
* local policy Jacobian differences.

The capability representation should be selected by **transfer prediction**, not ARI.

---

# 13. Strong definition of a capability atom

A cluster is not automatically an atom.

Use the following operational definition:

> **A capability atom is an equivalence class of corrective interventions whose localized policy repairs exhibit reproducible cross-event transfer.**

Therefore:

$$
event
\rightarrow
representation
\rightarrow
candidate\ cluster
\rightarrow
transfer\ validity
\rightarrow
validated\ atom.
$$

Required atom validity properties:

### Stability

Cluster/factor survives:

* support resampling;
* projection seed;
* nearby checkpoints.

### Predictive transfer

Representation similarity predicts:

$$
Train(e_i)
\rightarrow
Gain(e_j).
$$

### Interventional validity

Training on events loading on atom \(a\) preferentially changes behavior associated with the same atom.

If these do not hold, **do not call the clusters capability atoms**.

Use fingerprints only for:

* deduplication;
* pool coverage;
* diagnostics;
* diversity sampling.

---

# 14. Capability validity gate v2+

Current planned gate fixes ID collision, increases K, training dose, evaluation size, and column-centers the gain matrix.

Also measure:

### Within-vs-cross gain

$$
G_{aa} -
G_{a,\neg a}.
$$

### Ranking quality

For training cluster \(i\), whether predicted similarity ranks actually improved evaluation groups:

* NDCG;
* precision@k;
* top-quartile enrichment.

### Sign prediction

Especially important:

$$
P_{ij}<0
\Rightarrow
G_{ij}<0?
$$

Negative-transfer prediction may be more useful than exact magnitude correlation.

### Baseline representations

Compare:

* task/category labels;
* semantic embeddings;
* prompt gradient A1;
* random clusters;
* A3/A4 differential fingerprints.

The most interesting possible result is:

> A1 has excellent category ARI but poor transfer prediction; A3/A4 have lower category alignment but superior causal transfer prediction.

This would support:

> capabilities are not equivalent to benchmark categories.

---

# 15. If atoms validate: move from event-level to capability-residual distillation

Only after validity succeeds, define soft event loading:

$$
\alpha_{e,a}.
$$

Maintain capability residual:

$$
\delta_a^{(t)}
=
\sum_e
\alpha_{e,a}
\rho_e^{(t)}
\widehat{\Delta U}_e.
$$

Then:

$$
w_e^{(t)}
=
f(\widehat{\Delta U}_e)
\left(
\alpha_e^\top
\delta^{(t)}
\right).
$$

Potentially estimate signed capability transfer:

$$
M_{ba}
=
\mathbb E[
\Delta J_b
\mid
\text{train atom }a
].
$$

Use negative entries for predicted preservation risk.

This becomes:

> **capability-residual causal distillation**

rather than event-level correction.

But if the validity gate fails, keep the paper centered on event-level CRCD.

---

# 16. Explore stronger novelty directions adjacent to CRCD

You are explicitly encouraged to explore alternatives beyond the current design.

But any proposed innovation must satisfy:

1. still fundamentally be a **distillation mechanism**;
2. exploit the black-box teacher / student-agent setting;
3. have a clear nearest-prior comparison;
4. yield a falsifiable experiment;
5. not merely add generic RL, curriculum, or regularization.

Promising directions include:

---

## 16.1 Negative teacher evidence

Current CRCD only uses:

$$
\Delta U>0.
$$

But if:

$$
\Delta U<0,
$$

we have evidence that the expert/demo candidate is actually worse for this student state.

Can these events teach:

> when **not** to imitate the teacher?

Possible objective:

treat student's \(a^-\) as preferred over candidate \(a^+\) when intervention evidence supports it.

This turns CRCD from “selective teacher following” into **evidence-conditioned teacher trust**.

Test carefully.

---

## 16.2 Confidence-aware causal distillation

\(\widehat{\Delta U}\) from K rollouts is noisy.

Instead of thresholding sample mean, maintain a posterior:

$$
P(\Delta U>0|\text{rollouts}).
$$

Train only if:

$$
P(\Delta U>0)>\tau.
$$

Or weight by lower confidence bound:

$$
w_e=
LCB(\Delta U_e).
$$

This can reduce false-positive corrections.

---

## 16.3 Adaptive continuation allocation

Do not use fixed K for every event.

Spend continuation rollouts based on uncertainty.

Easy events:

$$
K\downarrow.
$$

Ambiguous near-zero events:

$$
K\uparrow.
$$

Stop sampling once the decision:

$$
\Delta U>0
$$

or

$$
\Delta U\le0
$$

is statistically stable.

This turns intervention verification itself into a budgeted sequential test.

---

## 16.4 Counterfactual action sets, not only binary correction pairs

Rather than:

$$
a^- \text{ vs } a^+,
$$

for some states evaluate multiple candidate actions:

$$
\{a_1,\dots,a_m\}
$$

through continuation rollouts.

Estimate:

$$
\widehat Q(h,a_j).
$$

Then distill a local ranking / soft target proportional to estimated downstream utility.

This would generalize CRCD from binary correction to:

> **counterfactual Q-distillation from black-box environment feedback**.

Investigate whether this is more novel/useful than binary DPO and whether cost is manageable.

---

## 16.5 Distill only the minimum sufficient action edit

For structured tool calls, \(a^+\) and \(a^-\) may differ only in:

* tool name;
* one argument;
* termination token;
* abstention decision.

Instead of training the full action string, identify the **minimal causal edit**.

Potential forms:

* span-level preference;
* field-level preference;
* structured action-component loss.

Hypothesis:

> smaller corrective support reduces collateral policy drift.

This could be a strong distillation-specific innovation.

---

## 16.6 Causal recovery window

The immediate action correction may not be sufficient.

After replacing \(a_t^-\) by \(a_t^+\), determine how many later steps remain causally altered.

Train only until trajectories re-enter an equivalent/recovered state.

So supervision support becomes:

$$
[t^*,t^{recover}]
$$

instead of one step or full tail.

This gives **adaptive supervision density based on behavioral recovery**, potentially stronger than fixed TurnOPD-like depth schedules.

---

## 16.7 Composition residuals

Identify cases where atomic behaviors appear individually solved but their composition fails.

Example:

* correct tool selection;
* correct argument formatting;
* correct state tracking;

but failure occurs at the transition between them.

Represent these as:

$$
\delta_{a\rightarrow b}
$$

or pairwise/higher-order residuals.

Distill only the junction state.

This may lead to **compositional corrective distillation**.

---

## 16.8 Minimal policy repair as constrained optimization

DPO may not be the optimal implementation.

Formulate:

$$
\max_\pi
\mathbb E_e[
\Delta U_e
\cdot
\Delta \log \pi_e
]
$$

subject to:

$$
\mathbb E_{h\sim mastered}
KL(\pi||\pi_0)
\le \epsilon.
$$

Try simple realizations:

* adaptive KL;
* Lagrangian;
* mirror descent;
* trust-region pairwise loss.

If one substantially improves the target-retention frontier over standard DPO, it could become the algorithmic centerpiece.

---

# 17. Theoretical angle

Try to formalize a small but clean policy-improvement result.

Suppose:

$$
Q^{\pi_0}(h,a^+)
>
Q^{\pi_0}(h,a^-)
$$

and an update locally increases:

$$
\log
\frac{\pi(a^+|h)}
{\pi(a^-|h)}
$$

while remaining sufficiently close to \(\pi_0\).

Investigate whether first-order policy improvement gives:

$$
\Delta J>0
$$

under reasonable assumptions.

The goal is **not** a grand theorem.

A useful proposition would be:

> Under exact intervention utility and sufficiently local trust-region updates, every accepted CRCD correction is a first-order policy-improving supervision event.

Also identify where the claim fails:

* imperfect \(\Delta U\);
* distribution shift after local action;
* non-Markov history;
* multiple equivalent actions;
* continuation policy mismatch.

A clean theorem + empirical calibration would considerably strengthen an ICLR submission.

---

# 18. Cross-benchmark priorities

## BFCL

Use for:

* CE vs pairwise mechanism;
* preservation;
* dose;
* invoke/abstain boundary;
* capability fingerprints;
* residual iteration;
* full official evaluation.

## ALFWorld

Very high priority.

Use for the core consequentiality claim because teacher/student divergence does **not** automatically imply an error.

Run:

* all divergence;
* first divergence;
* consequential-only;
* utility-weighted;
* CE vs preference;
* fixed vs adaptive preservation.

This benchmark may be more important scientifically than another +0.5 BFCL point.

## AppWorld

Once collection becomes available, reproduce the same event abstraction using replayable demo states.

Especially test whether expensive long demos can be compressed into a small number of consequential corrective interventions.

## τ²

Later.

Be explicit about user-simulator cost and protocol changes.

---

# 19. Evaluation should not be only Overall score

Always track:

### Target gain

$$
J_{\mathrm{target}}(\theta)
-
J_{\mathrm{target}}(\theta_0).
$$

### Preservation damage

$$
J_{\mathrm{mastered}}(\theta)
-
J_{\mathrm{mastered}}(\theta_0).
$$

### Repair efficiency

$$
\frac{\Delta J_{\mathrm{target}}}
{\# optimizer\ steps}.
$$

### Teacher efficiency

$$
\frac{\Delta J}
{\text{paid teacher output tokens}}.
$$

### Intervention efficiency

$$
\frac{\Delta J}
{\# consequential events}.
$$

### Drift

* KL to base on mastered states;
* behavior flip rate;
* invoke/abstain boundary shift.

### Residual contraction

$$
|E_{t+1}|/|E_t|.
$$

### Cross-event transfer

if fingerprints are used.

---

# 20. Required research discipline

For every new mechanism, explicitly record:

### Hypothesis

What causal/mechanistic claim are we testing?

### Nearest prior

What existing method already contains the closest component?

### Incremental novelty

Exactly what is different?

Avoid vague claims such as:

> “prior methods do not consider capabilities.”

Be precise.

### Minimal experiment

What is the cheapest experiment that can falsify the idea?

### Success pattern

What result would support the mechanism?

### Failure interpretation

If the result is negative, what exactly should be removed/downgraded?

---

# 21. Go / No-Go criteria

## Consequentiality

GO if filtering/weighting by \(\Delta U\) consistently improves either:

* final performance;
* target-retention frontier;
* sample/teacher efficiency;

over first-divergence DPO.

NO-GO as core novelty if first-divergence DPO performs equivalently.

---

## Utility magnitude

GO if estimated \(\Delta U\) predicts actual training gain.

If only its sign matters, simplify to binary consequentiality.

If neither sign nor magnitude predicts learning utility, re-examine the estimator and do not build the paper around it.

---

## Capability atoms

GO if fingerprint similarity predicts transfer significantly beyond:

* semantic/task category;
* random clustering;
* prompt-gradient similarity.

If not, downgrade fingerprints to pool diagnostics.

---

## Adaptive preservation

GO if it improves the Pareto frontier over fixed C3 anchors.

If equivalent, use fixed anchors for simplicity.

---

## Residual iteration

GO if cumulative-base retraining reproducibly beats continual fine-tuning and produces meaningful residual contraction.

Otherwise treat it as engineering rather than contribution.

---

# 22. Desired final method hierarchy

The paper should remain valid at three possible levels.

## Level 1 — minimum viable paper

**Counterfactual Consequential Correction Distillation**

Core:

$$
student\ state
\rightarrow
candidate\ correction
\rightarrow
paired\ continuation
\rightarrow
\Delta U
\rightarrow
local\ relative\ policy\ repair.
$$

This level must already be scientifically complete.

---

## Level 2 — stronger CRCD

Add:

$$
utility
\times
residual
$$

and adaptive preservation.

Core claim:

> learn only consequential unresolved corrections and protect only behaviors currently at risk.

---

## Level 3 — full capability CRCD

Only if atom validity succeeds:

$$
event
\rightarrow
validated\ capability\ atom
\rightarrow
capability\ residual
\rightarrow
signed\ transfer/risk
\rightarrow
capability\ selective\ distillation.
$$

Do not force Level 3 if the data does not support it.

---

# 23. Most important paper-level framing

Avoid framing the method as:

> “We apply DPO at the student's first error.”

That is too close to existing methods.

Prefer:

> **Expert disagreement is not supervision. CRCD turns a candidate correction into supervision only when a counterfactual intervention at the student's own state demonstrably improves downstream utility. It then performs the minimum relative policy repair necessary to move the student across that consequential decision boundary.**

If residual and preservation work, extend:

> **The correction is applied only while the corresponding error remains unresolved, and preservation constraints are activated only when mastered behavior is measurably at risk.**

This is the mechanism we should attempt to make both empirically unavoidable and algorithmically simple.

---

# 24. Immediate execution order

Unless current jobs already supersede these, prioritize:

### P0

1. Finish / inspect BFCL r3-union + C3 official result.
2. Finish ALFWorld CRCD event extraction and first official CRCD result.
3. Run all-disagreement vs first-divergence vs consequential-only vs utility-weighted.
4. Run reference-free preference vs base-centered DPO.
5. Finish capability validity gate v2 and make a hard Go/No-Go decision.

### P1

6. \(\Delta U\) calibration study.
7. CE vs pairwise policy-drift diagnostics.
8. adaptive/risk-triggered anchors.
9. paired local invoke/abstain boundary anchors.
10. round-4 residual iteration.

### P2

11. adaptive-K intervention verification.
12. causal recovery-window distillation.
13. minimal structured-action edit distillation.
14. multi-action counterfactual Q-distillation.
15. capability-level residual weighting if atoms validate.

---

# 25. Output expected from you

Do not merely implement experiments silently.

Produce a research report containing:

1. **What you changed**
2. **Why the change tests a mechanism rather than only tuning performance**
3. **Nearest related mechanism/prior work**
4. **Code paths changed**
5. **Exact experiment configuration**
6. **Results**
7. **Mechanistic interpretation**
8. **Whether the result strengthens or weakens CRCD novelty**
9. **Next cheapest falsification experiment**
10. **Any newly discovered mechanism-level idea worth exploring**

When suggesting a new idea, prefer:

> one strong new principle + a clean ablation

over:

> five extra losses/hyperparameters.

The end goal is an ICLR paper whose core can be explained in one sentence and whose key ablation makes that sentence empirically unavoidable.
