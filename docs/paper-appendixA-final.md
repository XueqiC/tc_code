# A. Additional Methodology Details

This section provides implementation details omitted from the main text, including the complete training procedure, gradient representation, acquisition estimators, reference-task interference measurement, guidance construction, importance weighting, preference-pair construction, and task-boundary calibration.

## A.1 Complete Training Algorithm

**Algorithm 1: Budgeted Teacher-Scaffolded Self-Distillation**

**Input:** support queries (\mathcal S); initial student (\pi_{\theta_0}); teacher (\pi_T); verifier (V); teacher-token budget (B); number of rollouts (K).

**1. Split and initialize**

1. Randomly partition

[
\mathcal S=\mathcal S_d\cup\mathcal S_c,
\qquad
\mathcal S_d\cap\mathcal S_c=\emptyset.
]

2. Reserve (\mathcal S_c) exclusively for final calibration.

3. Construct the initial gradient representation and dictionary from admissible training-side data.

4. Estimate the task-demand vector (r) from (\mathcal S_d).

5. Initialize acquired context set (\mathcal A=\varnothing), rollout buffer (\mathcal B=\varnothing), accumulated capability supply (s_0=0), and spent teacher budget (b=0).

**2. Repeat for rounds (r=0,1,\ldots)**

**Acquisition**

6. Compute remaining capability deficits

[
\delta_{r,j}=(r_j-s_{r,j})_+.
]

7. For each feasible candidate (x), estimate task coverage (C_r(x)), guidance-unlock probability (\hat\rho_r(x)), teacher cost (\hat c(x)), and reference-task interference (I_r(x)).

8. Compute

[
U_r(x)
======

\frac{\hat\rho_r(x)C_r(x)}
{\hat c(x)}
-----------

\lambda_I I_r(x).
]

9. While budget remains and (\max_x U_r(x)>0), acquire the highest-utility feasible demonstration, verify it, convert it into context (h(d_x)), update the remaining budget and marginal utilities, and add the context to (\mathcal A).

**Collection**

10. For every (x\in\mathcal S_d), sample (K) unguided trajectories from

[
\pi_{\theta_r}(\cdot\mid x).
]

11. Evaluate every trajectory with (V) and record its behavior-policy log-likelihood.

12. If all unguided trajectories fail and (h(d_x)\in\mathcal A), sample guided trajectories from

[
\pi_{\theta_r}(\cdot\mid x,h(d_x)).
]

13. Record each guided trajectory, verifier outcome, provenance, and behavior-policy log-likelihood.

**Optimization**

14. Compute clipped importance weights for guided trajectories.

15. Construct the verifier-weighted positive objective from all collected student trajectories.

16. Construct outcome-contrastive preference pairs when successful and failed trajectories share a valid common prefix.

17. Apply the first-divergence mask.

18. Admit a preference pair only when the length-normalized positive NLL satisfies the EMA learnability gate.

19. Update the student using

[
\mathcal L
==========

\mathcal L_{\mathrm{ver}}
+
\lambda_{\mathrm{pref}}
\mathcal L_{\mathrm{pref}}.
]

**Re-estimation**

20. Re-evaluate unguided student capability.

21. Update accumulated verified capability supply (s_{r+1}), guidance-unlock estimates, interference estimates, and acquisition utilities.

22. Stop if the unguided convergence criterion is satisfied, the budget is exhausted with no further training benefit, or all remaining candidates have non-positive acquisition utility.

**3. Calibration**

23. Freeze the final student and all components defining the routing score.

24. Evaluate the frozen nonconformity score on (\mathcal S_c).

25. Compute the split-conformal threshold (\gamma_\alpha).

**Output:** specialized student (\pi_{\theta^\star}) and deployment threshold (\gamma_\alpha).

---

## A.2 Gradient Features and Coupled Dictionary Construction

We compute gradient features with respect to the same restricted parameter subset used for specialization. To reduce optimizer-dependent scaling effects, raw gradients are transformed using the corresponding Adam preconditioner before dimensionality reduction.

Let (\tilde g(x)\in\mathbb R^D) denote the preconditioned gradient. We apply a fixed random projection (P\in\mathbb R^{p\times D}),

[
g(x)=P\tilde g(x),
\qquad p\ll D.
]

When both prompt-side and outcome-side gradients are available, we fit a coupled sparse dictionary with shared latent coefficients. Missing outcome views are handled through an explicit availability mask rather than zero imputation.

Dictionary hyperparameters, projection dimension, sparsity coefficient, atom normalization, fitting schedule, and sensitivity analyses are reported in Appendix B.

---

## A.3 Demand, Supply, and Saturation

For dictionary atom (j), task demand is estimated by aggregating its activation across (\mathcal S_d). To prevent a small number of high-magnitude queries from dominating the estimate, activations are normalized at the query level before aggregation.

A verified student trajectory contributes capability supply according to its projected gradient representation. Accumulated supply after round (r) is denoted (s_{r,j}).

The remaining deficit,

[
\delta_{r,j}
============

(r_j-s_{r,j})_+,
]

ensures that acquisition exhibits diminishing returns along already-covered task directions.

We evaluate alternative demand and saturation estimators in the ablation study.

---

## A.4 Estimating Guidance Unlock Probability

The acquisition score requires an estimate of

[
\rho_r(x)
=========

P(V=1\mid x,h(d_x)).
]

Before any guided observations are available, we use a common prior so that initial acquisition is determined by demand coverage, predicted teacher cost, and interference risk.

After guided trajectories have been collected, (\rho_r) is updated using empirical success statistics with smoothing across nearby task representations. We do not use teacher success itself as an estimate of student success under guidance.

The unguided success probability can optionally be incorporated to estimate the incremental guidance effect

[
\Delta\rho_r(x)
===============

## P(V=1\mid x,h(d_x))

P(V=1\mid x).
]

We compare absolute unlock probability and incremental unlock probability in the acquisition ablations.

---

## A.5 Reference-Task Interference and (\kappa^r)

The interference term (I_r(x)) estimates whether acquiring and subsequently learning from the region associated with candidate (x) is likely to interfere with capabilities outside the target task.

Let (\mathcal R) denote a fixed set of free reference tasks and let (g_{\mathcal R}) denote their gradient representation. We estimate the interaction between a candidate's predicted update geometry and the reference-task gradient geometry through the reference coupling statistic

[
\kappa^r(x).
]

Operationally, (\kappa^r) distinguishes three empirical interaction regimes: task-aligned transfer, approximately orthogonal interaction, and antagonistic coupling. The acquisition penalty (I_r(x)) is a monotone function of the antagonistic component of (\kappa^r(x)).

The exact normalization and aggregation used to compute (\kappa^r) are specified below together with sensitivity analyses. RQ5 evaluates whether these geometric regimes predict measured reference-task interference after specialization.

---

## A.6 Construction of Teacher Guidance Context

For an acquired query (x), the teacher produces a verified demonstration

[
d_x\sim\pi_T(\cdot\mid x).
]

The demonstration is transformed into a context block (h(d_x)) and prepended or otherwise inserted into the student's generation context according to the environment interface.

Teacher-generated actions are never copied into the optimization target. The context is used only during guided collection and is removed when computing the deployment-policy likelihood (\pi_\theta(\tau\mid x)).

We keep the guidance transformation fixed across methods unless explicitly varied in an ablation.

---

## A.7 Importance Weighting and Support

For a trajectory collected at round (r), we store the behavior-policy log-likelihood

[
\log\mu_r(\tau\mid x).
]

For guided trajectories,

[
\mu_r(\tau\mid x)
=================

\pi_{\theta_r}(\tau\mid x,h(d_x)),
]

whereas the target likelihood is evaluated without teacher context:

[
\pi_\theta(\tau\mid x).
]

The trajectory-level log importance ratio is

[
\log w_\theta(\tau,x)
=====================

## \log\pi_\theta(\tau\mid x)

\log\mu_r(\tau\mid x).
]

For numerical stability we compute the ratio in log space and apply clipping,

[
\bar w_\theta
=============

\min(w_\theta,c).
]

The exact importance-sampling identity assumes

[
\pi_\theta(\cdot\mid x)
\ll
\mu_r(\cdot\mid x).
]

This condition is not guaranteed by contextual guidance: a guidance context can, in principle, make some trajectories that remain possible under the unguided policy unreachable under the guided behavior policy. Our theoretical interpretation of guided learning therefore applies to the overlapping support. Clipping controls variance on observed trajectories but cannot recover probability mass absent from the behavior-policy support.

We report the empirical distribution of importance ratios and compare no correction, unclipped correction where numerically feasible, and clipped correction.

---

## A.8 Outcome-Localized Preference Construction

For each query, we search the student rollout buffer for successful-failure pairs sharing the same environment state before their first action divergence.

For a pair ((\tau^+,\tau^-)), define

[
t^\star
=======

\min{t:a_t^+\neq a_t^-}.
]

Tokens preceding (t^\star) receive no preference loss.

The positive learnability score is the length-normalized NLL

[
\ell^+(x)
=========

-\frac{1}{|\tau^+|}
\sum_{t=1}^{|\tau^+|}
\log
\pi_\theta(a_t^+\mid h_t^+).
]

We maintain an exponential moving average of this quantity,

[
m_r
===

\beta m_{r-1}
+
(1-\beta)\bar\ell_r^+,
]

and construct the admission threshold from the running statistic. This length normalization removes the systematic preference for shorter trajectories that arises when raw sequence NLL is used.

Only admitted pairs contribute to (\mathcal L_{\mathrm{pref}}).

---

## A.9 Single-Dose Positive Replay

The primary verifier objective is computed from the actual sampled rollout distribution and does not select the shortest successful trajectory.

When persistent positive replay is used, we retain at most one verified anchor per query per collection stage. The anchor is sampled uniformly from eligible successful trajectories rather than selected by trajectory length.

This replay mechanism is distinct from the policy-gradient estimator and is included only as an optimization aid. We evaluate whether additional positive replay improves performance beyond this single-dose setting.

---

## A.10 Repricing and Stopping Criteria

After every optimization round, the student is reevaluated without teacher context. Verified unguided behavior updates the estimated capability supply and the empirical need for guidance.

We then recompute

[
U_{r+1}(x)
]

for all unacquired candidates.

Teacher acquisition terminates when

[
\max_x U_r(x)\le0
]

or the remaining budget cannot fund a feasible positive-utility candidate.

Optimization terminates according to the predefined unguided convergence criterion. We do not use the final calibration set for early stopping or checkpoint selection.

---

## A.11 Task-Boundary Calibration

After all training decisions have been completed, we freeze the student, dictionary, and nonconformity-score construction.

For query (x), the deployment nonconformity score is

[
R(x)
====

|g_p(x)-D_pz(x)|_2.
]

For the calibration set

[
\mathcal S_c
============

{x_1^c,\ldots,x_m^c},
]

we compute scores

[
R_i=R(x_i^c).
]

For desired miscoverage level (\alpha), the threshold is the finite-sample corrected empirical quantile

[
\gamma_\alpha
=============

R_{(k)},
\qquad
k=
\left\lceil
(m+1)(1-\alpha)
\right\rceil,
]

with the usual convention when (k>m).

The resulting deployment rule is

[
\operatorname{route}(x)
=======================

\begin{cases}
\text{specialized student}, & R(x)\le\gamma_\alpha,\
\text{reject/escalate}, & R(x)>\gamma_\alpha.
\end{cases}
]

Because the score construction is frozen before (\mathcal S_c) is accessed, the calibration split is used only to determine the final threshold. The resulting guarantee is the standard split-conformal marginal guarantee under exchangeability; it should not be interpreted as a calibration-set-conditional or out-of-distribution detection guarantee.
