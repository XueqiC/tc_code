# 3. Method

## 3.1 Problem Formulation

We study **few-shot task specialization** of a language-model agent under a limited teacher budget. Let (\mathcal{T}) denote a target task with query distribution (p_{\mathcal T}(x)), and let

[
\mathcal{S}={x_i}*{i=1}^{N}, \qquad x_i\sim p*{\mathcal T},
]

be a small support set that characterizes the task. An agent executes a trajectory

[
\tau=(a_1,o_1,\ldots,a_H,o_H),
]

where (a_t) denotes the model action at step (t) and (o_t) denotes the corresponding environment observation. We assume access to a deterministic or stochastic task verifier

[
V(x,\tau)\in{0,1},
]

which evaluates whether a completed trajectory successfully solves query (x).

Our starting point is a generic student policy (\pi_{\theta_0}). We seek a specialized policy (\pi_{\theta^\star}) that maximizes expected task success,

[
J_{\mathcal T}(\theta)
======================

\mathbb E_{x\sim p_{\mathcal T}}
\mathbb E_{\tau\sim\pi_\theta(\cdot\mid x)}
[V(x,\tau)].
]

In addition, we have query access to a stronger teacher policy (\pi_T). A teacher call produces a demonstration (d) at token cost (c(d)). Given a total teacher budget (B), the acquired demonstration set (\mathcal A) must satisfy

[
\sum_{d\in\mathcal A}c(d)\le B.
]

The conventional approach uses (d\sim\pi_T(\cdot\mid x)) directly as an imitation target. Such an update, however, moves the student toward the teacher distribution:

[
\mathbb E_{d\sim\pi_T(\cdot\mid x)}
[\nabla_\theta\log\pi_\theta(d\mid x)]
======================================

-\nabla_\theta
D_{\mathrm{KL}}
\left(
\pi_T(\cdot\mid x)
|
\pi_\theta(\cdot\mid x)
\right).
]

This update can mix task-relevant capability transfer with changes induced merely by differences between teacher and student behavior distributions. Our method instead uses the teacher only to facilitate exploration. A purchased demonstration (d) is converted into a context (h(d)), under which the **student** generates a new trajectory. Teacher-generated tokens are never used as optimization targets.

Accordingly, our problem is to jointly determine (i) **where to spend teacher tokens**, (ii) **when teacher context is needed for student exploration**, and (iii) **how to convert guided student trajectories into improvements of the unguided deployment policy**. We refer to the resulting framework as **budgeted verifier-guided self-distillation with teacher context scaffolding**.

We partition the support set before training into a demand set (\mathcal S_d) and a calibration set (\mathcal S_c). Only (\mathcal S_d) is used for task characterization, teacher acquisition, and optimization; (\mathcal S_c) is reserved exclusively for post-training task-boundary calibration.

---

## 3.2 Task-Aware Teacher Acquisition

Because teacher demonstrations are used as exploration scaffolds rather than training targets, their utility is determined by how effectively they unlock successful student behavior in regions that the current student has not yet mastered.

### Representing task demand

We characterize the target task in the student's parameter space. For each support query (x\in\mathcal S_d), we extract a projected gradient representation (g(x)\in\mathbb R^p). We fit a sparse dictionary

[
D=[d_1,\ldots,d_M]\in\mathbb R^{p\times M}
]

and represent each query through sparse coefficients

[
z(x)
====

\arg\min_z
|g(x)-Dz|_2^2+\lambda_z|z|_1.
]

Aggregating activations over (\mathcal S_d) yields a demand vector

[
r=(r_1,\ldots,r_M),
]

where (r_j) quantifies the extent to which dictionary direction (d_j) is required by the target task. The corresponding active directions span an estimated task-demand subspace (\mathcal C_{\mathcal T}).

This representation also provides a diagnostic for supervision-induced drift. Given an expected update (u), we decompose it by projection,

[
u
=

\Pi_{\mathcal C_{\mathcal T}}u
+
(I-\Pi_{\mathcal C_{\mathcal T}})u,
]

and measure the relative task-extraneous component as

[
R_{\mathrm{ext}}(u)
===================

\frac{
|(I-\Pi_{\mathcal C_{\mathcal T}})u|*2
}{
|\Pi*{\mathcal C_{\mathcal T}}u|_2+\epsilon
}.
]

We use this quantity as an analysis tool rather than assuming that dictionary atoms intrinsically separate capability and distribution mismatch.

### Valuing teacher context

At round (r), let (s_{r,j}) denote the accumulated verified student capability associated with direction (d_j). The remaining task deficit is

[
\delta_{r,j}=(r_j-s_{r,j})_+.
]

For a candidate teacher demonstration (d_x) associated with query (x), let (\hat s_j(d_x)) denote its predicted supply along direction (d_j). Its marginal task coverage is

[
C_r(x)
======

\sum_{j=1}^{M}
\min{\delta_{r,j},\hat s_j(d_x)}.
]

A demonstration is useful, however, only if conditioning on it enables the student to produce successful behavior. We therefore estimate its **unlock probability**

[
\rho_r(x)
=========

\Pr_{\tau\sim
\pi_{\theta_r}(\cdot\mid x,h(d_x))}
[V(x,\tau)=1],
]

using a prior before guided observations are available and updating the estimate from observed guided rollouts thereafter.

We score each candidate by its expected unlocked task coverage per teacher token,

[
U_r(x)
======

\frac{
\hat\rho_r(x)C_r(x)
}{
\hat c(x)
}
-

\lambda_I I_r(x),
]

where (\hat c(x)) is the predicted teacher-token cost and (I_r(x)) estimates potential interference with reference capabilities. We estimate (I_r) using held-out reference-task gradients; the exact estimator and its relation to the (\kappa^r) mechanism are provided in Appendix A.

The capped coverage term introduces diminishing returns as task-demand directions become saturated. We greedily acquire feasible candidates with positive marginal utility until the budget is exhausted or

[
\max_x U_r(x)\le0.
]

Thus, the teacher budget is an upper bound rather than a prescribed expenditure.

---

## 3.3 Teacher-Scaffolded Student Exploration

The acquired demonstrations are not added to the training set. Instead, they are used only when the student cannot obtain verifier reward through its own unguided exploration.

For each query (x), we first draw (K) trajectories from the deployment policy,

[
\tau_{1:K}
\sim
\pi_{\theta_r}(\cdot\mid x).
]

If any trajectory succeeds, collection for that query remains unguided. If all (K) trajectories fail and a teacher demonstration has been acquired, we expose the student to its context (h(d_x)) and sample from the guided behavior policy

[
\mu_r(\tau\mid x)
=================

\pi_{\theta_r}(\tau\mid x,h(d_x)).
]

The resulting trajectory is still generated entirely by the student. The teacher changes the student's **exploration distribution**, but does not supply the trajectory on which the student is optimized.

This distinction follows from a simple property of self-generation. For an unguided student trajectory,

[
\mathbb E_{\tau\sim\pi_\theta(\cdot\mid x)}
[
\nabla_\theta\log\pi_\theta(\tau\mid x)
]
=0.
]

Self-generated behavior therefore has no intrinsic gradient toward an external source distribution. The verifier converts this otherwise zero-mean signal into a task-directed objective. Defining

[
J_x(\theta)
===========

\mathbb E_{\tau\sim\pi_\theta(\cdot\mid x)}
[V(x,\tau)],
]

we obtain

[
\nabla_\theta J_x(\theta)
=========================

\mathbb E_{\tau\sim\pi_\theta}
\left[
V(x,\tau)
\nabla_\theta\log\pi_\theta(\tau\mid x)
\right].
]

Teacher context is therefore needed only to address the cold-start regime in which the student rarely samples trajectories with (V=1).

Because unguided exploration is always attempted first, guidance usage automatically tracks student competence. As successful guided behaviors are internalized, the same queries begin succeeding under (\pi_\theta(\cdot\mid x)) and cease to invoke teacher context. The collection process consequently transitions from guided-heavy to increasingly unguided exploration without an externally specified teacher-annealing schedule.

---

## 3.4 Learning from Student-Provenance Trajectories

We optimize the unguided deployment policy using only trajectories generated by the student.

### Positive verifier signal

For unguided rollouts, the behavior policy coincides with the collection-time deployment policy. Guided trajectories, however, are sampled from

[
\mu_r(\tau\mid x)
=================

\pi_{\theta_r}(\tau\mid x,h(d_x)),
]

while the target policy is (\pi_\theta(\tau\mid x)). Under the standard absolute-continuity condition

[
\pi_\theta(\tau\mid x)>0
;\Rightarrow;
\mu_r(\tau\mid x)>0,
]

the deployment-policy reward gradient can be expressed by importance sampling:

[
\nabla_\theta J_x(\theta)
=========================

\mathbb E_{\tau\sim\mu_r}
\left[
w_\theta(\tau,x)
V(x,\tau)
\nabla_\theta\log\pi_\theta(\tau\mid x)
\right],
]

where

[
w_\theta(\tau,x)
================

\frac{
\pi_\theta(\tau\mid x)
}{
\mu_r(\tau\mid x)
}.
]

In principle, conditioning on teacher context can alter the support of the behavior policy, so the absolute-continuity assumption need not hold universally. We therefore interpret the identity above as the idealized objective under support overlap. In practice, we use the trajectories actually reachable under the behavior policy and clip the importance ratio,

[
\bar w_\theta(\tau,x)
=====================

\min{w_\theta(\tau,x),c},
]

which stabilizes optimization at the cost of bias but does not itself restore missing support.

The positive update is then estimated from the collected student trajectories (\mathcal B_r):

[
\hat g^{+}
==========

\frac{1}{|\mathcal B_r|}
\sum_{(x,\tau)\in\mathcal B_r}
\bar w_\theta(\tau,x)
V(x,\tau)
\nabla_\theta
\log\pi_\theta(\tau\mid x).
]

Teacher-generated trajectories receive zero optimization weight.

### Localized negative signal

Failed student trajectories can provide complementary information when paired with successful trajectories for the same query. We avoid treating an entire failure as negative evidence because successful and unsuccessful trajectories may differ only at a small number of consequential decisions.

Given a successful trajectory (\tau^+) and a failed trajectory (\tau^-) sharing the same environment history, let

[
t^\star
=======

\min{t:a_t^+\neq a_t^-}
]

denote their first action divergence. Preference supervision is applied only from (t^\star) onward.

We additionally require the positive trajectory to be sufficiently learnable. We define its length-normalized negative log-likelihood

[
\ell^+(x)
=========

-\frac{1}{|\tau^+|}
\log\pi_\theta(\tau^+\mid x)
]

and admit the preference pair only when (\ell^+(x)) falls below an exponential-moving-average threshold. This prevents negative updates from dominating before the student can represent the verified alternative.

The final optimization objective combines verifier-weighted positive learning with this localized preference term,

[
\mathcal L(\theta)
==================

\mathcal L_{\mathrm{ver}}(\theta)
+
\lambda_{\mathrm{pref}}
\mathcal L_{\mathrm{pref}}(\theta).
]

After each update, we recompute the remaining task deficits and teacher utilities. Student improvement therefore changes both whether existing guidance is invoked and whether additional guidance is worth purchasing.

---

## 3.5 Training and Deployment

**Training.** Starting from (\pi_{\theta_0}), we alternate between teacher acquisition, student exploration, and student-provenance optimization. Each round first re-estimates the student's remaining task deficits and purchases only teacher contexts with positive expected utility. The student then attempts each query unguided and invokes an acquired context only after unguided failure. Verifier-evaluated student trajectories are used for optimization, after which acquisition utilities are recomputed under the updated student. Training terminates when the unguided student satisfies the convergence criterion, the teacher budget is exhausted, or no remaining teacher context has positive estimated utility. Algorithm 1 and complete implementation details are provided in Appendix A.

**Deployment.** We use the specialized student only within the estimated task region. After training, all model and representation parameters are frozen, and the held-out calibration split (\mathcal S_c) is used to calibrate a fixed nonconformity score (R(x)) derived from the prompt-side task representation. Let (\gamma_\alpha) denote the corresponding split-conformal quantile. Under the standard exchangeability assumption, this construction provides the usual finite-sample marginal coverage associated with split conformal calibration. At inference time, queries satisfying (R(x)\le\gamma_\alpha) are routed to the unguided specialized policy (\pi_{\theta^\star}(\cdot\mid x)); queries outside the calibrated task region are rejected or escalated. Teacher context is never required at deployment.
