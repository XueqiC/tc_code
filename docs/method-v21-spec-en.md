# Method v2.1 Specification (English, 2026-08-25)

In one sentence, the method spends a metered token budget on teacher
demonstrations that serve as in-context guidance rather than as
imitation targets, lets the student generate its own verified
solutions under that guidance, and trains the student exclusively on
its own verified behavior with a single anchored objective, while the
acquisition rule prices every purchase and stops spending when the
marginal utility becomes non-positive.

## 0. Notation

**Task and data.**
Let $\mathcal{T}$ denote the target task with query distribution
$P_\mathcal{T}$. A support set
$S=\{q_1,\dots,q_k\}\sim P_\mathcal{T}$ with $k=50$ is the only
information revealed about the task. A trajectory
$\tau=(m_1,m_2,\dots)$ is a multi-turn interaction whose assistant
turns are denoted $y_i$. The verifier $V(q,\tau)\in\{0,1\}$ is the
benchmark's official evaluation of trajectory $\tau$ on query $q$.

**Models and budget.**
The teacher $\pi_T$ is a black-box API that returns text only and
charges per emitted token. The student is initialized from an
open-weight base model $\theta_0$; $\theta_r$ denotes the student
after training round $r$. The token budget is $b$, and $B_r$ is the
cumulative expenditure after round $r$.

**Capability decomposition.**
For any example $x$, the gradient feature $g(x)\in\mathbb{R}^d$ is
the Adam-preconditioned gradient of a low-rank adapter at the
student's initialization, compressed by a fixed random projection.
The prompt-view feature $g_p$ is computed from the query text alone;
the response-view feature $g_r$ requires a solution. A two-view
coupled dictionary $[D_p;D_r]$ is learned over both views. We call
each paired row of this dictionary a *capability atom*, a recurring
direction in gradient-feature space that represents a reusable unit
of skill. The sparse code $c(x)$ assigns each example a coefficient
vector over atoms; for unpurchased candidates it is obtained by a
lasso solve against the $D_p$ block only. The *task demand*
$w=\frac{1}{k}\sum_{q\in S}|c(q)|$, normalized and truncated to the
atoms carrying 90% of its mass, induces per-atom token quotas
$r_a=w_a\cdot b$; $\hat s_a$ denotes the supply of atom $a$ purchased
so far. For a reference task, $\kappa^p$ and $\kappa^r$ denote its
atom overlap with $\mathcal{T}$ in prompt view and response view
respectively; the former measures query confusability, the latter
measures training interference.

**Collection and training.**
For a query $q$, $d_q$ is a teacher demonstration and $h(d_q)$ is the
guidance block extracted from it (defined in Section 3). The student
draws $k_s$ independent trajectories per query at sampling
temperature $T_s$; both are implementation conventions, currently
$k_s=4$ and $T_s=0.7$, listed in the appendix. The *anchor set*
$\mathcal{A}_r$ is the collection of training rows derived from the
student's own verifier-approved trajectories available at round $r$;
we refer to its elements as anchors because their presence in the
training mixture preserves the student's output distribution during
fine-tuning. The pair set $\mathcal{P}_r$ contains
*outcome-contrastive preference pairs*, pairs of same-context
responses whose verifier outcomes differ. The training objective
$\mathcal{L}$, its preference weight $\beta$, and the gating
statistic are defined in Section 4.

## 1. Capability Decomposition

One backward pass per support query yields $g_p(q)$; response-view
features are computed for every example whose solution is available.
The dictionary is fit by

$$\min_{D_p,D_r,C}\ \|X_p-CD_p\|_2^2+\|X_r-CD_r\|_2^2+\lambda\|C\|_1,$$

so a single sparse code per example must reconstruct both of its
views. Sparsity is what makes the decomposition identifiable, since a
principal subspace is rotation-invariant and cannot attach identity
to individual directions. The fitted dictionary supplies three
quantities consumed later. First, the demand vector $w$ and the
quotas $r_a$ determine how much supervision each capability warrants.
Second, the reconstruction residual of a prompt under $D_p$,
thresholded at the least typical calibration query, defines the task
boundary; out-of-task prompts receive near-zero code mass on demanded
atoms and therefore never acquire positive value in Section 2.
Third, the overlaps $\kappa^p$ and $\kappa^r$ forecast, before any
expenditure, how training on $\mathcal{T}$ will move a reference
task, following the first-order relation
$\Delta L_{\mathrm{off}}\approx-\eta\sum_a S_a\langle d_a,\nabla L_{\mathrm{off}}\rangle$.

## 2. Budgeted Acquisition

Every candidate action $a$ is the purchase of a demonstration $d_q$
for some support query $q$, to be used as in-context guidance, or the
decision to stop. Each action is assigned the marginal value

$$v(a)=\frac{\rho(q)\cdot\sum_c\min\bigl(r_c-\hat s_c,\ c_c(q)\bigr)}{\hat t(a)}\;-\;\Bigl(\lambda_{\mathrm{anc}}\Delta_{\mathrm{anchor}}(a)+\sum_{c\in\mathrm{shared}}\hat s_c(a)\,\kappa^r_c\Bigr),$$

where $\hat t(a)$ is the expected token cost of the demonstration,
$\rho(q)$ is the estimated probability that the student produces a
verified solution for $q$ when guided (initialized from the measured
aggregate rate and updated online per query), and the subtrahend is
the *displacement cost*, an estimate of the harm that additional
purchases inflict on capabilities the student already holds, through
anchor dilution $\Delta_{\mathrm{anchor}}$ and response-view coupling
$\kappa^r$. Empirically the displacement cost does not steer
selection when interference is intrinsic to the task's own demanded
atoms; its operative role is to lower all prices as marginal utility
decays, so that expenditure halts endogenously once every action
satisfies $v(a)\le 0$. Acquisition greedily executes
$\arg\max_a v(a)$ and updates quotas after each purchase. The
$\min(r,\cdot)$ truncation makes the implicit coverage objective
monotone submodular, so greedy selection inherits the classical
near-optimality guarantee.

## 3. Demonstration-Conditioned Self-Generation

For each purchased demonstration the student, not the teacher,
produces the training data.

1. **Guidance construction.** The guidance block $h(d_q)$
   concatenates the final solution-bearing assistant turns of $d_q$
   (consecutive duplicates removed, most recent six turns, capped at
   6{,}000 characters) and is appended to the system prompt with an
   instruction to study the approach and then solve the task
   step by step.
2. **Sampling.** The current student $\theta_r$ draws $k_s$
   independent trajectories for $q$ at temperature $T_s$ under the
   guidance-augmented context, with full trajectories recorded.
3. **Verification.** Each trajectory is scored by the official
   verifier $V(q,\tau_j)$.
4. **Data admission.** For each query with at least one verified
   trajectory, one such trajectory is retained. The guidance block is
   removed from its context, so that training contexts match
   deployment contexts, and every assistant turn becomes one training
   row whose provenance is recorded as student-generated; these rows
   join $\mathcal{A}_{r+1}$. For queries that also have a failed
   trajectory, the pair of first assistant turns with differing
   verifier outcomes joins $\mathcal{P}_{r+1}$ as an
   outcome-contrastive preference pair; both sides come from the same
   policy under the same context, so the pair differs on the outcome
   axis alone. The observed per-query success rate updates
   $\rho(q)$ for the next round of pricing.

## 4. Training Objective

The student is trained on
$\mathcal{A}_{r+1}\cup\mathcal{P}_{r+1}$ with the single objective

$$\mathcal{L}(x)=\mathrm{NLL}\bigl(y^+\,|\,x\bigr)+\beta\cdot\mathbb{1}\bigl[y^-\text{ exists}\bigr]\cdot\ell_{\mathrm{OR}}\bigl(y^+\!\succ y^-\,|\,x\bigr),$$

where $\ell_{\mathrm{OR}}$ is a reference-model-free odds-ratio
preference loss and $\beta=0.25$ is a fixed weight audited in earlier
ablations. Four execution rules apply. The preference term on a row
activates only when the row's positive-response negative
log-likelihood is at or below an exponential moving average of
positive NLLs (decay 0.99), which recovers the ordering of a
likelihood-first curriculum inside one continuous run. The preference
loss is computed only from the first token at which $y^+$ and $y^-$
diverge, so the shared prefix receives no gradient pressure; we call
this *divergence masking*. A pair is admissible only if its two sides
share the context and differ in verifier outcome. Training concludes
with weight averaging of near-optimal checkpoints and a rollback that
fires only when a held-out probe, whose gold answers are obtained for
free by executing purchased demonstrations, degrades beyond a
one-question margin. Adapter rank, epochs, and learning rate follow
project-wide conventions listed in the appendix.

## 5. Iteration and Stopping

Training yields $\theta_{r+1}$, after which acquisition resumes with
updated $\rho(q)$ and shrunken quota gaps. Because a stronger student
converts more demonstrations into verified solutions while unmet
demand decays, the number of rounds is not a hyperparameter; the
procedure stops when every action's marginal value is non-positive.
All teacher output tokens are charged against $b$, including
demonstrations and, on benchmarks where the teacher additionally
serves as the user simulator, the simulator's tokens.

## 6. Certification and Deployment

A deployment-time request exposes only its prompt, so the boundary
predictor reuses the prompt-view machinery of Section 1. The
reconstruction residual under $D_p$ serves as the nonconformity
score, and a conformal threshold calibrated on a disjoint split gives
distribution-free coverage at level $1-\alpha$. The delivered student
is evaluated behaviorally on both sides of the boundary, and the
capability match is reported as finite-sample Clopper-Pearson bounds,
a certified lower bound on in-task coverage and a certified upper
bound on out-of-task leakage. At deployment the same boundary routes
in-specification queries to the student and refuses or escalates the
remainder.

## 7. Empirical Status

Guided self-generation raises the base student's verified-solution
rate on AppWorld training tasks from roughly 3% to 32.9%±2.8% across
four sampling seeds, covering 31 of 60 queries and yielding 400
anchor rows and 21 outcome-contrastive pairs. Round-1 training with
the objective of Section 4 is in progress against the previous
anchored result of 0.284±0.037 over the base student's 0.248±0.035.
The online estimate $\rho(q)$ is not yet wired into acquisition
pricing, and multi-round iteration awaits the round-1 verdict.
