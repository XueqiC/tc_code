# Method v2.1 Specification (English, rev. 2, 2026-08-25)

In one sentence, the method spends a metered token budget on teacher
demonstrations that serve as in-context guidance rather than as
imitation targets, lets the student generate its own verified
solutions under that guidance, and trains the student exclusively on
its own verified behavior with a single anchored objective, while the
acquisition rule prices every purchase and stops spending when the
marginal utility becomes non-positive.

## 0. Notation and Protocol Constants

**Task and data.**
Let $\mathcal{T}$ denote the target task with query distribution
$P_\mathcal{T}$. A support set
$S=\{q_1,\dots,q_k\}\sim P_\mathcal{T}$ with $k=50$ is the only
information revealed about the task. The support set is partitioned
once, before any purchase, into a demand split $S_d$ ($|S_d|=40$)
used for demand estimation and acquisition, and a calibration split
$S_c$ ($|S_c|=10$) reserved for boundary calibration and the
regression probe; the two splits stay disjoint throughout.
(Development-stage experiments used the 60 training tasks of the
AppWorld train split without this partition; the frozen protocol
applies the 40/10 partition of $k=50$ and all headline numbers are
rerun under it at freeze.)

A trajectory $\tau=(m_1,m_2,\dots)$ is a multi-turn interaction whose
assistant turns are denoted $y_i$; in AppWorld each assistant turn is
a Python code block executed by the environment, and each environment
turn is the execution output. The verifier $V(q,\tau)\in\{0,1\}$ is
the benchmark's official evaluation of the final environment state.

**Models and budget.**
The teacher $\pi_T$ is a black-box API that returns text only and
charges per emitted token. The student is initialized from an
open-weight base model $\theta_0$; $\theta_r$ denotes the student
after training round $r$. The token budget is $b$; $B_r$ is the
cumulative expenditure after round $r$. Student-side sampling and
demonstration re-execution consume local compute only and are never
charged against $b$.

**Teacher demonstrations.**
A demonstration $d_q$ is one complete teacher trajectory for query
$q$: the task context followed by the teacher's alternating code
turns and the environment's outputs, ending in a verifier-approved
final state (purchased demonstrations are execution-checked on
delivery and rejected ones are logged as waste). Demonstrations are
purchased once and retained; they are re-usable in every later round
at no additional cost.

## 1. Capability Decomposition

**Gradient features.**
For any text sequence $x$ with target segment $y$, the gradient
feature $g(x)$ is the Adam-preconditioned gradient, at the student's
initialization, of the causal language-modeling loss of $y$ given the
preceding context, restricted to a low-rank adapter and compressed by
a fixed random projection. The two views instantiate this with two
different targets. The *response view* $g_r$ takes the solution as
the target given the query as context; it exists only after a
solution is available. The *prompt view* $g_p$ takes the query text
itself as the target with empty context, so the backward pass is
against the autoregressive negative log-likelihood of the query
tokens; no response is required. The prompt view measures what
learning the query's surface statistics would demand and is the
quantity available before any purchase.

**Dictionary fit.**
The two-view coupled dictionary $[D_p;D_r]$ is fit on all examples
whose features are available at fitting time: support queries
(prompt view only), purchased demonstrations (both views), the
student's own verified trajectories as they accumulate (both views),
and reference-task corpora used for coupling estimates (both views,
free external data). An example missing one view contributes only the
reconstruction term of the view it has:

$$\min_{D_p,D_r,C}\ \sum_i \bigl[u^p_i\|x^p_i-c_iD_p\|_2^2+u^r_i\|x^r_i-c_iD_r\|_2^2\bigr]+\lambda\|C\|_1,$$

with $u^p_i,u^r_i\in\{0,1\}$ indicating view availability. Each
paired dictionary row is a *capability atom*, a recurring direction
in gradient-feature space representing a reusable unit of skill; the
sparse code $c(x)$ assigns coefficients over atoms, computed for
unpurchased candidates by a lasso solve against $D_p$ alone. The
dictionary is refit as purchases and verified trajectories
accumulate.

**Readings.**
The *task demand* $w=\frac{1}{|S_d|}\sum_{q\in S_d}|c(q)|$,
normalized and truncated to the atoms carrying 90% of its mass,
induces per-atom token quotas $r_a=w_a\cdot b$. The boundary is the
$D_p$ reconstruction residual thresholded by conformal calibration on
$S_c$ (Section 6). The overlaps $\kappa^p$ and $\kappa^r$ of a
reference task measure query confusability and training interference
respectively and follow the first-order relation
$\Delta L_{\mathrm{off}}\approx-\eta\sum_a S_a\langle d_a,\nabla L_{\mathrm{off}}\rangle$.

## 2. Budgeted Acquisition

**Token-denominated supply.**
Quotas $r_a$ are in tokens, so supply must be too. A purchase or an
admitted trajectory $x$ with token length $t(x)$ supplies atom $a$
with

$$s_a(x)=t(x)\cdot\frac{|c_a(x)|}{\sum_{a'}|c_{a'}(x)|},$$

its token cost amortized over atoms in proportion to its normalized
code. $\hat s_a$ accumulates these contributions.

**Marginal value.**
Every candidate action $a$ is the purchase of a demonstration $d_q$
for a demand-split query $q$ without one, or the decision to stop.
With $\hat t(a)$ the expected demonstration length and $\rho(q)$ the
estimated probability that the guided student produces a verified
solution for $q$ (initialized from the measured aggregate rate,
updated per query after each collection phase),

$$v(a)=\frac{\rho(q)\cdot\sum_{a'}\min\bigl(r_{a'}-\hat s_{a'},\ s_{a'}(q)\bigr)}{\hat t(a)}\;-\;\Bigl(\lambda_{\mathrm{anc}}\Delta_{\mathrm{anchor}}(a)+\sum_{a'\in\mathrm{shared}}s_{a'}(a)\,\kappa^r_{a'}\Bigr),$$

where $s_{a'}(q)$ uses the prompt-view code of $q$ scaled by the
expected verified-trajectory length. The subtrahend is the
*displacement cost*; empirically it does not steer selection when
interference is intrinsic to the task's demanded atoms, and its
operative role is to depress all prices as marginal utility decays so
that expenditure halts endogenously when every action satisfies
$v(a)\le 0$. The $\min(r,\cdot)$ truncation makes the implicit
coverage objective monotone submodular, so greedy selection inherits
the classical near-optimality guarantee.

**Round boundary.**
A round consists of one acquisition phase followed by one collection
phase and one training phase. During acquisition, purchases are made
greedily by $\arg\max v(a)$ with quota bookkeeping after each
purchase, until every remaining action has $v(a)\le 0$; the purchased
batch then goes to collection (Section 3) and training (Section 4) as
a unit. After training, $\rho(\cdot)$ and the quota gaps are
re-estimated under $\theta_{r+1}$; if some action's value returns
above zero, the next round begins, otherwise the procedure
terminates. The number of rounds is therefore not a hyperparameter.

## 3. Demonstration-Conditioned Self-Generation

1. **Guidance construction.** The guidance block $h(d_q)$
   concatenates the final solution-bearing assistant turns of $d_q$
   (consecutive duplicates removed, most recent six turns, capped at
   6{,}000 characters) and is appended to the system prompt with an
   instruction to study the approach and then solve the task step by
   step.
2. **Sampling.** The current student $\theta_r$ draws $k_s=4$
   independent trajectories for $q$ at temperature $T_s=0.7$ under
   the guidance-augmented context (implementation conventions, listed
   in the appendix). In every round, sampling runs over all owned
   demonstrations whose queries lack a verified trajectory, not only
   over newly purchased ones; re-attempts under a stronger student
   are free.
3. **Verification.** Each trajectory is scored by the official
   verifier.
4. **Data admission.** For each query with at least one verified
   trajectory, the *shortest* verified trajectory (fewest assistant
   turns) is retained, one per query, consistent with the single-dose
   saturation of anchors. The guidance block is removed from its
   context so that training contexts match deployment contexts, and
   every assistant turn becomes one training row with provenance
   recorded as student-generated; these rows join
   $\mathcal{A}_{r+1}$. For queries that also have a failed
   trajectory, a preference pair is formed at the *first assistant
   turn where the two trajectories differ*: because the environment
   is deterministic given the action prefix, identical assistant
   prefixes imply identical contexts, so the first differing turn is
   the first point where the two sides share a context and diverge in
   content. The pair joins $\mathcal{P}_{r+1}$ as an
   outcome-contrastive preference pair. (The round-1 implementation
   approximated this with the first assistant turn and discarded
   pairs whose first turns coincided; the first-divergence rule
   replaces it from round 2 onward.) Observed per-query success rates
   update $\rho(q)$.

## 4. Training Objective

The student is trained on $\mathcal{A}_{r+1}\cup\mathcal{P}_{r+1}$
with the single objective

$$\mathcal{L}(x)=\mathrm{NLL}\bigl(y^+\,|\,x\bigr)+\beta\cdot\mathbb{1}\bigl[y^-\text{ exists}\bigr]\cdot\ell_{\mathrm{OR}}\bigl(y^+\!\succ y^-\,|\,x\bigr),$$

with $\ell_{\mathrm{OR}}$ a reference-model-free odds-ratio loss and
$\beta=0.25$ a fixed weight audited in earlier ablations. The
preference term of a row activates only when the row's positive
negative log-likelihood is at or below an exponential moving average
of positive NLLs (decay 0.99), recovering a likelihood-first
curriculum inside one continuous run. The preference loss is computed
only from the first token at which $y^+$ and $y^-$ diverge
(*divergence masking*). Pairs must share the context and differ in
verifier outcome. Training concludes with weight averaging of
near-optimal checkpoints and a rollback guard based on the
*regression probe*: the calibration queries $S_c$ paired with
reference outcomes obtained by re-executing their purchased
demonstrations in the environment, which costs no teacher tokens.
The probe is a degradation detector on held-out queries, not a
generalization estimate; rollback fires only when probe execution
correctness drops beyond a one-question margin. Adapter rank, epochs,
and learning rate follow project-wide conventions listed in the
appendix.

## 5. Iteration and Stopping

Training yields $\theta_{r+1}$; acquisition then reprices all actions
under updated $\rho(\cdot)$ and quota gaps and either opens the next
round or terminates (Section 2). All teacher output tokens are
charged against $b$, including demonstrations and, on benchmarks
where the teacher additionally serves as the user simulator, the
simulator's tokens.

## 6. Certification and Deployment

The boundary predictor uses the $D_p$ reconstruction residual as a
nonconformity score with a conformal threshold calibrated on the
held-out calibration split $S_c$, giving distribution-free coverage
at level $1-\alpha$ over in-task queries. Out-of-task behavior is
evaluated on reference-task corpora, which are free external data and
never touch the budget; leakage is reported as a finite-sample
Clopper-Pearson upper bound, and in-task coverage as the
corresponding lower bound. At deployment the same boundary routes
in-specification queries to the student and refuses or escalates the
remainder.

## 7. Empirical Status (2026-08-25)

Guided self-generation raises the base student's verified-solution
rate on AppWorld training tasks from roughly 3% to 32.9%±2.8% across
four sampling seeds, covering 31 of 60 queries and yielding 400
anchor rows and 21 outcome-contrastive pairs. Round-1 training with
the Section 4 objective is in progress against the previous anchored
result of 0.284±0.037 over the base student's 0.248±0.035. Not yet
wired in: the online $\rho(q)$ estimate in pricing, the
first-divergence pair rule (round 2), and the 40/10 support
partition of the frozen protocol.
