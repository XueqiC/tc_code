# Method v3.0. Budgeted Distillation Governed by the Supervision Decomposition (English, 2026-08-25)

In one sentence, we show that the expected parameter update induced by
any supervision source decomposes into a capability component and a
distribution-mismatch component, that the only mismatch-free channel
is verifier-approved self-generation, and we therefore spend the
entire token budget on contexts that raise the student's verified
self-generation rate, train with an off-policy-corrected
verifier-reward policy gradient equipped with a protected negative
channel, and let rounds, stopping, and annealing all emerge
endogenously.

## 0. Notation and Protocol Constants

**Task and data.** The target task $\mathcal{T}$ has query
distribution $P_\mathcal{T}$. A support set
$S=\{q_1,\dots,q_k\}\sim P_\mathcal{T}$ with $k=50$ is the only
information revealed about the task. Before any purchase, $S$ is
partitioned once into a demand split $S_d$ ($|S_d|=40$) for demand
estimation and acquisition and a calibration split $S_c$ ($|S_c|=10$)
for boundary calibration and the regression probe; the splits remain
disjoint. A trajectory $\tau$ is a multi-turn interaction with
assistant turns $y_i$; in AppWorld an assistant turn is a code block
and an environment turn is its execution output. The verifier
$V(q,\tau)\in\{0,1\}$ is the benchmark's official evaluation.
(Development experiments used the 60-task train split without the
partition; headline numbers are rerun under the frozen 40/10 protocol
at freeze.)

**Models and budget.** The teacher $\pi_T$ is a black-box API that
returns text only and charges per emitted token. The student is
initialized from an open-weight base model $\theta_0$; $\theta_r$ is
the student after round $r$. The budget is $b$; student-side sampling
and demonstration re-execution consume local compute only and are
never charged.

**Demonstrations and guidance.** A demonstration $d_q$ is one
complete teacher trajectory for $q$, execution-checked on delivery,
purchased once and retained for free reuse in all later rounds. The
guidance block $h(d_q)$ concatenates the final solution-bearing
assistant turns of $d_q$ (consecutive duplicates removed, most recent
six turns, capped at 6{,}000 characters) and is appended to the
system prompt.

**Collection and training constants.** Samples per query $k_s=4$ and
temperature $T_s=0.7$ are implementation conventions listed in the
appendix. $\mathcal{A}_r$ denotes the *anchor set*, the training rows
derived from the student's own verifier-approved trajectories;
$\mathcal{P}_r$ the set of *outcome-contrastive preference pairs*;
$\rho(q)$ the online estimate of the guided verified-solution
probability; $\beta=0.25$ the audited preference weight; the gating
statistic is an exponential moving average with decay 0.99.

**Measurement instrument.** For a text sequence with target segment
$y$ given context $x$, the gradient feature is the
Adam-preconditioned low-rank-adapter gradient of the causal
language-modeling loss of $y$ given $x$ at the student's
initialization, compressed by a fixed random projection. The
*response view* $g_r$ takes the solution as target given the query;
the *prompt view* $g_p$ takes the query text itself as target with
empty context, so it is computable before any purchase. A two-view
coupled dictionary $[D_p;D_r]$ is fit over all examples with
available features, with a view-availability mask so that examples
missing one view contribute only the term of the view they have. Each
paired dictionary row is a *capability atom*, a recurring direction
in gradient-feature space representing a reusable unit of skill.
Sparse codes $c(x)$, demand $w$, token quotas $r_a=w_a b$, the
token-amortized supply $s_a(x)=t(x)\,|c_a(x)|/\sum_{a'}|c_{a'}(x)|$,
and reference-task overlaps $\kappa^p$ (confusability) and $\kappa^r$
(interference) are all read from this dictionary.

## 1. Core Mechanism. The Supervision Decomposition

For any supervision source, the expected parameter update of the
student decomposes into two orthogonal components, a *capability
component* supported on the task's demanded atoms and a *mismatch
component* supported on atoms of distributional discrepancy between
the supervision source and the student. Three statements form the
mechanism.

**(i) Imitation targets pay a mismatch tax.** When training targets
are drawn from the teacher's output distribution $q_T$, the expected
update points along $\nabla\,\mathrm{KL}(q_T\,\|\,\pi_\theta)$. Its
mismatch component grows with the teacher-student distribution
distance and dominates for small students, because surface
regularities carry the strongest and most easily reduced loss signal
and are absorbed first, overwriting the student's own distribution.

**(ii) Self-generated targets induce zero drift.** When targets are
drawn from the student's own distribution $y\sim\pi_\theta$, the
score-function expectation vanishes,
$\mathbb{E}\bigl[\nabla\,\mathrm{NLL}(y|x)\bigr]=0$; such data moves
the parameters nowhere in expectation, so the mismatch component is
zero by construction.

**(iii) Verifier filtering tilts the drift exactly toward task
success.** Training only on self-generated samples with $V=1$ makes
the expected update equal to
$\nabla\,\mathbb{E}_{y\sim\pi_\theta}[V(y)]$, the REINFORCE identity.
The only systematic drift is the gradient of the task success rate;
the capability component is maximal and the mismatch component
remains zero.

The dictionary renders the decomposition observable, since the
mismatch component lives on mismatch atoms and the capability
component on demanded atoms; their norm ratio is measurable for any
supervision source and predicts the damage ordering of supervision
formats. The first-order coupling relation
$\Delta L_{\mathrm{off}}\approx-\eta\sum_a S_a\langle d_a,\nabla L_{\mathrm{off}}\rangle$
is this decomposition written out along an off-task direction.

Three corollaries fix the shape of the method. First, external
behavior may enter only as context and never as a gradient target.
Second, the admissible update family is the off-policy-corrected
policy gradient on verifier-approved self-generated samples, with a
protected negative channel. Third, the budget should purchase only
contexts that raise $\mathbb{E}[V]$, because targets are taxed and
contexts are tax-free.

## 2. Measurement and Demand

One backward pass per support query yields $g_p$. The dictionary is
fit on everything with available features at the time, support
queries (prompt view only), purchased demonstrations (both views),
accumulated verified self-trajectories (both views), and free
reference-task corpora (both views, for coupling estimates), and is
refit as evidence accumulates. Its three readings are the quotas
$r_a=w_a b$ with $w=\frac{1}{|S_d|}\sum_{q\in S_d}|c(q)|$ normalized
and truncated to 90% mass, the boundary given by the $D_p$
reconstruction residual with a conformal threshold calibrated on
$S_c$, and the coupling forecasts $\kappa^p,\kappa^r$.

## 3. Budgeted Acquisition

Candidate actions are the purchase of a demonstration for a
demand-split query that lacks one, or stopping. With $\hat t(a)$ the
expected demonstration length,

$$v(a)=\frac{\rho(q)\cdot\sum_{a'}\min\bigl(r_{a'}-\hat s_{a'},\ s_{a'}(q)\bigr)}{\hat t(a)}-\Bigl(\lambda_{\mathrm{anc}}\Delta_{\mathrm{anchor}}(a)+\sum_{a'\in\mathrm{shared}}s_{a'}(a)\,\kappa^r_{a'}\Bigr).$$

Supply is token-amortized throughout. The displacement term does not
steer selection when interference is intrinsic to the task's demanded
atoms; its operative role is to depress all prices as marginal
utility decays so that expenditure halts endogenously at
$v(a)\le 0$ for every action. The $\min(r,\cdot)$ truncation makes
the implicit coverage objective monotone submodular, so greedy
selection inherits the classical near-optimality guarantee. A round
consists of one acquisition phase run to exhaustion, one collection
phase, and one training phase; re-pricing under $\theta_{r+1}$ either
opens the next round or terminates the procedure, so the number of
rounds is not a hyperparameter.

## 4. Collection and Update

**Graduated collection.** Each round samples first from the unguided
policy $\pi_\theta(\cdot|x)$; the guidance block is used only for
queries whose unguided attempts all fail. As the student strengthens,
queries graduate one by one to purely on-policy collection, the
guidance share decays to zero without any annealing constant, and the
collection distribution converges to the deployment distribution.

**Admission.** For each query the shortest verified trajectory is
retained, one per query, consistent with single-dose anchor
saturation. The guidance block is removed from its context so that
training contexts match deployment contexts, and each assistant turn
becomes one training row of student provenance in
$\mathcal{A}_{r+1}$. For queries that also have a failed trajectory,
the preference pair is formed at the first assistant turn where the
two trajectories differ, which is the first shared-context divergence
point because the environment is deterministic given the action
prefix; the pair joins $\mathcal{P}_{r+1}$. Observed success rates
update $\rho(q)$.

**Update rule.** Training minimizes the single objective

$$\mathcal{L}=-\,\mathbb{E}\bigl[w(\tau)\cdot\mathbb{1}[V{=}1]\cdot\log\pi_\theta(\tau|x)\bigr]+\beta\cdot\mathbb{E}\bigl[g\cdot\ell_{\mathrm{OR}}\bigl(y^+\!\succ y^-|x\bigr)\bigr],$$

whose first term is the verifier-reward policy gradient with the
truncated importance ratio
$w(\tau)=\min\bigl(\pi_\theta(\tau|x)/\pi_\theta(\tau|x,h),\,1\bigr)$
correcting guided collection; both densities are the student's own,
so the black-box constraint is untouched, and truncation at one
follows the V-trace convention. The second term is the protected
estimator of the negative direction. A naive group-baseline policy
gradient presses failed trajectories down uniformly over tokens and
reintroduces the mismatch tax; the protections restrict the pressure
to rows whose positive is already learnable (the gate $g$ activates
when the positive NLL is at or below the moving average), to tokens
from the first divergence onward (divergence masking), and to pairs
that share context and differ in verifier outcome
(outcome-contrastive admissibility). Training ends with weight
averaging of near-optimal checkpoints and a rollback guard on the
regression probe, the calibration queries $S_c$ paired with reference
outcomes obtained by re-executing their demonstrations at zero
teacher cost; rollback fires only beyond a one-question margin.

## 5. Iteration and Stopping

After training, re-pricing uses the updated $\rho$ and quota gaps;
the loop continues while any action has positive value and terminates
otherwise. All teacher output tokens are charged, including
demonstrations and simulator tokens on benchmarks where the teacher
doubles as the user simulator. In the limit, once every query has
graduated, the loop reduces to pure on-policy verifier-reward policy
gradient; guidance is a cold-start scaffold and nothing more.

## 6. Certification and Deployment

The boundary uses the $D_p$ residual with a conformal threshold from
$S_c$, giving distribution-free in-task coverage at level $1-\alpha$.
Out-of-task behavior is evaluated on free reference-task corpora and
reported as a finite-sample Clopper-Pearson upper bound on leakage,
with the corresponding lower bound on in-task coverage. The same
boundary routes deployment traffic, sending in-specification queries
to the student and refusing or escalating the rest.

## 7. Evidence and Predictions

Existing results restate as consequences of the mechanism: the
cloning floor of 0.197-0.215 invariant to data selection, scale, and
volume (the mismatch tax is a property of the target distribution);
the provenance-purity monotonicity 0.284/0.245/0.202; single-dose
anchor saturation (a stationarity property, independent of dose); the
channel contrast of 33% guided self-generation success against 0.215
from cloning the same demonstrations; BB-OPD at 0.197 (repairing the
state distribution does not reduce the tax); the collapse of ungated
preference (0.630) and of style-axis pairs (−0.087) as the negative
channel's mismatch tax; and the failure of displacement-based
selection alongside the success of endogenous stopping. Pending
direct validations: **T1**, the mismatch-to-capability norm ratio of
cloning versus self-generated gradients under the dictionary
projection; **T2**, a sweep of the imitation-target fraction from 0
to 1 with damage predicted monotone in the mismatch norm; **T3**,
predicting the observed damage ordering of the seven baselines from
round-0 mismatch measurements. In flight: round-1 training (three
seeds); to be wired: online $\rho$ pricing, first-divergence pairs,
the 40/10 split, the $w(\tau)$ correction, graduated collection, and
the naive-group-baseline control.

## 8. Relation to Prior Work

Imitation's false promise has been observed (Gudibande et al., 2023),
distribution gaps have been patched by student rewriting (SDFT, Yang
et al., 2024), context distillation names the compression of context
into weights (Snell et al., 2022), and ReST-style loops provide
batch generate-filter-train scaffolding. None of these state the
orthogonal decomposition of supervision updates with a measurement
instrument, the zero-drift property of self-generated targets
together with the verifier-tilt identity as a design principle, the
derivation of channel, update, and budget rules from that principle,
or the prediction of damage orderings across supervision formats from
measured mismatch components. The claimed novelty is the mechanism
and its measurability; the components are its consequences.
