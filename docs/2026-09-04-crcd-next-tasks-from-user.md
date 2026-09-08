Implement and Validate the Updated CRCD on AppWorld, ALFWorld, and BFCL

You are the primary research and coding agent for the CRCD project. Your task is to update the repository so that the method, implementation, experiments, and paper documentation all reflect the new unified formulation below, and then validate the same method on AppWorld, ALFWorld, and BFCL.

The three benchmarks play different scientific roles:

ALFWorld is the controlled transition benchmark: the existing base student is initially unskilled, prior CE acquires the behavior, and a later residual round tests what happens after the student becomes competent.

BFCL is the preservation benchmark: the base already has substantial useful behavior and exposes call/abstain, multi-turn, and memory boundary damage under aggressive imitation.

AppWorld is the long-horizon, stateful, teacher-expensive benchmark: it tests whether capability discovery and acquisition actually reduce black-box teacher tokens when continuation length is a decision variable.

Do not write three benchmark-specific methods. Environment adapters and official utilities may differ, but atom discovery, acquisition scoring, constrained mirror projection, dual updates, stopping, and certification must use the same rules.

Do not treat the following method as already validated. The current experiments motivate its design, but the new capability dictionary, acquisition objective, and unified distillation operator still require direct tests. Use explicit Go/No-Go criteria and preserve negative results.

0. Required Working Discipline

Before changing code:

Read all repository-level instructions, especially AGENTS.md, PROJECT_STATE*, the canonical method document, experiment plans, result ledgers, and recent run logs.

Inspect the current git status and preserve unrelated or user-authored changes.

Locate the current BFCL and ALFWorld pipelines, model wrappers, teacher clients, LoRA trainers, gradient-fingerprint code, drift diagnostics, event miners, evaluation scripts, and SLURM or shell launchers. Treat the existing ALFWorld evidence as an empirical regression suite, not merely background prose.

Determine whether AppWorld and its data are already installed. Do not assume they are available.

Produce a short audit before implementation covering:

reusable components;

missing AppWorld infrastructure;

current model and teacher configurations;

exact available GPUs and storage;

estimated teacher-token and compute costs;

any mismatch between this prompt and the repository.

Implement in small testable stages. Run unit tests and smoke tests before launching expensive teacher queries or official evaluations.

All result records must contain the git commit, config, random seed, input split, training-event count, optimizer steps, per-event repetition count, exact teacher-output tokens, student rollout count, and checkpoint path.

Never tune on AppWorld test_normal or test_challenge. Never inspect test-task details or task-wise test reports. Use train for teaching/data construction, dev for development and tuning, and test only once the method and hyperparameters are frozen.

For ALFWorld, separate development from final reporting. Because valid_unseen has already been used repeatedly in prior work, document that history, tune new components on train/valid_seen or a newly frozen development subset, and use valid_unseen only for predeclared final confirmation of the revised method.

AppWorld's official repository and paper are:

https://github.com/StonyBrookNLP/appworld

https://arxiv.org/abs/2407.18901

Respect AppWorld's release and anti-leakage restrictions. Do not expose encrypted task code, hidden evaluation content, test task details, or extracted ground-truth solutions in logs committed to the repository.

ALFWorld's official project and paper are:

https://github.com/alfworld/alfworld

https://arxiv.org/abs/2010.03768

Use the official environment and split conventions. Do not expose simulator-only information, admissible-command oracles, or privileged state to the model unless that information is part of the declared agent interface for every baseline.

1. Authoritative Problem Definition

The input is:

$$
(\mathcal S_{\mathrm{few}},\pi_0,T,\mathcal V,B),
$$

where:

$\mathcal S_{\mathrm{few}}$ is a small set of examples or task instances specifying the target task;

$\pi_0=\pi_{\theta_0}$ is a pretrained small language model;

the fixed agent scaffold turns generations from $\pi_\theta$ into tool calls or executable actions;

$T$ is a black-box teacher returning text only, with no logits or gradients;

$\mathcal V$ is an executable environment or verifier returning bounded trajectory utility;

$B$ is the total number of teacher-generated output tokens permitted during bootstrap, acquisition, and teacher-side verification.

The learned object is a small language model $\pi_{\theta^*}(y\mid s)$. Under the fixed scaffold, it induces the agent policy. The objective is:

\arg\max_\theta
\mathbb E_{x\sim p_{\mathrm{task}},\tau\sim\pi_\theta(\cdot\mid x)}
[R(\tau)]
$$

subject to:

$$
\operatorname{TeacherTokens}\le B.
$$

At deployment, the student must operate without the teacher. The output should also include a certificate describing representation coverage, atom-wise residual risk, and selective deployment risk.

The method has four coupled blocks sharing the same gradient-space capability representation:

Capability Atom Discovery

Budgeted Data Acquisition

Capability-Constrained Mirror Distillation

Certification

Do not replace this formulation with a semantic capability checklist, a hard CE-versus-DPO router, raw $\Delta U$ weighting, or a simple first-divergence pipeline.

1.1 Why standard distillation is insufficient here

The updated Method and Introduction must derive the algorithm from four observed failures:

Full-trajectory or local CE assumes every teacher token should be copied absolutely. This can install a missing policy, as in low-skill ALFWorld, but it globally changes token probabilities and overwrites mastered behavior in BFCL and in over-repeated residual rounds.

Pairwise correction assumes the useful behavior already exists in the student's support. It can repair a local margin while preserving the base, but it cannot reliably install a missing action language, plan structure, or long-horizon behavior; this explains its low first-round ALFWorld ceiling.

First divergence and raw outcome heuristics confuse disagreement, causal usefulness, and generalizable imitation. First divergence is often harmless; hard $\Delta U>0$ filtering discards zero-estimated events that still teach reusable behavior; raw $\Delta U$ magnitude fails permutation controls.

Uniform full demonstrations ignore teacher cost, sequential reachability, and what the student already knows. They spend the budget on redundant tokens and cannot say which capability remains unresolved or whether the acquired evidence covers deployment.

The new method therefore treats teacher text as evidence for capability constraints, not as labels that automatically determine a loss. Sparse differential-gradient atoms define what behavior an event can move; acquisition buys the evidence expected to reduce unresolved atom risk per teacher token; constrained mirror projection makes the smallest policy change satisfying those constraints; certification reports which constraints are supported and met.

2. Shared Notation

Use the following symbols consistently in code comments, documentation, saved schemas, and the paper.

Symbol

Meaning

$x_i$

Complete task instance

$s_i$

Full agent decision state, including task and interaction history

$y_i^S$

Student continuation at $s_i$

$y_i^T$

Teacher continuation at $s_i$

$Q_i^S,Q_i^T$

Posterior expected terminal utilities after student or teacher intervention

$\psi_i$

Fisher-whitened teacher-student differential gradient fingerprint

$u_k$

The $k$-th capability atom vector

$U=[u_1,\ldots,u_K]$

Capability dictionary

$z_i$

Sparse atom composition of event $i$

$J_k(\pi)$

Student performance on atom-weighted probes

$\rho_k$

Required target performance for atom $k$

$r_k=[\rho_k-J_k]_+$

Capability residual

Do not call an event fingerprint, hard cluster, or entire subspace a capability atom. Each column $u_k$ is an atom; $U$ is the dictionary; $z_i$ is an event composition; and $\operatorname{span}(U)$ is the capability subspace.

Do not use $y^+$ or $y^-$ to identify teacher and student. Teacher behavior is not assumed to be better. Use ownership superscripts $T$ and $S$.

3. Block I: Capability Atom Discovery

3.1 Event construction

An event is:

$$
e_i=(s_i,y_i^S,y_i^T,\mathcal O_i),
$$

where $\mathcal O_i$ stores matched continuation outcomes for the student and teacher interventions.

For BFCL, the continuation unit should match the existing response or decision unit used by the official evaluator. For ALFWorld, use one textual environment action as the basic decision continuation and allow a logged multi-action teacher bridge. For AppWorld, use one agent generation/executable code block as the basic decision continuation. A longer teacher recovery bridge may contain several decision units, but its boundaries must be logged.

3.2 Gradient fingerprint

All gradients are taken through the student. The teacher remains black-box.

At the fixed base checkpoint $\theta_0$, compute:

\nabla_\theta\log\pi_\theta(y_i^S\mid s_i)
\big|_{\theta=\theta_0},
$$

\nabla_\theta\log\pi_\theta(y_i^T\mid s_i)
\big|_{\theta=\theta_0},
$$

F^{-1/2}(g_i^T-g_i^S),
$$

\frac{\widetilde\psi_i}
{|\widetilde\psi_i|_2+\varepsilon}.
$$

Use the student model's trainable LoRA parameters unless profiling shows that another fixed parameter subset is required. Implement:

length-normalized continuation log-probability;

a diagonal empirical Fisher estimated on target-task student rollouts;

a deterministic fixed random projection or sketch when the full gradient is too large;

metadata sufficient to reproduce each fingerprint;

versioning so fingerprints from incompatible model/tokenizer/scaffold versions cannot be mixed.

Do not use raw unnormalized gradients as the main method. Retain them only as a baseline.

3.3 Sparse dictionary

Fit:

$$
\min_{U,Z}
\frac{1}{2n}|\Psi-UZ|_F^2
+
\lambda_z|Z|_1,
$$

subject to:

$$
|u_k|_2=1.
$$

Implement soft sparse coding, not only PCA or k-means. Each event may load on multiple atoms. Select $K$ and $\lambda_z$ using train-to-dev held-out reconstruction and transfer prediction. Do not tune them on official test data.

Also implement these baselines using the same fingerprint input and dimensionality:

PCA/SVD components without sparsity;

k-means on unwhitened gradients;

k-means on Fisher-whitened gradients;

frozen semantic embeddings of states/responses;

task/category labels available only for analysis;

random normalized dictionaries;

the existing A1/A2/A3 fingerprints.

3.4 Transfer prediction

Do not define atoms using a hard within-cluster versus cross-cluster gate. Test the continuous first-order prediction:

\eta z_j^\top U^\top U z_i.
$$

For a controlled small update on event $i$, measure the actual held-out change on event $j$. Evaluate:

Spearman correlation over off-diagonal pairs;

column-centered and row-centered Spearman;

NDCG@1, @3, and @5 for identifying the events with greatest positive transfer;

mean squared error after fitting one scalar calibration coefficient on train only;

AUROC or balanced accuracy for the sign of transfer;

bootstrap confidence intervals and a permutation test;

partial correlation controlling for benchmark category, response length, state depth, and same-side boundary membership.

The sparse dictionary is considered empirically meaningful only if it improves out-of-sample transfer prediction over PCA, raw gradient similarity, semantic/category baselines, and random dictionaries. A positive correlation alone is not sufficient if it is fully explained by task categories.

3.5 Atom inference and residuals

Fit an amortized posterior:

$$
q_\phi(z\mid s,y^S)
$$

to predict atom loading for a state before querying the teacher. Use an ensemble or another calibrated uncertainty estimator. Keep this predictor small and separate from the student policy.

For atom relevance $\bar z_{ki}=\mathbb E[|z_{ki}|]$, compute:

\frac{
\sum_i\bar z_{ki}
\mathbb E_{\tau\sim\pi(\cdot\mid s_i)}[R(\tau)]
}{
\sum_i\bar z_{ki}+\varepsilon
},
$$

\frac{1}{|\mathcal P|}
\sum_i\bar z_{ki},
$$

\operatorname{LCB}
\left[
\frac{
\sum_i\bar z_{ki}Q_i^T
}{
\sum_i\bar z_{ki}+\varepsilon
}
\right],
$$

$$
r_k(\pi)=[\rho_k-J_k(\pi)]_+.
$$

Store posterior uncertainty, effective atom-weighted sample size, and the provenance of every probe contributing to these quantities.

4. Block II: Budgeted Data Acquisition

4.1 Query definition

A teacher query is:

$$
q=(s,L),
$$

where $s$ is a complete agent state and $L$ is a requested continuation length or granularity.

For AppWorld, support at least:

one executable code block;

a short bridge of up to a fixed number of agent interactions;

a complete teacher trajectory until complete_task or the maximum interaction limit.

For ALFWorld, support at least:

one teacher action at a student-visited state;

a short teacher bridge of bounded action length;

a complete verified demonstration or recovery trajectory.

For BFCL, preserve the benchmark's existing response units and define analogous local versus full-response acquisition only where meaningful.

4.2 Counterfactual outcome posterior

For every acquired event, estimate:

\mathbb E[R\mid do(y_i^S),s_i],
$$

\mathbb E[R\mid do(y_i^T),s_i].
$$

Use matched downstream continuation policies and shared randomness where the environment supports it. Save all branch seeds and state-reconstruction identifiers.

Treat:

$$
\Delta U_i=Q_i^T-Q_i^S
$$

as a derived posterior random variable only. Do not:

hard-filter all events at $\Delta U>0$;

multiply the training loss by raw $\Delta U$;

treat $\Delta U=0$ as useless;

convert $\Delta U<0$ into an automatic reverse preference.

4.3 Acquisition objective

Define posterior residual risk:

\sum_kd_k\mathbb E[r_k(\pi_t)^2].
$$

For each candidate query, estimate:

\frac{
\mathbb E
[\mathfrak R_t-\mathfrak R_{t+1}\mid q,\mathcal D_t]
}{
\mathbb E[c_T(q)]
}.
$$

Select the query maximizing expected residual reduction per teacher-output token. Approximate the expectation using the atom-posterior ensemble, the utility posterior, and an empirical model of how one newly acquired event changes the constrained distillation solution.

First implement offline replay acquisition using a pre-collected teacher pool. Reveal each teacher response and outcome only when the simulated policy selects it. This permits cheap, deterministic comparison of acquisition methods before making new teacher calls. Only after offline replay passes should online acquisition be enabled.

Compare against:

random task/state selection;

full teacher demonstrations selected randomly;

first disagreement;

hard consequentiality selection;

uncertainty-only selection;

diversity or coverage-only selection;

raw gradient similarity to the few-shot set;

uniform sampling over semantic benchmark categories;

an oracle using realized post-training gain, reported only as an upper bound.

Plot target performance and certified residual against total teacher-output tokens, including bootstrap, acquired responses, teacher-side verification, and discarded queries. Also report student rollout and optimization cost separately.

4.4 Sequential unlocking

Do not downweight an event solely because its state is unreachable under the initial student. Regenerate candidate states after each distillation round. Report how atom residuals and state reachability change between rounds.

For AppWorld and ALFWorld, environment cloning or replay may be used for offline counterfactual training analysis, but the evaluation agent must never receive an undo, reset, hidden-state, admissible-action oracle, evaluation-code, or ground-truth-solution capability unavailable to a standard benchmark agent.

5. Block III: Capability-Constrained Mirror Distillation

This is the main distillation contribution. Do not implement the main method as a per-event switch among CE, DPO, or other existing losses.

5.1 Constrained objective

At every outer round, solve from the original student $\pi_0$ using all accumulated evidence:

$$
\begin{aligned}
\min_{\theta,\xi\ge0}
\quad &
\mathbb E_{s\sim\mathcal P}
[D_{\mathrm{KL}}(\pi_\theta(\cdot\mid s)|\pi_0(\cdot\mid s))]
+
\kappa\sum_kd_k\xi_k \
\text{subject to}
\quad &
J_k(\pi_\theta)
\ge
\rho_k-\epsilon_k-\xi_k,
\qquad k=1,\ldots,K.
\end{aligned}
$$

Use primal-dual optimization. Record the trajectory of every $\lambda_k$, $J_k$, $r_k$, $\xi_k$, and the mastered-state KL.

5.2 Mirror target

For each event, form a candidate continuation set:

{y_i^S,y_i^T,y_i^{(1)},\ldots}.
$$

Define normalized capability pressure:

\sum_k
\lambda_k
\frac{\bar z_{ki}}
{\sum_j\bar z_{kj}+\varepsilon}.
$$

The unique mirror target is:

\frac{
\pi_0(y\mid s_i)
\exp(\Lambda_iQ_i(y)/\eta)
}{
\sum_{y'\in\mathcal Y_i}
\pi_0(y'\mid s_i)
\exp(\Lambda_iQ_i(y')/\eta)
}.
$$

Use length-normalized sequence scores when candidate continuations differ in length. The core training objective is:

\sum_i
D_{\mathrm{KL}}
(q_{\lambda,i}|\pi_\theta(\cdot\mid s_i)).
$$

Add a soft off-atom displacement penalty:

|(I-UU^\dagger)F^{1/2}(\theta-\theta_0)|_2^2,
$$

and a mastered-probe KL term. If fingerprints use a fixed random sketch, apply the same sketch to the whitened parameter displacement before evaluating this penalty.

The complete practical objective is:

\mathcal L_{\mathrm{CMD}}
+\gamma_\perp\mathcal L_\perp
+\gamma_{\mathrm{probe}}
\mathbb E_{s\sim\mathcal P}
[D_{\mathrm{KL}}(\pi_\theta|\pi_0)].
$$

5.3 Required limiting-behavior diagnostics

For a teacher/student candidate pair, verify numerically that:

\log\frac{\pi_0(y_i^T)}{\pi_0(y_i^S)}
+
\frac{\Lambda_i}{\eta}(Q_i^T-Q_i^S).
$$

The expected mechanism is:

a large unsatisfied atom produces a concentrated, teacher-like target and supports capability acquisition;

a nearly satisfied atom produces a small relative log-odds change;

a satisfied atom has vanishing dual pressure, preventing repeated-event overtraining;

an update threatening another atom reactivates that atom's constraint;

a low-utility teacher continuation receives little target mass without an explicit negative-example rule.

Log target entropy, teacher target mass, student target mass, $\Lambda_i$, and atom constraint status for every training event.

5.4 Distillation baselines and ablations

Use identical acquired data, teacher-token budget, random seed sets, and evaluation protocol for:

student base;

teacher-response CE/SFT;

reference-centered pairwise training;

reference-free pairwise training where existing code supports it;

fixed CE-then-pairwise schedule;

KL-regularized CE with tuned global coefficient;

global mirror distillation with one constraint and no capability atoms;

CRCD without the off-atom penalty;

CRCD without atom-wise dual variables;

CRCD without utility uncertainty, using posterior means only;

full CRCD;

per-benchmark oracle selecting the best baseline, used only as a reference upper envelope.

Match or report separately:

unique acquired events;

teacher-output tokens;

optimizer steps;

trained response tokens;

per-event repetition count;

mastered-state KL;

gradient/Fisher displacement.

Do not call two runs dose-matched merely because they have the same number of optimizer steps.

6. Block IV: Certification

6.1 Atom representation coverage

On held-out event fingerprints, compute:

1-
\frac{
\min_Z|\Psi_{\mathrm{held}}-UZ|F^2
}{
|\Psi{\mathrm{held}}|_F^2
}.
$$

Report this together with held-out transfer prediction. Do not present reconstruction coverage alone as a success guarantee.

6.2 Atom-wise capability certificate

For every atom, report:

(d_k,\operatorname{LCB}[J_k],\rho_k,
\operatorname{UCB}[r_k],n_k).
$$

Apply simultaneous confidence bounds or multiple-comparison correction for a joint claim over atoms. If an atom has insufficient effective probe count, mark it unresolved.

6.3 Acquisition stopping

Given an explicit teacher-token shadow price $\nu$, stop only when the budget is exhausted or:

\nu\mathbb E[c_T(q)]
\right}
\le0.
$$

Report remaining posterior residual at stopping.

6.4 Selective deployment

Using untouched calibration examples, construct an acceptance score from atom residual, atom-assignment uncertainty, policy margin, and distance from atom support. Evaluate:

risk-coverage curve;

accepted-set empirical failure rate;

nominal versus realized risk at several target levels;

confidence interval coverage across seeds;

abstention or teacher-routing rate.

Do not claim a formal certificate unless its statistical assumptions, calibration split, and finite-sample procedure are explicitly implemented and tested.

7. AppWorld Validation Protocol

AppWorld is a long-horizon interactive coding and API benchmark. The agent writes executable code, interacts with stateful applications, and is evaluated using database-state unit tests that also detect collateral changes. Its official splits are train, dev, test_normal, and test_challenge; its aggregate metrics include Task Goal Completion (TGC) and Scenario Goal Completion (SGC).

7.1 Integration requirements

Build an adapter that records at each agent interaction:

task ID and split;

task instruction and information legitimately visible to the agent;

full prompt/history state $s_i$;

generated executable code block $y_i$;

execution output and error;

environment snapshot or deterministic reconstruction identifier for offline branch evaluation;

number of model and teacher tokens;

official online evaluation output;

whether complete_task was called;

number of interactions and API calls.

Use AppWorld's functional API form unless the existing agent scaffold has a justified alternative. The base student and teacher must receive the same visible task information and API documentation policy.

For train/dev only, the evaluator may be used offline to produce learning signals. Never place evaluation code, hidden solution fields, required API labels, or ground-truth solutions in model prompts.

7.2 Split audit and few-shot episode construction

Before selecting tasks, inspect train/dev metadata and write a split audit answering:

whether scenario families have multiple variants spanning train and dev;

how many support, acquisition, calibration, and held-out tasks are available per scenario;

whether scenario membership can be used for experimental grouping without being exposed to the model;

whether train/dev distributions support episodic few-shot specialization without leakage.

Preferred controlled protocol, if the split audit supports it:

define each target specialization episode by an AppWorld scenario or a leakage-safe collection of related scenario variants;

use $K\in{2,4,8}$ train examples as $\mathcal S_{\mathrm{few}}$;

use other train instances for teacher acquisition;

reserve disjoint train or dev instances for atom validation and calibration;

evaluate on unseen dev variants from the same target distribution;

repeat over multiple target episodes and support-set seeds.

If scenarios do not span splits or do not contain enough variants, do not fabricate a protocol. Instead, define target subsets using train/dev-only metadata such as required-app combinations or task-generator families, document the resulting assumptions, and ensure those labels are never provided to the method. Keep a second, broader protocol that treats AppWorld as one target distribution using few-shot train examples.

After all choices are frozen on train/dev, run one official evaluation on test_normal; run test_challenge only if compute permits. Inspect aggregate TGC/SGC only, following the benchmark rules.

7.3 AppWorld utility

Use the official state-based evaluator. Save:

per-task binary success;

fraction and identity class of passed versus failed requirements on train/dev;

TGC;

SGC;

collateral-damage failures;

interactions, API calls, runtime errors, and completion length.

The primary task utility for counterfactual rollouts should be based on official evaluation progress and terminal success. Do not invent a reward using hidden solution similarity. Treat collateral-damage requirements either as part of official utility or as an explicit safety constraint, and report the choice.

7.4 AppWorld experiments, expected results, and decisions

AW-1: Base and teacher characterization

Do: Evaluate the base student and teacher on the selected train/dev tasks using the same scaffold.

Measure: TGC, SGC, binary success, requirement pass fraction, collateral failures, runtime errors, interactions, response tokens, and student/teacher disagreement states.

Expected: The teacher should materially outperform the student and create enough behavioral differences to bootstrap fingerprints. If both models are near zero or the teacher is not substantially stronger, the current teacher/scaffold is not suitable and the main experiment should pause.

AW-2: Capability atom validity

Do: Learn the sparse Fisher-whitened dictionary on train events; evaluate reconstruction and controlled transfer on held-out train/dev events.

How: Use small norm-matched updates on a training event or small atom-specific batch, then measure held-out margin, official requirement progress, and task utility changes. Compare all representation baselines in Section 3.3.

Expected: Sparse atom composition should predict positive transfer better than PCA, raw gradients, semantic embeddings, AppWorld scenario labels, required-app/API labels, and random dictionaries. It need not recover human semantic categories. If it fails to improve held-out transfer prediction, retain fingerprints only as diagnostics and do not call them capability atoms.

AW-3: Unified distillation mechanism

Do: Train full CRCD and every distillation baseline using exactly the same acquired event pool.

How: Use low-, medium-, and high-budget event pools. Record atom residuals, dual variables, mirror-target mass, parameter displacement, and mastered-probe degradation throughout training.

Expected: Because AppWorld likely presents substantial missing behavior for a small base model, early CRCD mirror targets should concentrate on successful teacher continuations and approach the benefit of CE. As atom constraints become satisfied, dual pressure should decline and prevent repeated-event overtraining. Full CRCD should match or exceed the strongest fixed operator at the same teacher budget while causing no greater collateral damage. A failure to match CE in the low-mastery regime falsifies the claimed acquisition limit of the mirror target.

AW-4: Active acquisition

Do: Compare CRCD acquisition with random full demonstrations, first disagreements, consequentiality-only selection, uncertainty, diversity, and gradient similarity.

How: First simulate acquisition from a fixed teacher pool; then repeat online for the top methods. Evaluate the area under the performance-versus-teacher-token curve.

Expected: CRCD should reach a fixed TGC/SGC target using fewer teacher-output tokens and should allocate longer teacher continuations to residuals that cannot be resolved by local actions. If gains exist only when counted by retained events but disappear when counted by total teacher tokens, do not claim budget efficiency.

AW-5: Sequential unlocking

Do: Repeat residual mining after each distillation round and track state depth, reachability, and atom composition.

Expected: Later rounds should expose states and atoms absent from the initial student distribution, while residual on already covered atoms decreases. If the same event types recur without atom-residual contraction, self-iteration is not functioning as proposed.

AW-6: Certification

Do: Calibrate atom satisfaction and selective deployment on untouched dev instances.

Expected: Nominal selective-risk bounds should achieve empirical coverage across seeds and retain a nontrivial fraction of tasks. A certificate that attains nominal risk only by rejecting nearly all tasks is technically valid but practically vacuous and must be reported as such.

8. ALFWorld Validation Protocol

ALFWorld is the most important controlled mechanism benchmark because the project already contains both sides of the alleged regime transition. The same environment and student family exhibit an initially low-skill policy and, after successful behavior acquisition, a high-skill policy on which repeated absolute imitation can become destructive. Use the official textual environment and retain the established task-category reporting for look, pick/place, clean, cool, heat, and two-object tasks.

8.1 Integration and split requirements

Reuse and refactor the existing ALFWorld pipeline rather than building a disconnected implementation. Every trajectory/event record must include:

split, task ID, task category, and random seed;

goal, visible observation/action history, and inventory as the decision state $s_i$;

student and teacher actions or bridges with exact token counts;

action validity, environment observation, terminal success, and episode length;

replay or reconstruction metadata for counterfactual branches;

state depth and empirical reachability under the current student;

student checkpoint used for mining and the original base checkpoint used for projection.

Use train demonstrations and training environments for acquisition. Use valid_seen or a newly frozen subset for implementation decisions. Report valid_unseen as the established 134-episode confirmation set, but explicitly disclose that prior iterations have already examined it. Freeze the revised method before its final multi-seed valid_unseen evaluation. Never give the model simulator-only state, an admissible command list, or expert-plan metadata unavailable under the normal text-agent interface.

Primary metrics are official task success, paired per-episode success, success by task category, episode length among solved tasks, invalid-action rate, and exact teacher-output tokens. Use paired bootstrap intervals and exact McNemar tests where appropriate; the 134-task evaluation is too small for one-seed point differences of a few percentage points to carry a mechanistic claim.

8.2 Existing empirical anchors that must be reproduced

Before evaluating the new method, reproduce or load raw official records sufficient to verify these qualitative facts:

the original base is approximately 8.96% on the established 134-task valid_unseen evaluation;

on first-round consequential events, CE/SFT reaches about 71.1% mean over three seeds, with one established run at 74.63%, while the best pairwise variants remain around 25--28%;

consequential events are far more event-efficient for CE than first-divergence events, while estimated zero-$\Delta U$ events still contain learnable imitation signal;

raw $\Delta U$ magnitude weighting fails its permutation controls and must not be restored as the new method;

reversing negative-teacher events is harmful;

after the CE model becomes competent, repeated per-event CE is more fragile than pairwise repair, although matched low dose can make the two similar;

deeper teacher-prefix states are initially nearly unreachable and become reachable after earlier behavior is learned, demonstrating sequential unlocking.

These numbers are regression anchors, not fixed targets for the new method. If they cannot be reproduced from stored outputs or reruns, first diagnose evaluator, scaffold, data, and checkpoint differences.

8.3 ALFWorld experiments, expected results, and decisions

ALF-1: Base, teacher, and event-pool regression

Do: Reproduce the base, teacher/demo, first-divergence, estimated-zero-consequence, consequential, all-divergence, CE, and pairwise reference arms with exact event, token, step, and repetition accounting.

Expected: Recover the qualitative ordering above. This establishes that later differences come from the revised method rather than a changed scaffold or evaluator.

ALF-2: Capability atom validity

Do: Compute fixed-base Fisher-whitened event fingerprints, fit the sparse dictionary, and measure controlled held-out transfer across states, task categories, and trajectory depths.

How: Compare sparse atoms against raw gradient similarity, PCA, unwhitened and whitened k-means, semantic state embeddings, ALFWorld task categories, random dictionaries, and existing A1/A2/A3 fingerprints. Evaluate transfer both before and after the first behavior-acquisition round. Keep the atom basis tied to the original base/parameterization and version any re-estimated basis separately.

Expected: The sparse composition should predict which held-out events improve after a small update beyond coarse task categories and response similarity. Sequential atoms may span multiple human task labels. If it cannot beat the strongest gradient or category baseline, use the representation only for diagnostics and do not call its columns validated capability atoms.

ALF-3: Unified distillation in the acquisition regime

Do: Starting from the original low-skill base, train full capability-constrained mirror distillation on the same first-round event pools as CE and pairwise baselines.

How: Compare CE, pairwise, global mirror projection, full CRCD, CRCD without atom duals, and CRCD without mastered probes. Match acquired evidence and teacher tokens; sweep only predeclared global trust-region/dual hyperparameters on the development split. Log mirror-target teacher mass, target entropy, atom residuals, dual pressure, and mastered-probe KL.

Expected: Large violated constraints should make the mirror target sharply teacher-directed, allowing CRCD to approach the strong CE acquisition result without a manually selected CE loss. It should substantially exceed pairwise repair in this low-skill round. If full CRCD cannot approach CE at matched data and training budget, its claimed acquisition limit is false.

ALF-4: Unified distillation after competence is acquired

Do: Use the established competent CE checkpoint only to mine second-round residual states and outcomes. Then compare two clean protocols:

accumulated projection: solve again from the original base using round-1 plus round-2 evidence and constraints;

adapted-base stress test: start from the competent checkpoint to measure sensitivity to repeated residual supervision.

For both, compare CE, pairwise, global mirror, and full CRCD at matched unique events, trained tokens, steps, and per-event repetition.

Expected: In accumulated projection, satisfied atom duals should suppress redundant round-1 fitting while unsatisfied atoms receive pressure. In the adapted-base stress test, CRCD should behave conservatively like a relative repair and be at least as robust as pairwise under repeated events. The same equations must create this change; no round-dependent CE/DPO switch is allowed. If target concentration remains CE-like after capability satisfaction, the dual/residual mechanism has failed.

ALF-5: Teacher-token acquisition efficiency

Do: Build offline acquisition replay from the existing verified demo/event pool, hiding teacher responses until selected. Compare atom-residual value-of-information acquisition with random demos, random states, first divergence, consequentiality-only selection, uncertainty, diversity, category balance, and full-demo acquisition.

How: Evaluate curves against cumulative teacher-output tokens, not retained event count. Charge for teacher bridges, failed/discarded queries, and any teacher-side verification text. Separately report environment branch rollouts because existing verified demonstrations may have zero new teacher cost in replay.

Expected: CRCD acquisition should recover strong task success using fewer revealed teacher tokens and should learn to request longer bridges when a one-action answer cannot unlock a residual. Existing evidence predicts that first divergence will be inefficient and that hard consequentiality will be useful but incomplete. If improvement disappears under exact token accounting, claim only event selection efficiency.

ALF-6: Sequential unlocking and adaptive continuation length

Do: After every outer round, re-roll the current student, re-estimate reachable state mass, infer atom residuals, and allow the acquisition policy to choose action, bridge, or full-trajectory supervision.

Expected: Early rounds should spend budget on prerequisites and increase reachability of deeper states; later rounds should shift budget toward newly exposed residual atoms. Compare reachability computed under the current student with the misleading initial-policy estimate. Failure to expose new states or shrink covered residuals is a No-Go for the sequential-acquisition story.

ALF-7: Certification

Do: Freeze an untouched calibration set, construct simultaneous atom-wise residual bounds and selective risk scores, and evaluate on the frozen final split across at least three training seeds.

Expected: Unresolved atom mass and out-of-support distance should identify failure-prone episodes better than raw sequence likelihood, entropy, or task category. The certificate must be calibrated and nonvacuous; otherwise report it only as a diagnostic profile.

9. BFCL Validation Protocol

Use the existing official BFCL harness and preserve the current categories and reporting axes, including Overall, AST or function-call correctness where available, Relevance, Irrelevance, multi-turn, memory/stateful, and any currently established NL/MT/Rel/Irrel/Memory breakdown.

Do not use official evaluation examples for training or hyperparameter selection. Reuse the current leakage-safe training/event-mining pools and create an untouched calibration subset before further method development.

9.1 Existing empirical anchors that must be reproduced

Before testing the revised method, verify from official raw records or reruns:

base Overall is about 46.06;

corrective CE collapses Overall to roughly 8--19 and produces mastered-prompt KL around 0.4 in the established diagnostic;

base-centered pairwise training keeps mastered-prompt KL near 0.005 and cumulative retraining reaches roughly 47.37--47.91 over multi-seed or best balanced-anchor configurations;

cumulative retraining from the original base is safer than continued finetuning and saturates around round 3;

call/abstain preservation is a coupled boundary problem: one-sided correction or anchoring trades Relevance against Irrelevance;

raw $\Delta U$ magnitude is associated with both target-axis gain and boundary damage, so it is not a valid scalar training weight.

If these facts do not reproduce, resolve evaluator, data, tokenizer, scaffold, and checkpoint differences before interpreting the new method.

9.2 BFCL experiments, expected results, and decisions

BF-1: Reproduce the established baselines

Do: Reproduce the base, CE, pairwise, cumulative pairwise, and best preservation configurations with their current multi-seed settings.

Expected: The reproduction should recover the established qualitative facts: CE causes substantial policy drift and large performance degradation; pairwise correction provides modest but stable gains; and homogeneous one-sided correction shifts the relevance/irrelevance boundary. If these cannot be reproduced, stop and resolve the infrastructure discrepancy before testing CRCD.

BF-2: Capability atom validity

Do: Recompute event fingerprints using the new fixed-base Fisher-whitened sparse dictionary and run the same transfer protocol as AppWorld.

How: Compare with existing A1/A2/A3 fingerprints, semantic/category labels, side labels, PCA, and random dictionaries. Include partial correlation controlling for BFCL category and call-versus-abstain side.

Expected: The sparse atom model should improve held-out transfer prediction beyond the current conditional result and should represent the call/abstain boundary as related atom compositions rather than two independent semantic classes. If the result remains only marginally above category labels or cannot predict negative transfer, narrow the claim to coverage and diagnosis.

BF-3: Unified distillation mechanism

Do: Train global mirror distillation and full atom-constrained CRCD on the same cumulative event pools used by CE and pairwise baselines.

Expected: In BFCL, already-satisfied atom constraints should keep most dual variables small; the mirror target should produce limited relative probability changes instead of concentrated teacher imitation. Full CRCD should avoid the CE collapse, remain near or above the best cumulative pairwise Overall score, and preserve Irrelevance and other mastered axes better than global mirror distillation without atom constraints.

Minimum success criterion:

no catastrophic CE-like collapse;

Overall statistically indistinguishable from or better than the strongest pairwise baseline;

lower mastered-prompt KL or a better target-gain-versus-drift Pareto frontier;

no large single-axis degradation hidden by Overall.

BF-4: Atom-wise constraint behavior

Do: Plot $J_k$, $\rho_k$, $r_k$, $\lambda_k$, and atom-specific drift throughout training.

Expected: Dual pressure should rise only for violated atoms, fall after residual closure, and reactivate when another update harms a previously satisfied atom. If dual trajectories do not track measured capability residuals, the constrained interpretation is not supported.

BF-5: Dose and repetition

Do: Repeat selected event pools at increasing per-event repetition while comparing CE, pairwise, global mirror, and CRCD.

Expected: CRCD should be less sensitive to repeated homogeneous events because satisfied atom constraints lose dual pressure. It should peak over a wider update range than current CE/pairwise objectives. If it merely shifts the optimum number of steps without widening the safe region, the proposed dose-control claim fails.

BF-6: Acquisition and teacher-token efficiency

Do: Simulate active selection from the existing BFCL event pool before making new teacher calls. Evaluate performance versus exact teacher-output tokens or an explicitly justified proxy if GT actions were used at zero teacher cost.

Expected: Atom-residual acquisition should outperform random, first disagreement, category balancing, and gradient similarity at matched cost. Be explicit that experiments using benchmark ground truth instead of a black-box teacher do not by themselves establish teacher-token efficiency.

BF-7: Certification

Do: Construct atom-wise confidence bounds and selective risk calibration using a held-out calibration split. Evaluate across seeds and all BFCL axes.

Expected: Capability residual should identify high-risk boundary and stateful examples better than raw confidence or response entropy. If atom residual does not add predictive value, retain only standard selective-risk calibration and do not claim an atom-based certificate.

10. Cross-Benchmark Universality Test

The primary universality claim is not that AppWorld, ALFWorld, and BFCL share identical atoms. It is that the same discovery, acquisition, constrained distillation, and certification algorithms adapt to all three without benchmark-specific operator rules.

Freeze the following across AppWorld, ALFWorld, and BFCL wherever architecture permits:

student checkpoint family and LoRA parameterization;

fingerprint normalization and Fisher estimation procedure;

dictionary model-selection rule;

acquisition objective;

primal-dual update rule;

mirror-temperature selection rule;

trust-region normalization;

stopping and certificate construction.

Benchmark-specific environment adapters, candidate continuation lengths, and official evaluators are allowed. Do not manually specify which benchmark or round should use CE-like versus pairwise-like behavior.

Report the per-benchmark oracle envelope and normalized regret:

\frac{J_{b,\mathrm{oracle}}-J_{b,\mathrm{CRCD}}}
{J_{b,\mathrm{oracle}}-J_{b,\mathrm{base}}+\varepsilon}.
$$

The strongest result would be that one CRCD configuration approaches the best acquisition baseline on low-skill ALFWorld/AppWorld, automatically becomes conservative after ALFWorld competence is acquired, approaches the best pairwise/preservation baseline on BFCL, and improves the aggregate teacher-token Pareto frontier without a benchmark-specific loss switch.

The claim fails if CRCD requires manually choosing different distillation operators, different residual definitions, or different acceptance rules after observing official test performance.

11. Execution Order

Follow this order. Do not launch the full matrix before earlier gates pass.

P0: Infrastructure and falsification gates

Repository plus AppWorld/ALFWorld split and contamination audit.

Reproduce the existing ALFWorld mechanism anchors from saved records; run only missing regression arms.

Deterministic AppWorld base-agent smoke test on a few train tasks.

Unified event schema for AppWorld, ALFWorld, and BFCL.

Fisher-whitened fingerprint implementation with numerical tests.

Sparse dictionary implementation and held-out reconstruction test.

Offline transfer-prediction gate on existing BFCL/ALFWorld events and a small AppWorld pilot.

Mirror-target implementation with analytical and numerical checks of the log-odds equation.

Tiny primal-dual training smoke test showing that satisfied constraints reduce their dual pressure.

P1: Core method validation

ALFWorld ALF-1 through ALF-4 using existing data and checkpoints.

BFCL BF-1 through BF-5 using existing event pools.

AppWorld AW-1 through AW-3 on train/dev pilot tasks.

Cross-benchmark comparison with the same method rules.

Three training seeds for principal distillation comparisons.

Update Go/No-Go decisions before any new large teacher acquisition.

P2: Budget acquisition and certification

Offline acquisition replay on all three benchmarks where a fixed teacher pool exists.

Online AppWorld teacher acquisition at three token budgets.

ALFWorld residual self-iteration and adaptive continuation-length validation.

Residual self-iteration and sequential unlocking on AppWorld.

Atom-wise and selective-risk calibration on all three benchmarks.

Freeze all choices, then run final AppWorld official test evaluation and predeclared ALFWorld/BFCL confirmation runs.

12. Expected Main Result Pattern

The method is supported only if the following complete pattern appears:

Sparse gradient atoms predict held-out transfer beyond category and representation baselines.

Acquisition based on posterior atom-residual reduction improves performance per total teacher token.

A single capability-constrained mirror-distillation implementation behaves acquisition-like when large capability constraints are violated and conservative when the base already satisfies most constraints.

The same distillation rule approaches the strongest fixed baseline on AppWorld, ALFWorld, and BFCL without a benchmark-specific CE/DPO choice, and changes continuously from acquisition-like to conservative behavior as atom constraints become satisfied within ALFWorld.

Atom-level dual variables stop redundant fitting and improve the gain-versus-drift Pareto frontier.

The certificate provides calibrated and nonvacuous information about remaining risk.

Do not retroactively redefine success after seeing results. Use the following interpretations:

Atoms fail, distillation works: publish the distillation as constrained task projection and downgrade atoms to diagnostics.

Atoms work, acquisition fails: retain atom discovery and distillation, but do not claim teacher-token optimality.

Mirror distillation merely matches tuned CE/DPO with no robustness or budget benefit: distillation novelty is insufficient.

Certification is vacuous or miscalibrated: remove the formal certificate claim and report residual diagnostics only.

Only one or two benchmarks work: do not claim universality; identify the violated assumption.

13. Required Code and Artifact Outputs

Adapt filenames to the repository's existing organization instead of creating redundant parallel structures. At minimum, produce:

a shared event schema supporting AppWorld, ALFWorld, and BFCL;

deterministic gradient-fingerprint extraction with Fisher whitening;

sparse dictionary training, loading, and held-out evaluation;

atom-loading posterior inference;

counterfactual branch replay and utility-posterior estimation;

offline and online acquisition policies with exact token accounting;

capability-constrained mirror-target construction;

primal-dual trainer and atom-residual probes;

capability-subspace and mastered-state drift diagnostics;

AppWorld agent/environment adapter and official evaluation wrapper;

ALFWorld integration with the existing official environment and evaluation harness;

BFCL integration with the existing official evaluation harness;

certificate calibration and risk-coverage evaluation;

reproducible configs and launch scripts;

unit tests for equations, normalization, token accounting, split isolation, and deterministic replay.

Every saved event should minimally contain:

{
  "benchmark": "appworld_alfworld_or_bfcl",
  "split": "train_dev_calibration_or_test",
  "task_id": "...",
  "state_id": "...",
  "state_hash": "...",
  "student_checkpoint": "...",
  "student_continuation": "...",
  "teacher_model": "...",
  "teacher_continuation": "...",
  "student_branch_outcomes": [],
  "teacher_branch_outcomes": [],
  "student_output_tokens": 0,
  "teacher_output_tokens": 0,
  "fingerprint_version": "...",
  "fingerprint_path": "...",
  "atom_dictionary_version": "...",
  "atom_loading": [],
  "provenance": {}
}

Do not commit secrets, raw credentials, hidden AppWorld evaluation content, or large generated outputs that belong in external artifact storage.

14. Required Documentation Updates

Update the repository's canonical documents rather than leaving this method only in experiment notes.

Method document

Create or update the canonical Method document with:

problem definition and assumptions;

one notation table;

the four blocks in this prompt;

the sparse dictionary objective;

atom performance and residual definitions;

acquisition objective;

constrained policy projection;

mirror-target derivation;

certificate scope and limitations;

a distinction between implemented components, proposed components, and empirically validated claims.

Remove or clearly mark stale claims, including:

relative correction is universally better than CE;

task-family success alone selects the operator;

raw $\Delta U$ magnitude is a useful training weight;

negative teacher events should be reversed;

first disagreement is a reliable correction target;

fingerprint clusters have already been established as capability atoms;

event efficiency is automatically teacher-token efficiency.

Experiment plan

Create or update the experiment plan with:

the AppWorld, ALFWorld, and BFCL protocols above;

exact P0/P1/P2 ordering;

compute and teacher-token estimates;

baselines and ablations;

primary and secondary metrics;

Go/No-Go criteria;

expected result and alternative interpretation for every experiment.

Result ledger

For every completed experiment, record:

hypothesis;

exact data and split;

implementation and config;

teacher-token cost;

optimizer dose;

official result and uncertainty;

whether the predeclared criterion passed;

what claim is strengthened, weakened, or falsified;

links to raw outputs and checkpoints.

Project state

Update the project-state document after each major milestone so another agent can resume without reconstructing the work from chat history.

15. Required Response Format

In your next response, do not merely restate the method. Report:

Repository audit: relevant files, current infrastructure, and missing pieces.

Method consistency audit: mismatches between current code/docs and this formulation.

Implementation plan: concrete files to add or modify, with dependencies.

Experiment matrix: AppWorld, ALFWorld, and BFCL runs, exact data splits, budgets, metrics, seeds, baselines, and expected results.

Cost estimate: GPU hours, environment rollout count, and teacher-output tokens.

Immediate P0 actions: what can be completed without new expensive calls.

Blockers or decisions required from the user: only items that materially change the scientific design or cost.

After the audit, begin implementing all unblocked P0 items. Do not wait for confirmation on routine reversible code changes, diagnostics, documentation updates, or smoke tests. Pause before large teacher expenditures, final AppWorld test evaluation, or any change that violates benchmark restrictions.