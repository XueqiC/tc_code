# Draft: Introduction and Methodology (ICLR format)

Prose draft, 2026-08-09. Reader-facing. The internal quick-read version is
`problem-setting-and-method.md`; the executable detail is
`docs/superpowers/specs/2026-08-06-action-space-capability-shape-design.md`.

**Working title.** *Which Tasks Survive Distillation: Action-Space Notation as a
Training-Time Control on Agent Capability*

---

## 1 Introduction

Tool-using agents are increasingly deployed in narrow settings: a fixed set of tools, a
predictable band of requests, and a cost ceiling that rules out calling a frontier model on every
one. The usual route is to distil a large model into a small one on the large model's own traces,
evaluate the student on a tool-use benchmark, and ship once the score is close enough. What that
procedure cannot rule out is the failure it is most exposed to. A student can handle single
lookups reliably and still fail, repeatedly, on requests that need one tool's output to determine
the next call; if such requests are a modest share of the benchmark, the headline number will not
move. Nothing about the measurement was wrong. It reported how often the student succeeded across
the benchmark's mixture, while the deployment's fate turned on which requests it had stopped
being able to serve.

The gap between those two questions is not a matter of measurement precision. Two models with
the same average score do not solve the same tasks, and a narrow deployment is an intervention
on exactly the mixture the average was taken over, so a difference that is small in aggregate
can be concentrated where the traffic is. Hooker et al. (2019) established this for compressed
image models, where comparable top-line accuracy coexists with systematic divergence on narrow
subsets. What a practitioner needs before shipping is not a count of tasks the small model can
do but the identity of them, and whether that set contains what will arrive.

Existing practice addresses this in five ways, and each stops short of the question. The first
is to trust the aggregate and ship, which fails for the reason above. The second is to measure
more finely: capability profiling with item-response models separates a model's ability from
item difficulty and supports item-by-item comparison (InferenceDynamics, ACL 2026), which
reveals divergence where an average hides it. But it is diagnostic — it tells a practitioner
that two models differ on which items, not what to change in order to move the set. The third is
to avoid relying on the small model where it is weak, by learning a routing or deferral boundary
and escalating the rest (CAT 2024; Kag et al., ICLR 2023). This is often the right engineering
answer, and it is an answer to a different question, since it keeps the large model in the loop
and asks when not to trust the student rather than how to make the student cover the scope. The
fourth is to change what the model is given at inference time. Within agent research this is
where action-space representation has been studied: CodeAct compares code, JSON and text action
spaces by prompting one fixed model and reporting mean scores for each. That measures how well
an already-trained model can use a format it is handed, which is a different variable from what
a model acquires when trained in that format, and it is reported in aggregate, so it cannot
speak to which tasks. The fifth is to improve the distillation itself by choosing better
supervision — more trajectories, cleaner trajectories, trajectories selected for a target
capability — as in agent distillation work that transfers tool use into small students without a
notation arm to compare against.

What none of these vary is the *form* in which the supervision is written. Across all five, the
notation of a tool call is treated as a presentational detail: the trajectories are what they
are, and the choice of whether a call appears as executable code or as a JSON object is made by
convention or by whatever the teacher emitted. Our question is whether that choice is inert. If
two students are trained on the same teacher trajectories, differing only in the notation those
trajectories are written in, do they end up able to do the same tasks?

We keep the study to one manipulated variable and one measurement. Each teacher trajectory is
parsed into a notation-neutral form recording the reasoning, the tool, the arguments, and which
arguments were results of earlier steps; from that single record we generate both a code
rendering, where a call's result is bound to a name that later calls refer to, and a JSON
rendering, where each call is an object and dependencies are carried in an explicit reference
map. Generating both from one source means neither arm is advantaged by resembling the teacher's
own output format, and a machine-checked round-trip establishes that the two carry identical
content rather than assuming it. Two students are then trained, brought to the same measured
ability with an item-response model, and compared on which items they succeed at — a comparison
in which average scores are equal by construction, so any difference is about identity of tasks
rather than amount of capability. We ask whether the difference is organised by structure: which
tools a task requires, how its calls compose, and how deep its longest chain of result-consuming
steps runs. Since the question is only meaningful relative to a stated demand, we fix what the
deployment asks for by declaring a tool scope per request topic, with a per-topic coverage level
(§2.2).

Contributions. (i) We ask, and answer, whether action-space notation is a *training-time*
determinant of which tasks a distilled agent can perform, rather than an inference-time
presentation choice — evaluated on per-item outcomes at matched ability, which is the only
comparison under which "which tasks" is answerable. (ii) We show the difference is organised by
the structural coordinates above, rather than being idiosyncratic across tasks, and pre-register
the failure of that organisation as a kill condition. (iii) We provide a rendering apparatus in
which notational equivalence is a *checked property* rather than an assumption: two renderers
each paired with a reader, round-trip equality verified over randomly generated trajectories.
This is what makes the comparison interpretable, and in construction it caught three defects
that would have corrupted results silently rather than crashing — including a literal argument
being read back as a dependency, inflating the very structural quantity under study. (iv) We
quantify and report a confound that any such comparison must handle, in the direction that works
against us: on a matched tokenizer the same trajectory costs about 1.40× more tokens in JSON than
in code, and the ratio varies with composition shape (about 1.48× for branching against 1.34× for
independent calls), so the length asymmetry is collinear with the coordinate the study is stated
over. We therefore pre-register an explicitly one-sided claim structure, under which a code-arm
result is conservative and a JSON-arm result is declared unclaimable in advance.

> **Draft note.** Paragraph 1 rests on one empirical premise that is currently unsupported: that
> requests requiring one tool's output to determine the next call are disproportionately where a
> distilled agent degrades. It is stated as an exposure the shipping procedure cannot rule out
> rather than as an observed incident, which is honest but weak. Before submission it needs either
> a citable report of shape-correlated degradation after agent distillation, or one measurement of
> our own on a public tool-use benchmark.
>
> **Scope note.** The motivation must stay in terms readers share — a fixed tool set and
> dependency-carrying requests. It must not be anchored to any particular deployment's tool
> inventory: naming a specific system's components reads as arbitrary and invites the objection
> that the result is an artefact of that setup. If a concrete instantiation is wanted, use a
> public tool corpus (the StableToolBench / BFCL family we already evaluate on), not an internal
> one.

---

## 2 Methodology

### 2.1 Overview

The design has one manipulated variable and one outcome. Fixed: the teacher, the tool set,
the task panel, the student's base model and size, and the seeds. Manipulated: whether the
teacher's trajectories are written as code or as JSON for the purposes of training.
Outcome: the per-task success pattern of each student, compared at equal measured ability.

The pipeline is four stages. Teacher trajectories are parsed once into a notation-neutral
intermediate representation (§2.3.1). Two training corpora are generated from it, one per
notation, with equivalence machine-checked (§2.3.2). Two students are trained, identical in
every respect but corpus notation (§2.3.3). Both are evaluated on a shared task panel, placed
on a common ability scale, and compared at matched ability, with the resulting difference
regressed onto the structural coordinates of §2.2.4 (§2.3.4).

### 2.2 The task boundary: composed by topic, not learned as a predicate

#### 2.2.1 Topic-mediated scoping

We obtain the boundary in two stages, neither of which requires learning a predicate over
tool combinations. Observed requests are first assigned to a small set of topics. Each topic
then carries a declared tool scope — the tools a request on that topic may legitimately
require — and the boundary is the union of the scopes of the topics that occur. A tool need
never appear in the observed requests to be inside the boundary; it is inside if its topic is.

The reason this is tractable while the direct route (§2.2.2) is not comes down to what the
observed requests are being asked to establish. They are not being asked to select a
hypothesis; they are being asked whether the topic inventory has been seen. That is a
missing-mass question — how much request probability sits on topics not yet observed — and it
is estimable in the manner of Good–Turing rather than bounded as a learning problem. Being
confident that every topic carrying at least \(\varepsilon\) of the request mass has been
observed takes roughly \(k \gtrsim \varepsilon^{-1}\log(M/\delta)\) requests for \(M\) topics.
At \(M = 20\), \(\delta = 0.05\), this is about 120 requests at \(\varepsilon = 0.05\) and
about 600 at \(\varepsilon = 0.01\). Hundreds, and deployment logs hold thousands.

The structural reason the exponent disappears is that topic mediation converts one joint
decision over tool *combinations* into \(|T|\) independent per-tool decisions given a topic.
One cost of that conversion is stated rather than hidden: the scope map is an *input* to the
method, drawn from deployment documentation or configuration, and not a result of it.

The second cost — that the composed scope over-approximates what is truly demanded — we do not
leave as a caveat, because it can be quantified. Gibbs et al. (2025) reformulate conditional
coverage as coverage over a class of covariate shifts and show that when that class is
finite-dimensional, exact finite-sample coverage over every member of the class is
simultaneously attainable; their stated example is coverage over each of a given collection of
subgroups. A declared topic partition is such a collection. Calibrating on held-out requests
labelled by the tools the teacher actually executed therefore yields, at a chosen level
\(\alpha\) and per topic,

\[ \Pr\big(\text{tools the request needs} \subseteq \text{declared scope} \;\big|\; \text{topic}\big) \ \ge\ 1 - \alpha, \]

with the guarantee holding in finite samples and without distributional assumptions, and with
the procedure slotting into an ordinary split-conformal pipeline. The over-approximation is
thereby a tunable quantity rather than an unbounded one: \(\alpha\) sets how conservative the
scope is, and the direction of the residual error is the safe one, since the failure that
matters is a needed tool falling outside scope and an unneeded tool inside it costs only
tightness. What remains unclaimable is worst-case coverage — \(\alpha\) is a level over the
request distribution — and exchangeability between the calibration requests and deployment,
which in a constructed panel is an assumption we declare rather than verify.

Against this boundary, the property we require of the student is decomposable and directly
measurable: for each tool in scope, can it issue a well-formed call in code notation, and does
it select that tool when a request needs it. Both are counted per tool rather than inferred in
aggregate.

#### 2.2.2 Why the scope is declared per topic rather than predicted per request

The natural alternative is to skip the topic layer and decide scope for each incoming request
individually. What that asks for is *conditional* validity: for this request, the predicted tool
scope contains the tools the request actually needs, with stated probability. That property is
not available, and the reason both closes the direct route and dictates the form of §2.2.1.

Two results together settle the form. Barber et al. (2021) show that exact conditional coverage
is unattainable distribution-free, and that the natural relaxation does not rescue it: requiring
coverage at level \(1-\alpha\) over *every* subgroup carrying mass at least \(\delta\) admits no
procedure better than the trivial one of targeting marginal coverage at level \(1 - \alpha\delta\).
The same paper states the escape: relaxing to coverage over every subgroup drawn from a
*restricted* class \(\mathcal{X}\) does admit nontrivial prediction sets, provided \(\mathcal{X}\)
is sufficiently restricted. Gibbs et al. (2025) then supply the constructive half, by recasting
conditional coverage as coverage over a class of covariate shifts and obtaining exact
finite-sample coverage over every member of a finite-dimensional class — with coverage over each
of a given collection of subgroups as the worked case.

Our topic partition is such a collection, which is why topic-conditional coverage is not a
concession made for tractability. It is the strongest form the guarantee can take, and it is
attainable exactly rather than approximately. Conditioning finer than the declared partition —
per request — is what the impossibility result forbids; conditioning at the partition is what the
constructive result delivers. The current frontier continues past this point toward
sample-conditional guarantees (Sample-Conditional Coverage in Conformal Prediction, NeurIPS
2025) and localised variants, which we do not need here and note only to place the choice.

The applicability condition should be stated rather than assumed. The force of the impossibility
result depends on the conditioning variable being effectively nonatomic. Requests here are
natural-language strings that are almost surely distinct, so per-request conditioning is in that
regime; were requests drawn from a small finite set with repetition, conditional coverage would
be attainable by direct partition and this argument would not apply.

A second, cruder bound points the same way and we note it only in passing. Treating scope
determination as selecting a predicate over tool *combinations* from a finite class, the standard
sample-complexity requirement \(k \gtrsim (\log|H| + \log(1/\delta))/\varepsilon\) admits a class
of effectively one element at \(k = 64\), \(\varepsilon = \delta = 0.05\), while the class of
subsets of a tool universe grows combinatorially. The empirical record agrees: Ammons et al.
(POPL 2002), inferring specifications from executions, report a false rejection rate of five in
six at comparable example counts, with a pruner unusable as a result. We do not rest the design
on this bound, since it prices a hypothesis class of our own construction and the coverage
formulation above is both assumption-free and matched to what we actually require, which is
one-sided: missing a needed tool is the failure, including an unneeded one is not.

#### 2.2.3 Gradient spectra as a secondary instrument

A gradient-spectral characterisation of task demand is a component we adopt rather than a claim
we make. The mechanism is established: low-rank gradient features representing a target
capability is LESS (ICML 2024); the singular value decomposition of layer-wise gradients as a
characterisation of what data demands, with effective rank as the operative statistic, is Li
et al. (ACL 2026); embedding tasks and models in a shared space to select an adequate model is
Task2Vec's MODEL2VEC (ICCV 2019). We cite these for the mechanism and do not restate their
finding as ours. It enters our pipeline as corroboration for the structural organisation of
§2.3.4, never as the primary outcome.

Reusing it obliges us to carry its known confound. LESS documents that gradient norm
anticorrelates with completion length, calling this "a universal problem for influence
formulations that compute averaged token gradients"; since our arms differ in length by
construction (§2.4.2), an uncorrected spectral comparison would differ with no task-semantic
content. We apply LESS's own remedy — \(L_2\)-normalising per-example gradient features and
comparing by cosine — which fixes tr\(\Sigma\) and removes the first-order length effect. A
second-order residual survives normalisation in principle, since averaging over more tokens can
contract directions toward the mean and depress effective rank, so we do not argue it away. We
test for it with a format-permutation null: the *same* semantic trajectories are rendered in
both notations and the principal angles between the resulting subspaces are measured. Near-zero
angles establish that the instrument reads task content rather than notation; a non-trivial
angle disqualifies the instrument for this contrast and we report it as such.

The earlier objection that \(k\) gradient vectors span at most \(k\) dimensions, making
rank-\(r\) recovery from \(k \gtrsim r\) samples vacuous, applied to a small-\(k\) regime that
no longer obtains: we impose no sampling budget (§4.3.3 of the design), so \(k\) is taken well
above \(r\). Stable subspace recovery additionally requires an eigengap — by Davis–Kahan the
error scales inversely in that gap, so rapid but smooth decay is the worst case rather than the
best — and we therefore report the observed gap as a diagnostic alongside the angles.

#### 2.2.4 Structural coordinates

What we do characterise is the boundary's structure. Three coordinates, each computed exactly
from the intermediate representation with no model in the loop:

1. **Tool closure** — the set of distinct tools a task's solution requires.
2. **Composition shape** — whether calls form a chain, a branch, or a fan-out over a shared
   intermediate result.
3. **Dataflow depth** — the length of the longest chain of steps linked by one consuming
   another's result. One for a task whose calls are mutually independent.

These are the axes along which §2.3.4 asks whether a capability difference is organised, and
they are properties of the task, not of either notation — which is what makes them admissible
as the coordinate system for the comparison.

### 2.3 Shaping capability: notation as a training-time control

#### 2.3.1 A notation-neutral intermediate representation

Every teacher trajectory is parsed once into a record holding, per step, the model's
reasoning text, the tool invoked, the arguments supplied, and — critically — which arguments
were results of earlier steps rather than literals. Both training notations are generated
from this record, never from each other and never from the teacher's raw output.

This is not a convenience. Had either notation been the teacher's native format, that arm
would enjoy a fidelity advantage unrelated to notation, and the experiment would measure
proximity to the teacher instead of the affordances of a representation. Generating both from
a shared neutral form makes the two arms equidistant from the teacher by construction.

The representation admits only values that both notations can carry — a constraint discovered
empirically rather than assumed: a Python tuple survives the code notation and returns as a
list from the JSON notation, so admitting tuples would let the two arms disagree about which
trajectory they had been given.

#### 2.3.2 The two notations, and a checked equivalence

In the **code** notation, each call is a keyword-argument invocation whose result is bound to
a positional name, and a dependency is expressed by referring to that name:

```python
step0 = search_papers(query="graph networks")
step1 = summarize(text=step0, max_words=50)
```

In the **JSON** notation, each call is an object, and dependencies are carried in a separate
reference map keyed by parameter name:

```json
{"tool": "search_papers", "args": {"query": "graph networks"}}
{"tool": "summarize", "args": {"max_words": 50}, "from_steps": {"text": 0}}
```

The reference map is deliberately kept out of the argument object. An inline reference marker
is unsafe, not merely inelegant: a literal argument that happens to be an object carrying the
marker key is then read back as a dependency, inventing a dataflow edge the teacher never
wrote, in one arm only, and inflating the very coordinate of §2.2.4 that the study is stated
over. A separate key makes the collision impossible rather than detectable. We note that the
JSON arm is given a genuine reference mechanism on purpose; an arm forced to re-paste
referenced values would be handicapped rather than merely different, and the comparison would
measure the handicap.

Each notation is paired with a reader that recovers the intermediate representation from
rendered text, and we require \(\mathrm{parse}(\mathrm{render}(t)) = t\) for every trajectory,
checked over randomly generated trajectories including adversarial reasoning text. This
equality is what entitles the paper to the phrase "the same trajectories in two notations";
without it, an observed capability difference is indistinguishable from one arm having lost
information.

#### 2.3.3 Students

Two students are trained from the same base model at one fixed size, differing only in corpus
notation, with identical hyperparameters and seeds. We do not sweep student size; the compute
is spent instead on the control arms of §2.4, which is where the kill conditions live. No
budget is capped anywhere in the design, for either arm or for any baseline.

#### 2.3.4 Comparison at matched ability

Comparing which tasks two models solve is meaningless if one is simply stronger, so overall
level is factored out before the comparison rather than adjusted for afterwards. Both students
are evaluated on a shared task panel, and an item-response model is fitted that assigns each
model a single ability parameter and each task a difficulty parameter. The comparison is then
made between models at *matched ability*.

The reason this is the right instrument is precise: at equal ability, aggregate scores are
equal by construction, so the "perhaps one is just better" explanation is unavailable, and
what remains in the per-task residual pattern is entirely about *which* tasks. That residual
pattern is the paper's outcome variable. We then ask whether it is organised by the
coordinates of §2.2.4 — whether one notation's advantage grows with dataflow depth, with
branching, or with tool closure size. A difference with no structure is reported as a
measurement without the structural claim.

### 2.4 Controls, and what can kill the result

#### 2.4.1 Controls

**Interface perturbation.** Every tool is renamed and its schema rewritten while preserving
semantics, and both students are re-evaluated. If an advantage rests on having memorised the
particular interface strings seen in training, it should not survive; if it reflects structure,
it should.

**An unbounded scaffolding baseline.** Prompt and scaffold optimisation is run against the
undistilled small model with *no budget limit*, so the baseline is as strong as it can be made
rather than merely cost-matched. This makes the comparison strictly harder for us: the claim
that weight training was necessary must survive an opponent given unlimited optimisation. The
arm requires a pre-registered stopping rule — a fixed number of rounds without improvement on
a held-out split — and the consumed budget is reported, since an unbounded arm without a
stopping rule makes a negative result unfalsifiable.

**Floor exclusion.** Tasks on which both students score zero carry no information and are
excluded and reported as censored rather than modelled. A difference measured between two
floors is an artefact, and prior evidence says the floor is a live risk for small models on
tool-use panels.

**Truncation.** Generation limits are set high enough that no completion is truncated in
either arm, and this is asserted per arm and per structural bucket before analysis. A failure
counts as a limit failure only when the completion reached the limit; a completion that
stopped short and was simply unusable is a different failure that a larger limit would not
touch, and conflating the two would report a budget effect that was never there.

#### 2.4.2 The confound we have measured and have not yet solved

The two notations do not cost the same. On a matched tokenizer the same trajectory costs about
**1.40× more tokens in JSON than in code**, so "both arms see the same trajectories" and "both
arms see the same number of tokens" cannot both hold. We chose identical trajectories and
dropped budget caps, which makes the design **one-sided** and we pre-register it as such: the
uncontrolled variable favours JSON, so a code-arm win is conservative — obtained on fewer
tokens, with the confound running against the finding — while a JSON-arm win is not claimable.

The unresolved problem is sharper than the aggregate ratio. The ratio *varies with composition
shape*: about 1.48× on branching tasks against 1.34× on tasks whose calls are independent. The
length asymmetry is therefore largest exactly where §2.2.4's coordinates vary, which means
"code affords deeper composition" and "JSON simply pays more tokens for branching" predict the
same observation. Removing generation limits addresses the evaluation-side half of this, since
an uncapped JSON arm cannot be cut off preferentially on expensive shapes. It does not address
the training-side half, which follows from how per-example gradients behave over sequences of
differing length and is not a budget setting at all. A per-task length control is required
before the structural claim of §2.3.4 can be made, and designing it is open work.

The normalisation of §2.2.3 should not be mistaken for a solution to this. That remedy makes a
*measurement instrument* read task content rather than length; it does nothing about the fact
that the two training corpora differ in length per item, which shifts what the optimiser sees
during training and cannot be normalised away after the fact. The two are separate problems
that share a cause.
