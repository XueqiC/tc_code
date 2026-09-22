# ALFWorld baseline fidelity audit

Audit: Codex#16, 2026-09-22. Branch `alf-baseline-audit`; source HEAD
`4b760ec7d04ee8a17ee23c5cb65dfc3c72906132`.

Only this report was created. No existing source, configuration, bank, test, or
result was changed; no patches were applied. No GPU, model inference, teacher
request, environment rollout, or training was run. Paper retrieval was read-only.
Small in-memory CPU calculations exercised the existing loss and parser functions.

The principal findings are two definite SmartAD normalization mismatches, an
explicitly adapted SAD curriculum/objective, and a Kang collector that implements
teacher-first planning but substitutes prompted continuation for native prefill.
The K32 Kang training route and the legacy `kang` route must be distinguished.

## Sources and interpretation

- **SmartAD:** Guokai Tang and Feng Zhao, *SmartAD: Capacity-Aligned Agent
  Distillation for Small Language Models*, Findings of ACL 2026, pp. 27045–27057.
  [Citation](https://aclanthology.org/2026.findings-acl.1349/),
  [original PDF](https://aclanthology.org/2026.findings-acl.1349.pdf).
  Relevant locations: §3.2, Eqs. 3–4, printed p. 27048; §3.3, Eq. 5,
  pp. 27048–27049; §4.1 and Appendix A for the experimental recipe.
- **SAD:** Liu et al., *Structured Agent Distillation for Large Language Model
  Agents*, arXiv:2505.13820. Checked both
  [v1](https://arxiv.org/html/2505.13820v1) and
  [v5](https://arxiv.org/html/2505.13820v5).
  This matters: the collection notes cite v1, whereas the earlier local reference
  points to v5. The curriculum is Eq. 7 in v1 and Eq. 13 in v5; the hard-target
  formula is Appendix D.3, Eq. 13 in v1 and Eq. 19 in v5.
- **Kang:** Minki Kang et al., *Distilling LLM Agent into Small Models with
  Retrieval and Code Tools*, [arXiv:2505.17612v2](https://arxiv.org/html/2505.17612v2),
  §4, Fig. 3, §5. Read the referenced original repository locally at
  `/home/xueqi/hq/projects/tc-alignment/envs/baseline_repos/agent-distillation`,
  commit `8884b80ea3d22e53a2e1e4b9fd324600d13e0430` (abbreviated `K/` below);
  its working tree was clean.
  [Pinned upstream tree](https://github.com/Nardien/agent-distillation/tree/8884b80ea3d22e53a2e1e4b9fd324600d13e0430).

No SmartAD paper or original SmartAD/SAD implementation was found in this
worktree's `docs/` or `refs/`; `refs/` and the local `envs/baseline_repos` are
absent here. Kang's source was readable in the sibling main checkout. **Offline
materials alone did not establish SmartAD's original NLL normalization.** The
original PDF retrieved during this audit does establish it explicitly; the
previous “not locally recoverable” statement is therefore no longer a sufficient
description of the evidence. If only offline material were allowed, settlement
would require the Eq. 3 definition, the per-turn denominator, the outer averaging
unit, and the accompanying implementation paragraph, or equivalent original code.
None should be inferred from the phrase “trajectory NLL” alone.

Classification uses the requested categories: **changes the trained model**,
**changes selection only**, and **cosmetic/documentation**. Selection changes can
subsequently change a trained model through different data. “High” means a core
mechanism or conditioning distribution differs; “Medium” means a meaningful
experimental adaptation or conditional edge case; “Low” means metadata/reporting.
Rerun statements are conditional on the source revision and route that produced a
result. Historical prose is not proof that a particular checkpoint used this HEAD.

## SmartAD

### 1. Original method

The PDF describes the “macro-average of the negative log-likelihood (NLL)” across
assistant turns. Writing `S_m` for turn NLL sum and `n_m` for its supervised length,
Eq. 3 is `mean_m(S_m / n_m)`. Eq. 4 selects at the initial student parameters.
It samples ten trajectories at temperature 1, then filters correctness/tool
failures. Eq. 5 is `sum_i(w_i * CE_i) / sum_i(w_i)`, with weights 1/1.5/2.
Action spans are bounded Code blocks; the final block receives the final weight;
remaining supervised text is reasoning. See the [PDF, §§3.2–3.3](https://aclanthology.org/2026.findings-acl.1349.pdf).

### 2. Our implementation

- `alfworld_selection.py:13–19,112–129` pools NLL and token counts across turns,
  saves only trajectory aggregates, and minimizes `(mean_nll, candidate_id)`.
  `alfworld_cli.py:222–234` supplies unweighted, masked token NLL at the initial
  student. `read_selection`, `alfworld_selection.py:152–176`, rechecks that same
  micro-average argmin. The legacy `paper_data.py:163–190` uses the same statistic.
- `paper_losses.py:10,143–146` has the correct numeric weights but divides the
  weighted numerator by the **unweighted generated-token count**.
  `alfworld_training.py:136–143` then weights each row by `n_row / N_update`.
  Consequently the update loss is `sum_update(w_i * CE_i) / N_update`.
- `paper_losses.py:38–93` maps ALFWorld commands to action and the last verified
  turn's action to final. Text before the first action marker becomes reasoning;
  observations are masked. `alfworld_training.py:19–34` additionally supervises
  the shared encoder's native end-of-turn token, inheriting the last visible kind.

### 3–5. Deviations, minimal fixes, and reruns

**S1 — Wrong selection averaging unit. High; changes selection only.**
The two means are not interchangeable. CPU example: candidate A has turns
`(S,n)=(1,1),(270,90)`; B has `(50,20),(50,20)`. Current scores are A=2.978022,
B=2.5, selecting B; turn-macro scores are A=2, B=2.5, selecting A.

Minimal fix: compute `math.fsum(s/n for s,n in pieces)/len(pieces)`; save per-turn
sums/counts and a distinctly named `turn_macro_mean_nll`; use it in both selection
and artifact validation. Update the legacy selector or explicitly retire it.
Bump the artifact/rule version and reject old rule versions in `read_selection`.
Checking hashes alone is insufficient: that function currently checks student
identity but does not require the artifact's `rule` to equal the current rule.
Old trajectory totals cannot recover an average of turn means: rescore unless
per-turn receipts exist elsewhere. Rerun selection; retrain/evaluate only where
chosen rows change if this is the sole correction. Singleton tasks do not change.

**S2 — Weighted-loss denominator and reduction unit. High; changes the trained model.**
For a row with three reasoning tokens of NLL 1,2,3 and one action token of NLL 4,
the current weighted loss is `12/4=3`; weight normalization gives
`12/4.5=2.666667`. The scale depends on span composition, including terminal
turns, and is not one fixed learning-rate multiplier over training.

Minimal local denominator fix: divide SmartAD by `weights.sum()` with masked
positions contributing zero. **That one-line change is insufficient to establish
trajectory-level Eq. 5** under the current trainer: its row token weighting would
then produce `sum_rows(n_row/N_update * weighted_sum_row/weight_sum_row)`.
For the literal per-trajectory objective, accumulate each selected trajectory's
weighted numerator and weight sum, divide once per trajectory, then use a stated
trajectory batch reduction. Stream turn gradients if needed, but delay the
optimizer step until those units are complete. An update-wide weighted mean is
another defensible matched-protocol adaptation; identify it explicitly rather
than claiming the paper specifies that batching behavior. Retrain and reevaluate
affected SmartAD results; no new teacher data are needed for this correction.

**S3 — ALFWorld span adaptation and parser edge cases. Medium; changes the trained model.**
The last environment command is assigned final weight (`paper_losses.py:81`;
`pi1.py:195–196`), an ALFWorld terminal-action proxy. Preserve and name that choice
for an ALFWorld adaptation. If answer-producing spans exist, label those from
structured trajectory metadata instead. Do not infer that the entire terminal
turn is final: the current code correctly retains its leading reasoning.

Unlike a bounded action block, `ACTION:` keeps classifying subsequent unmarked
prose as action until another recognized marker. CPU reproduction:
`ACTION: take cup 1\nI should now finish.` is entirely action. Minimal robustness
fix: validate that the action is the terminal command line, as the scaffold
requires, or bound its span using the parsed executed command and classify
remaining authored prose separately. No original Code marker support is needed
for this ALF-only bank; do not advertise the parser as a direct CodeAct parser.
Relabel/rescore/retrain only if affected rows or the chosen final-span convention
change. The prevalence of these edge cases in current sealed banks was not measured.

**S4 — Different candidate acquisition. Medium; changes selection only.**
`tools/alfworld_teacher_pool.py:614–632,656–667` retries until a requested number
of successes or a cap, rather than implementing the source sampling protocol.
The first requested temperature is 0, later requests use the collector's sampling
temperature; transmitted settings are recorded at `:91–99`. The collection notes
explicitly say Luna omits temperature and uses provider defaults
(`docs/alfworld_baseline_collection_notes.md:102–108`). Duplicates are retained;
neither this nor a singleton argmin is itself a scoring bug.

Minimal fix: distinguish number of draws from successful-candidate target; expose
and record the effective sampling policy and budget caps. Use the source protocol
only for a reproduction run; retain the small budget protocol under an adaptation
label otherwise. The documented 87-candidate inventory is a prior export claim,
not revalidated bank evidence in this audit. Changing the pool requires collection,
selection, training and evaluation again; relabeling an existing pool does not.

**S5 — Hardcoded target count and undocumented tie convention. Low; cosmetic/documentation.**
`alfworld_selection.py:132–134` always records a target of 3, although collection
accepts other targets and the example command requests 4. Read the requested
target from the frozen candidate-set artifact, carry it through selection, and
calculate shortfalls from it. This affects metadata, not the argmin. Candidate-ID
tie breaking is a reproducibility extension, not a verified author rule; keep it
and say so. No numerical rerun for either documentation fix. A deliberately changed
tie policy would require checking exact tied winners before deciding on retraining.

## Structured Agent Distillation (SAD)

### 1. Original method

Appendix D.3 explicitly supplies a hard-target CE option with separate masked
**sums**, not separate length-normalized means. At unit coefficients and a
complete disjoint mask, its numerator is ordinary generated-token CE.
The [v1 source, Eq. 7 and Appendix D.3](https://arxiv.org/html/2505.13820v1)
also defines trajectory complexity from reasoning length, action length and
teacher entropy, followed by easy-to-hard ordering.

The [v5 source, §3.3, Appendix L and Appendix N](https://arxiv.org/html/2505.13820v5)
describes teacher-distribution KL and an explicit action head/discrete action
normalization domain. Its Appendix D.3 retains hard-target CE. Appendix N makes
entropy a curriculum signal, excluded from gradients. These are distinct claims:
the hard-label option exists, but implementing it over one language-model softmax
does not reproduce all of v5's distribution/action-head machinery. Neither
version inspected supplies enough detail to recover an exact exhaustive-pass
pacing implementation. Pin the intended version and variant in manifests.

### 2. Our implementation

- `paper_losses.py:127–131`: `sad_sum = sum_generated(CE)/n_row`.
  `alfworld_training.py:54–62,101–102` makes it the K32 default.
- `paper_losses.py:132–142`: `sad_mean` aliases legacy `sad`; compute each present
  reason/action+final mean, then average the present group means. There is no
  SmartAD weighting in either SAD branch.
- `alfworld_training.py:136–143`: token-weight rows, giving a single update-wide
  generated-token mean for `sad_sum`.
- `alfworld_curriculum.py:34–42`: count turns per package, shuffle **rows**, then
  stably sort by that package turn count in every pass. Equal-length trajectories
  can be interleaved and their turns reordered. `:27–55` gives exactly 3/10 full
  passes with token-budget batching. No teacher entropy or separate span lengths
  enter the schedule.

### 3–5. Deviations, minimal fixes, and reruns

**D1 — Mean-of-groups is an ablation, not the hard-sum formula. High for a claimed
paper result; changes the trained model.** With NLL 1,2,3 in reason and 4 in action,
`sad_sum=2.5` but `sad_mean=3`. The latter gives each action token three times the
per-token weight of reasoning in this example. Use `sad_sum` for the declared
equal-token hard-label baseline and retain `sad_mean` as a separately named
ablation. Rerun historical group-mean results if they are to represent the hard-sum
baseline. Current K32 default already selects `sad_sum`; no change needed there.

**D2 — “Sum” includes a variable update normalizer. Medium; changes the trained model.**
Current update loss is `sum(CE)/N_update`, not the literal unnormalized sum.
The collection notes correctly derive equal within-update token weights, but
that alone does not establish identical optimization. `N_update` varies because
whole turns overshoot the budget and endpoint batches flush; AdamW state,
epsilon, and clipping preclude assuming exact trajectory equivalence.

Minimal matched-protocol fix is reporting: name it `SAD hard-label token-mean +
turn-count curriculum`, recording the normalization. A literal-sum experiment
must instead accumulate raw masked sums with a declared trajectory/batch reduction
and its appropriate optimizer recipe. Reporting-only correction needs no rerun;
switching reduction requires retraining/evaluation. No new labels are needed.

**D3 — No teacher distributions or separate action head. High relative to full v5;
changes the trained model.** `K32PaperTrainer.logprobs` at
`alfworld_training.py:114–120` uses ordinary student token log probabilities;
`pi1.py:60–71` records no KL or auxiliary objective. The comment “two-head
text-only adaptation” at `paper_losses.py:136` means two masks here, not two heads.

Minimal fix under the available black-box data: label the baseline as the
Appendix D.3 **hard-label adaptation**, replace misleading head terminology, and
avoid attributing v5's separate-distribution effect to it. Full v5 reproduction
requires teacher token/action distributions, an explicit action mapping/head, and
verified normalization domains; those cannot be reconstructed from successful
text traces alone. Relabeling requires no rerun; full-objective reproduction
requires new sufficient teacher data and retraining/evaluation. An optional
contrastive/alignment extension mentioned elsewhere is not automatically a missing
mandatory component.

**D4 — Turn-count row sorting substitutes for the trajectory curriculum. High;
changes the trained model.** Equal-turn episodes can have very different reasoning
and command lengths. Dropping entropy is only one difference: both explicit
length terms have also been replaced, and the sampling unit is now a row.

Minimal text-only improvement: compute/store reasoning and action lengths per
complete trajectory using the same tokenizer/masks; state coefficients and
`gamma=0` explicitly; sort trajectory IDs and preserve turn order within each.
For a fuller implementation collect the defined teacher entropy and pin its
aggregation, coefficients and schedule. Do not substitute initial-student NLL
and call it teacher entropy. The v5 main-text combined score and Appendix N's
entropy discussion do not uniquely settle pacing; record the choice rather than
inventing stages. Any changed order/batches require retraining/evaluation; the
length-only improvement can reuse existing teacher text.

## Kang: first-thought prefix (FTP)

### 1. Original method and repository flow

**The first thought is teacher-generated, not student-generated.** FTP is used
during teacher trajectory acquisition; the student later learns that completed
text and needs no supplied FTP at deployment. This follows
[§4 and Fig. 3](https://arxiv.org/html/2505.17612v2).

The locally inspected pinned original implementation establishes the details:

1. `K/exps_research/unified_framework/processors/reasoning.py:91–110` constructs
   the separate reasoning request from the question and reasoning system prompt.
   `K/src/smolagents/prompts/teacher_model.yaml:1–12` requests reasoning plus an
   answer. This is a teacher reasoning experiment, not a student rollout.
2. `K/exps_research/first_thought_prefix/build_prefix_memory.py:28–37` stores
   `question -> "Thought: " + response.split("\n\n")[0] + "\n\n"`.
   There is no 40-word limit, whitespace cleanup, or correctness filter here.
3. `K/exps_research/unified_framework/experiment.py:77–80` restricts entries to
   questions present in memory. `processors/agent.py:112–128` looks up the
   question, registers a one-element prefix list, then runs the teacher agent.
4. `K/src/smolagents/agents.py:1488–1497` pops that element on the first model
   step. `K/src/smolagents/models.py:1264–1283` appends an assistant prefill,
   sets `add_generation_prompt=False`, `continue_final_message=True`, and
   prepends the prefix to the returned completion. Later steps retain the
   completed first turn in agent memory (`agents.py:515–526,1481–1486,1517–1527`).
5. `K/exps_research/unified_framework/filter_agent_training_data.py:43–84`
   filters scored logs; `K/exps_research/train_utils/preprocess.py:174–216`
   exports full conversation messages for SFT.
   `K/exps_research/finetune_sft.py:222–239` uses a multi-turn completion collator.

These pointers are under the [pinned original repository](https://github.com/Nardien/agent-distillation/tree/8884b80ea3d22e53a2e1e4b9fd324600d13e0430).

### 2. Our implementation and export

`tools/alfworld_teacher_pool.py:48–59` asks the configured teacher for an initial
ALFWorld plan using the goal/reset observation, then uses the same paragraph split
with `THOUGHT:` casing. `:301–322` captures the raw planning response before the
shared transport strips whitespace. `:62–88,348–357` stores one paid response per
task ID and reuses it across retries. No student generates a prefix here.

At `:363–375`, **step_index 0 only**, the collector adds a continuation instruction
and an assistant message containing the prefix. It joins prefix+reply unless the
reply already starts with the exact prefix, and stores the resulting complete
turn. Subsequent steps use command/observation history (`:378`), not full thoughts.

`attempt_material`, `:444–460`, stores the prefix and its paid CoT call ID.
`export_pool`, `:478–547`, preserves full teacher targets and separate commands,
replays commands for verification, seals `public/` and `sealed/`, and writes the
candidate-set index. `alfworld_support.py:440–450` transfers targets verbatim into
`teacher_react_turns`. Prefix records and memory live in the `.collection`
sidecar; planning and trajectory costs are separate. Failed purchases are charged.
Successful command replay verifies task execution, not that native prefill happened.

**Actual K32 route:** `configs/rtd/pi1_alfworld_k32_kang.yaml:1–16` chooses
`method: pi1_ce` over a fixed FTP bank; `tools/alf_pi1_train.py:113–118,170`
loads the full turns and runs plain CE. It does not call `first_thought` again.
`alfworld_cli.py:74` only offers SmartAD/SAD training; its `cost --method kang`
is accounting, not a Kang training command. The `kang` case in
`K32PaperTrainer` is not exposed by that CLI.

### 3–5. Deviations, minimal fixes, and reruns

**K1 — Prompted continuation is not native assistant prefill. High; changes the
trained model through acquired data.** The collector's request has no verified
equivalent of the original decoder continuation controls. Concatenating afterward
cannot establish that generation used the same prefix-conditioned token stream.
Exact echo removal also handles only exact echoes, not reformatted repetition.

Minimal fix for reproduction: use a backend supporting native assistant prefill,
set both original controls, retain the exact rendered prompt/request, and validate
that the prefix is an unfinished assistant turn. A hosted backend lacking that
feature must remain labeled `prompt-continuation FTP adaptation`; a stronger
instruction does not fix decoder semantics. Native-prefill correction needs fresh
trajectories and retraining/evaluation. Prefix CoT can be reused only if teacher,
question and planning contract are unchanged. Relabeling existing results needs
no computation rerun.

**K2 — Planning task/prompt and sampling are adapted. Medium; changes the trained model.**
The planning call explicitly forbids action and requests an initial household plan,
rather than a solution to the original reasoning task (`:48–54`). It uses the
ALFWorld reset observation, the Luna teacher, requested temperature 0 and the
collector's completion cap. Those are data-distribution choices. Preserve the
reset-only information boundary and record the exact prompt/teacher/effective
sampling; report this as ALFWorld planning FTP. Restoring a source-style reasoning
prompt or changing these settings requires new planning responses, trajectories,
training and evaluation. Task-ID memory keys and marker casing are sensible domain
adaptations, not evidence of a student-derived prefix or a broken extraction rule.

**K3 — Retry-to-success acquisition. Medium; changes selection only.**
The collector can try three trajectories with one reused plan (`:614–632,663–667`),
retaining a success, whereas the paper's §5 reports one sampled trajectory per
question followed by filtering. Minimal fix: record retry count/retention policy
and call the existing bank a budgeted adaptation; for the single-draw variant
retain only first attempts and exclude failed tasks, with full costs retained.
Where complete ledgers exist that subset can be exported offline; changing
sampling or adding attempts needs acquisition. A changed retained set requires
training/evaluation again. Reusing a question's prefix itself matches the memory
design and is not a separate error.

**K4 — Legacy training can still fabricate a retrospective prefix. High;
changes the trained model.** `paper_train.py:231–235` unconditionally calls
`paper_data.first_thought` for legacy `method == "kang"`; `paper_data.py:193–214`
builds a 40-word summary of already collected targets and prepends it. That path
neither acquires FTP trajectories nor checks acquisition provenance. `TeacherRow`
has no acquisition-method field (`paper_data.py:68–77`). The earlier audit's
claimed `KANG_PREFIX` guard / `paper_fidelity.py` are **absent at this HEAD**.

Minimal fix: keep the explicit pi1 FTP-bank route; rename/isolate the legacy
summary as `kang_action_list_summary`; carry and validate acquisition method,
prefix record/call identity and continuation mode in a dedicated FTP loader.
Reject plain or retrospectively rewritten rows from that route. Relabel an old
summary result without rerunning it; obtaining an FTP result requires a proper
bank and training/evaluation. An existing valid FTP bank can be reused for this
routing fix. Do not rerun current pi1 FTP training merely because the legacy
route exists.

**K5 — SAG configuration is a separate evaluation adaptation. Medium; changes
selection only (inference action selection).** `paper_kang.py` implements SAG,
not FTP. `:16–25,34–72` votes over valid replayed public outcomes, selecting the
first winner on ties and first raw sample if all are invalid. The original
`K/src/smolagents/agents.py:1461–1474,1501–1515` has the corresponding execution
outcome voting logic. ALFWorld's observation/admissible/done key is a task-specific
substitute for code-execution output. The local default is `n=3, temperature=.7`;
the [paper's §5](https://arxiv.org/html/2505.17612v2) reports `N=8, temperature=.4`.
Expose/name those settings and report greedy FTP and FTP+SAG separately. Matching
the paper settings needs evaluation only, not training or teacher acquisition.
SAG is not a selector over purchased support tasks and does not become inert just
because all support tasks were purchased.

## Shared deviations affecting the above baselines

**X1 — Command-only, windowed conditioning instead of full reasoning history.
High; changes the trained model and SmartAD selection.**
`alfworld_support.py:413–418` uses only the last eight command/observation pairs,
truncating historical observations to 300 characters. The collector appends the
executed command, not the teacher's full response (`alfworld_teacher_pool.py:378`).
`pi1.py:190–196` re-renders verified states into these same prompts for training
and selection. Thus the archive contains full target thoughts, but later prompts
do not contain them. For Kang this includes the completed FTP first turn.

Minimal fidelity fix: preserve authored assistant turns alongside verified
environment observations and render full history for collection, scoring and
training, with an explicit context-limit policy. Existing archived targets can
support a separately named offline recontextualization experiment, but cannot
retroactively change the history on which the teacher actually conditioned.
Original-flow acquisition requires recollection; changed scoring requires
SmartAD reselection; changed training contexts require retraining/evaluation.
Keeping the frozen ALFWorld deployment context is a valid comparison constraint
only when this limitation is explicit.

**X2 — Text marker masking can discard teacher-authored text. Medium;
changes the trained model and SmartAD selection if triggered.**
Targets loaded by pi1 are authored replies, with real observations already in
the prompt (`pi1.py:93–108,184–196`). Nevertheless `paper_losses.py:74–75,110–111`
masks any substring recognized as an observation, including a tokenizer token
touching that span. A CPU example with a teacher-authored line
`Observation: this is authored text` is masked. `pi1.encode_teacher_turn` explicitly
says not to infer observation provenance from words in the reply (`:96–98`).
The SmartAD/SAD wrapper therefore has a different effective supervision mask from
plain pi1 CE despite sharing its encoding.

Minimal fix: construct observation masks from structured message provenance;
reject ambiguous embedded environment text rather than silently treating all
marker-like prose as environment output. Record the actual mask/denominator in
receipts. If an inventory proves no such tokens exist, no numerical rerun is
needed; otherwise rescore affected candidates and retrain affected methods.
This concern does not apply to the current Kang pi1 route's all-authored-token CE.

**X3 — Matched local experiment rather than original training recipe. Medium;
changes the trained model.**
The local protocol uses K32 support, Gemma/Luna, an ALFWorld command scaffold,
40-step collection horizon, native Gemma boundaries, LoRA rank 16/alpha 32,
constant LR 1e-5, and 3/10-pass endpoints, with roughly 512 supervised tokens per
update. See `alfworld_training.py:45–62,122–149`, `alfworld_curriculum.py:27–55`,
`pi1.py:60–108`, and the Kang YAML. Compare each paper's experimental setup, not
just its loss formula. For example Kang §5 uses rank 64, two epochs, batch 8 and
LR 2e-4; SAD v1 Appendix B specifies a different schedule.

Minimal fix: name the matched ALFWorld adaptation, log all settings and boundary
semantics, and reserve “reproduction” for a separately registered original recipe.
Appending the student's native boundary is not intrinsically a bug: special-token
choices differ across tokenizers. Its inclusion in NLL and loss is an explicit
local contract, not verified exact original EOS handling. No rerun for accurate
labeling; changing the model, data, boundary or optimizer requires the corresponding
selection/training/evaluation reruns. Do not silently change the frozen comparison
protocol while repairing a baseline.

## Documentation reconciliation and validation

**DOC — Low; cosmetic/documentation.** Minimal corrections to make in a future
authorized implementation change (none applied by this audit):

- Replace SmartAD's unresolved-normalization claim with the original Eq. 3 source
  and its confirmed mismatch; correct the “follows” claim in
  `docs/PAPER_BASELINES.md:226–235`.
- `docs/alfworld_k32_training.md:48–76` and
  `alfworld_curriculum.py:12–18` still say the original curriculum passage is
  unavailable. It is now verified; the **implementation** remains a proxy.
- `docs/alfworld_k32_training.md:80–84` describes SAD group means as the main path;
  `docs/alfworld_baseline_collection_notes.md:176–206` and current code instead
  select `sad_sum`. Pin v1/v5 and identify the reduction as normalized.
- Separate current full-target K32 banks from the 2026-09-17 command-only audit.
  That audit's SmartAD/CE equality assumes weight-sum normalization; current code
  gives `1.5*CE` on an all-action row and `2*CE` on an all-final row.
  Its prefix guard is not present here, and its support-budget argument does not
  describe FTP or inference SAG. Historical measurements may remain valid for
  their actual archived code/data, not for this HEAD.
- Separate `docs/PAPER_BASELINES.md:256–262`'s retrospective summary route from
  the current FTP collector and pi1 route. Never describe FTP as continuing a
  student's prefix. No computation rerun for documentation alone.

CPU checks executed with `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`
and Python `-B`. Only selected function ASTs were evaluated; collector/model
initialization was not executed. For NLLs `[1,2,3,4,99]` and kinds
`[reason,reason,reason,action,observation]`, the actual loss functions returned:

| Method | Loss | Gradients with respect to token log probabilities |
|---|---:|---|
| Plain CE | 2.5 | `[-.25,-.25,-.25,-.25,0]` |
| `sad_sum` | 2.5 | `[-.25,-.25,-.25,-.25,0]` |
| `sad_mean` | 3.0 | `[-1/6,-1/6,-1/6,-.5,0]` |
| Current SmartAD | 3.0 | `[-.25,-.25,-.25,-.375,0]` |

Also reproduced the S1 ranking reversal and S3/X2 parser cases above. Five FTP
extraction cases (normal paragraphs, leading blank lines, preserved whitespace,
thought tags, and empty response) matched the vendored rule modulo marker casing.
The final worktree check showed only this new report; existing tracked files had
no diff, and report formatting checks passed. Existing
tests were read, not run: `test_baseline_run_losses.py:55–62,92–97` explicitly
asserts the current SmartAD token-count denominator, so passing those tests would
not validate Eq. 5. No empirical count of changed winners, checkpoint quality,
or affected marker cases is claimed. No original SmartAD/SAD code was available
to settle undocumented batching, tie handling or pacing beyond the paper text.

## Files read

Files read in full or in relevant excerpts; directory/name-only search hits are
excluded. Paths are relative to this worktree unless prefixed `K/` as defined above.

**Implementation and configuration:**

- `src/bfas/rtd/baselines/alfworld_selection.py`
- `src/bfas/rtd/baselines/alfworld_training.py`
- `src/bfas/rtd/baselines/paper_losses.py`
- `src/bfas/rtd/baselines/paper_kang.py`
- `src/bfas/rtd/baselines/alfworld_curriculum.py`
- `src/bfas/rtd/baselines/alfworld_cli.py`
- `src/bfas/rtd/baselines/pi1.py`
- `src/bfas/rtd/baselines/paper_train.py`
- `src/bfas/rtd/baselines/paper_data.py`
- `src/bfas/rtd/benchmarks/alfworld_support.py`
- `src/bfas/adapters/alfworld.py`
- `tools/alfworld_teacher_pool.py`
- `tools/alf_pi1_train.py`
- `configs/rtd/pi1_alfworld_k32_kang.yaml`

**Tests and local notes:**

- `tests/test_alf_baseline_mechanisms.py`
- `tests/test_baseline_run_losses.py`
- `tests/test_alfworld_teacher_pool.py`
- `docs/alfworld_baseline_collection_notes.md`
- `docs/alfworld_k32_training.md`
- `docs/PAPER_BASELINES.md`
- `docs/2026-09-17-baseline-degeneracy.md`
- `docs/2026-09-14-analysis-update-conditional-response-zh.md`
- `docs/2026-09-15-user-direction-unified-method-and-theory-zh.txt`
- Search excerpts from other `docs/` files matching SmartAD/SAD/Kang/fidelity
  were used only to locate citations; no additional findings rely on them.

**Original Kang repository:**

- `K/exps_research/first_thought_prefix/build_prefix_memory.py`
- `K/exps_research/unified_framework/experiment.py`
- `K/exps_research/unified_framework/processors/agent.py`
- `K/exps_research/unified_framework/processors/reasoning.py`
- `K/exps_research/unified_framework/filter_agent_training_data.py`
- `K/exps_research/train_utils/preprocess.py`
- `K/exps_research/finetune_sft.py`
- `K/src/smolagents/models.py`
- `K/src/smolagents/agents.py`
- `K/src/smolagents/prompts/teacher_model.yaml`

External documents read: the SmartAD ACL landing page/PDF and SAD v1/v5 and Kang
v2 HTML papers linked above. This report was reread during final validation.

## Final deviation table

“Yes” means necessary to produce a corrected result, not work performed in this
read-only audit. Documentation-only relabeling never retroactively repairs a model.

| Baseline | Deviation | Severity | Fix | Rerun needed |
|---|---|---|---|---|
| SmartAD S1 | Token-micro NLL instead of turn-macro NLL | High; selection only | Mean of turn means; per-turn receipts; version/revalidate artifacts and legacy selector | Rescore/reselect; train/evaluate if winners change |
| SmartAD S2 | Weighted CE divided by token count; row/update reduction differs | High; trained model | Weight-sum denominator with declared trajectory/batch reduction | Train + evaluate |
| SmartAD S3 | Terminal-command final proxy; action span can absorb trailing prose | Medium; trained model | Declare final mapping; validate/bound structured command spans | If labels change; rescore if masks change |
| SmartAD S4 | Success-target retries and different effective sampling/pool size | Medium; selection only | Explicit draw/retention policy; source-style pool or adaptation label | New pool/selection/train/evaluate if protocol changes |
| SmartAD S5 | Hardcoded target 3; tie convention not sourced | Low; documentation | Read frozen target count; disclose ID tie rule | No numerical rerun |
| SAD D1 | Legacy/group-mean objective differs from hard token sum | High when claimed literal; trained model | Keep `sad_sum` default; label `sad_mean` ablation | Legacy mean runs need train + evaluate for sum claim |
| SAD D2 | `sad_sum` uses variable update token normalization | Medium; trained model | Name normalized variant; explicit raw-sum experiment if needed | No for relabel; train + evaluate for reduction change |
| SAD D3 | Single-softmax hard labels omit full v5 KL/action head | High vs full v5; trained model | Pin hard-label variant, or obtain distributions + implement action head | No for relabel; new sufficient data + train/evaluate for full v5 |
| SAD D4 | Turn-count row sorting replaces trajectory complexity curriculum | High; trained model | Span-length trajectory score; explicit entropy policy and pacing | Train + evaluate; entropy may require new teacher data |
| Kang K1 | Prompt continuation substitutes for native prefill | High; trained model | Native prefill controls or explicit adaptation label | Recollect trajectories + train/evaluate for native flow |
| Kang K2 | ALF planning prompt/teacher/sampling substitutes for reasoning setup | Medium; trained model | Record exact planning contract and effective settings | New planning/data + train/evaluate if changed |
| Kang K3 | Multiple retries instead of single-draw filtering | Medium; selection only | Explicit retry policy or first-attempt subset | Train + evaluate if retained data change |
| Kang K4 | Legacy `kang` still adds retrospective 40-word summary | High; trained model | Separate legacy name; validate FTP provenance; use existing pi1 FTP route | Affected legacy runs only; valid FTP bank reusable |
| Kang K5 | SAG uses n=3/.7 and ALF outcome key | Medium; inference selection only | Expose settings; distinguish greedy FTP from FTP+SAG | Evaluation only if changed |
| All X1 | Prior thoughts absent; command/observation window replaces full history | High; trained model + selection | Preserve/render authored history with explicit context policy | Recollect for original acquisition; reselect/train/evaluate as affected |
| SmartAD/SAD X2 | Marker-like authored text can be masked as observation | Medium, conditional; trained model + selection | Provenance-based masks; inventory affected rows | Only if masks/winners change |
| All X3 | Local models/data/optimizer/exposure/native-token recipe | Medium; trained model | Label matched ALF adaptation; separate original-recipe experiment | No for relabel; affected stages if recipe changes |
| All DOC | Stale normalization, curriculum, guard and FTP/SAG descriptions | Low; documentation | Version-pin sources and distinguish code/data routes | No |
