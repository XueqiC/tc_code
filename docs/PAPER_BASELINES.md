# Budgeted paper baseline runner

`tools/baseline_run.py` implements the four **protocol adaptations** for
`google/gemma-4-12B-it` and the sealed `gpt-5.6-luna` banks. This delivery is CPU
validated only: no model-weight loading, GPU experiment, teacher call, or
benchmark performance result was produced.

```bash
# Run later on one explicitly selected GPU. Repeat --method for sad/kang/gad.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/baseline_run.py \
  --method smartad --benchmark alfworld \
  --bank /home/xueqi/hq/projects/tc-alignment-uni/data/rtd/v1_1_alfworld_luna \
  --budget-tokens 30000 --seed 0 --run-dir results/paper_baselines/alfworld_smartad

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/baseline_run.py \
  --method gad --benchmark bfcl \
  --bank /home/xueqi/hq/projects/tc-alignment-uni/data/rtd/v1_1_bfcl_luna \
  --budget-tokens 5000 --seed 0 --run-dir results/paper_baselines/bfcl_gad

# Prepare the ALFWorld B/2, B, 2B curve on CPU in one invocation.
.venv/bin/python tools/baseline_run.py \
  --method smartad --benchmark alfworld \
  --bank /home/xueqi/hq/projects/tc-alignment-uni/data/rtd/v1_1_alfworld_luna \
  --budget-tokens 15000,30000,60000 --seed 0 \
  --run-dir results/paper_baselines/alfworld_smartad --prepare-only

# Audit HotpotQA purchases on CPU without touching an active results tree.
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/baseline_run.py \
  --method smartad --benchmark hotpotqa \
  --bank /home/xueqi/hq/projects/tc-alignment-uni/data/rtd/v1_1_hotpotqa_luna \
  --budget-tokens 7500,15000,30000,60000 --seed 0 \
  --run-dir /tmp/hotpotqa_smartad_audit --prepare-only
```

Supply exactly one of `--budget-tokens` and the backward-compatible
`--budget-fraction`; there is no implicit CLI budget. Token caps are nonnegative
integers (zero permits only zero-cost packages). A single cap uses `--run-dir`
unchanged. A comma list requires distinct caps and treats `--run-dir` as a common
prefix, creating sibling directories `alfworld_smartad_B15000`,
`alfworld_smartad_B30000`, and `alfworld_smartad_B60000` in the example above.
Each contains its own manifest and purchased rows, and, when run, training and
evaluation artifacts. Levels run sequentially in the supplied order with the
same frozen purchase order and seed. Omit `--prepare-only` to train/evaluate each
level in one invocation, using a fresh prefix.

`--seed` accepts any non-negative integer and controls training randomness only:
adapter initialization, global scoring/dropout RNGs, the exposure schedule,
preconditioner rollouts, and GAD sampling/discriminator initialization. Dropout
remains disabled by the existing recipe. Seed zero retains the paths above;
seed 7 appends `_s7` to each path (for example, `alfworld_smartad_B15000_s7`).
Pass the same unsuffixed `--run-dir` prefix for each repeat. The seed is recorded
in the manifest, hyperparameters, preparation summary, and evaluation metrics,
so curve reports can group by method/budget and compute mean and spread.

Purchase order and usable demonstrations remain frozen at purchase seed zero.
For nonzero repeats, a guard compares purchase metadata and the exact bytes of
`purchased_rows.json` with matching seed-zero runs in the same parent directory,
including a reference under a different name. Before the first training update,
it also compares `training_rows.json` and SmartAD's `smartad_selection.json`
(scores and selected IDs); evaluation rechecks these artifacts. Mismatches or
missing required reference artifacts fail loudly. Preparation can verify purchases
while selection is still pending; if no seed-zero run exists, the manifest records
`no_seed_zero_run`. Reference runs are read only. Exposure schedules may differ;
evaluation task selection, decoding, and evaluation seeds stay fixed.

Append `--prepare-only` for a CPU purchase/manifest audit. It creates no metrics
and launches no workers. Training and evaluation use separate child processes
so the training model releases GPU memory before vLLM starts. Existing run
directories are refused; this runner does not implement resume. Local model
files are mandatory; Hub downloads are disabled. Banks must be absolute paths
and are opened read-only. All baseline artifacts live in the specified run
directory, except the existing serving/port locks.

Append `--smoke` to a fresh run for **2 student commits, 2 rows per commit,
at most 2 distinct preconditioner prompts, and 3 evaluation tasks**. ALFWorld
uses the first three valid_seen tasks; BFCL uses the first three sorted
simple_python IDs with the official partial evaluator. Purchase/selection
still use the complete purchased data. Smoke receipts are marked
`official_full: false` and written to `smoke_metrics.json` rather than
`official_metrics.json`; they are not paper results. Kang's additional SAG
campaign uses the same three tasks.
HotpotQA retains the full frozen 500-question dev split and rejects `--smoke`.

Workers flush timestamped JSON progress to `train.log` and `evaluate.log`,
including rendering, selection, preconditioner rollouts, training loss and
generated tokens/s, export, and evaluation. Long stages emit a heartbeat every
30 seconds. Stage events include `gpu_held`: false for evaluation export,
renderer preparation, digesting and scoring; true while the training model or
evaluation server holds the device (including startup). This describes phase
ownership, not measured utilisation or the enclosing Slurm allocation.
Training throughput excludes preconditioner refresh time and counts
supervised/generated tokens, not prompt tokens. Manifest status changes to
`training` and `evaluating` while the workers run.

Full intermediate evaluation weights (`adapter` and `hub_merged`) use the first
existing, writable scratch directory in `SLURM_TMPDIR`, `TMPDIR`, `/tmp` order.
Each export gets a private `baseline-export-<job-id>-<unique-suffix>` directory
(PID when outside Slurm), kept until both official and Kang evaluation finish.
The server and renderer read the merged snapshot there. Scratch is removed on
success, exceptions and handled worker termination; SIGKILL/node loss still
requires node/scheduler cleanup. If no candidate is usable, exports retain the
existing `run/export` path and retention behavior. Metrics, official evaluator
output, manifest, purchase rows, exposure schedule and logs remain under the
shared run directory. Checkpoint and export digests still hash the same file
bytes and relative names, before scratch cleanup.

The B7500 run's existing `compute.jsonl` shows completed selection, a
52-response preconditioner refresh, and eight student commits by the end of
the reported silent interval. It was not stuck in reference soft targets:
these baselines use hard-label CE and have no frozen-reference soft-target
loss. The refresh took about 18 minutes, followed by roughly three minutes per
commit. The shared forward journal invoked full Python garbage collection and
CUDA cache clearing twice per forward, a source of CPU overhead removed for
this runner. These are log/code findings, not a measured GPU speedup.

Scoring now captures hidden states before the LM head, projects only action
positions in chunks of 32, and checkpoints those chunks during backward so
vocabulary tensors do not accumulate across the trajectory. SmartAD's initial
student NLL uses the same chunked path under no-grad. Model parameters/buffers
and scoring tensors must stay on logical `cuda:0` within the inherited
`CUDA_VISIBLE_DEVICES`; a device mismatch raises instead of offloading a loss.
Rendering is cached and token/span lookup avoids a full span scan per token.
PyTorch CPU threads are bounded to four. The shared frozen-source soft-gradient
routine also checks reference hidden states, logits, and labels before loss
computation; it remains separate from these baseline objectives.

All evaluation paths use the runner-owned vLLM server with
`start_new_session=True` (POSIX `setsid`). Shutdown sends TERM to its whole
process group and KILL to any remaining descendants even if the server leader
has already exited. On interruption the parent lets the worker run its server
cleanup before forcing it to exit.

## Purchase contract

Choose the absolute **teacher-output-token cap B a priori**, before inspecting
bank costs, success rates, purchased trajectories, or evaluation results. Fix
the same B for every method on a benchmark, independently of bank size or the
certified usable cost basis. Proposed defaults and prespecified budget-curve
levels are:

| Benchmark | B/2 | B (proposed default) | 2B |
|---|---:|---:|---:|
| ALFWorld | 15,000 | 30,000 | 60,000 |
| HotpotQA | 10,000 | 20,000 | 40,000 |
| BFCL | 2,500 | 5,000 | 10,000 |

These are protocol choices, not budgets fitted to observed performance. The
runner supports ALFWorld, BFCL and HotpotQA. Pass the chosen cap(s) explicitly
on the command line. The requested HotpotQA CPU purchase audit separately uses
7,500 / 15,000 / 30,000 / 60,000; these audit caps do not change the proposed defaults.

The authority for the frozen **purchase order and charging** rule is
`paper/sections/appendix.tex`, “Sealed-pool baseline runs”, alongside
`bank_build.py` and the v1.1 certificate.
Start from all public query IDs, sorted lexicographically, then shuffle once
with Python `random.Random(0)`. All methods use this identical order. Do not
shuffle only successful packages or skip costly failures.

With `--budget-tokens B`, the integer cap is exactly B, without rounding or bank
normalization. For legacy `--budget-fraction` runs only, the cap remains
`B = ROUND_HALF_UP(fraction * certified usable cost basis)`, with the unrounded
product recorded. Inspect the next package's recorded
`cost` in privileged sealed replay accounting. Purchase it only if cumulative
actual charges remain at most B; **stop at the first overflow**, even if later
packages would fit. This is an offline actual-content-cost simulation, distinct
from RTD's prospective public-upper-bound reservation. The method receives only
purchased usable behaviors. Unavailable/protected/failed attempts in the
inventory still cost their recorded tokens and supply no positives. Costs keep
the bank's exact/estimated confidence; historical collection costs are neither
replaced nor relabeled as new calls. Certificate and package digests are checked.

Historical read-only audit of the supplied banks on 2026-09-11 using the legacy
25% fraction rule (not the proposed absolute caps above):

| Bank | Usable basis | B at 25% | Charged | Purchased attempts | Usable trajectories | Training turns |
|---|---:|---:|---:|---:|---:|---:|
| ALFWorld Luna | 118,792 | 29,698 | 29,229 | 13 | 5 | 49 |
| BFCL Luna | 1,418 | 355 | 106 | 3 | 1 | 1 |

The next charges are 6,795 and 4,203 respectively. The low BFCL exposure follows
the specified prefix rule; it is not expanded to fill the remaining budget.
Purchasing succeeds even when no usable trajectory fits, but training then
fails explicitly and retains the charged-attempt manifest.

## Student data and optimization

Use the bank's native Gemma prompts/targets, including its tool schemas and
native call syntax. **Preserve the prompt/target boundary**: the current Gemma
chat template includes the empty thought prefill in the prompt. The older Qwen
baseline boundary shift described in the historical appendix is not applied to
Gemma. Teacher trajectory observations and prior turns remain masked prompt
context. Authored native termination tokens are trained once, without an extra
EOS after a terminal turn/handoff marker. Trailing whitespace after that marker
is not an additional generated action. Complete prompts are never truncated.

LoRA rank 16, alpha 32, dropout 0, and the seven target modules are read from
`configs/rtd/v1_1_alfworld_luna.yaml`: q/k/v/o projections and gate/up/down
projections. Student seed defaults to 0 and follows `--seed`. Load the same local
BF16/eager Gemma backbone and FP32 LoRA coordinates via the RTD loader, with its gradient
checkpointing and generation/scoring consistency guard.

Match the D0 control's **optimizer and student update budget**: 24 commits,
40 rows per commit, eta `1e-5`, no Adam/clipping/weight decay on the student.
Use the shared `FrozenStep` and `rms_diagonal` implementations; refresh P at
steps 1 and 13 from two fresh student responses per distinct purchased prompt,
with relative damping .01 and mean diagonal 1. Rows are sampled with replacement
using a separate training-seed RNG; the exposure schedule is saved. SmartAD's
selection and Kang's prefix change the method-specific training rows.

These are budgeted few-shot baseline runs, **not P1 recorded-V0-exposure arms**:
there is no V0 schedule input or return-gradient feedback/control loop. The
preconditioner uses purchased training prompts; this does not claim identical
numeric P, training/feedback exposures, or GPU cost to D0's full P1 campaign.
No calibration or evaluation score selects hyperparameters/checkpoints.

## Method definitions and deviations

**SmartAD.** For each task, score all its purchased verified trajectories with
the initial base-equivalent student (fresh LoRA has zero initial delta). Select
the lowest generated-token mean NLL; break ties by package ID. Score before any
update and retain complete selected trajectories. The loss is weighted token
CE divided by generated token count: reason 1, action 1.5, final decision 2.
Plain ALFWorld commands are actions and the last verified episode command is
the terminal decision. BFCL tool calls are actions; non-tool final prose is the
final decision. Selection scores/IDs are saved. This follows the selection and
segment weighting of [SmartAD](https://aclanthology.org/2026.findings-acl.1349/),
with the requested fixed weights and benchmark-specific terminal-action label.

**SAD.** Compute separate mean CE on REASON and ACT spans, then average the
present span groups; final decisions join ACT. Recognize `[REASON]`/`[ACT]`,
THOUGHT/ACTION, and native Gemma/tool delimiters without inserting pseudo-tags
into deployment text. Observation spans always have weight zero. This is a
text-only adaptation of [SAD](https://arxiv.org/abs/2505.13820): teacher
distribution/logit alignment is replaced with hard labels; there is no
teacher-logit KL or feature-alignment term. ALFWorld's converted bank contains
commands rather than the collector's private reasoning. Missing reasoning is
not fabricated; action-only rows reduce to action CE. The empty template
prefill is prompt context, not a fake reasoning example.

HotpotQA keeps native bank prompts and uses the 100-token `agent_action` cap.
Numbered `Thought n:` and `Action n:` spans and standalone `search`, `lookup`
and `finish` actions are recognized. The last user-turn prefill distinguishes
an inherited thought continuation from an action-only retry, including a retry
that produces plain text. Wikipedia observations are masked. Only `finish`
gets the final-decision weight; earlier reasoning in the terminal row stays
reasoning. SAD groups both tool actions and terminal answers under ACT.

**Kang / Agent Distillation.** Derive a deterministic, at-most-40-word
retrospective summary from each purchased trajectory's text and prepend it as
THOUGHT to its first supervised response. Later prompts/targets stay unchanged.
This replaces the original teacher-generated CoT first-thought procedure with
offline extraction, as required by the no-new-teacher-call constraint. At
inference, the student produces the thought itself; no teacher-derived prefix
is supplied for evaluation tasks. [Original paper](https://arxiv.org/abs/2505.17612).

On ALFWorld and BFCL, Kang has an additional full evaluation with **n=3, temperature=.7** and stable
per-task/per-step seeds. Vote by execution result, tie-breaking by first sample.
ALFWorld probes replay the already executed commands into fresh environment
workers, check that they reproduce the current public state, and execute each
candidate there. Votes use observations/admissible commands/terminal status,
never success/reward labels. BFCL executable multi-turn tasks use deep copies
of the current official harness objects; only the winner executes on the live
objects. Errors have no vote; when all candidates fail, retain the first raw
candidate for normal official handling. For BFCL schema-only tasks and
external/file-backed memory/search tasks, no cloneable execution environment is
available: use official decoded-AST consistency (final text for no-call answers).
Each vote key records `executed`, `decoded_ast`, or `final`; this fallback is an
explicit departure from fully executable SAG, never a correctness-oracle vote.
HotpotQA currently reports Kang's official greedy score only, with
`kang_self_consistency.status: unsupported` in `metrics.json`. Its manifest
records no SAG sample count or temperature. A HotpotQA SAG implementation
would need to sample complete ReAct episodes and vote over normalized final
answers; a second greedy campaign is never labeled SAG.

**GAD.** A separate small discriminator (257 byte symbols, embedding width 32,
GRU hidden width 64, conditional pair MLP) reads complete prompts and responses.
It learns Bradley–Terry preference for the purchased teacher response over four
fresh current-student responses at the **same exact purchased prompt**. A
missing teacher response causes that prompt to be skipped before sampling.
Teacher-token charges do not recur when responses are reused.

Train the discriminator with AdamW, lr `1e-4`, weight decay 0, one update per
prompt. Its raw scores give detached group advantages `(r-mean(r))/max(std(r),
1e-6)` with population standard deviation. Constant-reward groups have zero
advantage. The student receives the clipped, token-mean GRPO surrogate (clip
.2); no gradient passes through the discriminator or sampled tokens. Each
response group receives one student update, so old log probabilities are the
detached current pre-commit score. Preserve the RTD generation/score guard.

Use four student SFT warmup commits with concurrent discriminator training,
then four rounds of five alternating discriminator/PG commits: **24 total
student commits**. Discriminator steps and sampling are additional compute and
are recorded, not claimed compute-matched to SFT. These fixed steps replace the
paper's epoch-based warmup/training, and KL coefficient is explicitly zero.
Prompt coverage is restricted to the purchased prompt support; “on-policy”
means fresh responses from the current student there, not new teacher responses
at previously unseen rollout states. [GAD paper](https://arxiv.org/abs/2511.10643).

## Evaluation and artifacts

ALFWorld uses `ALFWorldAdapter.evaluate` inside the existing vLLM serving lane,
ReAct, valid_seen **140**, greedy, 40 steps, 256 action tokens. BFCL uses the
installed official CLI with `google/gemma-4-12B-it-FC` / `gemma4_fc`, full **5,217**
generation entries, temperature **.001**, and native vLLM backend. Its runtime,
outputs and locks inside `BFCL_PROJECT_ROOT` use the run directory; installed
harness files remain unchanged. BFCL's official Overall CSV is the headline;
do not substitute an average of category accuracies. Existing RTD completeness
validators reject missing/duplicate tasks, missing score categories or errors.

HotpotQA uses `HotpotQAAdapter(seed=0)` with live retrieval through the same serving
lane on the merged student export. It evaluates the frozen first **500 dev
questions**, temperature **0**, seven steps, at most 14 model calls, and
**100 generated tokens per call**, with the byte-identical six-shot ReAct
prompt. Answer **EM** is the headline (`overall_accuracy_percent = 100 * em`);
answer F1 is separate. The ported validator checks ordered IDs, frozen question
and gold hashes, horizons, decoding settings and recomputed EM/F1. Receipts
include prompt/evaluator/split source hashes, dataset and retrieval mode,
checkpoint/export hashes and reserved GPU time. `BFAS_HOTPOTQA_OFFLINE=1`
explicitly selects replay and binds cached Wikipedia snapshots. The shared
campaign lives in `bfas.rtd.benchmarks.adapter_evaluation`. A cheap retrieval
preflight runs before training workers and before evaluation export/serving.
Partial or errored evaluation artifacts have `status="failed"`, `complete=false`,
null result scores, and an error reason. The policy is released even when
rendering or the campaign fails.

The local `envs/hotpotqa/data` is already populated and `envs/hotpotqa/cache`
links to `/home/xueqi/hq/projects/tc-alignment-ws/envs/hotpotqa/cache`. On a tree
without `envs/hotpotqa`, link that directory to the shared ws environment.
Runtime data/cache and the read-only bank are not copied into Git. Evaluation
disables teacher-pool imports and reads cached Wikipedia without creating cache
locks. The paper runner uses the ported registry providers; this integration
does not migrate the shared RTD experiment driver to the sibling tree's newer
feedback RNG or checkpoint scheduling protocol.

The completed-bank SmartAD `--prepare-only` audit is saved in
[`paper_baselines_hotpotqa_audit.json`](paper_baselines_hotpotqa_audit.json),
including the bank certificate, source identities, package IDs and charges.
The full manifests and purchased rows are under
`/tmp/tc-alignment-base-hotpotqa-smartad-audit_B<cap>`.

| Token cap | Charged tokens | Packages (usable / unavailable) | Positive rows |
|---:|---:|---:|---:|
| 7,500 | 6,879 | 6 / 0 | 32 |
| 15,000 | 6,879 | 6 / 0 | 32 |
| 30,000 | 25,622 | 7 / 4 | 39 |
| 60,000 | 59,466 | 14 / 10 | 79 |

Both smaller budgets stop before the same 8,221-token package: the next
cumulative purchase would cost 15,100. The 30k and 60k runs stop before packages
costing 5,487 and 541 tokens respectively. These are deterministic prefix
purchases, including paid unavailable packages, with zero new teacher calls,
new teacher tokens or GPU hours. They do not measure model performance.

`metrics.json` and `official_metrics.json` contain the official single-sample
score and benchmark-specific details. On ALFWorld/BFCL, Kang additionally writes `kang_metrics.json`
and includes `kang_self_consistency` in `metrics.json`; the SAG score has a
separate sampling label and is never presented as greedy. Both campaigns run
the same tasks and official checkers. Each vote is archived in
`kang_votes.jsonl` under its campaign directory. This doubles evaluation
campaigns for Kang and adds its candidate/probe cost.

`manifest.json` records B, the requested `budget_tokens` (or legacy
`budget_fraction` and its exact product), the complete frozen order,
purchased IDs and per-attempt charges/confidence/digests, teacher tokens,
positive counts, method hyperparameters, configuration/code hashes, base and
tokenizer identities, checkpoint identity, hardware, status and GPU hours.
GPU hours mean single-device reserved wall time during training or serving,
including startup/environment waits; CPU model export and BFCL scoring are
excluded. This is a resource-allocation measure, not CUDA-kernel utilization.
Failures retain charges, logs and measured resource time. `compute.jsonl`
records student commits, P refreshes, score checks and discriminator rewards.
Prepared runs have GPU hours zero and no benchmark score.

CPU validation command (no GPU or teacher/API execution):

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m pytest -q tests/test_baseline_run*.py
```
