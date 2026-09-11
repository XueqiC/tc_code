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
  --budget-fraction 0.25 --seed 0 --run-dir results/paper_baselines/alfworld_smartad

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/baseline_run.py \
  --method gad --benchmark bfcl \
  --bank /home/xueqi/hq/projects/tc-alignment-uni/data/rtd/v1_1_bfcl_luna \
  --budget-fraction 0.25 --seed 0 --run-dir results/paper_baselines/bfcl_gad
```

Append `--prepare-only` for a CPU purchase/manifest audit. It creates no metrics
and launches no workers. Training and evaluation use separate child processes
so the training model releases GPU memory before vLLM starts. Existing run
directories are refused; this runner does not implement resume. Local model
files are mandatory; Hub downloads are disabled. Banks must be absolute paths
and are opened read-only. All baseline artifacts live in the specified run
directory, except the existing serving/port locks.

## Purchase contract

The authority is the frozen **purchase** rule in `paper/sections/appendix.tex`,
“Sealed-pool baseline runs”, alongside `bank_build.py` and the v1.1 certificate.
Start from all public query IDs, sorted lexicographically, then shuffle once
with Python `random.Random(0)`. All methods use this identical order. Do not
shuffle only successful packages or skip costly failures.

The integer cap is `B = ROUND_HALF_UP(fraction * certified usable cost basis)`;
the unrounded product is also recorded. Inspect the next package's recorded
`cost` in privileged sealed replay accounting. Purchase it only if cumulative
actual charges remain at most B; **stop at the first overflow**, even if later
packages would fit. This is an offline actual-content-cost simulation, distinct
from RTD's prospective public-upper-bound reservation. The method receives only
purchased usable behaviors. Unavailable/protected/failed attempts in the
inventory still cost their recorded tokens and supply no positives. Costs keep
the bank's exact/estimated confidence; historical collection costs are neither
replaced nor relabeled as new calls. Certificate and package digests are checked.

Read-only audit of the supplied banks on 2026-09-11:

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
projections. Student seed is fixed at 0. Load the same local BF16/eager Gemma
backbone and FP32 LoRA coordinates via the RTD loader, with its gradient
checkpointing and generation/scoring consistency guard.

Match the D0 control's **optimizer and student update budget**: 24 commits,
40 rows per commit, eta `1e-5`, no Adam/clipping/weight decay on the student.
Use the shared `FrozenStep` and `rms_diagonal` implementations; refresh P at
steps 1 and 13 from two fresh student responses per distinct purchased prompt,
with relative damping .01 and mean diagonal 1. Rows are sampled with replacement
using a separate seed-zero RNG; the exposure schedule is saved. SmartAD's
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

**Kang / Agent Distillation.** Derive a deterministic, at-most-40-word
retrospective summary from each purchased trajectory's text and prepend it as
THOUGHT to its first supervised response. Later prompts/targets stay unchanged.
This replaces the original teacher-generated CoT first-thought procedure with
offline extraction, as required by the no-new-teacher-call constraint. At
inference, the student produces the thought itself; no teacher-derived prefix
is supplied for evaluation tasks. [Original paper](https://arxiv.org/abs/2505.17612).

Kang has an additional full evaluation with **n=3, temperature=.7** and stable
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

`metrics.json` and `official_metrics.json` contain the official single-sample
score and per-category scores. Kang additionally writes `kang_metrics.json`
and includes `kang_self_consistency` in `metrics.json`; the SAG score has a
separate sampling label and is never presented as greedy. Both campaigns run
the same tasks and official checkers. Each vote is archived in
`kang_votes.jsonl` under its campaign directory. This doubles evaluation
campaigns for Kang and adds its candidate/probe cost.

`manifest.json` records B and its exact product, the complete frozen order,
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
