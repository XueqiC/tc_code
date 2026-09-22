# ALFWorld K=32 training mechanisms

Implementation is additive: `tools/alf_baseline.py` dispatches to the new
`baselines/alfworld_*.py` modules. No edits to `pi1.py`, `paper_train.py`, existing
losses, any `SCOPES` file, or input banks. Tests and preparation used no GPU,
network, teacher/API request, or environment replay. No student training or
real student NLL selection has been run in this delivery.

## SmartAD selection

Statistic: **sum of unweighted supervised-token NLL across all turns, divided
by the total supervised-token count of the complete trajectory**. Choose the
minimum; break exact ties by candidate ID. It is neither total trajectory NLL
nor a macro average of turn means. Both total and mean NLL and their denominator
are recorded. Observation spans are excluded; the shared encoder's native
boundary is included and inherits the last supervised span kind.

The default `--statistic token_mean` preserves this legacy token-wide mean.
The [fidelity audit](alfworld_baseline_fidelity_audit.md) verified that SmartAD
Eq. 3 instead averages each assistant turn's mean NLL. The additive
`--statistic turn_mean` implements that statistic. New v2 score receipts retain
per-turn sums/counts and both statistics; see the
[fidelity variant commands](alfworld_baseline_fidelity_fixes.md) for offline
recomputation and the limitations of old trajectory-only receipts.

The initial student uses the same local snapshot, tokenizer, shared
`pi1.encode_teacher_turn`, scoring backend, precision and fresh base-equivalent
LoRA construction as training, before any optimizer step. Selection binds
snapshot/tokenizer/encoder/source hashes and versions. A subsequent encoder
change, including the native-boundary fix, invalidates old selection artifacts.
All three training seeds (0, 1, 2) use the same frozen selection.

`selection.json` contains every task, all verified candidate IDs, their scores
and purchase cost, task-level total purchase cost, the chosen ID, the choice
reason, shortfall flags, and the full chosen training rows. It is independently
hashed; training checks the argmin and row hashes and reads **only this artifact**,
without reopening the bank or reselecting. No dev/eval successes are used.
Scoring publishes one immutable receipt per completed trajectory; `--resume`
continues incomplete selection with identity checks and a process lock. Completed
artifacts cannot be replaced. An interrupted trajectory is rescored in full.

Coverage is **not N=3 for all**: **28/32 tasks reached 3, one has 2, one has 1;
the remaining two have zero**. There are 87 verified ledger candidates over 30
tasks. The two-candidate task still makes a real argmin choice; only the
one-candidate task says “single candidate, no choice.” Both are flagged below
the requested 3; zero-candidate tasks are explicit exclusions with retained costs.

## SAD curriculum variants

The fidelity audit verified SAD's trajectory complexity formula (v1 Eq. 7,
v5 Eq. 13): reasoning length, action length, and teacher entropy. The new
`--sad-curriculum trajectory_cost` uses authored reason/action token lengths
(final actions included), with `--sad-alpha 1 --sad-beta 1` by default. Teacher
entropy is unavailable from the black-box teacher and explicitly omitted.
It sorts complete trajectories by cost, preserves turn order within each,
and uses seeded trajectory ties in each exhaustive pass. Native boundary tokens
remain supervised but are excluded from the authored-length complexity score.

The unchanged default `--sad-curriculum turn_count` is an explicit proxy: in **each complete pass**,
order rows by increasing number of teacher turns in their verified trajectory.
Draw a seeded row permutation first; stable sorting preserves that random order
among equal-length trajectories. Every row is seen once per pass. Thus short
episodes precede long episodes, with deterministic seed-dependent ties. Full
teacher-forced context is retained even when turn rows are reordered. The
frozen schedule contains every pass order, every batch, and each row's difficulty.
The run and endpoint manifests contain the curriculum description and deviation.

The legacy schedule's frozen metadata retains its historical source-verification
notes for reproducibility. Those notes predate the audit. Neither variant claims
an exact original pacing recipe; both retain the matched 3/10 exhaustive passes.

## Training protocol and deviations

`K32PaperTrainer` subclasses `PaperTrainer` and uses its parameter/device setup.
It imports the shared encoder and calls `span_ce`: SmartAD reason/action/final
weights 1/1.5/2; SAD defaults to the equal-token hard-label `sad_sum`, with
`sad_mean` available as a mean-of-groups ablation. SmartAD retains token-count
normalization by default; `--smartad-variant weight_norm` selects `smartad_wsum`
and divides by the sum of weights across each optimizer update. Appended special-token IDs are
only classified; label construction stays exclusively with pi1. Thus the
`alf-eval-vllm` boundary fix (native token 106) flows through automatically.

The optimizer recipe comes from pi1's registered configuration and
`plain_ce_hyperparameters`: AdamW, LR 1e-5, betas (0.9, 0.999), epsilon 1e-8,
weight decay .01, gradient clip 1, LoRA rank 16/alpha 32/dropout 0, constant LR,
no warmup. Complete turns are accumulated to at least 512 supervised tokens
per update; batches flush exactly at 3 and 10 passes. There is one continuous
10-pass run per seed (0 and 1), with both endpoints saved. Each row's existing
span loss is weighted by its supervised-token count within the update by default, matching
pi1's reduction across rows. This differs from the old paper trainer's row-macro
reduction and preconditioned optimizer, as required by the matched pi1 protocol.
No teacher logits, feature alignment, or additional auxiliary objectives are added.
For `weight_norm`, rows instead contribute in proportion to their weight sums,
so the entire update equals `sum(w * NLL) / sum(w)`. This is explicitly an
update-wide matched-protocol adaptation, not a mean of per-trajectory losses.

CPU validation with the actual local Gemma tokenizer and boundary-fixed encoder
read all 424 D0 turns without truncation: 9,649 supervised tokens (5,869 reason,
3,435 action, 345 final), including native token 106 on all 424 turns. SAD's
endpoints are exactly 28,947 and 96,490 tokens. The seed-0 schedule has 184 AdamW
updates and begins with rows from 4-turn episodes, followed by 6-turn episodes;
the full difficulty range is 4–38 turns. This checks that the schedule is active
on the real bank, not merely on a synthetic fixture.

All interpretation/protocol differences are explicit: token-mean selection
normalization, incomplete candidate coverage, unverified SAD curriculum,
the requested hard-label losses and pi1 optimizer/exposure protocol, and the
benchmark-specific final-action/native-boundary span classification. Training
outputs always require fresh directories; training resume is not implemented.
Only selection offers `--resume`.

## Data cost

Costs come from each bank's own collection ledgers. The reader checks the
episode ledger's hash against the bank audit, reconstructs the latest reported
state of each request ID in `usage.jsonl`, and reconciles every attempt with
`teacher_ledger.jsonl`. Reservations and settlements are not separate purchases.
Unsettled reservations fail closed rather than being labeled actual cost.

Charge all support-task purchases, including failures, unselected candidates,
and tasks with no successful candidate. Kang includes both `cot` planning and
trajectory phases; planning is not double counted through the episode ledger.
Every training/endpoint manifest carries `teacher_data_cost`, ledger hashes,
per-task/per-attempt totals, phase totals, and the collector's price schedule.
Tokens are actual reported prompt+completion usage; cached input is a subset,
not extra tokens. USD is an estimate at the recorded prices, not an invoice.
No allocation of project totals across methods or seeds occurs.

| Data | Prompt tokens | Completion tokens | Total tokens | Estimated USD |
|---|---:|---:|---:|---:|
| D0 (CE and SAD, each) | 614,588 | 67,052 | 681,640 | 0.2033800 |
| Kang planning + trajectory | 703,318 | 90,983 | 794,301 | 0.2498432 |

Kang planning accounts for 17,592 tokens; trajectories account for 776,709.
Read-only cost manifests were generated in
`results/alfworld_k32_baseline_costs_v1/{ce,sad,kang}/manifest.json`.
For CE or Kang runs produced by another entry point, `cost --run-manifest`
copies the run receipt into a new directory with `teacher_data_cost`, after
checking its bank manifest hash. It records the original receipt hash and
directory for resolving checkpoint paths; it does not edit the original run.
SmartAD's final cost will be bound and recorded when its sealed bank is available;
no guessed or partial cost manifest is emitted.

## Commands

This `alf-baselines` checkout currently lacks `pi1.py`; the optional `--pi1-root`
imports its canonical module read-only from another worktree, without copying it.
The commands below use `alf-eval-vllm`, which already has the boundary fix.
After merging pi1, use the merged local config and omit `--pi1-root`.

```bash
PY=.venv/bin/python
PI1=/home/xueqi/hq/projects/tc-alignment-vllm
CFG="$PI1/configs/rtd/pi1_alfworld_k32.yaml"
MODEL=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
GPU_UUID=GPU-REPLACE-WITH-FULL-UUID

# (a) Requires the frozen bank AND matching .collection/candidate_sets.json.
$PY tools/alf_baseline.py select \
  --pi1-root "$PI1" --config "$CFG" --model-path "$MODEL" --gpu-uuid "$GPU_UUID" \
  --bank artifacts/alfworld_k32_smartad_n3 \
  --output results/alfworld_k32_smartad_selection_v1
# To continue interrupted scoring, repeat this exact command with --resume.

# (b) SmartAD: only the selection artifact supplies training data/cost.
for seed in 0 1 2; do
  $PY tools/alf_baseline.py train --method smartad --seed "$seed" \
    --pi1-root "$PI1" --config "$CFG" --model-path "$MODEL" --gpu-uuid "$GPU_UUID" \
    --selection results/alfworld_k32_smartad_selection_v1/selection.json \
    --output "results/alfworld_k32_smartad_v1/seed-$seed"
done

# (c) SAD hard-label adaptation with the explicitly unverified schedule above.
for seed in 0 1 2; do
  $PY tools/alf_baseline.py train --method sad --seed "$seed" \
    --pi1-root "$PI1" --config "$CFG" --model-path "$MODEL" --gpu-uuid "$GPU_UUID" \
    --bank "$D0" --output "results/alfworld_k32_sad_v1/seed-$seed"
done

# Read-only accountant, useful for CE/Kang table entries from other trainers.
CUDA_VISIBLE_DEVICES='' $PY tools/alf_baseline.py cost --method ce \
  --bank "$D0" --output results/alfworld_k32_ce_cost_new
CUDA_VISIBLE_DEVICES='' $PY tools/alf_baseline.py cost --method kang \
  --bank artifacts/alfworld_k32_kang_ftp --output results/alfworld_k32_kang_cost_new
# Optional: add --run-manifest /path/to/existing/run/manifest.json to either
# cost command to produce a cost-annotated run receipt in its fresh output.

# CPU tests, using the actual boundary-fixed encoder from its owner.
CUDA_VISIBLE_DEVICES='' BFAS_PI1_ROOT="$PI1" $PY -m pytest -q \
  tests/test_alf_baseline_mechanisms.py tests/test_baseline_run_losses.py \
  tests/test_baseline_run_training.py tests/test_alfworld_teacher_pool.py
```

The shared pi1 `select_device` and `verify_cuda_device` functions enforce
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, a full UUID, the driver's PCI mapping, and the
same UUID on logical `cuda:0` before model allocation. All model/tokenizer loads
are local-only. The command never downloads missing weights.

The test command above passed **139 tests** with `CUDA_VISIBLE_DEVICES=''`.
It covers independent total/token-mean/turn-mean selection oracles, deterministic
and interrupted selection, selected-row integrity, curriculum ordering and seed
reproducibility, actual optimizer updates against full autograd, both exposure
endpoints, the imported boundary fix, mocked GPU identity checks, cost manifests
for all four methods, real D0/Kang ledger sums, and related existing regressions.

At implementation time `artifacts/alfworld_k32_smartad_n3` and its
`candidate_sets.json` are absent; only its completed-looking raw collection
ledgers are present. They were not exported, rewritten, or treated as verified
bank state. Selection deliberately requires the frozen export before running.
