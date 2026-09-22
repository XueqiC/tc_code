The inference and training fixes have separate identities. Inference uses
`alfworld-evaluation-scoring-c26d-v2`; plain CE training uses
`pi1-alfworld-k32-plain-ce-v2` and records `target_boundary` in its run identity
and manifest. Existing v1 manifests are not silently upgraded.

To measure the inference fix, hold the checkpoint fixed and compare archived
v1 evaluation with a new v2 campaign. To measure the training fix, evaluate the
old and corrected training checkpoints under the same v2 harness. No evaluation
or training GPU job was run during this change.

The only changed ALFWorld `SCOPES` symbols are `HFBackend` and `VLLMBackend`.
The inventory, server identity symbols, command parser, action-selection
functions, prompts, and `official_episode` remain frozen. Parser diagnostics
observe exact membership and returned admissible strings, then freeze the
fallback flag immediately before environment execution. New per-turn fields
are `parser_fallback`, `parser_candidate`, and (only on fallback)
`parser_fallback_reason`. Existing fields retain their values.

This worktree lacked the pi1 module, CLI, config, tests, and their shared-trainer
and exposure-schedule support. Those prerequisites were brought in from the
local sibling checkout. The real bank is read from that checkout; it is not
copied or modified. Existing training arguments retain their meanings. The
optional `--bank` override locates the read-only bank in this isolated worktree.

Run from `/home/xueqi/hq/projects/tc-alignment-vllm`:

```bash
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ALF_BANK=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
ALF_MODEL=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/alf_pi1_train.py \
  --preflight --bank "$ALF_BANK" --model-path "$ALF_MODEL"
```

The complete printed sample is in `alfworld_turn_boundary_preflight.json`.
Its requested tail fields are:

```text
last_8_input_ids: [236787, 817, 531, 16083, 5127, 236743, 236770, 106]
last_8_label_ids: [236787, 817, 531, 16083, 5127, 236743, 236770, 106]
decoded_last_tokens: ": go to diningtable 1<turn|>"
tensor_shape: [1, 756]
assertions_passed: true
```

The preflight uses FrozenRenderer, PaperTrainer.encode, and the same
teacher-forcing tensor preparation used by the chunked scorer. Training forwards
one complete turn at a time and accumulates gradients across rows; there is no
padding or truncation. The preflight asserts the final label is the single native
special token 106, unmasked and retained after tensor construction.

Set `GPU_UUID` to the caller-chosen full GPU UUID. These commands are provided
for later execution, not run here. Use fresh output directories:

```bash
# Exactly one AdamW step over four real rows; verifies finite loss and nonzero
# loss derivatives at the boundary positions using the production trainer loop.
.venv/bin/python -B tools/alf_pi1_train.py --check-step \
  --seed 0 --gpu-uuid "$GPU_UUID" --output /tmp/alf-boundary-check \
  --bank "$ALF_BANK" --model-path "$ALF_MODEL"

# Normal training: same seed/output/GPU/model interface, endpoints from tokens.
.venv/bin/python -B tools/alf_pi1_train.py \
  --seed 0 --gpu-uuid "$GPU_UUID" --output /tmp/alf-pi1-boundary-s0 \
  --bank "$ALF_BANK" --model-path "$ALF_MODEL"
```

The measured bank denominator is 9,225 authored tokens + 424 boundaries = 9,649
supervised tokens per pass. Therefore three complete passes become 28,947
(+1,272), and ten become 96,490 (+4,240). The requested 28,099 / 92,674 endpoints
add the boundary tokens only once across repeated passes and are incompatible
with supervising the boundary on every turn exposure. The schedule computes
endpoints from encoded row lengths; neither endpoint is hard-coded.

CPU verification command (the real bank test is also read-only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ALF_PI1_BANK="$ALF_BANK" PYTHONPATH=src:tests:. OMP_NUM_THREADS=1 \
.venv/bin/python -B -m pytest -q -p no:cacheprovider -m 'not integration' \
  tests/test_alf_pi1_train.py tests/test_alf_vllm.py \
  tests/test_alfworld_boundary_diagnostics.py tests/test_alfworld_boundary_identity.py \
  tests/test_alf_eval_shard.py tests/test_rtd_alfworld_evaluation.py \
  tests/test_rtd_alfworld_identity.py tests/test_rtd_alfworld_rollout.py \
  tests/test_baseline_run_training.py tests/test_baseline_run_losses.py \
  tests/test_baseline_run_hotpotqa.py tests/test_baseline_run_seeds.py \
  tests/test_rtd_baselines_exposure.py tests/test_rtd_gemma4_scoring.py
```

Result: **425 passed, 2 deselected, 1 warning in 247.39 seconds**. The warning
is the tiny CPU PEFT export fixture's missing tokenizer-config warning. The two
environment integration tests are deselected; the requested inference tests
use stubbed servers. A separate final parser/identity/shard run passed all
39 tests. `git diff --check` passed.

The v1 source fixtures preserve the original projection hash
`a475f62fd4014f1ec43d1e1ea0b885283a43a2038b396ab9aec7964557f010e9` and run the
existing backend contract test against the archived v1 classes. The v2 frozen
projection hash is `4d666d0189b01a779e707d6f355e7bbe9584df21d4f96f2bd9828d91a6d79a7d`.

A broader check also ran `tests/test_rtd_runtime_memory.py`. Five parametrizations
of `test_training_worker_command_preserves_parent_environment` fail because the
test passes `{'output_root': 'unused'}` to `run_campaign`, which requires
`config['benchmark']`. Both that test file and `src/bfas/rtd/cli.py` are unchanged
from HEAD. Those unrelated CLI failures are not fixed by this boundary change.
Re-executing the original HEAD `run_campaign` function reproduces all five
failures (`KeyError: 'benchmark'`, 5 failed in 0.13 seconds).
