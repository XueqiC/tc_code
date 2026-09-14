# Table 1 baseline fidelity ledger

Source rule: use the official repository where one exists; otherwise implement
from the **paper**. Paper reimplementation is the normal case for SmartAD and
SAD, which have no released official repository. Their method names are
`smartad` and `sad` throughout configuration, manifests, metrics, table labels
and run directories. Implementation provenance belongs in the manifest; metrics
carry no per-arm fidelity or caveat fields.

This change is implementation and CPU validation only. No teacher/API requests,
GPU work or SLURM submissions were made. Everything under `archive/` remains
sealed. The runner version is `budgeted-paper-baselines-v2-fidelity`; use fresh
names such as `hotpotqa_smartad_v2`, `hotpotqa_kang_first_thought_prefix_v2`, and
`alfworld_sad_v2`. No archived score is corrected by these code changes.
`results/paper_baselines/` is absent in this worktree; the archived manifests and
local `data/rtd/v1_1_{hotpotqa,alfworld}_luna/` banks were available.

| Method | Implementation source and mechanism | Implementation and experimental setting | Teacher-output-token accounting |
|---|---|---|---|
| SmartAD (`smartad`) | Paper: Eq. 5 normalizes weighted NLL by weight sum (reason 1, action 1.5, final 2); Eq. 3 averages assistant-turn mean NLLs for selection. | Both normalizations are corrected; observations are excluded and candidate counts are recorded. Archived cells had one usable candidate per represented task. | Loss/selector fixes cost **0** teacher tokens. K=2 spends both candidates within the same **30,000**-token cell, reducing task coverage. |
| Kang (`kang_first_thought_prefix`; historical arm `kang_action_list_summary`) | Official repository: question-keyed first-paragraph CoT memory, continued as the first teacher-agent assistant prefix. | The new acquisition path reserves and charges both stages and preserves produced targets. The legacy retrospective action-list target rewrite remains explicitly **NOT the published mechanism**. | Each task's CoT pass and fresh prefixed trajectory share **one 30,000-token ledger**. Existing unprefixed trajectories cannot be converted retrospectively. |
| SAD (`sad`) | [Paper](https://arxiv.org/abs/2505.13820): the published objective aligns teacher **distributions per span**. | Our black-box teacher exposes **no logits to any method in this experiment**; SAD therefore optimizes the same REASON/ACT span structure against **hard labels**, averaging CE over present groups. This is the shared experimental setting. | The implementation and naming changes cost **0** teacher tokens; trajectory acquisition uses the same cell budget. |

SmartAD source: [Tang & Zhao, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1349.pdf),
Eqs. 3 and 5. Table 6 reports **30.92 at K=1 → 32.68 at K=10** for the 3B
student under uniform token-level loss. This measures what selection buys when
the output-token budget is not the binding constraint: the paper varies K,
without imposing our fixed 30,000-token cell allowance. It does not predict the
net benefit after reducing task coverage here. The paper samples K=10 at
temperature 1.0 before filtering; K=2 is our within-budget ablation.

The table aggregator reads historical `sad` directories as `sad`, and historical
`kang` directories as `kang_action_list_summary`, without modifying them.
Its default Kang row describes the historical summary arm, not newly acquired
FTP. Teacher/student, benchmark, training schedule and existing SAG differences
still apply; the acquisition change implements the official FTP mechanism
rather than reproducing an entire official experiment.

## Kang repository audit and acquisition contract

Official checkout: `envs/baseline_repos/agent-distillation/`, commit
`8884b80ea3d22e53a2e1e4b9fd324600d13e0430`, from
[Nardien/agent-distillation](https://github.com/Nardien/agent-distillation).
Read before edits:

- `exps_research/build_prefix_memory.py`: exactly
  `"Thought: " + response.split("\n\n")[0] + "\n\n"`; no stripping or word limit.
- `exps_research/unified_framework/processors/reasoning.py`: direct reasoning
  messages and `model(messages=messages, prefix=prefix)`. This processor removes
  `Thought: ` when consuming a prefix for **reasoning** mode.
- `exps_research/unified_framework/processors/agent.py` and
  `src/smolagents/agents.py`: the agent receives the stored prefix unchanged,
  registers it as a one-element list and consumes it only at the first step.
- `src/smolagents/models.py`: the vLLM adapter appends an assistant message,
  disables a new generation prompt, sets `continue_final_message=True`, and
  restores the prefix in the returned content. This is assistant continuation,
  not a user instruction asking the model to repeat a summary.
- `src/smolagents/prompts/teacher_model.yaml`: the default CoT system prompt
  is copied exactly and checked against the checkout by a CPU test.
- `scripts/inference/run_agent_teacher_train.sh`,
  `scripts/inference/run_cot_student.sh`, and
  `scripts/training/train_agent_ftp.sh`: prefix-memory flag, separate CoT mode,
  and training on a separately generated prefixed trajectory dataset.

The new `paper_acquisition.py` has two separable public functions:
`build_prefix_memory(questions, teacher, *, ledger, config, output_path=None)`
and `acquire_with_prefix(question, teacher, *, ledger, config, prefix_memory,
attempt_id, purchase_trajectory)`. They also exist as `KangAcquisition` methods.
Set `KangAcquisitionConfig.kang_mode="kang_first_thought_prefix"` (default).

The supplied teacher returns a `Ledger.online_request` receipt with `cost`,
`confidence`, `usage`, and `output`. `output` is the newly generated suffix;
the wrapper restores the prefix before the existing trajectory loop consumes
it. The provider adapter must expose `supports_assistant_prefix=True` and
implement real continuation. There is deliberately no provider client in this
module, and an adapter without prefix support fails locally before an agent
request. This change does not establish that the current Luna API supports
that capability; that must be supplied by a compatible adapter, not silently
approximated using user messages.

The existing environment purchase loop is injected as
`purchase_trajectory(question=..., model=...)`; every teacher turn must use
that model callback. The wrapper charges each call, passes the prefix only on
turn zero, and tags returned bank payload provenance or `TeacherRow` objects
with the acquisition method. A callback that merely returns cached data without
calling the model is rejected. Bank conversion must retain provenance and full
produced targets, and account for both CoT and agent charges from the acquisition
ledger (including failed requests); the training replay accountant does not
invent or perform missing CoT purchases. Training accepts FTP only on rows bearing that provenance;
it does not rewrite the first target. `--method kang` now selects FTP;
`--method kang --kang-mode kang_action_list_summary` explicitly selects the
legacy implementation.

Every CoT completion is charged in full, including discarded paragraphs.
Duplicate question occurrences each issue a request; the last occurrence wins
in the question-keyed dictionary, matching the official memory builder's
overwrite semantics. Unique question inputs therefore issue one request each.
Reservations precede calls; settlement precedes output validation. Billed
invalid output stays charged; unknown usage keeps its reservation. There is no
automatic retry. Use a fresh run name and attempt IDs for a redo.

## Fixed-budget cells: B = 30,000 total teacher OUTPUT tokens

**B = 30,000 is the Table 1 protocol for every method.** A fresh Kang cell
starts with one ledger containing 30,000 tokens. In acquisition order it pays
one complete CoT pass, then one trajectory purchase, then moves to the next
task. SmartAD at K=2 pays candidate 1, then candidate 2 for that task, then moves
on. Failures, rejected candidates and paid stages of unfinished pairs remain
charged. Stop at the first unaffordable reservation; do not skip an expensive
task or give prefixes or second candidates a separate allowance.

Sources are the **ordered `charges` arrays** in the seed-zero manifests:

- `archive/table1/table1_hotpotqa_{kang,smartad}/manifest.json`
- `archive/table1/table1_alfworld_{kang,smartad}/manifest.json`

Kang and SmartAD have identical archived charges within each benchmark. All
`_s1` and `_s2` counterparts also have identical charges: they repeat training
seeds, not acquisitions to multiply by three. A charge covers a whole
trajectory attempt, not one agent turn or just retained training tokens.

| Benchmark | Archived attempts / unique tasks | Archived usable packages | Output spend | Min / median / mean / max tokens per attempt | Confidence |
|---|---:|---:|---:|---|---|
| HotpotQA | 17 / 16 | **4** | 29,663 | 116 / 856 / 1,744.8824 / 6,560 | All 17 exact |
| ALFWorld | 13 / 13 | **5** | 29,229 | 136 / 2,341 / 2,248.3846 / 6,541 | All 13 estimated |

**CoT assumption:** actual plain-CoT lengths are **unmeasured**. The proxy is
the benchmark's mean archived trajectory charge, rounded up to integer tokens:
**1,745 HotpotQA / 2,249 ALFWorld per CoT**. Reserve **4,096 output tokens per
CoT request**, settle the entire completion including discarded paragraphs,
and release the unused reservation. Do not shrink the cap to spend the last
tokens. Prefixing can change trajectory length and success; for this projection
the fresh trajectory costs the same as its archived counterpart. Those episode
costs are known replay amounts, not online bounds: actual acquisition still
reserves each agent turn using `agent_max_output_tokens` and can stop earlier.

For SmartAD, candidate 1 uses the next archived attempt's cost; candidate 2 uses
the same rounded benchmark mean as its **unmeasured candidate-length proxy**.
Both are charged sequentially, even if candidate 1 fails. The offline model
reserves each trajectory's projected cost; these are estimates, not provider
caps. K counts attempts, not guaranteed correct candidates. The first 10
HotpotQA / 7 ALFWorld records have distinct task IDs, so completed pairs below
are also distinct tasks; HotpotQA's later duplicate is not reached.

For expected usable packages, let `p = archived usable / archived attempts`
(**4/17**, **5/13**). Project `completed pairs × p`, assuming unchanged task
yield. For SmartAD this explicitly assumes **no rescue by candidate 2**; it
credits no unmeasured correctness gain. The archive-outcome replay column
instead preserves the actual ordered `usable` flags. Neither is a measured
fresh-acquisition result.

| Benchmark / method | Archived usable packages | Complete pairs / tasks within B | Expected usable packages (unchanged yield) | Usable if archived outcomes recur | Charged / remaining tokens | Budget stop |
|---|---:|---:|---:|---:|---:|---|
| HotpotQA / Kang FTP | **4** | **10** | **2.35** | **3** | 27,499 / 2,501 | CoT 11 needs a 4,096 reservation. |
| ALFWorld / Kang FTP | **5** | **7** | **2.69** | **4** | 26,528 / 3,472 | CoT 8 needs a 4,096 reservation. |
| HotpotQA / SmartAD K=2 | **4** | **10** (20 candidates + 1 paid partial-pair candidate) | **2.35** | **3** | 29,584 / 416 | Task 11's first candidate costs 2,085; its second needs 1,745 and cannot fit. |
| ALFWorld / SmartAD K=2 | **5** | **7** (14 candidates + 1 paid partial-pair candidate) | **2.69** | **4** | 28,869 / 1,131 | Task 8's first candidate costs 2,341; its second needs 2,249 and cannot fit. |

Both paid partial-pair SmartAD candidates are unusable in the archive. A valid
first candidate of a partial pair in a new run could still be retained with its
actual K=1 recorded. SmartAD selects at most one correct package per task.
Second candidates can rescue failed first candidates: under the additional,
unverified assumption of independent candidate correctness with probability p,
completed pairs yield `n × (1 - (1-p)^2)` ≈ **4.15 / 4.35** packages. There are
no paired-candidate measurements here to choose between those yield models or
to estimate selection's accuracy benefit.

At average costs, two stages buy roughly **half the task coverage** of one
trajectory per task: `B / (2 × mean)` ≈ **8.60 / 6.67** pairs before integer
stopping, or **2.02 / 2.57 expected usable packages** at the archived yield
rates, next to the archived **4 / 5**. The ordered model completes **10 / 7**
pairs because early trajectories are relatively cheap. Usable outcomes also
cluster early; halving task coverage need not literally halve retained
packages. The ordered model preserves those effects instead of reordering
purchases to get a preferred count. The budget remains 30,000 in every case.

Reproduce the ordered model from the repository root (local arithmetic and
one in-memory ledger per cell; no teacher calls or file writes):

```bash
PYTHONPATH=src python - <<'PY'
import json
import math
from pathlib import Path
from bfas.rtd.ledger import BudgetError, Ledger

for benchmark in ("hotpotqa", "alfworld"):
    path = Path(f"archive/table1/table1_{benchmark}_kang/manifest.json")
    charges = json.loads(path.read_text())["charges"]
    other = Path(f"archive/table1/table1_{benchmark}_smartad/manifest.json")
    assert charges == json.loads(other.read_text())["charges"]
    mean = sum(c["tokens"] for c in charges) / len(charges)
    proxy = math.ceil(mean)
    p = sum(c["usable"] for c in charges) / len(charges)
    for method in ("kang_first_thought_prefix", "smartad"):
        ledger = Ledger(30_000)
        complete = []
        for i, charge in enumerate(charges, 1):
            t = charge["tokens"]
            stages = ([("cot", proxy, 4096), ("trajectory", t, t)]
                      if method == "kang_first_thought_prefix" else
                      [("candidate1", t, t), ("candidate2", proxy, proxy)])
            try:
                for stage, cost, cap in stages:
                    key = f"{i}:{stage}"
                    ledger.reserve(key, cap)
                    ledger.settle(key, cost, confidence="estimated",
                                  usage={"completion_tokens": cost})
            except BudgetError:
                break
            complete.append(charge)
        print(benchmark, method, dict(
            B=ledger.budget, archived_usable=sum(c["usable"] for c in charges),
            completed_pairs=len(complete), expected_usable=round(len(complete)*p, 2),
            replay_usable=sum(c["usable"] for c in complete),
            spent=ledger.spent, remaining=ledger.remaining, stopped_at=f"{i}:{stage}"))
PY
```

## Unbudgeted counterfactual: preserve the current trajectory-attempt count

This asks what it would cost to preserve **all 17 / 13 archived attempts**,
including failures, and add a CoT or second candidate to each. It is **not the
cost or budget of running a Table 1 method**. The earlier figures use the
unrounded mean proxy; rounding as in the ordered model adds just 2 / 8 tokens.

| Framing | Benchmark | Preserved allocation | Added CoT / second candidate tokens | Fresh trajectory tokens | Total output tokens |
|---|---|---|---:|---:|---:|
| **Unbudgeted counterfactual — Kang** | HotpotQA | 17 CoTs + 17 prefixed attempts (archive: 4 usable) | 29,663 | 29,663 | **59,326** |
| **Unbudgeted counterfactual — Kang** | ALFWorld | 13 CoTs + 13 prefixed attempts (archive: 5 usable) | 29,229 | 29,229 | **58,458** |
| **Unbudgeted counterfactual — SmartAD** | HotpotQA | 17 first + 17 second candidates | 29,663 | 29,663 | **59,326** |
| **Unbudgeted counterfactual — SmartAD** | ALFWorld | 13 first + 13 second candidates | 29,229 | 29,229 | **58,458** |

These preserve attempt allocation, not guaranteed usable counts. For a measured
Kang CoT mean c, the formulas become `17*c + new_agent_tokens` and
`13*c + new_agent_tokens`. A median-length CoT proxy contributes 14,552 / 30,433
tokens instead. One pass per attempted occurrence includes failures and
HotpotQA's duplicate; prefix reuse for that retry is a different allocation.
SmartAD's HotpotQA row is also attempt-matched: exactly two candidates for each
of 16 unique tasks would cost about **55,836** at the mean, also unbudgeted.
Reusing archived first candidates reduces new expenditure to the second batch,
but the first batch's spend still belongs in the cell total. Neither method
receives a budget increase.

## Archived-ledger dry-run diagnostic

The existing CLI remains read-only and cannot construct a teacher or launch a
worker. It checks whether a **CoT-only reservation envelope fits an existing
ledger**, not a fresh Table 1 plan. `--attempt-manifest` supplies request counts,
including failures, not online question text. `--questions questions.json`
instead accepts exact training questions.

```bash
python tools/kang_prefix_acquire.py --dry-run \
  --attempt-manifest archive/table1/table1_hotpotqa_kang/manifest.json \
  --ledger archive/table1/table1_hotpotqa_kang/manifest.json \
  --cot-max-output-tokens 4096 --run-name hotpotqa_kang_first_thought_prefix_v2

python tools/kang_prefix_acquire.py --dry-run \
  --attempt-manifest archive/table1/table1_alfworld_kang/manifest.json \
  --ledger archive/table1/table1_alfworld_kang/manifest.json \
  --cot-max-output-tokens 4096 --run-name alfworld_kang_first_thought_prefix_v2
```

In those already-spent ledgers, 17 / 13 CoT reservations at 4,096 each give
CoT-only envelopes of **69,632 / 53,248**, versus **337 / 771** remaining;
the diagnostic shortfalls are **69,295 / 52,477**. This shows that appending
all prefixes after spending the archived allowance cannot fit. It does not
imply a larger protocol budget: a fixed-budget redo buys fewer trajectories
from the start. No reservation is written; archived allowances do not authorize
online spending.

For a durable RTD JSONL ledger, supply `--ledger path/to/ledger.jsonl
--initial-budget B`. Read-only resume validates event hashes and held
reservations, refusing torn journals without repairing them or creating side
files. Agent request caps are separate and excluded from this CoT-only envelope.

Normalization and method-name changes consume zero teacher tokens. Repeating
student training seeds on one bank does not repurchase its teacher data.

## CPU verification

`tests/test_baseline_fidelity.py` exercises exact paragraph extraction, question
keys, duplicate occurrences, ledger reservation/settlement/failure behavior,
first-turn-only prefix continuation, real-ledger dry runs, SAD/SmartAD names
without metrics caveats, shared CoT/trajectory budgeting, legacy Kang opt-in,
preserved FTP training targets, and archive write guards. The baseline
loss/data tests cover single-kind cancellation, mixed-kind
weighting, observation gradients and Eq. 3 turn averaging. No test constructs a
provider client or loads a real model.

Follow-up baseline/table run: **291 passed** with:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_baseline_fidelity.py tests/test_baseline_run*.py \
  tests/test_table1_aggregate.py
```

The follow-up also executed the accounting example above, checked all 12 seed
manifests' charge arrays, and compared all **179 tracked archive files** byte for
byte with HEAD; they are unchanged. `git diff --check` passed.

The earlier baseline-fidelity run also exercised RTD ledger/broker/acquisition
checks. It found one unrelated existing failure,
`tests/test_rtd_v11_acquisition.py::test_v10_one_window_byte_identical_to_prechange_cpu_fixture`,
in an exact JSON comparison at floating-point precision
(`2.2841295676302185` versus `2.284129567630219`). The same failure reproduces in
isolation after loading the original `HEAD:src/bfas/rtd/ledger.py` into the test
process; its fixture was not altered. That separate suite was not rerun for
this follow-up.

## Files changed in this follow-up

- Documentation: [baseline_fidelity_ledger.md](baseline_fidelity_ledger.md),
  [PAPER_BASELINES.md](PAPER_BASELINES.md).
- Implementation: [paper_fidelity.py](../src/bfas/rtd/baselines/paper_fidelity.py),
  [paper_losses.py](../src/bfas/rtd/baselines/paper_losses.py),
  [paper_train.py](../src/bfas/rtd/baselines/paper_train.py),
  [paper_evaluation.py](../src/bfas/rtd/baselines/paper_evaluation.py),
  [baseline_run.py](../tools/baseline_run.py),
  [table1_aggregate.py](../tools/table1_aggregate.py).
- Tests: [test_baseline_fidelity.py](../tests/test_baseline_fidelity.py),
  [test_baseline_run_training.py](../tests/test_baseline_run_training.py),
  [test_baseline_run_seeds.py](../tests/test_baseline_run_seeds.py),
  [test_table1_aggregate.py](../tests/test_table1_aggregate.py).

## Changed paths across the original task and follow-up

Implementation and accounting:

- [paper_losses.py](../src/bfas/rtd/baselines/paper_losses.py)
- [paper_data.py](../src/bfas/rtd/baselines/paper_data.py)
- [paper_train.py](../src/bfas/rtd/baselines/paper_train.py)
- [paper_acquisition.py](../src/bfas/rtd/baselines/paper_acquisition.py) (new)
- [paper_fidelity.py](../src/bfas/rtd/baselines/paper_fidelity.py) (new)
- [paper_evaluation.py](../src/bfas/rtd/baselines/paper_evaluation.py)
- [ledger.py](../src/bfas/rtd/ledger.py)
- [baseline_run.py](../tools/baseline_run.py)
- [kang_prefix_acquire.py](../tools/kang_prefix_acquire.py) (new; dry-run CLI only)
- [table1_aggregate.py](../tools/table1_aggregate.py)

CPU tests:

- [test_baseline_fidelity.py](../tests/test_baseline_fidelity.py) (new)
- [test_baseline_run_losses.py](../tests/test_baseline_run_losses.py)
- [test_baseline_run_data.py](../tests/test_baseline_run_data.py)
- [test_baseline_run_hotpotqa.py](../tests/test_baseline_run_hotpotqa.py)
- [test_baseline_run_training.py](../tests/test_baseline_run_training.py)
- [test_baseline_run_seeds.py](../tests/test_baseline_run_seeds.py)
- [test_table1_aggregate.py](../tests/test_table1_aggregate.py)

Documentation:

- [baseline_fidelity_ledger.md](baseline_fidelity_ledger.md) (this new ledger)
- [PAPER_BASELINES.md](PAPER_BASELINES.md)
