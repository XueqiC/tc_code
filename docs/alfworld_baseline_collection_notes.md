# ALFWorld SmartAD / Kang FTP collection

Collection only. No NLL selection, loss change, student training, or evaluation
change is implemented here. No live purchase was made during development.

## Purchase commands

Run from `/tmp`, with credentials already in the environment. The sealed K=32
source is read from the main worktree because it is absent from this isolated
worktree. Both new output directories are outside `data/`. The price flags reuse
the completed D0 collection's accounting assumptions; they are not a claim about
current provider pricing.

```bash
cd /tmp
repo=/home/xueqi/hq/projects/tc-alignment-baselines
source_bank=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_20260921
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export BFAS_TEACHER=gpt-5.6-luna

"$repo/.venv/bin/python" -B "$repo/tools/alfworld_teacher_pool.py" \
  --source "$source_bank" \
  --out "$repo/artifacts/alfworld_k32_smartad_n4" \
  --method smartad --candidates-per-task 4 --attempts-per-task 12 \
  --workers 3 --rate-limit-retries 7 --max-tokens 4000000 --max-usd 5 \
  --usd-per-mtok-in 0.20 --usd-per-mtok-out 1.20 --usd-per-mtok-cached 0.01

"$repo/.venv/bin/python" -B "$repo/tools/alfworld_teacher_pool.py" \
  --source "$source_bank" \
  --out "$repo/artifacts/alfworld_k32_kang_ftp" \
  --method kang-ftp --attempts-per-task 3 \
  --workers 3 --rate-limit-retries 7 --max-tokens 1200000 --max-usd 5 \
  --usd-per-mtok-in 0.20 --usd-per-mtok-out 1.20 --usd-per-mtok-cached 0.01
```

The commands are provided for a later authorized purchase. Rerun the same command
to resume; token/dollar caps may increase, but method, candidate target, attempt
cap, source, collector identity, teacher and prices are bound to the output.
Budget/attempt exhaustion can leave shortfalls; four requested candidates are not
a guarantee of four successful or four distinct candidates.

## Artifacts and cost semantics

For either `--out X`, `X/public/` and `X/sealed/` are the standard audited bank.
Every attempt has a sealed payload; successful candidates retain
`payload_kind=teacher_react_turns`, with executed commands stored separately.
`payload.collection` carries candidate ID, sampling parameters, full purchase cost,
phase costs, and trajectory hashes. The existing `payload.cost` remains completion
tokens for compatibility; `collection.purchase_cost.tokens` includes prompt plus
completion (including hidden reasoning). Cached input is already part of prompt
tokens and is never added twice. Decimal dollar estimates use the supplied rates.

`X.collection/` contains:

- `teacher_ledger.jsonl`: one episode row per attempted trajectory, including failures.
- `usage.jsonl`: one reservation and at most one reported settlement per logical
  HTTP purchase, tagged `cot` or `trajectory`. Each multi-turn attempt sums its
  calls once. HTTP 429 retries share that reservation; other transport failures
  retain the uncertain envelope and are not retried within the purchase.
- `candidate_sets.json`: task-indexed attempts and usable candidates, opaque bank
  query IDs, stable task/attempt candidate IDs, sampling, per-attempt costs,
  attempted/verified/distinct counts, shortfalls and task totals. Identical
  candidates remain separate paid rows. Distinctness is exact full teacher payload;
  command-only distinctness is also reported. Bank/ledger/usage hashes bind the index.
- `summary.json`: per-task and aggregate attempts, verified and distinct candidates,
  total tokens, costs, uncertain calls, shortfalls, and separate CoT/trajectory costs.
- Kang additionally writes `prefix_memory.json` (`task_id -> exact prefix`) and
  `prefix_records.json` (original reasoning request/response, prefix and paid call
  ID). The summary hashes these files; sealed candidate metadata includes the
  prefix and CoT call ID. One CoT purchase is reused across retries of that task.
  Its cost belongs to the attempt that bought it; failed attempts and this shared
  planning cost must remain in the method's total even if only a later candidate
  is selected. A paid but lost CoT response blocks that task without repurchase.

Use the whole collection's costs for method comparisons, not just selected
candidate costs. Acquisition success uses the existing ALFWorld metric; exported
verified counts additionally require the existing CPU command replay to pass.

## Declared adaptations / deviations

- Both methods use the sealed K=32 ALFWorld train tasks, initial observations,
  existing ReAct scaffold/admissible commands, 40-step horizon, Luna teacher and
  official environment success/replay checks instead of their original domains,
  prompts, models and correctness checks. Deployment contexts and the frozen
  evaluation harness are unchanged. No original-method performance claim is made.
- SmartAD buys independently sampled trajectories until N successes or a bounded
  attempt/budget cap (default `3*N` attempts). It retains the collector's schedule:
  requested temperature 0 on task attempt 0, then 0.7 on every later attempt.
  Luna's existing transport omits temperature, so its actual sampling is provider
  default; the archive records both requested and transmitted values, without
  claiming a deterministic first sample or guaranteed diversity. No new sampler,
  seed manipulation, prompt diversification or deduplication is introduced.
- Kang replaces the vendored math/QA reasoning prompt with an ALFWorld planning
  question using only the goal and initial observation. Memory keys are task IDs;
  `Thought: ` becomes ALFWorld's `THOUGHT: `. The extraction is exactly
  `response.split("\n\n")[0] + "\n\n"`, with no whitespace normalization or
  thought-tag removal. The HTTP response callback preserves the raw string before
  the shared transport strips it. The plain planning call requests temperature 0
  (also omitted for Luna) and uses the existing 2,048-token completion cap.
- Vendored Kang uses a vLLM assistant prefill with `continue_final_message=True`
  and prepends that prefix to the returned completion. The existing hosted
  transport has no such control. This collector instead appends the exact prefix
  as an assistant message, asks for its continuation in the system instruction,
  and joins prefix plus continuation into the first supervised turn. An exact
  echoed prefix is not appended twice. This is a prompt-based continuation
  adaptation, not native forced decoder prefill. It seeds only the first turn;
  later turns retain the existing ALFWorld command/observation history window.
- Kang plans once per task before its first trajectory, then shares that prefix
  across the bounded trajectory retries (default three). Planning and failed
  trajectories are charged. Uncertain purchases retain conservative bounds;
  a missing paid planning response is reported rather than purchased again.

Reference implementation inspected read-only:
`envs/baseline_repos/agent-distillation/exps_research/first_thought_prefix/build_prefix_memory.py`,
`exps_research/unified_framework/experiment.py`, `processors/agent.py`, and
`src/smolagents/models.py` in that same repository.

## Token planning estimates

The existing D0 journal reports **681,640 tokens / 32 tasks = 21,301.25 per task**:
42 attempts, 30 successful tasks, 424 verified turns; 319,995 tokens on successful
attempts and 361,645 on failures. These estimates include failed attempts, not just
the roughly 10,666 tokens per successful trajectory.

- SmartAD N=4: use **about 85.2k tokens/task**, or **2.73M for K=32** (linear D0
  scaling). The command allows 4M as headroom. Repeated difficult tasks and the
  larger attempt cap may consume more; shortfalls are reported at the cap.
- Kang FTP: use **about 23–26k tokens/task**, or **0.74–0.83M for K=32**. This is
  D0's 21.3k plus planning and first-turn prefix overhead, assuming similar agent
  lengths/retry rates. The reset-only planning prompt's conservative byte bound
  averages 1,197.84 (maximum 1,524), plus at most 2,048 completion tokens: average
  reservation bound about 3,246 tokens for CoT alone. Actual CoT cost may be lower;
  prefix length and changed trajectory success are unmeasured. The command caps at
  1.2M. These are estimates from existing data, not a pilot purchase.

## Offline validation

```bash
cd /tmp
repo=/home/xueqi/hq/projects/tc-alignment-baselines
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$repo/src:$repo/tests" "$repo/.venv/bin/python" -B -m pytest \
  -q -p no:cacheprovider \
  "$repo/tests/test_alfworld_teacher_pool.py" \
  "$repo/tests/test_rtd_alfworld_identity.py" \
  "$repo/tests/test_appworld_teacher.py"
```

Tests use HTTP stubs and fake CPU environments. They cover N candidates, duplicate
retention, costs, effective sampling, failed-attempt shortfalls, resume, parallel
prefix storage, exact paragraph extraction, first-turn seeding, separate planning
bills, missing paid responses, and one reservation/settlement per successful
logical purchase including 429 retries. No credentials are inspected, no live
teacher is called, and no GPU is used.

Result on this worktree: **287 passed in 44.34s**. `git diff --check` passed;
changes are confined to the collector, its tests and this notes file. No symbol
listed in `alfworld_identity.SCOPES` was edited.

## SAD fidelity check against the paper (arXiv 2505.13820v1, read 2026-09-22 01:40 CDT)

- **Curriculum exists** (Eq. 7 + Algorithm 1 line 3): trajectories are sorted easiest -> hardest by
  `C(tau) = alpha*len(reasoning) + beta*len(actions) + gamma*entropy(pi_T(tau))`. No weighting, only ordering.
- **Implemented**: short-to-long ordering on the two length terms with alpha = beta.
  **Deviation**: the `gamma*entropy(pi_T)` term needs the teacher's token distribution, which a
  black-box text-only teacher (gpt-5.6-luna) does not expose; it is dropped and must be declared.
- **Hard-label branch** (Appendix D.3): `L = lambda_cot * sum_t m_r(t)*CE + lambda_act * sum_t m_a(t)*CE`
  with lambda_r = lambda_a = 1 - a masked token-SUM. Since every generated token carries exactly one
  mask and the weights are equal, this is plain token-level CE over generated tokens.
  **Deviation**: our `span_ce("sad")` normalises as mean-of-group-means (reason group and action group
  weighted equally regardless of length). The paper-literal hard-label SAD is therefore "plain CE +
  curriculum"; the mean-of-means form is a different objective and should be reported as such, or the
  sum form adopted as the default with mean-of-means kept as an ablation.

## SmartAD bank as exported (from the paused ledger; no further purchase)

- 87 usable candidate packages over 32 tasks: **28 tasks with 3 candidates, 1 with 2, 1 with 1, 2 with none**.
  SmartAD therefore trains on 30 tasks. Never describe this as N=3 for all tasks.
- Selection statistic: trajectory-wide per-token mean NLL under the untrained student, frozen to an artifact.
