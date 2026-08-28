# BFAS unified pipeline specification

Target: a single package `src/bfas/` such that running any benchmark is
`python -m bfas.run --benchmark <name> --arm <name> --seeds 3`. The
protocol is identical across benchmarks; only the adapter differs.
Every failure mode we hit this week is prevented by contract and by
build-time assertions, not by care.

## Protocol constants (module `bfas/protocol.py`, the ONLY place)

- SUPPORT_FRACTION = 0.05, SUPPORT_MIN = 50, SUPPORT_MAX = 250
- CALIB_FRACTION = 0.2 (of support)
- SUPPORT_SEED = 50, stratified over adapter-declared categories
- Pool = official train split if the benchmark has one, else the full
  task set; in the latter case support ids are excluded from evaluation
- TEACHER_ATTEMPTS = 3 (attempt 1 greedy, then temperature 0.7)
- Unguided student sampling: adaptive rounds with Beta(1,1) posterior
  per task; a task keeps sampling while 0.15 < posterior mean < 0.85
  and its round count < ROUNDS_CAP = 8; p-hat = posterior mean
  (Laplace smoothed). Temperature 0.7.
- Guided sampling: only for tasks with posterior mean < 0.5 that have a
  verified teacher demo; same round policy.
- Training: advantage weight (1 - p_hat) on the positive objective
  (existing trainer, AW_V3 path); guided-collected rows carry
  mu fields for the clipped importance ratio; lambda_pref stays 0.
- Seeds: 0, 1, 2.

## Adapter contract (`bfas/adapter.py`, abstract base)

Every benchmark implements:

1. `task_pool() -> list[TaskRef]` where TaskRef = (task_id, category).
2. `official_eval_split_disjoint() -> bool` — if True, evaluation uses
   the benchmark's own eval split unchanged; if False, evaluation is
   the pool minus support.
3. `rollout(policy, task_ids, temperature, guided_demos=None) ->
   list[Rollout]` — executes ONE rollout per task id, through the SAME
   serving path the final evaluation uses. Returns
   `Rollout(task_id, verified: bool, turns: list[Turn], raw)` where:
   - `verified` comes from the benchmark's own checker run on this
     exact rollout (positive evidence; NEVER inferred as the
     complement of a failure list).
   - `Turn(prompt: str, target: str)` pairs where `prompt` is
     byte-identical to what the serving stack renders at that point
     (the adapter must obtain it from the serving renderer itself,
     not re-implement it) and `target` is the exact emission the
     policy produced, reconstructed from structured fields when the
     serving stack stores calls structurally.
4. `teacher_demo(task_ids, attempts) -> dict[task_id, Demo]` — Demo
   carries the verified demonstration and a compact worked-example
   string for guidance contexts. Per-attempt verification through the
   same checker as rollouts.
5. `evaluate(policy_ref, out_dir) -> dict` — official metric on the
   evaluation side only, returning the benchmark's headline number
   plus per-category numbers.
6. `serving_probe() -> None` — starts/stops whatever server the
   benchmark needs and asserts one round-trip works.

`policy` is a lightweight handle: either a HF adapter dir or a base
model name; the pipeline owns vllm server lifecycle where needed
(port registry, one server per lane, GPU_UTIL configurable).

## Build-time audits (`bfas/audit.py`, run automatically in the
pipeline after pool construction; failure aborts before training)

- No pool row contains a `messages` key.
- Every row's prompt ends with the adapter-declared generation marker
  (adapter exposes `generation_suffix() -> str`).
- Round-trip format check: for 3 random rows, the adapter re-renders
  the row's context through the serving renderer and the result must
  equal the stored prompt exactly.
- Target-content policy: adapter exposes
  `target_policy(category) -> {"call_required", "prose_ok"}`; audit
  enforces it per row.
- p-hat sanity: values in [0,1]; at least one row with advantage > 0,
  otherwise the pipeline records a zero-update decision trace and
  skips training (the stopping branch), copying the base policy as
  the arm's checkpoint.
- Verified counts cross-checked against the checker's own summary
  totals per category; mismatch aborts.

## Arms (`bfas/arms.py`)

ours (advantage pool), sft, sad, agentkd, ddpo, pbsd, bbopd (each a
pool transform over teacher demos / rollouts, reusing the existing
appworld_train AW_DISTILL modes), star (self rows, plain), base
(evaluation only). Pool transforms live here, NOT in adapters.

## Reuse, do not rewrite

- Trainer: `src/appworld_train.py` unchanged (env-var interface).
- AppWorld adapter wraps `src/appworld_eval.py`,
  `src/appworld_teacher.py`, the existing support split
  `configs/support_split.json`, and the awb4/awb8 artifact layout.
- BFCL adapter wraps the official `bfcl_eval` package: generation via
  `bfcl generate --run-ids`, verification via `bfcl evaluate`
  per-round score files (positive summary totals), serving prompts via
  the model's registered handler `_format_prompt`, targets via the
  structured `tool_calls` field with the category target policy
  (multi-turn rows are EXCLUDED from ours training by default —
  `mt_training=False` — every construction degraded stateful
  categories; keep the plumbing behind the flag).
- ALFWorld adapter: local env (envs/alfworld sandbox), official
  unseen-split evaluation, train games as pool. Teacher-as-agent runs
  through the same chat client (`appworld_teacher.generate_reply`)
  with an ALFWorld prompt loop mirroring `src/alfworld_eval.py`.
- tau2 adapter: defer to a stub raising NotImplementedError with the
  planned interface documented (user-sim complexities; phase 3).

## CLI (`bfas/run.py`)

`--benchmark`, `--arm`, `--seeds`, `--gpu`, `--dry-run` (build pools
and run audits only). Writes artifacts under
`results/bfas/<benchmark>/<arm>_s<seed>/` and one line per finished
arm into `notes/exp_log.md`.

## Tests (`tests/test_bfas_conformance.py`, runnable without GPU)

- Split determinism and sizing rule on synthetic pools (5%/clamp/80-20).
- Audit failures fire on: a row with `messages`; a prompt with the
  wrong suffix; a call-required row with prose target.
- BFCL verdict extraction against a fixture score directory with a
  known pass/fail layout, including a category whose score file is
  missing (must yield unverified, not verified).
- Beta-posterior sampler: task solved 3/3 stops early; task 0/3 keeps
  sampling until cap; p-hat values are posterior means.

## Addendum (2026-08-28): export-identity control

The pipeline must include a serving-identity audit: before any arm's
evaluation is trusted, a zero-training control (base weights passed
through the identical load-save-export-serve path) must score within
noise of the directly served base model on the benchmark's official
metric. If it does not, the export path is reconstructing a different
architecture (e.g., dropped wrapper tensors, fallback attention
implementations) and every trained-arm number through that path is
invalid. Implement as `bfas audit-serving --benchmark <name>`, run
once per benchmark port and cached.

## Addendum 2 (2026-08-28, user directive): benchmark-native serving only

Evaluation must use each benchmark's OWN prescribed harness path for
local models — no custom export/serving constructions. For BFCL that is
`bfcl generate --backend vllm --local-model-path <ckpt>` (the harness
launches and owns the server); base uses the same flow with the hub
model. A trained checkpoint must first be made hub-faithful (trained
tensors overlaid onto the original hub checkpoint layout, original
config/tokenizer untouched — tools/bfcl_hub_merge_export.py) so the
harness sees exactly the artifact class it documents. The
export-identity audit (Addendum 1) then verifies the whole chain.
Custom pre-started servers are allowed ONLY where the benchmark's own
docs prescribe pointing at an external endpoint.
