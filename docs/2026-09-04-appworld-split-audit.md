# AppWorld repository + split audit and base-agent smoke (2026-09-04)

Scope: CRCD spec `docs/2026-09-04-crcd-next-tasks-from-user.md` §0 (audit before
implementation), §7.1 (adapter records), §7.2 (split audit), §11 P0 items 1 and 3.
Only train/dev metadata was inspected; no test task directory was opened, and no
ground-truth solution/evaluation content is reproduced here. Tools:
`tools/appworld_audit.py` (`split`, `smoke`), `tools/appworld_smoke.sh`; machine
readable outputs in `results/analysis/appworld_audit_split.json` and
`results/analysis/appworld_audit_smoke.json`; logs in `logs/t4_appworld*.log`.

## 1. What is installed

| piece | location | state |
|---|---|---|
| AppWorld package | `envs/appworld-official/.venv` (python 3.12), editable install of `envs/appworld-repo` | `appworld 0.2.0.dev0`, repo commit `a072b7a` (2026-02-17, "Fix bundle files #210") |
| AppWorld data | `envs/appworld-official/data` (symlinked as `envs/appworld-repo/data`) | data `0.2.0` (Oct 2025 changelog); 733 task dirs; `datasets/{train,dev,test_normal,test_challenge}.txt` |
| Agent scaffold | `envs/appworld-repo/experiments/code/simplified/react_code_agent.py` (`simplified_react_code_agent`) | official ReAct code agent; prompt `experiments/prompts/react_code_agent/instructions.txt`; local patch: `reasoning_content` may be `None` (vLLM) |
| Our adapter | `src/bfas/adapters/appworld_official.py` (`AppWorldOfficialAdapter`) | drives `appworld run <exp> --root envs/appworld-repo --without-setup --clear-first`, student = vLLM `bfas-policy`, teacher = same agent on Azure/ollama; evaluation = official `evaluate_dataset` (run automatically by `appworld run`, or `appworld evaluate <exp> <dataset>` offline) |
| Launchers | `tools/aw_collect_official.sh` (teacher = Azure gpt-5.4, vLLM lane via `bfas.run`) | teacher credentials exhausted (Azure P1/P2 403, ollama weekly cap) -> collection lane stopped |
| Legacy | `envs/appworld-venv`, `envs/appworld-data*`, `src/bfas/adapters/appworld.py` | in-house harness, superseded; not used |
| Student | `Qwen/Qwen3.5-4B` in the HF cache; served with `envs/vllm-serve/.venv/bin/vllm` 0.27.1 | |

Task counts: train 90, dev 57, test_normal 168, test_challenge 417 (= 732 listed ids
+ 1 extra dir). Task metadata lives in `data/tasks/<id>/specs.json`
(`instruction`, `supervisor` {name, email, phone}, `datetime`, `db_version`,
`canary_string`) and `data/tasks/<id>/ground_truth/` (`metadata.json`: `difficulty`,
`num_apps`, `num_apis`, `num_api_calls`, `num_solution_code_lines`, timing;
`required_apps.json`, `required_apis.json`; plus solution/evaluation files that
must never be read or prompted). The scenario (generator family) is the task-id
prefix: `<scenario>_<variant>`, e.g. `82e2fac_1`; `appworld.task.load_task_ids`
only exposes `difficulty` as a filter, there is no separate family field beyond the
id prefix. All 147 train/dev tasks have `mode = full`.

## 2. Adapter records vs §7.1

What one official run writes per task under
`experiments/outputs/<exp>/tasks/<id>/`: `logs/lm_calls.jsonl` (every request:
`input.messages`, `seed`, `temperature`; `output.choices`, `usage`
{prompt/completion/total}, `prompt_token_ids`), `logs/api_calls.jsonl`
(`{method,url,data}` per API call), `logs/environment_io.md` (executed code block +
execution output per interaction), `logs/logger.jsonl`, `dbs/*.jsonl` (final DB
state), `misc/usage.json` (token totals), and after evaluation
`evaluations/<dataset>.json` (`aggregate` TGC/SGC, `individual[<id>]`) plus
`tasks/<id>/evaluation/report.md`.

| §7.1 field | already recorded by the run | surfaced by the adapter (`Rollout.meta`) | gap |
|---|---|---|---|
| task id + split | id yes; split implicit (dataset file) | `task_id`, `category` = scenario | add `split` |
| instruction / legitimately visible info | in the first user message (`instruction`, supervisor name/email/phone, app descriptions) | inside `Turn.context.messages` | ok |
| full prompt state `s_i` | `input.messages` of every call | `Turn.prompt` (rendered) + `Turn.context` | ok |
| generated code block `y_i` | response text; code extracted by the agent's regex | `Turn.target` = whole response; `is_call_target` = contains ```` ```python ```` | add extracted code (done in `appworld_audit.task_record`) |
| execution output / error | `environment_io.md` | not read | add parser (done in `appworld_audit.parse_environment_io`) |
| snapshot / reconstruction id | final `dbs/`; `AppWorld.save_state/load_state` exist but the agent never calls them; env `random_seed` in config | none | record (task_id, random_seed, executed code sequence) as deterministic replay key; call `save_state` per step if branch evaluation needs mid-episode snapshots |
| model / teacher tokens | `usage` per call, `misc/usage.json` | `tokens` (sum of `total_tokens`), `TeacherEpisode.tokens_spent` | split prompt vs completion (done in `task_record`) |
| official online eval output | `evaluations/<dataset>.json` | `verified` = `individual[id].success`; `evaluate()` copies the json | keep per-task `individual` entry (pass/fail counts) |
| `complete_task` called | derivable from `api_calls.jsonl` URL `supervisor/complete_task` | none | add (done in `task_record`) |
| interactions / API calls | `environment_io.md` count; `api_calls.jsonl` count | `steps` = number of LM calls | add both counts (done in `task_record`) |
| §0 record provenance (git commit, config, seed, split, checkpoint) | config jsonnet is deleted by `OfficialRun.cleanup` | not stored | keep the config text and git hash in the result record |

Prompt policy: the official prompt template exposes only the instruction, the
supervisor's identity and app descriptions; API docs are discovered by the agent
through `apis.api_docs.*` (functional API form). Student and teacher run the very
same template, so §7.1 "same visible information" holds by construction. No
ground-truth field is read on the prompt path.

## 3. Split audit (§7.2)

Data: `results/analysis/appworld_audit_split.json`.

| | train | dev | test_normal | test_challenge |
|---|---|---|---|---|
| tasks | 90 | 57 | 168 | 417 |
| scenarios (id prefix) | 30 | 19 | not inspected | not inspected |
| variants per scenario | 3 (all 30) | 3 (all 19) | | |
| scenarios spanning train and dev | **0** | **0** | | |
| task-id overlap train/dev | 0 | | | |
| difficulty 1 / 2 / 3 | 36 / 36 / 18 | 30 / 24 / 3 | | |
| tasks using 1 / 2 / 3 apps | 60 / 24 / 6 | 36 / 21 / 0 | | |

The three variants of a scenario always share the same required-app set
(0 scenarios differ), so a scenario is a coherent generator family, but a family
never crosses the train/dev boundary.

Required-app combinations (the only train/dev-only label that spans splits):

| required-app combination | train tasks | train scenarios | dev tasks | dev scenarios | spans train+dev |
|---|---|---|---|---|---|
| spotify | 42 | 14 | 30 | 10 | yes |
| phone+venmo | 15 | 5 | 18 | 6 | yes |
| file_system | 9 | 3 | 3 | 1 | yes |
| phone | 6 | 2 | 0 | 0 | no |
| file_system+spotify | 3 | 1 | 0 | 0 | no |
| phone+simple_note | 3 | 1 | 0 | 0 | no |
| phone+simple_note+venmo | 3 | 1 | 0 | 0 | no |
| simple_note+spotify | 3 | 1 | 0 | 0 | no |
| file_system+phone+venmo | 3 | 1 | 0 | 0 | no |
| simple_note | 3 | 1 | 0 | 0 | no |
| venmo | 0 | 0 | 3 | 1 | no |
| file_system+simple_note | 0 | 0 | 3 | 1 | no |

Per-app presence (task counts): spotify 48/30, phone 30/18, venmo 21/21,
file_system 15/6, simple_note 12/3 (train/dev). Only five apps are required across
train+dev; gmail/amazon/todoist/splitwise appear in the world but are never required.

Answers to the §7.2 questions:

1. *Do scenario families have multiple variants spanning train and dev?* No. Every
   family has exactly 3 variants and lives entirely in one split. Train and dev are
   scenario-disjoint by construction (this matches the benchmark's design).
2. *Support / acquisition / calibration / held-out per scenario?* Within a family
   only 3 tasks exist, so K in {2,4,8} few-shot support plus disjoint acquisition,
   calibration and held-out variants of the *same* scenario is impossible. Per
   spanning app-combination: `spotify` 42 train / 30 dev tasks (14/10 scenarios),
   `phone+venmo` 15/18 (5/6), `file_system` 9/3 (3/1).
3. *Can scenario membership be used for grouping without exposure?* Yes. The
   scenario id is only the task-id prefix, and required apps / difficulty are read
   from `ground_truth/` by the audit tool alone; none of these reaches the prompt
   (`task_record` stores them next to, not inside, the interaction). The adapter's
   `category` = scenario already groups rollouts this way.
4. *Do train/dev support episodic few-shot specialization without leakage?* Only
   at the app-combination level. Because dev scenarios are unseen families, any
   few-shot episode built from train scenarios and evaluated on dev measures
   transfer to new task generators of the same app family, never memorisation of a
   scenario.

### Recommended protocol

The spec's preferred scenario-spanning protocol is **not supported** by the data;
do not fabricate it. Use the fallback prescribed in §7.2:

*Primary (app-family episodes, train/dev metadata only).*

- Target families = the three app combinations that span both splits:
  `spotify` (14 train / 10 dev scenarios), `phone+venmo` (5 / 6),
  `file_system` (3 / 1, small -- report but do not tune on it).
- Per family and per support seed: pick K in {2,4,8} *train* tasks from
  distinct scenarios as S_few; the remaining train tasks of the family are the
  acquisition pool for the teacher; hold out 1 train scenario (3 tasks) of the
  family as the atom-validation/calibration set (never in S_few or acquisition);
  evaluate on all dev tasks of the family (unseen scenarios). For `spotify` this
  gives 42 - K - 3 acquisition tasks and 30 dev tasks; for `phone+venmo`
  15 - K - 3 and 18.
- Family labels are used only by the driver to select task ids; the prompt is the
  untouched official template. Cross-family dev tasks (`venmo`,
  `file_system+simple_note`, 6 tasks) are reported as an out-of-family control.
- Repeat over 3 support-set seeds per family; report TGC and SGC on the family's
  dev subset and on full dev.

*Secondary (single target distribution).* Treat AppWorld as one distribution:
S_few = K train tasks from distinct scenarios (stratified by difficulty 1/2/3),
acquisition = the remaining train tasks minus a held-out set of 5 train scenarios
(15 tasks) kept for calibration, evaluation = all 57 dev tasks. This is the
protocol the existing `configs/support_split.json` mechanism already implements
(support/demand/calibration over the 90 train ids).

*Final reporting.* After everything is frozen on train/dev: one run on
`test_normal` (aggregate TGC/SGC only); `test_challenge` only if compute permits.
No test id is opened during development; `tools/appworld_audit.py` reads only the
test id lists for counting.

## 4. Deterministic base-agent smoke (§11 P0 item 3)

See section 4 results below (filled by the smoke run; raw data in
`results/analysis/appworld_audit_smoke.json`).

SMOKE_RESULTS_PLACEHOLDER

## 5. Blockers for AW-1

- Teacher: no working teacher endpoint (Azure P1 and P2 weekly budgets exhausted,
  ollama weekly cap). AW-1 base characterization can run now on rai; the teacher
  half must wait for quota.
- Adapter gaps listed in section 2 (split tag, execution output, code block,
  complete_task flag, API-call counts, replay key, config/git provenance) are
  implemented in `tools/appworld_audit.py::task_record` and should be folded into
  `AppWorldOfficialAdapter._rollouts_from_run` before AW-1 records are produced.
- Existing partial artefacts from the earlier collection lane
  (`experiments/outputs/simplified_react_code_agent/bfas/s0_*`, 30 verified
  gpt-5.4 demos in `results/bfas/appworld/`) are reusable as a fixed teacher pool
  for offline replay.
