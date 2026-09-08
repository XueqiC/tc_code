# AppWorld intervention-event miner (AW-1, P1) — design and pilot (2026-09-04)

Scope: `docs/2026-09-04-crcd-next-tasks-from-user.md` §8/§9, item AW-1 of the
unified CRCD brief.  Deliverables: `tools/appworld_event_mine.py` (miner + env
worker), `tests/test_appworld_event_mine.py` (13 unit tests, mocked env/student),
this note.  Zero new teacher tokens: the teacher side is the 40 archived, verified
gpt-5.4 demos (`results/bfas/appworld/collect_shared/demos.json`; cost already in
`data/teacher_ledger/appworld.jsonl`).  Notation follows `docs/METHOD.md` §2
(ownership superscripts T/S; ΔU is a derived random variable, never a filter).

## 1. What the demos are

All 40 demos were collected under the OFFICIAL scaffold
(`simplified_react_code_agent`, `src/bfas/adapters/appworld_official.py`), not the
legacy in-house harness.  Each demo turn stores the agent's conversation
(`context.messages`) and the teacher's reply (`target`); every target is a single
```` ```python ```` block (0 prose-first, 0 multi-block, 0 code-free targets over the
40 demos), and the message list of turn *t* is exactly the instruction messages
(23 messages: official prompt with its few-shot dialogue + the task) followed by *t*
(assistant, "Output:" user) pairs — verified for all 40.  Demo lengths: 7–22 turns
(median 12).  All 40 demo task ids are in the demand split of
`results/bfas/appworld/ours_s0/support_split.json` (support 50 = demand 40 +
calibration 10); calibration ids are never mined.

## 2. Design

**Decision state s_t.**  The official agent's message list at teacher turn *t*:
instruction messages + the replayed teacher replies (`fixed content + "\n\n"`, exactly
as the agent appends them) + the live execution outputs (`"Output:\n```\n" + out + "```\n\n"`).
The row stores both the rendered prompt (student chat template with generation prompt,
ends in `<|im_start|>assistant\n<think>\n`) and `_render_context = {"messages": [...]}` so
`AppWorldOfficialAdapter.rerender` can rebuild it.  State hash for the unified schema =
sha1(prompt).

**Student query and thinking mode.**  The same request the official agent sends: chat
completions to the vLLM lane (`bfas-policy`), `temperature`, `seed`; plus an adaptive
`max_tokens` allowance (`--max-tokens`, default ceiling 4096) and, in the default `--thinking off` mode,
`chat_template_kwargs = {"enable_thinking": false}`.  This is the deployment
configuration of the AppWorld base chain (`tools/awoff_base_chain.sh` serves with
`--default-chat-template-kwargs '{"enable_thinking": false}'`; the official dev-57
14.0% TGC was obtained that way; `tools/appworld_smoke.sh` was patched to the same flag
on 2026-09-04 after T4 found that without it Qwen3.5-4B's default thinking floods
`content` with ~25k tokens, overflows the 32k context and kills the episode with HTTP
400).  The miner sends the kwarg per request *and* expects the server flag; the prompt
it stores is rendered with `enable_thinking=False` (ends in
`<|im_start|>assistant\n<think>\n\n</think>\n\n`), i.e. exactly the server-side prompt.
`--thinking parser` is the alternative (server with `--reasoning-parser qwen3`, thought
returned as `reasoning_content` and dropped as the official agent config does).  At
start-up `VLLMStudent.check_server` sends one short request and refuses to mine if
`</think>` text appears in `content`.  Code is extracted with a port of the agent's
`extract_code_and_fix_content` (first full ```` ```python ```` block, or an unterminated
trailing one), so the *executed* action is exactly what the deployment scaffold would
execute — including the case where the code block sits inside the model's thinking.

**Context budget and overflow.**  Every capped student request, including branch
continuations, uses `max_tokens_eff = max(256, min(student_max_tokens,
context_len - prompt_tokens - 64))`.  `--context-len` defaults to 32768 and must match
the server's context window.  Prompt tokens are counted with the already loaded
`--student` tokenizer, which should match the server tokenizer, using the same chat
template, generation prompt and thinking setting as rendering.  If that tokenizer is
unavailable, use the latest response's `usage.prompt_tokens` for an exact matching
conversation prefix plus `ceil(newly_appended_chars / 4)`; without matching usage,
estimate the entire message content at chars/4.  The bounded, thread-safe usage cache
matches message prefixes so concurrent branches and unrelated tasks cannot overwrite
one another's baseline.  `--max-tokens 0` retains the existing uncapped request behavior.

An HTTP 400 mentioning the context length gets exactly one retry with `max_tokens=256`
and the same messages and seed, independently of the transient-error retry budget.
If that retry also overflows during probing, emit an overflow row with
`_event_overflow=True`, `_student_finish_reason="overflow"`, `_student_max_tokens=256`
and the diagnostic in `_event_errors`, then follow the teacher and visit the next probe
state.  It consumes a probe slot but starts no branches: K=0, empty continuation
records, prior-only Q values of 0.5, and no student reply or observed agreement rate
(`_student_agree_rate=null`, `_student_samples=0`).  Incomplete sample sets are discarded
and do not contribute to `_agree_trace`; overflow rows count separately from agreements
and divergence events.  Every regular row records the effective allowance of its chosen
probe sample in `_student_max_tokens`, including a successful retry at 256, regardless
of later continuation allowances.  Other 4xx errors remain non-retried errors, transient
retries remain unchanged, and exhausted context retries within a continuation retain
the existing per-continuation error reporting in `_event_errors`.

**Equality test (`same_action`).**  Two replies are the same action iff the canonical
forms of their executed code blocks agree: `ast.dump(ast.parse(code))` when the block
parses (formatting, comments, blank lines and spacing are ignored), otherwise the
comment-stripped, whitespace-collapsed lines.  Different API names, argument values,
keyword order, variable names or wrapping (`print(x)` vs `x`) are different actions;
semantic equivalence of different programs is not attempted.

**Effect-level equality (`--equality api_effects`, added on the coordinator's request).**
The bridge returns, for every executed block, the API calls AppWorld's `RequestTracker`
recorded during it (`method`, `url`, `data`; admin calls are never tracked, `api_docs`
look-ups are calls).  Calls are normalised by dropping auth keys (`access_token`, `token`,
`authorization`, `api_key`, `session_id`), masking JWT-shaped strings and ISO timestamps
and sorting keys.  At s_t the teacher's block and each student sample that is not already
AST-equal are executed after the same prefix in scratch envs (fresh env, prefix replay,
one block; ≈3.5 s each, no LM call); two blocks are the same action iff their normalised
call sequences are equal.  Blocks that make no API call on either side (pure computation /
printing) agree iff they are AST-equal **or** print the same masked output — my reading
of "printed output + AST"; the strict AND would collapse to AST equality.  Caveat: a
same-effect student block may bind different REPL variables than the teacher's, and the
trunk keeps executing the teacher's later blocks, so "agreement" here means the
environment cannot tell the two apart, not that the student's own later code would.
`code_ast` remains the default and the fallback; the mode is recorded per row
(`_equality`).

**Probe selection, divergence and branching.**  `--probe-select first` (default) preserves
the original scan for the first `--max-probes` divergences; on the base student these
were concentrated at t ≤ 2.  `spread` instead preselects up to P = `--max-probes` evenly
spaced decision states, reserving one slot for the final state (quantiles 1/P, …, 1.0,
with P capped by demo length); `late` selects the last P states.  The final state is
immediately before executing the teacher's final block, normally `complete_task`.
`--probe-positions '0.25,0.5,0.75,1.0'` overrides the policy: finite fractions in [0, 1]
map to `ceil(fraction * demo_length) - 1`, clamped to valid indices, deduplicated, sorted
and capped at P.  Fixed quantiles need no randomness; sampling remains deterministic
given `--seed`.  Each preselected state gets an independent fresh environment and teacher
prefix replay, so earlier agreements or branch outcomes cannot change later probes.
Only states with complete sample sets contribute to `_agree_trace`.  At each probed s_t the student is
sampled `--probe-samples` times (default K, seeds `seed + 10t + j`), using the unchanged
`code_ast` or `api_effects` equality.  If all samples agree at a preselected state, emit
an agreement row without branches: `_event_k = 0`, wins `[0,0]`, empty continuation
records, zero branch episodes, Q^T = Q^S = 0.5 (prior only), ΔU = 0; raw pass fractions
are zero placeholders with no observations.  Otherwise the first disagreeing sample
is y^S and the teacher reply is y^T; K concurrent student continuations follow each,
with matched seeds `seed + 1e5·probe + 1000(j+1) + step`.  Probe numbering includes
selected agreements, keeping later seeds independent of earlier outcomes.  Events with
ΔU ≤ 0 are written like any other.  All rows retain the existing schema and add
`_probe_select` (effective policy, `explicit` for positions) and `_probe_pos`
(`(t + 1) / demo_length`, so the final decision state is 1.0).

**Environment reset.**  AppWorld is stateful and the REPL keeps variables across
steps; `AppWorld.save_state/load_state` snapshot the app databases only (not the REPL
namespace; `load_state` re-runs the preamble), so DB snapshots cannot fork a branch.
Every continuation therefore starts a **fresh AppWorld instance** (own
`experiment_name` under `experiments/outputs/simplified_react_code_agent/bfas_mine/<run>/`,
removed after evaluation unless `--keep-outputs`), re-executes the prefix code blocks
c_0..c_{t-1}, executes the branch action, and only then lets the student continue.  Each
env lives in its own worker process of the official venv (`--bridge` mode of the same
file, JSON lines over stdin/stdout, `APPWORLD_ROOT=envs/appworld-repo`, data 0.2.0, code
`a072b7a`, `random_seed=--env-seed` (default 1) + `set_random_seed` as the agent does).
Replay cost is negligible: env start ≈3 s, one `execute` ≈0.07 s, `evaluate` ≈0.2 s
(measured, see §4).  The prefix outputs of every branch are compared with the trunk's
(`_replay_output_mismatch`) and the trunk's with the demo's recorded outputs
(`_prefix_matches_demo`) after masking JWT access tokens, whose `exp` claim follows the
wall clock and differs between otherwise identical replays.

**Utility.**  One continuation is scored by the official evaluator
(`AppWorld.evaluate()`): `U = success` (all task tests pass; the TGC unit).  The pass
fraction `passed/num_tests` is recorded per branch as a secondary utility
(`_event_pass_frac`, and its Laplace form `_event_pass_frac_q`) but Q uses success.
Q^T = (Σ_j U_j^T + 1)/(K + 2), Q^S likewise, ΔU = Q^T − Q^S (`_event_u_plus`,
`_event_u_minus`, `_event_dU`, `_event_wins = [wins_T, wins_S]`, `_event_k`).
Episode caps: `--max-steps` 50 (official agent `max_steps`) on the whole episode, optional
`--max-cont-steps` on the continuation alone (continuation length is a decision variable
in §4 of the brief; it is recorded per row).

**Cost per probe.**  2K continuations.  Each continuation = prefix replay (≈3 s + 0.07 s
per step) + up to (max_steps − t − 1, or max_cont_steps) student calls; a student call
under this scaffold is 100–26 000 completion tokens (T4's smoke: mean ≈16 s per call on
GPU 4 with runaway thinking hitting the context limit), so LM calls dominate.  Plus K
probe samples at every replayed state.  `teacher_tokens = 0` on every row.

**Row fields.**  ALFWorld-miner names so the existing tooling applies unchanged:
`task_id, teacher ("demo_replay:gpt-5.4"), turn_index, prompt, response (y^T),
_rejected (y^S), _render_context, token_hint, _task_phat, _traj (task#pN),
_seed_category (scenario prefix), _event_turn, _event_good / _event_bad (executed code),
_event_u_plus, _event_u_minus, _event_dU, _event_k, _event_wins, _event_pass_frac,
_event_pass_frac_q, _event_seeds, _event_cont_steps, _event_completed, _event_errors,
_prefix_len, _prefix_codes (for re-estimation à la alf_event_reestimate), _demo_len,
_prefix_matches_demo, _replay_output_mismatch, _student_agree_rate, _student_samples,
_student_finish_reason, _agree_trace, _student_checkpoint, _student_max_tokens, _event_overflow,
_student_thinking, _student_reasoning_chars, _env_seed, _max_steps, _max_cont_steps, _teacher_tokens (0), _cost, _utility, _scaffold`.

## 3. Unified event schema (T1, `src/bfas/events_schema.py`)

Proposed converter (for T1 to drop into `src/bfas/events_schema.py`; not applied here):

```python
def from_appworld_row(row, tokenizer, *, source, split="train", student_checkpoint=BASE_STUDENT):
    """AppWorld miner row (tools/appworld_event_mine.py) -> UnifiedEvent."""
    ev = from_alfworld_row(row, tokenizer, source=source, split=split, student_checkpoint=student_checkpoint)
    ev.benchmark = "appworld"
    ev.provenance.update({k: row[k] for k in (
        "_event_pass_frac", "_event_pass_frac_q", "_event_cont_steps", "_event_completed",
        "_agree_trace", "_student_agree_rate", "_prefix_codes", "_prefix_matches_demo",
        "_replay_output_mismatch", "_env_seed", "_max_steps", "_max_cont_steps",
        "_student_max_tokens", "_student_thinking", "_equality", "_utility", "_scaffold", "_cost",
    ) if k in row})
    return ev.validate()

CONVERTERS["appworld"] = from_appworld_row
```

The rows carry exactly the fields `from_alfworld_row` reads (`prompt`, `response`,
`_rejected`, `_event_k`, `_event_wins`, `_event_u_plus/minus`, `_event_good/bad`,
`_prefix_len`, `_event_turn`, `_traj`, `teacher`), so conversion is
`from_alfworld_row(row, tok, source=...)` followed by `benchmark = "appworld"`
(validated in `test_event_row_field_set_and_schema_mapping`).  The clean solution is a
`from_appworld_row` entry in `CONVERTERS` that sets `benchmark="appworld"` and copies
the AppWorld-specific provenance (`_event_pass_frac`, `_agree_trace`, `_prefix_codes`,
`_env_seed`, `_max_cont_steps`, `_student_max_tokens`) — one line of mapping, left to T1
since `events_schema.py` is outside this task's write list.  `student_checkpoint` =
`Qwen/Qwen3.5-4B`, `teacher_model` = `demo_replay:gpt-5.4`, `teacher_output_tokens` of
y^T counted by the tokenizer as for ALFWorld (the acquisition ledger stays at 0 new
tokens; the demo cost is in `data/teacher_ledger/appworld.jsonl`).

## 4. Environment checks (no LM)

Replaying the full demo code through the bridge (fresh env, `random_seed=1`):

| task | turns | evaluator | tests | output mismatches vs demo | wall |
|---|---|---|---|---|---|
| 229360a_1 | 16 | success | 6/6 | 0/15 | 5.4 s |
| 287e338_2 | 7 | success | 2/2 | 1/6 (JWT `exp` only) | 4.8 s |
| 29caf6f_1 | 8 | success | 8/8 | 1/7 (JWT `exp` only) | 5.6 s |

So the archived demos still pass under the installed data/code, the environment is
deterministic given the code sequence up to wall-clock-derived token expiries (masked),
and replay-from-scratch is cheap (≈3 s start + 0.07 s/step).

## 5. Tests

`.venv/bin/python -m pytest -q tests/test_appworld_event_mine.py` — 18 passed:
scaffold port (code extraction, message formats), equality test, JWT masking, Laplace
Q, ΔU sign for a crafted teacher-better case (wins [2,0] → ΔU = +0.5) and a
student-better case (wins [0,3] → ΔU = −0.6, emitted unchanged), all-agree → no
episodes, matched-K bookkeeping (K seeds shared between branches, 2K episodes per
probe), reset-between-branches invariant (every continuation starts a fresh env,
replays the prefix, then the forced action, start/stop paired), probe/continuation
caps, row field set + unified-schema conversion, demo/split loading (calibration
excluded), dry-run writes nothing, request body per thinking mode, server check
refusing leaked thinking, effect-level equality (normalisation, no-call fallback, an end-to-end api_effects run on the fake env), resume/shard selection.

## 6. Pilot

Configuration (the only one whose numbers count): vLLM 0.27.1, `Qwen/Qwen3.5-4B`
served on rai GPU 2 (49 GB card, `--gpu-memory-utilization 0.4`, 19.1 GB resident,
`--max-model-len 32768 --max-num-seqs 256 --default-chat-template-kwargs
'{"enable_thinking": false}'`, port 8989); miner `--thinking off --max-tokens 4096
--temperature 0.7 --k 2 --max-probes 2 --max-cont-steps 20 --max-steps 50 --env-seed 1
--seed 0`, tasks `287e338_2` (spotify, 7-turn demo) and `29caf6f_1` (phone+simple_note,
8-turn demo), both demand.  Start-up server check: `finish=stop`, `reasoning_chars=0`;
`reasoning_replies=0` over the whole run, no `</think>` in any stored y^S, no HTTP 400,
no errored continuation.  Output `data/appworld_events/pilot_2task.jsonl`, log
`logs/t5_appworld_mine.log`.  (A first attempt against a server without the flag was
aborted before it wrote any row; see open issue 1.)

| task | t | y^T (teacher code) | y^S (student code) | wins [T,S] | Q^T | Q^S | ΔU | pass-frac [T,S] | cont. steps [T],[S] | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| 287e338_2 | 0 | `show_api_descriptions('spotify')` | `show_account_passwords()` + login | [0,2] | 0.25 | 0.75 | **−0.50** | 0.50 / 1.00 | [7,4],[4,7] | 27 s |
| 287e338_2 | 2 | `show_api_doc('supervisor','show_account_passwords')` | `print(apis.supervisor.show_account_passwords())` | [0,1] | 0.25 | 0.50 | **−0.25** | 0.50 / 0.75 | [5,10],[2,7] | 35 s |
| 29caf6f_1 | 0 | `show_api_descriptions('phone')` + `('simple_note')` | `show_api_descriptions('simple_note')` | [0,0] | 0.25 | 0.25 | 0.00 | 0.44 / 0.69 | [20,20],[20,11] | 81 s |
| 29caf6f_1 | 1 | `show_api_doc('phone','login')` + `('phone','search_contacts')` | `show_contact_relationships()` | [0,0] | 0.25 | 0.25 | 0.00 | 0.50 / 0.44 | [6,17],[20,20] | 87 s |

Totals: 4 events (2 with ΔU < 0, 2 with ΔU = 0, 0 with ΔU > 0), 16 continuations,
190 student calls, 23,706 student completion tokens, 0 teacher tokens, 249 s wall
(71 s + 170 s per task), `prefix_matches_demo = True` and 0 replay mismatches for all
rows.  The student disagreed with the teacher's exact code at every probed state
(`agree_trace` all 0.0; the divergences are all at t ≤ 2).  Under thinking-off the base
completes the Spotify task from its own first action (2/2) but not from the teacher's
identical-information action (0/2) — at K = 2 this is within noise and is reported as
observed, not interpreted.  Continuations that hit the 20-step cap (`completed=False`)
are the main cost driver (≈4 s per student call, 20 calls).

**Cost per probe (measured).**  2K = 4 concurrent continuations: 27–87 s wall, 22–71
student calls, 2.6k–9.2k completion tokens; env work ≈ 3 s start + 0.07 s per replayed
step per continuation.  Per continuation ≈ 11 student calls on average (bounded by the
continuation cap), ≈ 15 s of wall time at 4-way concurrency.

**Projection for AW-1 (40 demos, K = 3, max-probes 4, same caps).**  Worst case
40 × 4 = 160 probes × 6 = 960 continuations (+ 3 probe samples at each of ≈ 12 replayed
states per demo ≈ 1,400 cheap calls); ≈ 11 calls per continuation → ≈ 12k student calls,
≈ 1.5M student completion tokens, 0 teacher tokens.  Wall on one GPU with 6-way
concurrency per probe: 160 × (60–90 s) ≈ 2.7–4 h plus ≈ 0.5 h of trunk sampling —
call it 3–5 h for a single miner process; two processes sharded with
`--start-index/--limit` against one server roughly halve it (the server was far from
saturated at 4 concurrent 4B requests).  Memory: 19 GB GPU at 0.4 utilisation; 7 bridge
processes × ≈ 0.7 GB RSS.

## 6b. AW-1 base-half run (launched 2026-09-04)

Launched 2026-09-04 13:20 on rai GPU 2 (`GPU-20b20454…`, vLLM pid 3299206, port 8989,
same server flags as the pilot: thinking off, 0.4 memory utilisation ≈ 19 GB, 100% util
while mining).  Miner pid 3307538, **single process** (two sharded processes were not
validated, so not used):

```
PYTHONPATH=src .venv/bin/python tools/appworld_event_mine.py --split demand --k 3 --max-probes 4 \
  --max-cont-steps 20 --max-tokens 4096 --thinking off --equality api_effects --resume \
  --port 8989 --run-tag aw1_base_k3 --out data/appworld_events/aw1_base_k3.jsonl
```

Output `data/appworld_events/aw1_base_k3.jsonl`, finished-task sidecar
`aw1_base_k3.jsonl.done` (re-run the same command with `--resume` after an interruption;
the server must be up), log `logs/t5_aw1_mine.log`.  Rows: 40 demand demos, K = 3,
≤ 4 probes each (≤ 960 continuations), 0 teacher tokens.

First task (`229360a_1`, 16-turn demo): 5 states replayed, 4 probes, 4 events, 560 s,
427 student calls, 56k completion tokens, 24 continuations, no errors,
`prefix_matches_demo = True`.  Effect-level agreement at t = 0 and t = 4 was 0.67 (two of
three samples made the teacher's API calls with different code); the first probe gave
ΔU = +0.20 (wins [1,0]), the other three ΔU = 0 (wins [0,0]) with most continuations
hitting the 20-step cap.  Probe wall 119–159 s at 6-way concurrency, i.e. ≈ 140 s per
probe; expected duration for 40 demos ≈ 40 × (4 × 140 s + ≈ 30 s trunk/effect replays)
≈ 6.3 h if every demo spends its 4 probes, less where effect-level agreement skips
states — finishing roughly 19:30–20:00 rai time.  The caller monitors the log; the vLLM
server (pid 3299206) must be stopped after the FINAL line.

## 7. Open issues

1. **`max_tokens` and thinking mode.**  The official agent sends no `max_tokens` and
   the chat template thinks by default; the miner (like the base chain) disables
   thinking and caps `max_tokens` at 4096 (recorded per row; `--max-tokens 0` removes the
   cap).  The pilot numbers below come from this configuration; the first pilot attempt,
   run against a server without the flag, was aborted before it produced any event
   (0 rows, moved to `_trash/pilot_2task_thinking_leaked.jsonl`).
2. **Prompt of stored events vs. demo turns.**  Event prompts end in
   `<think>\n\n</think>\n\n` (thinking off) while the archived demo `Turn.prompt`s
   were rendered with the default template (`<think>\n`); any CE arm that mixes demo
   turns and event rows should re-render both with one convention
   (`AppWorldOfficialAdapter.rerender` / `_render_context`).
3. **Env seed of the demos.**  Teacher runs used `random_seed = 0 + run_serial`; the
   demos do not record the serial.  With `--env-seed 1` all three replayed demos
   reproduce their recorded outputs (JWTs aside) and pass, so seed dependence looks
   negligible, but `_prefix_matches_demo` is stored per row to catch exceptions.
4. **y^S selection with K probe samples.**  The first disagreeing sample is taken as
   y^S; the agreement rate is stored.  With `--probe-samples 1` the miner reduces to the
   ALFWorld one-sample rule.
5. **Reachability.**  `_agree_trace` gives per-step agreement along the teacher prefix,
   not the probability that the student reaches s_t on its own; a depth-reachability
   estimate as in `tools/alf_reachability.py` is a separate run.
6. **Equality test strictness.**  In the pilot (code_ast) the student never reproduced
   the teacher's code exactly (the teacher batches two doc look-ups per block, the
   student issues one, or skips the look-up), so every replayed state was a divergence
   and the probes were all spent on the first 2–4 steps.  `--equality api_effects`
   (section 2) is the response; the AW-1 run uses it.  The pilot's ΔU rows were mined
   under code_ast and are not directly comparable to AW-1 rows.
7. **Concurrency.**  Each branch continuation holds one official-venv process
   (≈0.7 GB RSS) and one open AppWorld world; the pool keeps 1 + 2K of them.  Tasks
   are mined sequentially; a `--tasks`/`--start-index` sharding is available for
   several miner processes against one vLLM server.
