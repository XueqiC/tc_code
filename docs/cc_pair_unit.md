# C20: condition-contrast training

`src/bfas/arms.py` provides pair transforms and objective math. The standard BFCL
campaign trains through `src/appworld_train.py`: its `train_ddpo` uses
`-logsigmoid(beta * ((logpi_plus-logpi_minus) - (base_plus-base_minus)))`.
That function and the default row-loading, SFT, retention and export path are
unchanged. `AW_DISTILL=pair_unit` opts into `bfas.pair_unit`; setting
`AW_CC_PAIRS_PATH` with `AW_DISTILL=ddpo` selects the flattened comparison.

## Producer contract

C19's `tools/cc_pairs.py` and its output were not present when C20 was implemented.
The reader accepts JSONL records, a JSON list, or an object with a `pairs` or
`confused_pairs` list. A list of IDs (also `pair_ids` / `confused_pair_ids`) resolves
against sibling `candidates.jsonl`. Each pair has this form:

```json
{"pair_id":"p1", "type":"1", "seed_function":"seed_f", "sides":[
  {"prompt":"serialized state 1", "y_plus":"teacher action 1", "y_minus":"rejected action 1"},
  {"prompt":"serialized state 2", "y_plus":"teacher action 2", "y_minus":null}
]}
```

`pair_type` aliases `type`; `s1` / `s2` alias the sides list. `seed_function` may
be on each side. Original sides must share it. Optional `side_id`, `task_id`,
teacher and provenance fields are preserved. Missing/null/empty `y_minus` means
positive-only base-centred margin; no incorrect negative is invented.

C19 selection fills an empty teacher target from the same state's checker-verified
base greedy response, or a verified T=0.7 sample if greedy has no usable response.
It preserves the response bytes, sets `y_plus_from_student: true`, and records
`y_plus_provenance` as `student_verified_greedy` or `student_verified_sample`.
A checker-failed student call supplies `y_minus` when available, with
`y_minus_provenance: student_wrong`. Without a verified nonempty positive, the
candidate stays empty and selection excludes the pair with a reported reason.
Historical `teacher_text`, teacher provenance and probe likelihoods remain intact;
evaluation validates the derived training fields against the source candidate and probe.

`split.json` must explicitly list `train_pair_ids` (aliases `train` or
`train.pair_ids`). Only selected training IDs enter training; explicit conflicting
splits and train/held-out ID overlap are errors. The reader also supports per-pair
`split: "train"` when called without a split file. C19 remains responsible for
official calibration/certification exclusion and parent-function isolation.

Arm C shuffling permutes whole right sides within each type. It first tries a
seeded perfect matching requiring different seed functions and original pair
IDs. If infeasible, it retries the entire type with a seeded derangement requiring
only different original pair IDs; same seed functions are then allowed. Every
prompt, label, side occurrence and provenance is preserved, and no pairs are
discarded. A type with only one pair still errors because no derangement exists.
Arms A/B retain their original pairs and training behavior.

For example, stage-1's six `binding` pairs (five from `simple_javascript_42`, one
from `live_parallel_multiple_10-9-0`) use the fallback and retain all six pairs:
four re-pairings have the same seed function and two have different seed functions.
Arm C's `cc_manifest.json` records `shuffle_same_seed_by_type`, a map from type to
the number of output re-pairings whose two sides share a seed function, including
zero for types matched entirely across seeds. These are pair occurrence counts,
not unique seed-function counts. The CPU pre-flight uses the same rule and seed
and prints the same counts before training.

## Objective and anchors

For each state, `m = beta * ((logpi-base)(y_plus) - (logpi-base)(y_minus))`.
With no rejected text, the second term is zero. These are summed response log
probabilities including EOS, using the existing tokenizer and truncation.
The joint objective is `max(relu(gamma-m1)^2, relu(gamma-m2)^2)`;
`AW_PAIR_REDUCTION=sum` adds the two terms instead. Arm A uses the existing
logistic DDPO formula independently on each flattened side, including the same
positive-only fallback on data that legacy DDPO would otherwise drop.

Anchors come from `AW_CC_ANCHOR_PATH`, defaulting to
`data/bfcl_sft/anchors_single_base.jsonl`. A mixed pool is allowed: only `_anchor`
rows enter retention, each requiring `_rejected`. Use `none` for no anchors.
Anchors keep their logistic DDPO loss, weight preparation and unit row mass;
they never enter a worst-side objective or shuffle. Pair losses carry two row
units of mass in a batch, and singleton anchors one. The existing adaptive
call/abstain probe selection, dual update, anchor sampling and loss scaling are
also supported by `AW_ANCHOR_ADAPTIVE=1`. The three-arm runner pins it to zero:
policy-dependent anchor sampling cannot guarantee identical exposure.

## Matched campaign

`tools/cc_three_arms.sh [port]` runs selected arms sequentially on the one visible
GPU, then calls `tools/bfcl_std_campaign.sh` for official BFCL v4 `all` evaluation.
It requires fresh output tags, validates current-run score production and checks
actual token totals, side/anchor multiset hash, model, settings and hostname
before evaluation. It respects SLURM's GPU assignment. `scripts/cc_arms_hpg.slurm`
requests fsu-compsci-dept, one B200, four hours; no job array.

Main environment settings (all arms share them):

| Runner setting | Default | Trainer setting |
| --- | --- | --- |
| `CC_ARMS` | `A B C` | A=`ddpo`; B/C=`pair_unit` |
| `CC_LR` | `5e-6` | `AW_DDPO_LR` |
| `CC_BETA` | `0.1` | `AW_DDPO_BETA` |
| `CC_GAMMA` | `1.0` | `AW_PAIR_GAMMA` |
| `CC_REDUCTION` | `worst` | `AW_PAIR_REDUCTION` (`worst`, `sum`) |
| `CC_EPOCHS` | `1` | `AW_DDPO_EPOCHS` |
| `CC_STEPS` | `0` (derive from epochs) | `AW_PAIR_STEPS` |
| `CC_BATCH_UNITS` | `4` | `AW_PAIR_BATCH_UNITS` |
| `CC_SHUFFLE_SEED` | `0` | `AW_PAIR_SHUFFLE_SEED` |
| `CC_SEED` / `CC_STUDENT` | `0` / `Qwen/Qwen3.5-4B` | CLI seed/student |
| `CC_PAIRS_PATH` / `CC_SPLIT_PATH` | `data/cc_pairs_v1/{confused_pairs,split}.json` | `AW_CC_*` |
| `CC_ANCHOR_PATH` | `data/bfcl_sft/anchors_single_base.jsonl` | `AW_CC_ANCHOR_PATH` |
| `CC_TAG_PREFIX` | `cc_v1` | output tags `cc_v1_A_s0`, etc. |

The batch schedule shuffles pair units and singleton anchors identically across
arms; A flattens each chunk before its independent logistic losses. Each pass
consumes every side and anchor once. Explicit steps must cover complete passes
(`ceil((pairs+anchors)/CC_BATCH_UNITS)` steps per pass); partial passes are
rejected because shuffled pair membership changes their token exposure. Token
totals match at pass boundaries; per-step token counts can differ. This shared
CC batch schedule is opt-in and does not change the legacy DDPO row schedule.

The section-3 execution gate defaults to 20 training pairs and two types;
`CC_MIN_PAIRS` / `CC_MIN_TYPES` allow explicit smaller smoke experiments. There
are no teacher calls. Pair construction/confusion mining is a separate task.

Each output contains `cc_manifest.json` and `cc_steps.jsonl`. Every step logs
both margins for every event pair (also in A), singleton anchor margins, actual
full-vocabulary conditional **KL(policy || base)** at the observed chosen and
rejected response prefixes before that update, and exact post-truncation input,
response and prompt token counts. KL is token-weighted over these prefixes; it
is not a rollout-distribution trajectory KL or a sampled log-ratio surrogate.
Reference caching, KL diagnostic forwards and adaptive probe tokens are counted
separately from optimizer-exposure tokens. Backward/checkpoint recomputation is
not counted as additional data exposure. All branches count even when the hinge
is inactive. No KL regularizer is added to the objective.

## C21: held-out pair evaluation

`tools/cc_pairs.py evaluate` evaluates every original pair in `train`,
`heldout_seed`, and (if present) `heldout_within_seed`. It skips only IDs listed
as `excluded_cross_partition`. It uses the full C19 records, including prompts,
teacher texts, ground truth and checker metadata; the minimal training-only
records shown above do not contain enough information for AST evaluation.
Base probe records must match the candidate hashes and state IDs. Duplicate,
missing, stale or incomplete results, inconsistent checker versions, overlapping
partitions and train/held-out seed leakage are errors. Inputs are read-only.

```bash
.venv/bin/python tools/cc_pairs.py evaluate \
  --model results/appworld_students/cc_v1_B_s0/adapter \
  --pairs data/cc_pairs_v1_stage1/confused_pairs.json \
  --split data/cc_pairs_v1_stage1/split.json \
  --base-results data/cc_pairs_v1/probe_results.jsonl

# Existing vLLM server. --tokenizer overrides the server model card's root
# when that path is only accessible on the remote host.
.venv/bin/python tools/cc_pairs.py evaluate \
  --base-url http://127.0.0.1:8976/v1 --tag cc_v1_B_s0 \
  --tokenizer Qwen/Qwen3.5-4B

# Submit only when GPU evaluation is desired; 30-minute limit per tag.
sbatch scripts/cc_eval_hpg.slurm cc_v1_A_s0 cc_v1_B_s0 cc_v1_C_s0
```

The pair, split and base-results paths above are the CLI defaults. `--tag`
defaults to the checkpoint's parent for `hub_merged`/`adapter`, or the served
model name. Output defaults to `results/cc_eval/<tag>/`; `--out` overrides it.
`pair_eval.jsonl` contains both per-side greedy AST outcomes, raw outputs,
repair/damage flags against base greedy (never the four samples), teacher-forced
own/other likelihoods, κ and base κ, and side/pair flip flags. `summary.json` and
`report.md` provide overall, partition and type tables. Reports become explicitly
incomplete when a rerun starts and are marked complete only after all scoring
and checks succeed.

Generation runs once per distinct state, greedy with thinking off and a default
512-token output limit. Local `--model` loads the merged HF checkpoint directly
and reuses the probe's native CE scorer: separately tokenized prompt/response,
full response including EOS, fp32 reduction, no truncation. `--base-url` loads
only a tokenizer locally, discovers the served ID (`--served-model` disambiguates
multiple models), and uses vLLM's [echoed prompt log-probabilities](https://docs.vllm.ai/en/v0.15.0/api/vllm/entrypoints/openai/completion/protocol/)
on the same concatenated token IDs. This has the same likelihood definition but
server numerical precision can differ from native CE; the backend is recorded.
A server lacking echoed prompt log-probabilities fails explicitly. The remote
tokenizer must match the served checkpoint. No teacher requests are made.

κ(s1)=logπ(a2|s1)−logπ(a1|s1), with the symmetric definition on s2. A pair flips
when `max(base κ1, base κ2)>0` and `max(current κ1, current κ2)<=0`: neither side
still prefers the opposite target. Both the conditional rate over base-positive
pairs and the fraction of all pairs are reported, alongside per-side flips and
the number of pairs with any side flipped. Zero denominators are JSON `null`
and Markdown `N/A`. Side repair/damage counts include pair occurrences;
distinct-state counts are also reported. Call rates deduplicate `state_id` in
each group, report model/base/delta overall and by `should_call`/`should_abstain`,
and count malformed `<tool_call>` attempts as calls while failing their AST
outcome. These metrics complement the separate official BFCL v4 campaign.

The SLURM wrapper requests fsu-compsci-dept, one B200, and 90 minutes for the
default three tags. Set `CC_EVAL_TAGS='tag1 tag2'` or pass positional tags;
request a longer `sbatch --time` for more than three tags. It preserves the probe
launcher's `SLURM_SUBMIT_DIR`, PATH, cache and checker-venv conventions and the
assigned GPU. Each tag runs sequentially in a fresh native model process with a
30-minute timeout; failures stop the list. It searches `appworld_students` then
`bfcl_students`, preferring `hub_merged` and falling back to `adapter`, which the
trainer saves as a merged HF model. This fallback handles the official campaign's
removal of `hub_merged`. Overrides: `CC_PYTHON`, `CC_PAIRS_PATH`, `CC_SPLIT_PATH`,
`CC_BASE_RESULTS`, `CC_EVAL_OUT`, and `CC_SEED`.
