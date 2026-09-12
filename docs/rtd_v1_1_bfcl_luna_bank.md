# Luna BFCL bank for RTD v1.1 / unified

Built offline on 2026-09-11 from the read-only `tc-alignment-g4` inputs for
teacher `openai/gpt-5.6-luna-FC`, with `google/gemma-4-12B-it` student rendering.
The seed-50 demand split and frozen 40-parent support manifest are unchanged.
No API calls, model weight loads, GPU probes, or training runs were made.

| Audit field | Value |
| --- | ---: |
| Demand tasks / verified tasks | 40 / 23 |
| Available packages / training states | 20 / 20 |
| Ledger attempts / sealed attempt payloads | 74 / 74 |
| Failed attempts, retained as paid unavailable inventory | 51 |
| Verified episodes without adapter training turns | 3 |
| All recorded completion tokens | 228,545 |
| All exact / estimated completion tokens | 228,545 / 0 |
| Failed-attempt completion tokens | 200,717 |
| Verified-without-turns completion tokens | 26,410 |
| Usable recorded cost / budget denominator | 1,418 |
| Exact / estimated usable cost | 1,418 / 0 |
| Maximum usable episode cost | 237 |
| Uniform usable `demo_attempt` cap | 256 |
| Public cap sum | 5,120 |
| 10% / 25% budget ceilings | 142 / 355 |
| CPU preflight support / rendered states | 33 / 33 |

Attempt indices 0, 1, and 2 contain 40, 17, and 17 rows, respectively, costing
93,190, 67,211, and 68,144 completion tokens. Every source gateway row is copied
without changing its recorded cost, verification, temperature, or demo payload.
Every row has one sealed payload with the same `historical_response` and cost.

The accounting rule is the GPT-5.4 recipe: failed attempts and verified episodes
without usable adapter turns stay paid in the sealed inventory and historical
total. They are outside the **usable-content** budget denominator. Thus
`200717 + 26410 + 1418 = 228545`; nothing is dropped from historical spend.
Gateway episode costs already include generated memory prerequisites and failed
completions. They are neither re-estimated from visible text nor charged again
from retained result copies.

The three verified episodes without per-state training turns are
`memory_rec_sum_99-student-19` (25,826 tokens), `multi_turn_long_context_78`
(202), and `multi_turn_miss_param_1` (382). Their empty adapter turns are
preserved; no synthetic training states are constructed.

## Source evidence and retention

The source ledger is
`/home/xueqi/hq/projects/tc-alignment-g4/data/teacher_ledger/bfcl.jsonl`.
Its 74 luna rows contain exact gateway completion counts and the 23 serialized
verified demos. The local snapshot is `data/teacher_ledger/bfcl_luna.jsonl`.
Its `.provenance.json` sidecar records the source hash and original line numbers,
split hash, retained result/journal hashes and lines, and exclusions. The bank
certificate binds that sidecar, and `sealed/audit.json` preserves it in full.

The collection docs and `tools/bfcl_teacher_demos.sh` in g4 were inspected.
The shell merger's model-specific filename would be
`data/bfcl_demos_openai_gpt_5_6_luna_FC_verified.json`. That file, model-specific
luna SFT demo files, and a merged luna result directory were absent at build
time. The generic `data/bfcl_demos_ds_verified.json` contains 27 IDs and provides
no luna-specific attribution; it and the GPT-5.4 adapter cache were not used.
The gateway's embedded verified demos are the pool's source instead.

Under g4's harness root
`envs/bfcl/gorilla/berkeley-function-call-leaderboard`, seven
`result_bfas_*/openai_gpt-5.6-luna-FC` directories survive, with seven
`bfas_usage.jsonl` journals containing 25 records and 892 output tokens.
All retained result totals reconcile with their journals. Six web-search
results have unique task/cost matches among the copied ledger rows. The seventh
is a zero-token provider failure with no matching ledger row. These files add
no new charge. The ledger does not retain UUID attempt identifiers, so the
sidecar describes these as candidate matches, not proven UUID attribution.
No `score_bfas_*` directories survive; no positive verdict is inferred from
their absence.

Exactness here refers to the gateway's recorded provider completion counts;
most raw per-call usage and official scores are no longer independently
re-auditable. g4's adapter sums provider counts into `TeacherEpisode` and
deletes result/score directories in its cleanup path. The snapshot does not
substitute a different target or attach a GPT-5.4 usage proxy. Its accounting
status is `exact-gateway-ledger`, with raw harness retention explicitly marked
partial. The zero extra-call fields describe this offline import.

## Configs and validation

`configs/rtd/v1_1_bfcl_luna.yaml` copies `v1_1_bfcl.yaml`, changing only the bank
path, student, Gemma call format, and the 60 GB resource declaration already
used by the Gemma config. `configs/rtd/unified_bfcl_gemma4_luna.yaml` copies
`unified_bfcl_gemma4.yaml`, changing the bank path and its displayed integer
budget caps to `[142, 355]`. Every other YAML value, including frozen P1 values,
is unchanged. Both configs pass the existing protocol validators.

The legacy historical-token constants and public-cap mapping are frozen
protocol fields. Actual luna costs and the 256-token class cap come from the
bound bank certificate. The 142-token first checkpoint is below one class-cap
reservation; it cannot acquire a package under hard reservation. The second
checkpoint permits a reservation. Budget fractions and the certificate's
next-power-of-two cap rule are unchanged.

Validation completed:

- All 20 rows passed `bfcl_pool_render_gemma4.py --verify` with the local cached
  Gemma tokenizer; the bank renders the original pool through the same converter.
- V0 CPU preflight passed in `--mode run`; D3 CPU preflight passed in the
  standalone default `--mode smoke`. Both reported 20 available packages,
  33 support/rendered states, zero replay spend, and no deferred checks.
- D3 `--mode run` correctly requires a completed V0 replay schedule. None is
  fabricated for this bank build; that frozen P1 requirement remains in force.
- 49 focused CPU tests passed: harness/gateway converters, native student
  rendering and bank certificates, and preflight. Coverage includes both
  teachers, failed-cost retention, mixed-teacher rejection, journal mismatch,
  and preservation of original demo contexts.
- All 74 sealed payload checksums and their costs/original ledger rows were
  checked, as were the source snapshot, certificate bindings, and exact config
  differences. GPU execution and model initialization are outside CPU preflight.

## Reproduce

Run from this checkout. `data/` is a writable local overlay; the new pool,
ledger, and bank are local files. Existing shared inputs remain untouched.
The `BFCL_PROJECT_ROOT` override keeps harness loader locks under `/tmp` and
prevents silent omission of memory parents from the read-only installation.

```bash
export PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BFCL_PROJECT_ROOT=/tmp/rtd-luna-bfcl-runtime
RTD_PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
G4_ROOT=/home/xueqi/hq/projects/tc-alignment-g4

"$RTD_PY" tools/bfcl_ledger_from_harness.py \
  --source-ledger "$G4_ROOT/data/teacher_ledger/bfcl.jsonl" \
  --teacher openai/gpt-5.6-luna-FC \
  --harness "$G4_ROOT/envs/bfcl/gorilla/berkeley-function-call-leaderboard" \
  --out data/teacher_ledger/bfcl_luna.jsonl
"$RTD_PY" tools/bfcl_pool_from_harness.py --from-ledger \
  --ledger data/teacher_ledger/bfcl_luna.jsonl \
  --out data/bfcl_sft/pool_luna_sft.jsonl
"$RTD_PY" tools/bfcl_pool_render_gemma4.py \
  --in data/bfcl_sft/pool_luna_sft.jsonl \
  --out data/bfcl_sft/pool_luna_gemma4_sft.jsonl --verify
"$RTD_PY" tools/rtd_bank_build.py --benchmark bfcl \
  --pool data/bfcl_sft/pool_luna_sft.jsonl \
  --ledger data/teacher_ledger/bfcl_luna.jsonl \
  --config configs/rtd/v1_1_bfcl_luna.yaml --out data/rtd/v1_1_bfcl_luna
"$RTD_PY" tools/rtd_preflight.py \
  --config configs/rtd/v1_1_bfcl_luna.yaml --arm V0 --mode run
"$RTD_PY" tools/rtd_preflight.py \
  --config configs/rtd/unified_bfcl_gemma4_luna.yaml --arm D3
"$RTD_PY" -m pytest -q tests/test_bfcl_harness_converters.py \
  tests/test_rtd_v11_students.py tests/test_rtd_preflight.py
```

Bank creation refuses an existing output directory. Reproduction requires a
fresh bank path; preserve the certified bank when checking it again.
