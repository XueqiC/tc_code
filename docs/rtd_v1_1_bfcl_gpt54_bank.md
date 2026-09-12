# GPT-5.4 BFCL bank from surviving harness artifacts

Built offline on 2026-09-10 for RTD v1.1/unified, with the unchanged seed-50
40-parent support split and `google/gemma-4-12B-it` student. No API calls or model
weight loads were made.

| Certificate field | Value |
| --- | ---: |
| Available packages / training states | 21 / 21 |
| Ledger attempts (including unavailable inventory) | 120 |
| Recorded usable cost | 1,519 output tokens |
| Exact usable cost | 710 output tokens |
| Estimated usable cost | 809 output tokens |
| Uniform demo-attempt cap | 256 output tokens |
| Public cap sum | 5,376 output tokens |
| 10% / 25% budget ceilings | 152 / 380 output tokens |
| Surviving provider usage, all attempts | 205,053 output tokens |

The certificate and its sealed audit label collection cost
`exact-plus-unknown-rerun`. The 809 estimated tokens are **original-attempt usage
proxies for seven changed adapter re-run targets**, not recovered costs for those
targets. The remaining 710 tokens belong to 14 unchanged demos. The denominator
is cached evidence accounting, not the complete teacher bill.

## Evidence limits

- Attempt 1 has all 72 usage records: 40 tasks and 32 memory prerequisites,
  totaling 147,589 tokens. Attempt 2 retains 18 prerequisite usage records,
  totaling 57,464 tokens. Attempt 3 retains no usage counts. All 80 task rows
  across attempts 2–3 contain missing usage in their attributed components;
  their ledger spend is only the recovered lower bound, marked estimated.
  Quota failures can discard earlier stateful completions, so missing counts
  are never asserted to be exact zero.
- Prerequisites use the adapter's dependency lists and nested completion-count
  summation. Each generated prerequisite is charged once, to its first dependent
  task in demand-split order. The two vector-memory tasks share a chain; it is
  not billed twice. Every charged component has its result id, file hash, line,
  provider count, and file-mtime timestamp in ledger provenance.
- The merged directory contains 27 entries, but its two web-search entries have
  **no score files**. The old merge inferred success from absence among failed
  ids; the converter requires a reconciled positive score. Thus 25 attempt-1
  demos qualify. Quota-error strings that the irrelevance checker accepts are
  also excluded from successful demos.
- Four scored stateful demos have no training turns under the adapter contract.
  They stay in unavailable paid inventory. Together with the two unscored web
  entries, these account for the six merged entries absent from the 21-row pool.
  No episode was flattened into a fabricated state.
- Saved adapter turns are preferred, including empty turns, and their source is
  recorded. Its extra verified `multi_turn_miss_func_198` is absent from the
  merged scored successes and is not added as purchased evidence.
- The adapter re-run log shows three generation passes. All three corresponding
  UUID result directories were deleted by `teacher_demo`; serialized demos have
  no usage. The exact number of extra API calls (including retries) and their
  cost are **unknown**. Matching outputs do not establish API cache reuse. The
  existing harness files are reused as accounting evidence only.

## Reproduce

The converters are CPU-only. Their output directories must be writable. In the
restricted `tc-alignment-uni` checkout, `data/` is a local overlay: existing
shared inputs remain symlinks and the newly generated outputs are local files.
The shared checkout's `data/` and harness files were not modified.

```bash
export PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BFCL_PROJECT_ROOT=/tmp/rtd-gpt54-bfcl-runtime
RTD_PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python

"$RTD_PY" tools/bfcl_ledger_from_harness.py \
  --adapter-log /home/xueqi/hq/projects/tc-alignment-g4/logs/bfcl_adapter_collect_gpt54.log
"$RTD_PY" tools/bfcl_pool_from_harness.py
"$RTD_PY" tools/bfcl_pool_render_gemma4.py \
  --in data/bfcl_sft/pool_gpt54_sft.jsonl \
  --out data/bfcl_sft/pool_gpt54_gemma4_sft.jsonl --verify
"$RTD_PY" tools/rtd_bank_build.py --benchmark bfcl \
  --pool data/bfcl_sft/pool_gpt54_sft.jsonl \
  --ledger data/teacher_ledger/bfcl_gpt54.jsonl \
  --out data/rtd/v1_1_bfcl_gpt54
```

Bank creation refuses an existing output directory. The `BFCL_PROJECT_ROOT`
override puts harness locks in `/tmp`; without it, the read-only installation
silently omits memory parents when the adapter catches its loader exception.

The ledger sidecar is `data/teacher_ledger/bfcl_gpt54.provenance.json`. Its hash
and accounting status are bound into `public/cap_certificate.json`, and the
full sidecar is preserved in `sealed/audit.json`. The pool has its own
`pool_gpt54_sft.provenance.json` exclusion/source manifest.

Validation: 21 Gemma rows rendered with `--verify`; 78 focused CPU tests passed
(converters, Azure adapter, gateway ledger, adapter conformance, student
rendering, BFCL certificates, ALFWorld conversion, and WebShop banks).
