# Additive ALFWorld fidelity variants (Codex#17)

Implemented on `alf-baseline-audit`, 2026-09-22, from
`alfworld_baseline_fidelity_audit.md`. Existing defaults, bank bytes, acquisition,
mask behavior and results remain unchanged. These are matched ALFWorld
adaptations with explicit corrected mechanisms, not full original-paper
reproductions. All work and validation here used CPU only, with no model
inference, teacher requests, or LONI access. No files in `results/` or
`artifacts/` were written.

## Registrations and commands

| Command | New option | Default |
|---|---|---|
| `tools/alf_baseline.py select` | `--statistic {token_mean,turn_mean}` | `token_mean` |
| `tools/alf_baseline.py train --method smartad` | `--smartad-variant {token_norm,weight_norm}` | `token_norm` |
| `tools/alf_baseline.py train --method sad` | `--sad-curriculum {turn_count,trajectory_cost}` | `turn_count` |
| SAD trajectory cost | `--sad-alpha FLOAT --sad-beta FLOAT` | `1`, `1` |

Selection records both the CLI statistic and the rule: `turn_mean` is
`mean_of_turn_means`, calculated as `fsum(S_turn / n_turn) / turns`.
Every new v2 `scores/<candidate_hash>.json` retains indexed `turn_scores`
containing `total_nll` and `supervised_tokens`, plus trajectory `mean_nll` and
`turn_macro_mean_nll`. Counts use the existing mask and include the shared
encoder's native boundary. Ties still use candidate IDs. `candidates.json`
retains every candidate's full frozen rows so a different winner can be
materialized without loading a tokenizer or reopening a bank. `manifest.json`
binds that inventory, student identity and scoring rule. `read_selection`
validates the known version/rule, score arithmetic and selected argmin.
Recognized v1 token-mean artifacts and resumptions remain supported unchanged;
unknown or mismatched rules are rejected. The legacy `paper_data.select_smartad`
selector retains its old behavior and is not used for the new registration.

Offline recomputation (writes a new directory, refuses overwrite):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  tools/alf_smartad_recompute.py \
  --scores /path/to/previous-selection/scores \
  --output /tmp/smartad-turn-mean --statistic turn_mean
```

The default statistic for this tool is `turn_mean`; it can also reconstruct
`token_mean` from the same receipts. It records source receipt hashes and zero
model calls, preserves all purchase costs, and refuses incomplete or mismatched
candidate inventories. Trajectory totals cannot recover turn means: when
per-turn receipts are absent the tool refuses before creating output. Complete
candidate rows are also needed (`candidates.json` beside `scores/`).
The existing sibling-worktree `results/smartad_selection_rai/scores` contains
five trajectory-only receipts and no per-turn receipts. Running the tool on
that directory produced the expected refusal; no replacement selection was
fabricated. A complete turn-mean selection there requires rescoring.

`weight_norm` registers `span_ce(..., 'smartad_wsum')`, dividing weighted NLL by
the sum of nonmasked weights (reason 1, action 1.5, final 2). Row gradients are
weighted by their weight sum divided by the update's weight sum. Consequently
each optimizer update implements `sum(w * NLL) / sum(w)`, including unequal
row lengths/compositions. This deliberately retains the token-budget optimizer
updates and is labeled an update-wide reduction, not a per-trajectory average
specified by the paper. Hyperparameters, hashed training identities and endpoint
manifests record the variant and reduction. No teacher data or selection change
is needed for this loss-only comparison.

`trajectory_cost` calculates each package's
`alpha * authored_reason_tokens + beta * authored_action_or_final_tokens` under
the existing span classifier. Observation-marked spans and appended native
boundaries do not enter complexity. The native boundaries still enter training
loss/exposure. Costs and component counts are frozen in the schedule. Each pass
orders trajectories easy-to-hard, uses seeded ties between trajectories, and
keeps original turn order within trajectories. Batches can still split
trajectories according to the existing complete-turn token budget. Gamma is
recorded as zero with entropy explicitly **unavailable for the black-box
teacher**, not an observed zero-entropy estimate. SAD's `sad_sum` default and
`sad_mean` ablation remain separately selectable.

## X2 read-only inventory

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  tools/alf_mask_inventory.py \
  --bank /home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0 \
  --bank /home/xueqi/hq/projects/tc-alignment-baselines/artifacts/alfworld_k32_smartad_n3 \
  --model-path /home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
```

The n3 bank is absent in this worktree and in the main checkout; the existing
bank in `tc-alignment-baselines` was scanned read-only. The tool audits each
bank, loads the actual pi1 rows and frozen prompts, and uses
`encoded_spans`/`token_kinds` with only the cached tokenizer. It reports tokens
masked among authored target IDs separately from appended boundaries, including
per-row package/task/index identifiers when affected. It writes only stdout.

| Bank | Packages | Turns | Authored tokens | Native boundaries | Masked authored tokens | Affected rows/packages |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 30 | 424 | 9,225 | 424 | **0** | **0 / 0** |
| SmartAD n3 | 87 | 1,343 | 32,093 | 1,343 | **0** | **0 / 0** |

Provenance: D0 sealed-manifest SHA256
`dcc043b0649e29b0fbec4319ab3f9c62b86b6f71f62156ac781586a8df4d9f3b`;
n3 sealed-manifest SHA256
`df529867d4fd6db67378945a3078c3c046ed03d73d020c2260c47f957875b4aa`;
tokenizer identity hash
`6e5d85e93f519ee8c8ce8db1432ce8fc2fdfdfcfc37c0148058127580300793d`.
These counts cover all usable authored rows. No X2 rerun is needed for these
snapshots; the parser remains unchanged and synthetic marker cases still expose
the known masking behavior.

## Kang adaptation label

The current K32 Kang bank is a **prompt-continuation FTP adaptation**: the
teacher first produces an ALFWorld plan from the reset observation, the collector
extracts its first-thought prefix, and the trajectory request asks for continuation
using that prefix as an assistant message. This API does not supply native
assistant-prefill decoding, so concatenation and exact-echo removal cannot be
claimed equivalent to original native prefill. The existing retry-to-success
budget policy and per-task reused plan remain unchanged; the K32 pi1 route
trains on complete authored turns. New Kang exports record usable attempt-zero
package IDs, excluding failed first attempts and all later retries. This enables
an offline first-attempt subset while retaining the complete original planning,
failure and retry costs in the source collection ledger; it does not change
sampling or turn that subset into a native-prefill reproduction.

New `kang-ftp` exports add `first_attempt_package_ids` and the adaptation label
to `.collection/candidate_sets.json`, and write a JSON list at
`.collection/first_attempt_package_ids.json`. To materialize that list:

```bash
.venv/bin/python tools/alf_bank_subset.py \
  --source /path/to/kang-bank \
  --package-ids /path/to/kang-bank.collection/first_attempt_package_ids.json \
  --output /tmp/kang-first-attempt
```

An empty list means no usable first attempts; the existing subset tool correctly
refuses an empty training bank. Existing exports were not regenerated here.

## Rerun implications

| Fix | Existing results that need new computation to report the variant |
|---|---|
| SmartAD turn mean | Reselect all token-mean selections, including `smartad_selection_rai`; its old totals require rescoring. Retrain/evaluate downstream models only if chosen rows change when this is the sole fix. Singleton candidates stay selected. |
| SmartAD weight normalization | Retrain and reevaluate token-normalized SmartAD checkpoints, including applicable `table1_alfworld_smartad*` runs if reported as this corrected loss. Selection and teacher banks are reusable; archived legacy recipes remain separate. |
| SAD trajectory cost | Retrain/evaluate SAD checkpoints using turn-count ordering, for each reported seed and 3/10-pass endpoint. No recollection or selection is needed. Historical SAD results using other legacy routes need separate recipe qualification. |
| X2 | None for the two scanned banks: zero affected tokens. Future banks with affected tokens require a separate provenance-mask fix and affected rescoring/retraining. |
| Kang export/label | No rerun for metadata or labeling, including existing `kang_v2` results. Using a smaller first-attempt subset requires training/evaluation again; no new teacher requests. Native-prefill reproduction remains unavailable with this API. |

CPU tests cover mean equivalence and ranking reversal, direct/offline selection
equivalence, legacy v1 compatibility, receipt validation/refusal, loss and gradient
arithmetic, an independent full-update AdamW oracle, trajectory ordering and
exposure, sealed synthetic banks with/without authored markers, and Kang export
metadata. The validation command and final pytest summary are recorded below.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_alf_baseline_fidelity.py \
  tests/test_alf_baseline_mechanisms.py tests/test_baseline_run_losses.py \
  tests/test_alfworld_teacher_pool.py
```

```text
153 passed, 1 skipped in 22.66s
```

The skip is the pre-existing optional real Kang bank cost check: that bank is
absent from this worktree. Synthetic collection tests use stub transports and
fake CPU environments only. `git diff --check` passed.
