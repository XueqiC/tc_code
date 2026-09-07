# RTD v1 status — C25l / T0–T6 / protocol v1.0.2

## C25l: recover completed campaigns after cleanup failure

Rai R0 round 1 finished scoring at **46.81%**, with complete artifacts in
`results/bfcl_std/rtd_R0_d9f417e2eea03ee5_r1/{data_overall.csv,resultdir,scoredir}`.
The shell then failed with `up_harness_outputs: command not found` and
`tag: unbound variable`, so the coordinator never published `evaluation-1.json`.
The frozen file itself had the full function name: the project log traces the
truncation to Bash reading shifted offsets after an edit during execution.

Cleanup now uses a per-tag EXIT trap, retains its tag outside function-local
scope, guards missing ownership variables, and isolates cleanup failures while
preserving the original exit status. Generation/scoring/copy failures return
nonzero explicitly. The script was replaced atomically to preserve the source
inode used by an already-running shell.

Before exporting or leasing a port, `evaluate()` checks for a completed campaign
under the tag lease. Reuse requires the verified checkpoint, current
hardware/data/base/harness guards, the exact saved stage binding, unchanged
merged/adapter export hashes, complete generation/scoring coverage, and a valid
aggregate. It logs **`reusing completed campaign`** and publishes the missing
evaluation record through the usual validation/reporting path. Reuse records
zero new campaign seconds/GPU reservation; the original attempt's timing and
nonzero exit remain in `compute.jsonl`. Incomplete current attempts retain the
existing archive-and-retry behavior; binding mismatches fail closed.

R0 also needs its historical tag: that campaign predates C25g/C25j and omitted
explicit base/tokenizer fields from its evaluation identity. Lookup follows only
the existing manifest-bound identity audit/migration chain, checks its hashes and
data binding, and derives the old tag from the complete historical identity.
The verified checkpoint still binds the immutable manifest's base/tokenizer.
The published record retains both the current `identity` and original
`campaign_identity`; the old stage binding, tag, exports and results are preserved.
Incomplete historical artifacts cannot trigger generation under an old identity.

The existing resume coordinator already evaluates each saved checkpoint before
launching the next training worker. CPU regressions cover R0/R1 proceeding from
completed round-1 scores directly to the round-2 worker, including R0's historical
identity shape. Both frozen scoring projections remain unchanged:

- Campaign: `66279841742a043566b960c20cc486718a6411157cf4f2f48c70a1e0272a9b22`.
- Evaluation/export: `a8897c3ae5df2911e072200f03322b4aa0a69ef36f95f2453d53b06fe8beda46`.

From `/home/xueqi/hq/projects/tc-alignment` on rai, run each command in its own
session after its existing coordinator/campaign has exited. R1 retains
`GPU_UTIL=0.6` for subsequent evaluations. Omit `--config` to use each saved
manifest, and retain `--acknowledge-code-drift` for continued training:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-20b20454-ae9f-7860-6801-430d68842a27 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u tools/rtd_experiment.py resume --arm R0 --run-dir results/rtd_v1/rai_R0 --acknowledge-code-drift > logs/rtd_resume_rai_R0_c25l.log 2>&1
```

```bash
GPU_UTIL=0.6 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u tools/rtd_experiment.py resume --arm R1 --run-dir results/rtd_v1/rai_R1 --acknowledge-code-drift > logs/rtd_resume_rai_R1_c25l.log 2>&1
```

Read-only CPU inspection verified R0's historical tag/stage/checkpoint binding,
the current scoring guard, and complete coverage of **5,106 scored tasks**.
Both arms' base, data, merged-export and adapter hashes matched their bindings.
At inspection, R1's bound tag was `rtd_R1_3b907b1e9e45c30b_r1`, but its aggregate
had not yet been copied into `results/bfcl_std`; its reuse path applies once the
campaign completes. These commands were documented, **not executed**. Live GPU
identity verification remains part of resume. No production run artifacts or
identity supplements were changed by C25l.

Validation passed: **281 CPU tests in 143.03 seconds**, with six existing toy
tensor/PEFT warnings. This includes all `tests/test_rtd_*.py` and
`tests/test_bfcl_std_campaign.py`, cleanup return/nounset/exit failures, completed
campaign reuse, historical audit-chain refusals, and both frozen scoring
projections. `bash -n` and `git diff --check` also passed.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_rtd_*.py tests/test_bfcl_std_campaign.py
bash -n tools/bfcl_std_campaign.sh
```

## C25j: scoring identity and audited parked-run updates

The [v1.0.2 protocol](rtd_v1_protocol.md) now lists the exact scoring files and
mixed-file source selectors. `bfcl-evaluation-harness-scoring-v3` excludes locks,
ports, launchers and evaluation/resume coordination. It retains BFCL package/data,
checker semantics, generation settings, export/merge math and complete-score
aggregation. Base/tokenizer and round weights remain content-bound. The rest of
the source tree is recorded as code-drift metadata; continuing training still
requires `--acknowledge-code-drift` when code changed.

Both CPU commands completed successfully:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python tools/rtd_experiment.py update-identity --run-dir results/rtd_v1/rai_R0
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python tools/rtd_experiment.py update-identity --run-dir results/rtd_v1/rai_R1
```

| Evidence | rai_R0 | rai_R1 |
| --- | --- | --- |
| Manifest-bound supplement | [6b5b84b4…](../configs/rtd/legacy_identities/6b5b84b47cac8e2dddc26e71aeec5004ad175ad1e0f3cf2b40aff60f08c32c26.json) | [6319fc3b…](../configs/rtd/legacy_identities/6319fc3b01a6076591bbff403cd0c20231818d65a2cf9f7585ca801b8bc06269.json) |
| Manifest mtime, ns | `1788731029299098054` | `1788735258210661466` |
| Old inventory checked | 189 files | 189 files |
| New scoring inventory checked | 190 files | 190 files |
| Changed old inventory members | Only `tools/bfcl_std_campaign.sh` | Only `tools/bfcl_std_campaign.sh` |
| Round-1 adapter SHA-256 | `e5ab727990cc9168086efc3aaefee7ef1e4578a58d250c238a8c360731f2d826` | `1563d3eafca87c0c89247b137d3e0ddd47d4fc4f9bbf4c1c7e3cd7c38270b2e0` |
| Checkpoint/model/tokenizer verification | Passed | Passed |
| Fresh-process scoring guard | Passed | Passed |
| Run files with unchanged sizes/mtimes | 31/31 | 31/31 |

Both supplements move from harness hash
`59fca38454f397049d968798dd90ebfa8076faa4291b336147965c8b950542e9`
to `f0faf0ac081ce34e9e1cad8235ccc7a77755c5f397e391dca8a327a8f1381a5e`.
The campaign's full old/new file SHA-256s are respectively
`2ae66bb6c6e860dfa2348bc7b7d7abe5e7d7b3890f1ba10a6227abf70e683569`
and `68a42837909f27e4b6811ba8e7eb96d8f63681604bbdfd6d26c7d27386a188a1`.
The old set did not contain the later-added lock helpers; the audit does not
invent old hashes for them. Their current hashes are source metadata.

The newer campaign and `evaluation.py` both passed comparison with the
[reviewed pre-run source bundle](../configs/rtd/identity_evidence/c25j.json).
Campaign scoring projection:
`66279841742a043566b960c20cc486718a6411157cf4f2f48c70a1e0272a9b22`;
evaluation/export projection:
`a8897c3ae5df2911e072200f03322b4aa0a69ef36f95f2453d53b06fe8beda46`.
Both match their historical projections. The archived export source predates
both manifests at `1788720479885000000` ns. Its timestamp is corroboration,
not a cryptographic historical binding; the campaign additionally has a matching
full-file hash in both prior supplements. Full per-file evidence and old
identities are retained in each supplement's `identity_updates` entry. Original
audits, C25g migrations, manifest bindings and legacy source identities remain.

After syncing the C25j code and `configs/rtd/identity_evidence/c25j.json`, run
this exact CPU command from the HPG repository root (no `.git` required):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/rtd_experiment.py update-identity --run-dir results/rtd_v1/R0
```

No GPU launches, training, production exports or official evaluation were run.
No run files under `results/` were modified. The two identity supplements are
the only run-related writes. The already-running old rai R0 evaluator retains
its old in-memory guard; subsequent resume/evaluation must use a fresh process.
Existing generated artifacts and completed scores are not relabeled.

C25j final validation passed **253 CPU tests in 138.03 seconds** across every
`tests/test_rtd_*.py` plus `tests/test_bfcl_std_campaign.py` (six existing toy
tensor/PEFT warnings). Coverage includes lock/launcher/logging exclusions,
checker/sampling/merge/aggregate sensitivity, split and legacy supplement
updates, reviewed historical projections without git, timestamp/content refusals,
missing scoring files, model/tokenizer/checkpoint corruption, concurrent edits,
immutable run files, retained audit history and idempotence. Python compilation,
shell syntax, CLI help and whitespace checks passed.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_rtd_*.py tests/test_bfcl_std_campaign.py
```

## C25g: git-free evaluation harness identity

hpg R0 job **41270793** failed after **19 seconds** because C25e required
`git rev-parse HEAD` in the synced BFCL checkout, which has no `.git`.
`bfcl-evaluation-harness-content-v2` now hashes content only. Its canonical
SHA-256 covers relative file paths, source/resource file sizes and SHA-256s,
the complete `bfcl_eval/data` manifest (names, sizes, SHA-256s), checker bridge,
parser, campaign, completeness checker, export tool, and saved `evaluation_*`
config. The current tree contains **113 source/package files, 71 data files,
and 5 tools**. Generated leaderboard results/scores, git internals and Python
bytecode caches are excluded.

Virtualenv entry points and installed package metadata no longer enter the
content guard: rai/hpg install paths and wrapper shebangs differ. Git HEAD is
best-effort `evaluation_harness_metadata`, separate from the content identity,
in new manifests, supplements and evaluation results. It is read only when
the leaderboard or Gorilla parent has a `.git` directory; missing git,
exit 128 and timeouts cannot abort identity construction. Metadata changes
cannot veto resume or completed-score reuse. Equal files/config produce equal
harness identities regardless of absolute checkout path or git availability.
The separate hardware, base, data, checkpoint and config guards still apply.

Both existing **rai R0 and R1** supplements have migrated from
`bdb1d984ddcba5eee848150d1e3b2a3455ab9f72a19f5e88fd06327ebc824108`
to content hash
`59fca38454f397049d968798dd90ebfa8076faa4291b336147965c8b950542e9`.
Each retains its full original manifest binding, original C25e/C25f audit and
legacy source identity. `identity_migrations` records the previous complete
git-based identity/hash and the verification evidence: original checkout
content hash, all tool hashes, evaluation config and saved manifest data hash
matched before migration. No run manifest, recovery state or checkpoint was
rewritten; run-file sizes/mtimes were unchanged, and both content guards passed.

For another existing git-based **legacy supplement**, the same CPU command
performs this verified migration and appends its audit note:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/rtd_experiment.py audit-legacy --run-dir results/rtd_v1/<run>
```

Migration verifies the previous content binding plus the saved data hash; it
does not require git or old mtimes on a synced copy. Evaluation itself never
silently migrates a supplement. A legacy run without a supplement still uses
C25f's explicit mtime audit, now covering the source/tools **and data** in the
content identity. Old completed evaluations remain bound to their old identity
and are not automatically relabeled by supplement migration.

C25g focused validation: **78 CPU tests passed**, including copies at different
absolute paths with/without `.git`, absent/failing git, data edits/additions/
renames/deletions, source/tool/config drift, git metadata changes during resume
and cached-score reuse, and migration binding/audit preservation and refusals.
The full RTD/campaign suite passed **201 CPU tests in 133.75 seconds**. A copy
of the real 189 harness files under a different absolute path, without `.git`
or a virtualenv, produced the exact same identity/hash. Python syntax,
whitespace and `audit-legacy --help` checks passed.
No GPU jobs, training, production export or official evaluation were launched.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_rtd_*.py tests/test_bfcl_std_campaign.py
```

## C25f: rai R1 legacy identity audit and reproducible command

The resume in `logs/rtd_resume_rai_R1_c25e.log` stopped with
`ValueError('legacy run needs an audited, manifest-bound evaluation identity supplement')`.
Like rai R0, `results/rtd_v1/rai_R1/manifest.json` predates the split identity
format. Its supplement is now generated by this CPU command from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/rtd_experiment.py audit-legacy --run-dir results/rtd_v1/rai_R1
```

For another legacy run without a supplement, replace the path after `--run-dir`. The command reads
the saved configuration, enumerates the same files used by
`evaluation_harness_identity`, and requires **every file mtime to be strictly
earlier than the manifest file's mtime**, using integer nanoseconds. It refuses
newer or equal mtimes, an already split identity (including partial split
fields), invalid saved config hashes, missing required harness files, or changes
to the manifest/harness inventory or file stats while hashing. No current YAML,
model, bank, GPU, training, or evaluation campaign is loaded or launched.

The command writes only `configs/rtd/legacy_identities/<full manifest hash>.json`.
The hash is the existing canonical `digest(manifest)`, preserving checkpoint and
recovery bindings. It pins the current evaluation identity and labels the old
combined hash `legacy-mixed-harness`, with `files: null`. JSON stdout includes
the supplement path, whether it was created, both identities, and the audit
evidence: manifest mtime, every harness file's path/mtime, file count, latest
mtime/path, and recording time. Repeating the command rechecks the evidence and
prints a fresh audit, preserving the existing supplement unchanged; it refuses
to replace a different identity binding.

The generated R1 supplement is
[`configs/rtd/legacy_identities/6319fc3b01a6076591bbff403cd0c20231818d65a2cf9f7585ca801b8bc06269.json`](../configs/rtd/legacy_identities/6319fc3b01a6076591bbff403cd0c20231818d65a2cf9f7585ca801b8bc06269.json).
Its evidence records:

| Evidence | Value |
| --- | --- |
| Harness files checked | 120; all strictly earlier than the manifest |
| Manifest mtime (ns since Unix epoch) | `1788735258210661466` |
| Latest harness mtime (ns since Unix epoch) | `1788720142084297290` |
| Latest harness file | `tools/bfcl_std_campaign.sh` |
| Original combined harness hash | `195ba8c391820afc5122db83112fe5f265ad771189a59df4812fc3b93a6be5b3` |
| C25f evaluation harness hash (before C25g migration) | `bdb1d984ddcba5eee848150d1e3b2a3455ab9f72a19f5e88fd06327ebc824108` |

The evaluation harness matches R0's audited identity. Mtimes corroborate
continuity; they cannot prove historical contents cryptographically or recover
the old separate RTD source hash. Training resume still requires
`--acknowledge-code-drift`, and all existing harness/checkpoint/config/data/base/
hardware guards remain in force. This command audits evaluation identity only;
it does not validate recovery or round checkpoints or resume the run.

C25f validation: **53 CPU tests passed** (25 new audit cases plus 28 existing
evaluation/resume cases), including accepted fake runs, per-file mtime evidence,
nanosecond/equal-time refusals across harness components, partial/full split
identity refusals, alternate venv metadata, changes during hashing, full-manifest
binding, and repeat audits that preserve existing supplements. CLI help and the
real R1 command passed. The real supplement passes `guard_harness`, and all R1
run-file sizes/mtimes remained unchanged. No files in `results/` were modified
and no GPU work was launched.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_rtd_legacy_audit.py tests/test_rtd_evaluation_resume.py
```

## C25e: end-of-round evaluation identity and rai R0 resume

`results/rtd_v1/rai_R0` finished round 1 in
`logs/rtd_run_rai_R0b.log`: **12 committed steps, spend 188/825753**.
The coordinator then rejected evaluation because the old `harness_hash`
included RTD source files edited by C25d during training. CPU inspection verified
the round-1 adapter and round-state hashes, the recovery snapshot/manifest/ledger
bindings, and recovery at **round 2, step 1, `round_start`** with exactly one
completed checkpoint. Round 1 does not need retraining.

New manifests store two identities. As revised by C25g above,
`evaluation_harness` and its `harness_hash` cover BFCL package/data contents,
checker bridge and parser, campaign/completeness/export tools, and evaluation
configuration; git HEAD is optional metadata outside this identity.
These remain hard guards, alongside checkpoint,
config, base-model, data and hardware hashes. Evaluation identity includes the
harness hash, and completed scores still require matching artifacts and complete
task coverage. The harness is checked again after a campaign finishes.

`rtd_source` stores a source hash and per-file hashes. Source changes append
`code_drift` events with old/new hashes and differing filenames to the run's
hash-chained `code_drift.jsonl`; evaluation JSON and budget-report rows retain
those records. Training resume requires **`--acknowledge-code-drift`** when source
changes. Standalone evaluation of a trained checkpoint requires no acknowledgment,
including reuse of a previously completed evaluation after another source edit.
The original manifest remains immutable so existing checkpoint/recovery bindings
continue to verify.

The pre-C25e R0 manifest contains only a combined hash. Its reviewed supplement,
[`configs/rtd/legacy_identities/6b5b84b47cac8e2dddc26e71aeec5004ad175ad1e0f3cf2b40aff60f08c32c26.json`](../configs/rtd/legacy_identities/6b5b84b47cac8e2dddc26e71aeec5004ad175ad1e0f3cf2b40aff60f08c32c26.json),
pins the audited evaluation harness to the **full original manifest hash**;
lookup never relies on the run directory name. All 120 evaluated harness files
had modification times before the saved manifest. This corroborates the reported
RTD-only edits, but cannot prove historical contents cryptographically or recover
the old separate source hash. Legacy drift records therefore explicitly label
the old hash `legacy-mixed-harness` and mark the file diff unavailable. Other
legacy manifests require their own audited supplement; there is no automatic
trust of the current harness or acknowledgment bypass for a harness mismatch.

From `/home/xueqi/hq/projects/tc-alignment` on rai, resume on the **same RTX 6000
Ada UUID saved in the manifest**:

```bash
mkdir -p logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-20b20454-ae9f-7860-6801-430d68842a27 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u tools/rtd_experiment.py resume --arm R0 --run-dir results/rtd_v1/rai_R0 --acknowledge-code-drift > logs/rtd_resume_rai_R0_c25e.log 2>&1
```

Omit `--config`: resume uses the saved manifest configuration exactly, including
the **4096-token action cap**. C25d's current YAML introduces 512/1024 caps and
memory settings and is correctly rejected as a different config for this run.
The runtime preserves the legacy scalar cap when per-benchmark caps are absent;
continuation uses the updated RTD implementation under the explicit acknowledgment.

The coordinator verifies `round-1/checkpoint.json`, evaluates that endpoint
without starting a round-1 training worker, and only then starts the worker
through round 2. Workers reload the original recovery state. A failed evaluation
cannot advance training; completed evaluations are validated and reused.

CPU export is explicit: both base and PEFT loads use `device_map='cpu'`; the
Hub export/verification subprocess receives `CUDA_VISIBLE_DEVICES=''` and
`--verify`. Evaluation calls `tools/bfcl_std_campaign.sh` with the bound merged
model and `BFCLSTD_PRESERVE_GENERATION=1`. The campaign runs
`tools/bfcl_generation_check.py` before scoring, retries missing IDs at most twice,
and refuses incomplete generation. RTD independently checks generation IDs,
duplicates, score categories/counts and aggregate validity before publishing.

C25e validation: 28 new CPU tests passed, covering fake campaign resume/retry,
source-only drift and cached evaluation, legacy binding, worker acknowledgment,
CPU export placement, saved config, alternate BFCL venv and hard provenance
failures. The full RTD/campaign suite passed **146 tests in 129.70 seconds**;
the final affected-file run passed **48 tests in 9.59 seconds**, including the
five additional cases for a real tiny CPU PEFT export, edits during evaluation,
and legacy/current runtime token limits. Completed-training campaign resume also
passes without a code-drift acknowledgment. The real-run CPU audit verified its
saved config/base/data/bank/harness against current files and reached the expected
acknowledgment refusal; it substituted saved hardware metadata to avoid CUDA, so
live hardware verification remains part of the actual resume command. Python
compilation, shell syntax, CLI help and whitespace checks passed.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_rtd_*.py tests/test_bfcl_std_campaign.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_rtd_evaluation_resume.py tests/test_rtd_runtime_memory.py
```

**No GPU training, production model export, official evaluation, or teacher calls were
launched by C25e.** The resume command above is prepared for execution.

## Earlier C25d evidence

The fresh-directory commands below describe C25d experiments with the new caps;
use the C25e resume command above to continue the existing `rai_R0` run.

## C25d: capped sampling outcomes and round-start memory

C25d accepts actions that reach their configured token cap without EOS. The
official BFCL checker sees the sampled text unchanged, including its last token;
its verdict supplies the reward. A syntactically valid capped call can pass.
There is no retry, filtering, forced EOS, or automatic zero reward. Native CE,
REINFORCE, source losses, RMS and the KL pilot include exactly the sampled action
tokens, with an EOS term only when EOS was sampled. The leave-one-out baseline
and rollout denominator include capped outcomes. Empty/missing prompts or state,
tokens after EOS, and missing generation scores remain errors. Context room is
also a deterministic generation limit; prompts are never cut to make room.

`ActionTrace`, `SourceSample`, and serialized feedback `TaskRollout` carry
`truncated`. `score_consistency.eos` records an empty EOS position list for capped
samples. The compute ledger writes every feedback rollout and a
`window_truncation` event with reference/actual counts, totals, and an explicit
actual-feedback-reuse flag. `steps/*.json` and `trajectory.json` preserve these
counts. Report JSON/CSV and Markdown show counts by committed decision window;
actual feedback reused at the same parameters counts once. Journal totals also
include failed/repeated attempts and are reported separately from committed
totals. Source truncation and source-slot exposure counts remain available too.

The executable config now declares per-benchmark **new-token** limits:

```yaml
max_action_tokens_by_benchmark:
  bfcl: {single_turn: 512, multi_turn: 1024}
max_state_batch_size: 2
memory_peak_budget_gb: 38
memory_reserve_gb: 2
memory_state_estimate_gb: 16
```

These BFCL limits apply to source states and every feedback harness action,
including subsequent multi-turn calls. Missing BFCL entries take these defaults;
positive integer overrides are supported. Other benchmark keys can declare the
same single-/multi-turn mapping. The legacy scalar `max_action_tokens` is only a
fallback for a backend without matching benchmark limits. Resolved limits are
bound into the run config/backend identity and recorded on each sampled action.
Greedy endpoint evaluation retains its separate campaign settings.

R0 diagnosis from `results/rtd_v1/rai_R0/compute.jsonl` (decimal GB):

| Sequence | Sub-phase inferred from execution order | Prompt + action tokens | Allocated peak GB | Reserved peak GB |
|---:|---|---:|---:|---:|
| 208 | Source generation/teacher-forced score check | 7,065 + 84 | 17.661 | 45.554 |
| 344 | Source RMS preconditioner forward | 7,065 + 84 | 24.027 | 45.684 |
| 346 | Source RMS preconditioner forward | 5,757 + 89 | 24.114 | 49.241 |

The old measurements were cumulative high-water marks, so these are the points
where the recorded maxima increased, not isolated backward peaks. All 36 source
samples (18 states × 2) were generated sequentially. The next 36 teacher-forced
forwards belong to RMS. There is no KL pilot event in this journal: first-round
pilot is deferred until evidence is purchased. Thus the near-card-capacity
reservation was already present in source score checking and grew again during
RMS, with roughly half as much live tensor memory. This supports allocator cache
growth across differing sequence sizes as the immediate cause of R0's reported
47.5 GB process usage; it was not a batch of all states or a KL trial. R1's
`logs/rtd_run_rai_R1b.log` independently confirms the old capped-action exception.

Consecutive state/source batches now use `torch.cuda.mem_get_info(device)` on
logical `cuda:0` within inherited visibility. Available workspace is the smaller
of `(free - reserve)` and `(peak budget - (total - free))`; the latter includes
other processes and non-PyTorch memory. Batch size is capped by
`max_state_batch_size` and the configurable per-state workspace estimate. At
about 8–9 GB resident on the 48 GB card, these defaults choose one state. Each
action/gradient still executes separately in its original order, with the same
RNG draws and reductions. A singleton whose estimate exceeds headroom is flagged
as `singleton_exceeds_estimate`, never dropped. The estimate is a scheduling
bound, not a guarantee that an indivisible long state fits.

Caches are released between batches and sub-phases. Generation, score checks,
initial hidden projection, source preconditioner, pilot gradient, and KL trials
have separate `subphase_memory` records in `compute.jsonl`, including failures.
KL vocabulary tensors are scoped to one action and only scalar trial values
survive to the unchanged ordered mean. Nested phase resets preserve maxima in
ancestor phases and `step_memory`; report JSON/CSV include per-phase peaks.
`memory_batch` records show free/total memory, estimated workspace, and batch size.
**The target is under 40,000,000,000 bytes peak on rai's 48 GB card. C25d has not
measured that target: only CPU checks were run, and no GPU jobs were launched.**

Use fresh directories because both the harness and cap configuration changed.
Run these full-config commands sequentially on rai's physical GPU 2; the saved
failed runs remain intact. The inherited device binding is checked by each worker.

```bash
mkdir -p logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u tools/rtd_experiment.py run --arm R0 --config configs/rtd/v1_bfcl_c25.yaml --run-dir results/rtd_v1/rai_R0_c25d > logs/rtd_run_rai_R0_c25d.log 2>&1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u tools/rtd_experiment.py run --arm R1 --config configs/rtd/v1_bfcl_c25.yaml --run-dir results/rtd_v1/rai_R1_c25d > logs/rtd_run_rai_R1_c25d.log 2>&1
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python tools/rtd_experiment.py report --run-dir results/rtd_v1/rai_R0_c25d --run-dir results/rtd_v1/rai_R1_c25d --out results/rtd_v1/rai_report_c25d
```

Inspect phase/step peaks and truncation while a rerun is in progress (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
for arm in ('R0', 'R1'):
    path = Path(f'results/rtd_v1/rai_{arm}_c25d/compute.jsonl')
    if not path.exists():
        continue
    for line in path.read_text().splitlines():
        if not line.endswith('}'):
            continue  # a concurrent final append may still be incomplete
        row = json.loads(line)
        if row['kind'] in {'subphase_memory', 'step_memory'}:
            print(arm, row.get('round'), row.get('step'), row.get('operation'),
                  row.get('context'), row['status'],
                  'allocated_GB', round(row['peak_allocated_bytes']/1e9, 3),
                  'reserved_GB', round(row['peak_reserved_bytes']/1e9, 3),
                  'under_40GB', max(row['peak_allocated_bytes'], row['peak_reserved_bytes']) < 40e9)
        elif row['kind'] in {'window_truncation', 'device_binding'}:
            print(arm, row)
PY
```

These peak records measure the PyTorch allocator. Validate total process memory
on the card as well; the scheduler's free-memory readings include context/library
and other device allocations, but are boundary samples rather than process peaks.

CPU validation: **122 tests passed** in the RTD/campaign suite (124.06 seconds),
then **3 affected tests passed** (4.45 seconds) after avoiding unnecessary cache
flushes for already-cached pool states. The follow-up includes a new exact
batch-size equivalence check for source samples, RMS, pilot eta, committed steps,
gate, sampling RNG and final parameters. Capped HF fixture outputs, actual
official BFCL checker pass/fail behavior, capped REINFORCE/LOO, reuse accounting,
trajectory/report persistence, visible-device memory sizing, and nested/failed
phase peaks all passed. Python compilation and whitespace checks passed. The
two suite warnings are the existing toy tensor-to-float and PEFT Conv1D notices.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_rtd_*.py tests/test_bfcl_std_campaign.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_rtd_checks.py -k 'capped_rollouts_survive or state_batch_size'
```

## Earlier C25c evidence

C25c fixes the worker device boundary and backward memory use after the project
lead's full rai R0 run exhausted the 96 GB card. The lead reports that the
two-slot smoke fit. **C25c ran CPU tests only; no GPU training, production model
load, teacher call, or official evaluation was launched. The full-run 45 GB
target still requires measurement with the commands below.** C25b's earlier
score-consistency diagnosis is retained below as historical evidence.

## C25c: device binding and sequential checkpointed backward

`logs/rtd_run_rai_R0.log` failed in `round_start -> rms_diagonal -> score_source`,
before the first eight-slot update. Journal sequence 323 records **6,457 prompt
tokens + 135 action tokens**, and the allocation failed inside eager attention's
FP32 softmax. The run had already sampled prompts up to 7,065 tokens. Thus even
a single source graph, retained across all decoder layers without checkpointing,
exhausted memory. The log reports 92.33 GiB allocated by PyTorch and 93.55 GiB
process use, while requesting another 2.59 GiB.

The worker launch is in `src/bfas/rtd/cli.py:run_campaign`, called from the
experiment CLI. Audit found no RTD config/device override, environment stripping,
or nonzero logical CUDA index. The old `subprocess.run` inherited the environment
implicitly. Its saved manifest identifies GPU UUID
`97762062-28db-d85e-cb46-294b082f94df`, the 96 GB RTX PRO 6000 at physical index 1
in a read-only `nvidia-smi` inventory. Physical index 2 is the 48 GB RTX 6000 Ada,
UUID `GPU-20b20454-ae9f-7860-6801-430d68842a27`. CUDA's default `FASTEST_FIRST`
enumeration can differ from the physical/PCI order on these mixed cards. This
explains how numeric visibility can select the unexpected card; the old journal
did not record visibility/order, so it cannot prove the historical environment.

The launcher now defaults `CUDA_DEVICE_ORDER=PCI_BUS_ID` **before importing
torch**, honors an explicitly supplied order, and leaves `CUDA_VISIBLE_DEVICES`
verbatim. Each worker receives an explicit copy of the entire parent environment,
uses `cuda:0` relative to visibility, and verifies the coordinator's GPU UUID
before model load. The coordinator prints the binding, and the worker writes a
`device_binding` event with visibility, ordering, logical device, UUID, name,
and total capacity. GPU UUID visibility is also supported directly.

Backward now computes each complete source/teacher action separately and releases
its graph before the next side or slot. Accumulation, `g_i^T - g_i^S`, the gate
contraction, RMS `P`, and the update use only ordered LoRA coordinates. A 21M
FP32 gradient vector occupies about 84 MB (80 MiB); these are parameter-shaped
LoRA tensors, not base-model gradient buffers. Only the current slot's vectors
and the running result are retained, so eight slots do not require eight graphs
or an eight-by-model gradient array.

Every production decoder layer uses non-reentrant activation checkpointing for
grad-enabled scoring, covering RMS, pilot/update gradients, REINFORCE, and gate
VJP. The wrapper preserves eval mode and explicitly rebinds the scoring LoRA
snapshot during recomputation: ordinary HF training-only checkpointing would be
inactive in eval mode, and an unbound replay could use the resident student's
weights after `functional_call` restores them. Source/teacher weights are applied
before each backward, then slot gradients are accumulated with the same mean.
Full actions, EOS-inclusive CE, eager attention, and the full configuration's
eight-slot exposure are preserved.

`compute.jsonl` now contains `step_memory` records with `round`, `step`,
`start_phase`, `status`, `peak_allocated_bytes`, and `peak_reserved_bytes`.
Peaks reset once per step attempt; nested forward timers do not reset them.
Step 1 includes source/geometry setup. Failed attempts are recorded, and a resumed
step's `start_phase` identifies the measured suffix. `compute_end` also records
the current step's high-water marks, including failures. These are PyTorch
allocator measurements; CUDA context/library allocations are additional process
memory. Production peak memory and checkpointing runtime remain unmeasured.

Use new run directories: the failed run's harness and hardware bindings differ
from C25c. Its original log, manifest, journal, and recovery checkpoint remain
available for diagnosis. Exact full-config reruns on rai's physical GPU 2:

```bash
mkdir -p logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u tools/rtd_experiment.py run --arm R0 --config configs/rtd/v1_bfcl_c25.yaml --run-dir results/rtd_v1/rai_R0_c25c > logs/rtd_run_rai_R0_c25c.log 2>&1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u tools/rtd_experiment.py run --arm R1 --config configs/rtd/v1_bfcl_c25.yaml --run-dir results/rtd_v1/rai_R1_c25c > logs/rtd_run_rai_R1_c25c.log 2>&1
```

Run the arms sequentially on that card. The UUID form can replace
`CUDA_VISIBLE_DEVICES=2` with
`CUDA_VISIBLE_DEVICES=GPU-20b20454-ae9f-7860-6801-430d68842a27`.
For the 80 GB A100, use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3`
and fresh run directories. These full commands include the configured official
evaluation after each round. **None of these GPU commands was executed by C25c.**

After a rerun, inspect the binding and report per-step peaks (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
for run_dir in (Path('results/rtd_v1/rai_R0_c25c'), Path('results/rtd_v1/rai_R1_c25c')):
    if not (run_dir/'compute.jsonl').exists():
        continue
    for line in (run_dir/'compute.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'device_binding':
            print(run_dir.name, row)
        elif row['kind'] == 'step_memory':
            print(run_dir.name, row['round'], row['step'], row['start_phase'], row['status'],
                  'allocated_GiB', round(row['peak_allocated_bytes']/2**30, 3),
                  'reserved_GiB', round(row['peak_reserved_bytes']/2**30, 3))
PY
```

C25c CPU regressions compare all eight sequential slots with a batched loss for
learned, scalar, half, and teacher-only gates; assert saved activations are freed
before each side; verify LoRA-only gradients, RMS, and gate VJP; compare exact
BF16 Qwen3.5 checkpointed gradients at distinct functional snapshots; and test
environment inheritance, early ordering, UUID mismatch refusal, and failed-step
peak accounting. Validation totals are recorded in the validation section below.

## C25b: source sampling/score failure

The original `logs/rtd_smoke_R1.log` and `results/rtd_v1/rai_smoke_R1/compute.jsonl`
show a **573-token prompt, 38-token complete action**, followed by a successful
teacher-forced forward and the old sequence-score assertion. The deterministic
smoke pool identifies the first state as **`live_relevance_7-7-0`**, state hash
`08b2a63eca295b1f36e7340f3eabd22d1fecb9f1c76a8b236b992ba86f76d046`.
Its prompt ends in `<|im_start|>assistant\n<think>\n\n</think>\n\n`;
the tokenizer's EOS is **248046**. Generation enforces exactly one terminal EOS,
and `SourceSample` retains the original token IDs including EOS. Scoring
re-encodes the same already-rendered prompt, without another chat-template or
thinking-prefix application. Both attention masks are all ones; the score mask
selects only the action, including its final EOS. No boundary/EOS bug was found
in these paths.

**The original generated tokens and either score were never persisted.** The
assertion preceded the `source_sample` journal append, and the only recovery
checkpoint is before the round. Their exact per-token values and max/mean
differences cannot be recovered from these artifacts. The forensic record
[c25b_diagnosis.json](../results/rtd_v1/rai_smoke_R1/c25b_diagnosis.json) records the
known state/prompt and explicitly marks missing measurements. The old log,
manifest, recovery checkpoint, and journal are preserved.

The identified numerical mechanism is **cached BF16 generation versus full
BF16 teacher forcing**, with FP32 log-probability calculations in both paths.
HF generation casts logits to FP32, applies unwarped categorical sampling, and
sums selected FP32 log-softmax values as Python floats. The scorer computes
native CE in FP32 and sums in FP32. Qwen3.5 also selects different linear-attention
implementations: cached single-token recurrent/conv-state updates during decoding,
versus a chunked delta rule/full convolution in teacher forcing. Thus FP32
reduction does not remove differences already present in BF16 model outputs.
This mechanism is reproduced on CPU; its exact contribution to the old GPU
action remains unmeasured.

The CPU regression uses a random **four-layer Qwen3.5** with linear and full
attention, BF16 base, FP32 LoRA, eager attention, C25 rank/alpha/target modules,
573 prompt tokens, and HF `generate` at T=1/top_p=1/top_k=0. Model seed 8 and
sampling seed 5 produce 21 tokens including EOS:

| Measurement | Nats |
|---|---:|
| Mean absolute token difference | 0.000409523646 |
| Maximum absolute token difference | 0.001040458679 |
| Absolute sequence difference | 0.001580476761 |
| Old sequence tolerance | 0.001355696845 |

This actual backend execution fails the old assertion while comfortably passing
the requested numerical bounds. Full tokens, both per-token score arrays,
EOS/masks and measured dtypes are in
[CPU diagnostic.json](../results/rtd_v1/c25b_cpu_repro/diagnostic.json).

`score_consistency_tolerance: {mean_abs: 0.05, max_abs: 1.0}` is configurable in
the C25 YAML; both **nats/token** bounds must pass, with EOS included. Missing
configuration uses these defaults, and invalid/nonfinite tolerances are refused.
The native CPU oracle keeps tight defaults and the double-precision sequence
check remains available to tests. Prompt mismatch, incomplete EOS/masks, missing
score coverage, nonfinite values, and inconsistent recorded totals still fail.

Every source sample and every REINFORCE action writes a hash-chained
`score_consistency` event to **`compute.jsonl` before tolerance enforcement**.
It contains state/action identifiers, prompt and generated IDs/text, generation
and rescored per-token log probabilities, EOS handling, explicit masks/shifted
positions, model/logit/reduction dtypes, backend identities, both errors, bounds,
and pass/fail. Source events include the full state/parent hashes and sample
index; feedback events include task/action identity and the prompt-token state
hash. The manifest records the tolerance contract and journal location.
REINFORCE differentiates the **same `score_tokens` native-CE tensor** returned by
the teacher-forced audit; diagnostics detach only the values written to disk.

The lead should use the fresh C25b smoke command below. Config/harness hashes
changed, so the old run cannot be resumed under this patch. The rerun will capture
the original state's newly sampled action even if a tolerance check fails.

## Active protocol and bank

The project lead's [v1.0.1 decision](rtd_v1_protocol.md) supersedes C23/C24's
131,072-token fallback, hidden-usage denominator and 4×2 single-turn dose.
The original section-14 template is retained in `configs/rtd/v1_bfcl.yaml`.
The executable configuration is
[configs/rtd/v1_bfcl_c25.yaml](../configs/rtd/v1_bfcl_c25.yaml).

* Public class caps: demo attempt **65,536**; generator item **8,192**.
* Usable bank: **420 packages = 84 demos + 336 generator items**.
* Public-cap sum: **8,257,536** output tokens.
* Cumulative ceilings after rounds 1/2/3: **825,753 / 2,064,384 / 4,128,768**.
* Recorded bank usage: **55,370 = 21,203 exact + 34,167 estimated**. It does not
  set caps or ceilings. Hard reservation uses caps; reveal settles actual usage.
* Parent support: **40**, hash folds **22/18**. There are 33 runnable support
  parents (28 single-turn, 5 multi-turn); seven memory/web-search parents lack
  an initial-state adapter and are explicitly unavailable. There are 309/111
  usable packages in inner folds 0/1. No calibration parents enter training.
* Full archive: 698 records; 36 stateful demand attempts, 135 prerequisite/stale
  records and 107 protected generator items remain unavailable. 247/302 event
  rows attach to paid packages. Historical teacher totals remain **1,233,607
  exact demo tokens + 142,727 estimated generator tokens**.
* Active rebuilt bank: `data/rtd/v1_bfcl_c25/`; C23/C24 banks are preserved.
  Public manifest SHA256:
  `2d27d0f4dcfd1076ac2fbdba6ff3d2789caf5971c14a5da61634d710d00084e8`.

Generator packages remain estimated historical items: provider batch, rejected
writer draft, repair and verification histories cannot be recovered. These caps
are the lead's public retrospective convention, not a retry-wide online bill
bound. There is no online teacher API implementation in this runner.

## Recomputed affordability

Every usable, dependency-free package individually fits all three ceilings.
The capacity columns assume purchasing only that class at its cap and are bounded
by bank inventory; they do not claim simultaneous capacity in both classes.
There is still at most **one new package per decision**, four decisions per round.

| Checkpoint | Cap ceiling | Individually affordable demos/items | Single-class capacity demos/items |
|---|---:|---:|---:|
| 10% | 825,753 | 84 / 336 | 12 / 100 |
| 25% | 2,064,384 | 84 / 336 | 31 / 252 |
| 50% | 4,128,768 | 84 / 336 | 63 / 336 |

The following are **zero-prior-spend illustrations**, using the full checkpoint
ceiling as remaining budget. The actual runner recalculates
`b = ledger.remaining / remaining_windows_in_this_round` at each decision;
prior-round actual spend persists when authorization rises. A package need not
fit b individually: b constrains expected cost, while the hard ledger uses the
full remaining authorization.

| Checkpoint | Decision step | Windows left | b | Single-class capacity demos/items |
|---|---:|---:|---:|---:|
| 10% | 1 | 4 | 206,438.25 | 3 / 25 |
| 10% | 4 | 3 | 275,251 | 4 / 33 |
| 10% | 7 | 2 | 412,876.5 | 6 / 50 |
| 10% | 10 | 1 | 825,753 | 12 / 100 |
| 25% | 1 | 4 | 516,096 | 7 / 63 |
| 25% | 4 | 3 | 688,128 | 10 / 84 |
| 25% | 7 | 2 | 1,032,192 | 15 / 126 |
| 25% | 10 | 1 | 2,064,384 | 31 / 252 |
| 50% | 1 | 4 | 1,032,192 | 15 / 126 |
| 50% | 4 | 3 | 1,376,256 | 21 / 168 |
| 50% | 7 | 2 | 2,064,384 | 31 / 252 |
| 50% | 10 | 1 | 4,128,768 | 63 / 336 |

## Delivered implementation

| Block | Implementation |
|---|---|
| T0–T2 | Existing cap/bank/broker/ledger, full-state transport and frozen features; v1.0.1 caps and public-cap affordability adopted. |
| T3–T4 | Existing fixed-P map, train-only RMS/KL pilot, official BFCL feedback, insertion labels and Bayesian budget selector. Stable nonnegative KL arithmetic added for tiny pilot steps. |
| T5 | `tools/rtd_experiment.py`, backed by `src/bfas/rtd/{cli,experiment,runtime,persistence,evaluation}.py`; all eight subcommands, R0/R1, recovery, merged official evaluation, budget reports, fixed-ledger retraining and scalar/P=I/random component configs. |
| T6 | `tests/test_rtd_checks.py`: all seven section-13 properties, complete toy three-round runs, crash/charge recovery, native HF sampling and score equivalence, scalar/fixed-collection paths, and score completeness. |

Per round the runner rotates folds first, freezes the source policy, samples two
independent complete actions per legal state at T=1/top_p=1, fits train-only
normalization and RMS P, and calibrates eta from purchased evidence at KL .005.
Newly purchased generated states are sampled only after reveal under the same
frozen source and initial representation; they use the frozen moments, without
refitting P or feature normalization. No hidden generated question is a candidate
feature. Old replay draws parents then deduplicated states uniformly; new replay
draws uniformly within the package. Teachers are drawn uniformly over cached
request records and fixed for that block. Same-package aliases add no mass;
identical answers from independently paid requests retain empirical multiplicity.

There are 12 committed schedule steps and windows 1/4/7/10 per round. Each decision
saves the common start and virtual old update, samples reference feedback, chooses
once including empty, reveals/settles a selected package, applies the same-start
.75/.25 mixture, obtains actual feedback, updates phi for the next step, merges
the package, and refreshes the posterior. R0 uses zero acquisition values and
fixed a=.5 with the same prior, cost regression and budget procedure. Empty never
redraws. First-purchase calibration occurs after reveal and before insertion
measurement; its identity reference remains unchanged. Round eta is then frozen.

Single-turn feedback uses up to **8 parents × 4 full sampled generations**, scored
by the official AST/relevance bridge; multi-turn uses up to **4 × 2** from fresh
official simulator states. Available parent counts are recorded. Smoke uses a
stable hash slice of two single-turn parents per fold, two slots, one decision
window and one rollout per feedback parent; its explicit zero baseline replaces
undefined single-sample LOO. Ordinary runs retain same-task LOO.

**Backend decision:** local **HF generate with KV caching**, unwarped categorical
sampling (`top_k=0`, T=1, top_p=1, repetition penalty 1). The same HF model recomputes
full action likelihood including EOS. This avoids starting/reloading vLLM for
every source/reference/actual policy. LoRA coordinates stay FP32; the base model
uses BF16 and eager attention. Gradients accumulate per complete slot. The streamed
sigmoid VJP implements equation (5) by contracting `(gT-gS)` with `P*gJ` and the
frozen gate Jacobian, exactly matching the general autograd oracle without a
second backward through fused LM kernels. GPU throughput is unmeasured. Truncated
outputs and sampling/score discrepancies beyond the recorded tolerance fail explicitly; no forced EOS or retries
condition the policy on short outputs.

`teacher.jsonl` is a durable hash chain of cap reservations, authorization changes,
reveals, exact/estimated usage, dependencies and releases. `compute.jsonl` retains
source actions, decision distributions/public caps, rollouts, scores, gradient
counts, nested compute intervals and failed/repeated attempts. GPU seconds denote
synchronized time with the GPU reserved, not a hardware-counter measure of kernel
occupancy. Nested timings are not summed twice; unclosed intervals are reported
as incomplete. `steps/` records the exact old/new texts, masks implicit in prompt
versus EOS-inclusive action IDs, source/teacher tokens, raw slots and weighted
exposure. No-op steps report zero supervised exposure.

Recovery atomically publishes a SHA-bound checkpoint pointer after each phase.
Choice and RNG are durable before reveal; a reveal ahead of the checkpoint is
admitted only for that recorded choice. Committed steps, pending evidence,
posterior/controller state, initial/source LoRA tensors, feature moments, P, eta
and RNG states persist. Base checkpoint, tokenizer, harness/source code, config,
public data/integrity manifest, official data and hardware hashes bind the run.
Torn final journal writes are preserved separately; complete events are never
rolled back. A per-run lock excludes concurrent writers. Round LoRA checkpoints
and frozen round state are retained independently of the rolling recovery files.

With the shipped configuration, `run` releases the training process after each
round, evaluates that round's merged checkpoint, then restores the next round.
The campaign uses `tools/bfcl_std_campaign.sh` and the existing Hub-faithful export
convention, with an RTD-only retained-generation option and explicit greedy
endpoint temperature 0. Existing campaign defaults remain unchanged. Expected
coverage is derived from official loaders: currently **5,217 generation IDs**
including prerequisites, **5,106 scored IDs**. Missing, duplicate or unexpected
IDs/categories, inconsistent score summaries or changed checkpoint artifacts
prevent aggregation. Saved evaluations are idempotent. Reports use actual spend
on the x-axis and show cap ceilings, exact/estimated spend, exposure, feedback,
GPU time and evaluation status. Missing evaluation means a blank score.

## Commands (not launched in C25)

From the project root, prepare/re-audit the bank on each machine. Ingestion refuses
an existing directory; use the second command if C25 data are already present.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python tools/rtd_experiment.py audit --build-bank --config configs/rtd/v1_bfcl_c25.yaml
PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python tools/rtd_experiment.py audit --config configs/rtd/v1_bfcl_c25.yaml
```

Exact rai C25b smoke rerun command for the project lead (GPU 1; not run by C25b):

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u tools/rtd_experiment.py smoke --arm R1 --config configs/rtd/v1_bfcl_c25.yaml --run-dir results/rtd_v1/rai_smoke_R1_c25b > logs/rtd_smoke_R1_c25b.log 2>&1
```

Exact hpg submissions, each **fsu-compsci-dept, one B200, 240G, eight hours**:

```bash
mkdir -p logs
sbatch --export=ALL,RTD_ARM=R0,RTD_CONFIG=configs/rtd/v1_bfcl_c25.yaml scripts/rtd_run_hpg.slurm
sbatch --export=ALL,RTD_ARM=R1,RTD_CONFIG=configs/rtd/v1_bfcl_c25.yaml scripts/rtd_run_hpg.slurm
```

The default outputs are `results/rtd_v1/R0` and `results/rtd_v1/R1`. To guarantee
both arms use the identical allocated GPU, the following single-allocation
submission runs those same launchers sequentially; choose this **instead of** the
two submissions above for the strict same-machine comparison:

```bash
sbatch --job-name=rtd_R0_R1 --account=fsu-compsci-dept --qos=fsu-compsci-dept --partition=hpg-b200 --nodes=1 --ntasks=1 --gres=gpu:b200:1 --cpus-per-task=14 --mem=240G --time=08:00:00 --output=logs/%x_%j.out --wrap='RTD_ARM=R0 RTD_CONFIG=configs/rtd/v1_bfcl_c25.yaml bash scripts/rtd_run_hpg.slurm && RTD_ARM=R1 RTD_CONFIG=configs/rtd/v1_bfcl_c25.yaml bash scripts/rtd_run_hpg.slurm'
```

Total runtime for both arms within eight hours is unmeasured. Resume requires the
same saved hardware hash (including host/GPU UUID/software); use the same
allocation/device. Separately scheduled jobs can receive different GPUs, and
reports explicitly mark that hardware mismatch rather than claim matched arms.

Rai/local resume and explicit endpoint evaluation (R0 paths can replace R1):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python tools/rtd_experiment.py resume --arm R1 --config configs/rtd/v1_bfcl_c25.yaml --run-dir results/rtd_v1/R1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. .venv/bin/python tools/rtd_experiment.py evaluate --run-dir results/rtd_v1/R1 --round 3 --port 9170
```

HPG resume uses `RTD_COMMAND=resume` and `RTD_RUN_DIR`, on the original hardware:

```bash
sbatch --export=ALL,RTD_ARM=R1,RTD_COMMAND=resume,RTD_RUN_DIR=results/rtd_v1/R1,RTD_CONFIG=configs/rtd/v1_bfcl_c25.yaml scripts/rtd_run_hpg.slurm
```

CPU ledger inspection, report generation, and the scalar gate configuration:

```bash
.venv/bin/python tools/rtd_experiment.py replay-ledger --run-dir results/rtd_v1/R1
.venv/bin/python tools/rtd_experiment.py report --run-dir results/rtd_v1/R0 --run-dir results/rtd_v1/R1 --out results/rtd_v1/report
.venv/bin/python tools/rtd_experiment.py swap-component --config configs/rtd/v1_bfcl_c25.yaml --component gate --value scalar_sigmoid --out configs/rtd/v1_bfcl_scalar.yaml
```

Fixed-ledger 2×2: generate one final-collection config from each completed
acquisition run, then **retrain both D0/R0 and D1/R1 from theta0** for each config.
The adaptive runs are not reused as diagonal cells. Dependencies and original
full package usage are charged once in each new fixed-evidence experiment.

```bash
.venv/bin/python tools/rtd_experiment.py replay-ledger --run-dir results/rtd_v1/R0 --out configs/rtd/fixed_A0.yaml
.venv/bin/python tools/rtd_experiment.py replay-ledger --run-dir results/rtd_v1/R1 --out configs/rtd/fixed_A1.yaml
# For each config: launch once with RTD_ARM=R0 and once with RTD_ARM=R1.
```

No commands in this section were submitted to Slurm or run on a GPU by C25.

## Validation and remaining evidence

CPU checks:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_rtd_*.py tests/test_bfcl_std_campaign.py
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_checker_bridge.py -k 'not test_driver_closes_worker_when_run_fails'
.venv/bin/python -m compileall -q src/bfas/rtd tools/rtd_experiment.py
bash -n scripts/rtd_run_hpg.slurm tools/bfcl_std_campaign.sh
```

**C25c: 109 tests passed** in the combined CPU RTD/campaign run, in 115.83 seconds
(102 RTD, including 20 C25c device/memory regressions, plus seven campaign
regressions). Follow-up assertions for all 36 step-memory records, unchanged
checkpointed generation, and reduced saved tensors passed in the affected
22-test subset (17.10 seconds). The BF16 CPU Qwen test saves less than half as
many tensor bytes for backward with checkpointing and produces exactly matching
scores/LoRA gradients; this is not a production GPU peak measurement. Python
compilation and whitespace checks passed. **No C25c GPU runs were launched.**

**C25b: 89 tests passed** in the combined CPU RTD/campaign run, in 108.02 seconds
(82 RTD, including all 24 C25/T6 checks and 10 new C25b regressions, plus seven
campaign regressions). The new tests reproduce the BF16 Qwen3.5 discrepancy,
verify source-failure persistence and REINFORCE/CE gradient equality, exercise
each tolerance independently, and refuse prompt/coverage/total/nonfinite errors.
Python compilation and whitespace checks passed. Prior C25 checker regression:
**14 passed, one deselected**; prior C25 shell syntax checks passed.
The two warnings concern a toy tensor-to-float conversion and PEFT's automatic
Conv1D layout setting, not test failures. The excluded checker-driver test is the
pre-existing empty-protocol `KeyError: micro_update`
reported by C23/C24, outside the RTD runner. Correctness tests establish numerical
and transaction behavior, not benchmark performance or the production GPU runtime.

Still unmeasured: rai GPU smoke time/score parity, production BF16/LoRA runtime,
R0/R1 official budget curves, fixed-ledger/scalar results, and stronger same-pool
text-only and same-feedback RL/meta-reweighting baselines (T7). Archived stateful
teacher bridges and memory/web-search feedback adapters remain unavailable.
Existing A/B/C evaluations remain independent; C25 did not submit, cancel or
poll their jobs, and does not claim new A/B/C results.
