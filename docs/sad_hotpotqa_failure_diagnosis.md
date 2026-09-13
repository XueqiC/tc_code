# SAD HotpotQA seed 2: investigation, not a completed fix

Investigated on 2026-09-13. No training job was submitted, no teacher was
called, and no production files were changed on LONI. The local change only
improves the existing fatal identity error. The parameter mutation responsible
for the production failure has **not** been reproduced or fixed.

## Observed production sequence

Read both `train.log` and `compute.jsonl` under
`loni:/work/xueqic/hq/tc-hotpotqa/`:

- `results/_failed_cells/table1_hotpotqa_sad_s2_1019751/`
- `results/paper_baselines/table1_hotpotqa_sad_s2/` (attempt 1019797)

Both attempts have the same following journal sequence:

| Sequence | Event |
| --- | --- |
| 1962 | Commit student step 12, loss 3.8725088357925412. |
| 1964 | Begin step 13; refresh preconditioner over 22 prompts, 44 draws. |
| 1965–1968 | Generate a fresh 100-token capped first action. |
| 1969–1972 | Teacher-force that action; score-consistency check passes. |
| 1973–1975 | Teacher-force it again for its gradient. The progress log then records rollout 1 complete. |
| 1976–1979 | Generate a fresh 100-token capped second action for the same 1,691-token prompt. |
| 1980 | Fail step 13 in `PaperTrainer.sample` → `checked_score_action` → `score_action`, before the second action's token scoring starts. |

The first and second actions have different action hashes:

```
953cf7f5d6f01013ae195d78799b1a2b935bb3159f1fdb4b822949cac504338e
1d4a6b5a380ebca76fddf5164c4ccb00c060d7cf4c6aa37fcc96ff5f26324e48
```

Both generation records report policy
`d8638e6fa4841056319845291f4e81d22c6b842f1ebec47714c70a70ae3090e0`.
The first passing check reports backend
`c063465c88a3a21a45e16ffed477dbba0a76badb1a7cf39d4cb802567252d5c9`.
The second draw's ticket seed is `7961497690828205980` and its HF batch seed
is `107912798531799237` in both attempts. Failures occur at
11:43:04 UTC and 12:48:35 UTC respectively.

There is no student commit between these draws, no old action reused from
step 12, and no new step entered between generation and the failing check.
`rms_diagonal` consumes gradients without updating student parameters.

## What the available evidence cannot establish

The fatal guard did not record the two compared IDs. Its exception therefore
does not directly establish which clause failed or identify changed tensors.
The source-level inference is a policy mismatch: `backend_id` is assigned at
construction and copied into the action; HotpotQA's action-limit context only
changes `max_action_tokens`. No backend-ID mutation appears on this path.
This is an inference, not a measurement of the failing scorer's backend ID.

The remaining interval is the second call to `HFGenerateBackend.sample_action`
and its return to the check. That path includes temporary parameter installation
and restoration, but the logs do not establish that restoration changed any
parameter. They do not contain pre/post parameter snapshots or the scorer's
current policy hash. Assigning blame to restoration, checkpoint replay, or an
intervening update without those observations would be speculation.

The local and remote source hashes match for `paper_train.py`,
`return_gradient.py`, `functional_step.py`, `runtime.py`, `generation_batch.py`,
`source_scoring.py`, `checkpointing.py`, and `behavior/deltas.py`. The manifest
hashes agree for the files they cover. A fresh remote import also has bytecode
and constants matching the current source for the relevant sampling, identity,
and guard functions.

CPU probes with a tiny randomly initialized Gemma 4 model, LoRA, actual HF
batch generation, chunked scoring, and gradient checkpointing did not reproduce
the mismatch. This includes the actual `PaperTrainer` through step 13 with
40 slots per step, and repeated generation/scoring/backward with resident
parameter aliases and explicit garbage collection. These probes do not
reproduce the production weights, dimensions, or CUDA execution.

## Archived cells

Audited the local `results/paper_baselines/table1_*` journals corresponding to
all nine archived ALFWorld cells and three archived HotpotQA cells:

- Every journal's sequence, previous hash, and event content hash verifies.
- Each ALFWorld cell has 196 generated actions and 196 passing checks.
- Each HotpotQA cell has 88 generated actions and 88 passing checks.
- Across all 2,028 actions, generated action hashes and policy IDs match the
  paired check; each generation policy matches its preconditioner's recorded
  source snapshot ID. Every cell has two preconditioners and 24 commits.

The executed code additionally verifies the policy again before differentiable
action scoring. There is no evidence that a stale action was silently scored
under different parameters in these cells. **This does not certify that the
unidentified underlying defect could not silently change parameters elsewhere.**
In particular, these journals do not bind every committed tensor state to the
next generation. Until the actual mutation is identified, a blanket assurance
that completed cells are unaffected is not justified.

## Local diagnostic change

`score_action` retains the same backend and policy predicates, ordering,
short-circuit behavior, and fatal failure before token scoring. It now names
the failed clause and both compared IDs in the error. It neither disables
verification nor catches and continues. No regeneration was introduced.

Two tiny CPU tests deliberately violate each invariant and verify the error
details and that token scoring never starts. Both fail against the old generic
error. They are **diagnostic tests, not the requested reproduction of the
production failure**. A true regression test and corrective fix remain pending
an observed explanation of the parameter change.

Validation: `tests/test_rtd_return_gradient.py` (14 passed) and
`tests/test_baseline_run_training.py` (17 passed): **31 passed in 38.29s**.
CUDA visibility was empty, Hugging Face/Transformers offline modes were set,
and an inherited `sitecustomize.py` rejected IPv4/IPv6 socket connections in
the test runner and its Python subprocesses. Local Unix IPC remained available
for the existing BFCL checker fixture. `git diff --check` passed.

An instrumented reproduction would need the scorer's current policy ID and
parameter identities/bytes at generation entry, parameter installation,
generation exit, and restoration. No such production rerun was performed.
