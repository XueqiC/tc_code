# CPU startup preflight

Run the smoke startup checks without loading a model or using a GPU/API:

```bash
.venv/bin/python tools/rtd_preflight.py \
  --config configs/rtd/unified_alfworld_gemma4.yaml --arm D3
```

The default is the same smoke configuration, including the arm preset and smoke
overrides, used by `tools/rtd_experiment.py smoke`. For a full run, add
`--mode run --replay-schedule <V0-run>`. The default replay mode requires a
completed schedule. V1 and all unified arms (D0/D1/D2/D3 and D3 variants) also
accept `--replay-mode streaming --replay-poll-seconds 60 --replay-timeout-seconds 129600`
to follow a live V0. P1 still requires an explicit schedule for a full run. A streaming
source that has not published a schedule is reported as deferred; preflight
does not wait. Any available schedule is checked for identity and content.

Training consumes committed V0 steps in order, with the same purchases, source
states, weights, and repetitions; each student samples its own fresh actions.
The last checkpoint waits for V0's `complete=true` marker and final schedule hash.
Unified runs publish identical completed `replay_*` manifest fields in either
mode, including `replay_consumed_steps` and `replay_schedule_hash`. The frozen
`config.replay_mode` retains the launch mode and its polling/timeout settings for
resume; config hashes and campaign identities retain that provenance. Earlier
checkpoints remain valid as replay progress advances.

`smoke` and `run` execute this preflight before campaign dispatch, hardware
probing, or model loading. The check uses the production config/arm adapters,
bank audit and manifest builder, backend settings (generation batch, score
tolerance, memory policy and LoRA configuration), executor settings, support
and ledger constructors, replay readers, and feedback renderer. It uses the
real cached tokenizer with `local_files_only=True`, checks native termination
tokens and action caps, and renders/tokenizes every frozen support reset.
BFCL checker construction is checked too. Missing local model/tokenizer files
are CPU prerequisite failures; preflight never downloads them.

Output ends in `OK` (exit 0) or the first exception type/message (exit 1).
The bank, training ledger and run directory are untouched. A provisional
manifest substitutes an explicit CPU marker for the GPU hardware identity;
temporary replay bookkeeping is removed on exit. Production still creates
and verifies its actual hardware-bound manifest. Initial LoRA parameter hashes
are checked by the production replay reader after initialization; all other
available replay identity fields are checked on CPU.

An `OK` is a startup check. Actual model/LoRA initialization, generation versus
teacher-forced score agreement, gradients and GPU peak memory remain untested.

The four unified Gemma configs retain their frozen `unified-p0-1` identity and
P1 values. `runtime_config` supplies `method=rtd_v1_1` and
`protocol_version=1.1.0` to the backend and executor. In particular, D12 batch
validation must receive this adapted config, rather than the manifest config.
