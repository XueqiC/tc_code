# Shared vLLM ALFWorld evaluation

The official shard driver accepts `--server-json`, written only after the
launcher sees its unique model name at `/v1/models`. The server listens on
127.0.0.1 at the explicitly required `--port`. An occupied port is rejected
before any GPU probe. Each completion names that launch, so reusing a port for
another job cannot silently change the evaluated model.

Use a **local merged model snapshot**. PEFT overlays remain supported by the
reference `HFBackend`; this server path rejects them. All clients must run on
the server's host. They need the local tokenizer and can hide CUDA completely.

Run from `/tmp` so third-party tools never use the project root as their working
directory. The commands below use a new state directory. Set `MODEL`, `GPU_ID`,
`PORT` and `N` explicitly; `PORT` has no default. `N` is the number of simultaneous
episode workers (for example, 8). The two terminals use the same variables.

```bash
export REPO=/home/xueqi/hq/projects/tc-alignment-vllm
export MODEL=/absolute/path/to/merged-snapshot
export GPU_ID=0
export PORT=19327       # choose an unused port for this job
export N=8
export RUN=/tmp/alf-vllm-campaign
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONDONTWRITEBYTECODE=1
cd /tmp
```

Launch in terminal 1 and leave the supervisor running:

```bash
"$REPO/envs/vllm-serve/.venv/bin/python" -B "$REPO/tools/alf_vllm_server.py" \
  --model "$MODEL" --gpu "$GPU_ID" --port "$PORT" --state-dir "$RUN/server"
```

Wait for `Ready: .../server.json`. The state directory contains `server.json`,
`hardware.json`, `port`, `server.log`, and private compiler caches. The launcher
uses the server interpreter to capture exactly one generating GPU, driver,
host class, and package versions. Missing PEFT is recorded as `not-installed`,
because the serving environment does not need it. It sets
`VLLM_USE_FLASHINFER_SAMPLER=0`, removes `VLLM_ATTENTION_BACKEND`, uses
`--no-enable-log-requests`, and disables model generation defaults with
`--generation-config vllm`. Ctrl-C/SIGTERM stops this launch's process group.

In terminal 2, prepare a **new binding** with the re-frozen harness, then run N
shards. Preparation verifies the launch's model/tokenizer/context against the
campaign, and consumes its hardware identity without probing client CUDA:

```bash
export CUDA_VISIBLE_DEVICES=""
"$REPO/.venv/bin/python" -B "$REPO/tools/rtd_alfworld_evaluate.py" prepare \
  --model "$MODEL" --data-root "$REPO/envs/alfworld/data/json_2.1.1" \
  --server-json "$RUN/server/server.json" --out "$RUN/binding.json"

pids=()
for ((i=0; i<N; i++)); do
  "$REPO/.venv/bin/python" -B "$REPO/tools/alf_eval_shard.py" \
    --binding "$RUN/binding.json" --server-json "$RUN/server/server.json" \
    --output-root "$RUN/campaigns" --tag vllm --shard "$i" --of "$N" \
    >"$RUN/shard-$i.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
```

After every shard succeeds, finalise using the existing coordinator. It loads
neither a model nor an environment when all 140 records are present; the server
may be stopped before this step. If tasks are missing, rerun the failed shards
first (the coordinator's generation fallback remains HF).

```bash
CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/python" -B \
  "$REPO/tools/rtd_alfworld_evaluate.py" run \
  --binding "$RUN/binding.json" --output-root "$RUN/campaigns" --tag vllm
```

The backend sends local prompt token IDs, temperature 0, exactly 256 max tokens,
neutral penalties/sampling controls and no minimum length. Only the locally
bound tokenizer's EOS stops generation (`ignore_eos=true` disables the engine's
own EOS, while `stop_token_ids` explicitly enables the bound EOS).
`skip_special_tokens=false` is explicit. Returned token IDs are decoded with
the same local tokenizer call as HF, excluding only a terminal EOS. Completion
usage includes that EOS. A `length` finish must have 256 tokens and no terminal
EOS; `stop` must end at EOS, even when it is token 256. Disagreements raise.
Prompt length plus 256 must fit the configured context before any HTTP request.

The added AST selectors are `VLLMBackend` in `alfworld_evaluation.py` and
`SERVER_VERSION`, `checked_server_identity` in `alfworld_server.py`. The frozen
scoring-files digest is
`a475f62fd4014f1ec43d1e1ea0b885283a43a2038b396ab9aec7964557f010e9`.
Existing scientific selectors, HFBackend, ReAct constants, parsing, environment
stepping, valid_seen/140/40/greedy/256, and record/envelope keys and hashing are
unchanged. Fresh bindings intentionally reject the old harness hash; no
continuity supplement is manufactured. Task partitioning, locks, hard-link
publication and journal chain are preserved. Shard journal entries identify the
server launch and measure client wall time; client GPU seconds stay zero, so
concurrent clients do not claim duplicate server GPU time.

Tests (no server, GPU job, or real HTTP/socket calls):

```bash
cd /tmp
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO/src:$REPO:$REPO/tests" "$REPO/.venv/bin/python" -B -m pytest \
  -q -p no:cacheprovider \
  "$REPO/tests/test_alf_vllm.py" "$REPO/tests/test_alf_eval_shard.py" \
  "$REPO/tests/test_rtd_alfworld_evaluation.py" \
  "$REPO/tests/test_rtd_alfworld_identity.py" \
  "$REPO/tests/test_rtd_hardware_identity.py"
```
