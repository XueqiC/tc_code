#!/usr/bin/env python3
"""Greedy served-student episodes on the frozen train support; CPU client only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from bfas.rtd.benchmarks.alfworld_diagnostics import EpisodeParserDiagnostics
from bfas.rtd.benchmarks.alfworld_evaluation import VLLMBackend, official_episode
from bfas.rtd.benchmarks.alfworld_state import _observed, canonical_hash, validate_full_state
from bfas.rtd.benchmarks.alfworld_support import RealStepper, audit_verified_bank, validate_support
from bfas.rtd.transport import FullState
from tools.alf_bank_subset import read_json
from tools.alfworld_teacher_pool import bank, write_json

# official_episode's guard names the evaluation split. Its environment is injected;
# this adapter supplies only frozen TRAIN worlds. Persisted records say train.
LOOP_CONTRACT = dict(split='valid_seen', evaluation_temperature=0.0, max_steps=40, max_action_tokens=256)


def frozen_inputs(source, *, support_size=32):
    source = Path(source).resolve()
    audit_verified_bank(source)
    support = validate_support(read_json(source / 'public/support.json'))
    tids = support['historical_task_ids']
    if (support['m'] != support_size or len(tids) != support_size or
            any(support['tasks'][t]['excluded'] for t in tids)):
        raise ValueError('requires the complete unprotected frozen support')
    by_id = {}
    for request in read_json(source / 'public/reset_requests.json').values():
        tid = request['task_id']
        if request != support['tasks'][tid]['request'] or by_id.setdefault(tid, request) != request:
            raise ValueError('reset requests differ from frozen support')
    resets = read_json(source / 'public/reset_states.json')
    if set(by_id) != set(tids) or set(resets) != set(tids):
        raise ValueError('all support tasks need frozen reset requests and states')
    for tid in tids:
        state = FullState(**resets[tid])
        validate_full_state(state, expected_world_hash=by_id[tid]['world_hash'])
        if (json.loads(state.task_json) != by_id[tid] or len(json.loads(state.history_json)) != 1 or
                by_id[tid]['max_episode_steps'] != 40):
            raise ValueError('requires frozen 40-step task resets')
    return support, by_id, resets


class StepperBridge:
    """Adapt RealStepper to the evaluation loop without another episode loop."""
    def __init__(self, request, reset, factory):
        self.request, self.reset = request, reset
        self.stepper = factory(request)
        self.cursor = None

    def state(self, observation):
        if observation.world_hash != self.request['world_hash'] or observation.index != self.cursor:
            raise ValueError('stepper returned a different world or step index')
        return dict(op='state', observation=observation.observation,
                    admissible=list(observation.admissible), done=observation.done, won=observation.won)

    def _read(self):
        self.cursor, observation = self.stepper.reset(self.request)
        if [_observed(observation)] != json.loads(self.reset['history_json']):
            raise ValueError('live reset differs from frozen reset state')
        return self.state(observation)

    def step(self, command):
        self.cursor, observation = self.stepper.step(self.cursor, str(command))
        return self.state(observation)

    def close(self):
        self.stepper.close()


def real_stepper(request):
    return RealStepper(environment_hash=request['environment_hash'])


def rollout_support(source, output, backend, *, stepper_factory=real_stepper, support_size=32,
                    server_identity=None):
    source, output = Path(source).resolve(), Path(output)
    support, requests, resets = frozen_inputs(source, support_size=support_size)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(version='alf-pi1-support-rollouts-v1', split='train',
                    support_manifest_hash=support['manifest_hash'],
                    source_manifest_sha256=bank.file_hash(source / 'sealed/manifest.json'),
                    episode_loop='alfworld_evaluation.official_episode',
                    episode_loop_contract=LOOP_CONTRACT, server=server_identity)
    entries = []
    index = dict(identity=identity, complete=False, tasks=entries)
    write_json(output / 'index.json', index)
    try:
        for tid in sorted(requests):
            def factory(task_id):
                return StepperBridge(requests[task_id], resets[task_id], stepper_factory)
            with EpisodeParserDiagnostics(factory) as (diagnostics, observed_factory):
                record = official_episode(tid, backend, env_factory=observed_factory, identity=LOOP_CONTRACT)
            diagnostics.annotate(record)
            record.update(split='train', identity_hash=canonical_hash(identity),
                          request_state_hash=canonical_hash(requests[tid]),
                          reset_state_hash=FullState(**resets[tid]).state_hash,
                          commands=[t['command'] for t in record['turns']],
                          observations=[t['observation'] for t in record['turns']])
            filename = canonical_hash(tid) + '.json'
            write_json(output / filename, record)
            entries.append(dict(task_id=tid, file=filename, sha256=bank.file_hash(output / filename)))
            write_json(output / 'index.json', index)
    finally:
        backend.close()
    audit_verified_bank(source, expected_manifest_sha256=identity['source_manifest_sha256'])
    index['complete'] = True
    write_json(output / 'index.json', index)
    return index


def served_backend(server_path, model_path, tokenizer_path):
    from bfas.rtd.benchmarks.alfworld_identity import model_identity, tokenizer_identity
    from bfas.rtd.benchmarks.alfworld_server import checked_server_identity
    server = checked_server_identity(read_json(server_path))
    model, tokenizer = Path(model_path).resolve(), Path(tokenizer_path or model_path).resolve()
    manifest = dict(hardware=server['hardware'], hardware_hash=server['hardware_hash'],
                    paths=dict(model_path=str(model), tokenizer_path=str(tokenizer), checkpoint=None),
                    checkpoint=model_identity(model), evaluation_harness=dict(tokenizer=tokenizer_identity(tokenizer)),
                    config=dict(max_context_tokens=server['max_context_tokens']))
    return VLLMBackend(manifest, server=server), server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server-json', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True, help='the merged model used by this server')
    parser.add_argument('--tokenizer-path', type=Path)
    args = parser.parse_args(argv)
    # Validate inputs and destination before any server request.
    frozen_inputs(args.source)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    backend, server = served_backend(args.server_json, args.model_path, args.tokenizer_path)
    index = rollout_support(args.source, args.output, backend, server_identity=server)
    print(json.dumps(dict(output=str(args.output), tasks=len(index['tasks']), complete=index['complete'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
