"""An enumerable CPU policy on real C26-B fixtures, for the unmodified RTD math.

Only the backend/environment boundaries are substituted. Source, paid bank,
parser, complete action scoring, LOO, pilot, VJP, ledger and StateStore are real.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bfas.rtd.functional_step import lora_parameters
from bfas.rtd.persistence import atomic_json
from bfas.rtd.benchmarks.registry import ALFWorldFeedbackContext
from test_rtd_alfworld_rollout import TinyBackend, TinyTokenizer, EnumerableEnv, render
from test_rtd_alfworld_state import FakeStepper


class RunnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_theta = torch.nn.Parameter(torch.tensor([.35, -.2, .4, .1], dtype=torch.float64))
        self.eval()

    def save_pretrained(self, directory, *, safe_serialization):
        from safetensors.torch import save_file
        Path(directory).mkdir(parents=True)
        save_file({n: p.detach().contiguous() for n, p in lora_parameters(self).items()},
                  str(Path(directory) / 'adapter_model.safetensors'))
        atomic_json(Path(directory) / 'adapter_config.json', dict(peft_type='LORA', fixture=True))


class RunnerTokenizer(TinyTokenizer):
    def save_pretrained(self, directory):
        atomic_json(Path(directory) / 'tokenizer_config.json', dict(fixture=True, chat_template='fixture'))

    def encode(self, text, *, add_special_tokens=False):
        # Authored teacher targets may be tokenized; student sampled IDs are
        # always consumed directly by score_source/checked_score_action.
        if text in ('go to table 1', 'put apple 1 on table 1'):
            return [11, 21]
        return super().encode(text, add_special_tokens=add_special_tokens)


class RunnerBackend(TinyBackend):
    def __init__(self):
        super().__init__()
        self.model, self.tokenizer = RunnerModel(), RunnerTokenizer()

    def values(self, prompt_ids, action_ids, parameters):
        return super().values(prompt_ids, action_ids, {'theta': parameters['lora_theta']})

    def sample_action(self, prompt, parameters, generator, *, temperature=1., top_p=1.):
        # TinyBackend's sampler names theta, while the production runner enforces
        # LoRA-only names. Translate that single coordinate name at the boundary.
        class Coordinates(dict):
            def __getitem__(self, name):
                return super().__getitem__('lora_theta' if name == 'theta' else name)
        return super().sample_action(prompt, Coordinates(parameters), generator,
                                     temperature=temperature, top_p=top_p)

    def score_source(self, source, parameters):
        return self.values(self.tokenizer.encode(source.behavior.state.prompt), source.token_ids, parameters).sum()

    def score_behavior(self, behavior, parameters):
        return self.values(self.tokenizer.encode(behavior.state.prompt),
                           (*self.tokenizer.encode(behavior.text), 0), parameters).sum()

    def initial_hidden(self, prompt, initial_parameters, *, initial_snapshot_id):
        assert self.identity(initial_parameters) == initial_snapshot_id
        index = json.loads(prompt)['index']
        return torch.tensor([1., index, len(prompt) / 1000., .3], dtype=torch.float64)

    def source_kl(self, actions, source_parameters, updated_parameters):
        values = []
        for action in actions:
            old = torch.stack([self.values(action.prompt_ids, (10+z, 20+a, 0), source_parameters).sum()
                               for z in (0, 1) for a in (0, 1)])
            new = torch.stack([self.values(action.prompt_ids, (10+z, 20+a, 0), updated_parameters).sum()
                               for z in (0, 1) for a in (0, 1)])
            values.append((old.exp() * (old-new)).sum().clamp_min(0))
        return torch.stack(values).mean()


class RunnerEnv(EnumerableEnv):
    def observation(self):
        if not self.commands:
            return FakeStepper(self.request).observation(0)
        return super().observation()


def feedback_context(self, round_number, backend, journal):
    return ALFWorldFeedbackContext(self.protocol, round_number, render, RunnerEnv, journal)


@pytest.fixture
def multi_trial_window(tmp_path, monkeypatch):
    """Four parents, two public trials each; only the second trial has a package.

    Use the real verified-bank auditor with tiny counts instead of the production
    135-parent/data-installation audit. All runner/provider/math paths are real.
    """
    from bfas.rtd.benchmarks import alfworld_bank as bank, alfworld_config, registry
    from bfas.rtd.benchmarks.alfworld_state import canonical_hash, parent_hash
    from bfas.rtd.benchmarks.alfworld_support import (
        audit_verified_bank, freeze_support, seal_verified_bank, verify_package,
    )
    from bfas.rtd.experiment import RTDExperiment
    from bfas.rtd.persistence import digest
    from test_rtd_alfworld_state import make_payload

    groups = {0: [], 1: []}
    for i in range(32):
        tid = f'pick_and_place_simple-Apple-None-Table-{i}/trial-1'
        fold = int(parent_hash(tid), 16) % 2
        if len(groups[fold]) < 2:
            groups[fold].append(tid)
    tids = sorted(t for group in groups.values() for first in group
                  for t in (first, first.replace('trial-1', 'trial-2')))
    atomic_json(tmp_path / bank.SUPPORT, dict(demand=tids, calibration=[]))
    (tmp_path / bank.PROBE).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / bank.PROBE).write_text('')
    (tmp_path / bank.LEDGER).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / bank.LEDGER).write_text(''.join(json.dumps(dict(task_id=t)) + '\n' for t in tids))
    for tid in tids:
        world = tmp_path / 'envs/alfworld/data/json_2.1.1/train' / tid
        atomic_json(world / 'game.tw-pddl', dict(grammar={'task': [{'rhs': 'Your task is to: put apple on table.'}]}))
        atomic_json(world / 'traj_data.json', dict(task_id=tid))
        (world / 'initial_state.pddl').write_text('unique world ' + tid)
    environment = dict(fixture='C26-H CPU environment', max_episode_steps=40)
    environment['environment_hash'] = canonical_hash(environment)
    support = freeze_support(tmp_path, environment)
    records, payloads, requests, resets = [], {}, {}, {}
    for tid in tids:
        request = support['tasks'][tid]['request']
        payload = verify_package(make_payload(request), request, FakeStepper(request), render)
        assert payload['status'] == 'usable'
        resets[tid] = payload['verification']['states'][0]
        if tid.endswith('trial-2'):
            q = payload['query_id']
            records.append(bank.public_record(q, request))
            payloads[q], requests[q] = payload, request
    directory = tmp_path / 'bank'
    archive = bank.Archive(records, payloads, requests, [], dict(source_files=[]))
    seal_verified_bank(directory, archive, support, payloads, resets)
    audit = audit_verified_bank(directory)
    assert audit['usable_packages'] == 4
    monkeypatch.setattr(alfworld_config, 'bank_audit', lambda root, config: dict(audit, bank_path=str(directory)))
    monkeypatch.setattr(registry.ALFWorldExperimentSupport, 'feedback_context', feedback_context)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda *a, **k: pytest.fail('CPU window initialized CUDA'))
    config = dict(alfworld_config.default_config(), smoke_override=dict(parents_per_fold=2,
        slots=8, rollouts=2, windows=1, baseline='leave_one_out_same_task', max_seconds=900))

    def engine(name, *, resume=False, after_save=None, arm='R1', gate='linear_sigmoid', seed=1):
        cfg = dict(config, gate=gate, training_seed=seed)
        manifest = dict(bank_path=str(directory), budget_ceilings=[4 * 1310720] * 3,
                        arm=arm, config_hash=digest(cfg))
        providers = registry.get_benchmark(cfg)
        return RTDExperiment(cfg, manifest, tmp_path / 'results' / name, RunnerBackend(),
            providers.support_protocol(tmp_path, cfg), smoke=True, resume=resume, after_save=after_save)

    return SimpleNamespace(engine=engine, bank=directory, support=support, payloads=payloads)
