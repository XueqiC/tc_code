"""Real-bank CLI smoke through round checkpoint; only model/environment IO is fake.

Run with a CPU Python and the installed data/rtd/v1_1_alfworld bank. No teacher,
network, accelerator, or TextWorld worker is allowed, including during preflight.
"""
from collections import Counter
import json
from pathlib import Path
import runpy
import socket
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.rtd import cli, runtime
from bfas.rtd.benchmarks import alfworld_support, registry
from bfas.rtd.benchmarks.alfworld_state import Observation
from bfas.rtd.functional_step import lora_parameters
from bfas.rtd.persistence import ComputeJournal, digest
from bfas.rtd.unified.arms import ARMS
from bfas.rtd.unified.experiment import P1Experiment
from test_rtd_checks import TinyLM
from test_unified_generation_batch import BankTokenizer


class ContractTokenizer(BankTokenizer):
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id if token == '<turn|>' else self.unk_token_id

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        # Text-only Gemma template for this bank. Real prepare_renderer checks
        # every frozen reset byte-for-byte, and the adapter renders later turns.
        assert tokenize is False
        text = '<bos>' + ''.join(
            f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in messages)
        return text + ('<|turn>model\n<|channel>thought\n<channel|>' if add_generation_prompt else '')


class ContractLM(TinyLM):
    """Tiny differentiable policy with a fake HF generate boundary on CPU.

    Keep the production batching implementation, GenerationConfig, RNG tickets,
    ActionTrace metadata, parameter installation and teacher-forced scoring.
    """
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Identity()
        self.generation_settings = []

    def get_output_embeddings(self):
        return self.head

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output.logits = self.head(output.logits)
        return output

    def generate(self, *, input_ids, attention_mask, generation_config):
        assert input_ids.device.type == attention_mask.device.type == 'cpu'
        assert not self.training
        c = generation_config
        self.generation_settings.append(c.to_dict())
        assert (c.temperature, c.top_p, c.top_k) == (1., 1., 0)
        assert c.do_sample and c.num_beams == c.num_return_sequences == 1
        assert c.typical_p == c.repetition_penalty == 1
        assert c.use_cache and c.return_dict_in_generate and c.output_scores
        sequences, scores = input_ids, []
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        for _ in range(c.max_new_tokens):
            logits = self(sequences, attention_mask=attention_mask).logits[:, -1]
            scores.append(logits)
            token = torch.multinomial(logits.softmax(-1), 1)
            token[done] = c.pad_token_id
            sequences = torch.cat((sequences, token), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(token)), dim=1)
            done |= torch.isin(token[:, 0], torch.tensor(c.eos_token_id))
            if done.all():
                break
        return SimpleNamespace(sequences=sequences, scores=tuple(scores))


@pytest.mark.parametrize('arm', ARMS)
def test_unified_real_alfworld_complete_first_round(tmp_path, monkeypatch, arm):
    bank = ROOT/'data/rtd/v1_1_alfworld'
    if not (bank/'public/support.json').is_file():
        pytest.skip('real ALFWorld v1.1 bank is not installed')
    resets = json.loads((bank/'public/reset_states.json').read_text())
    environments, runs = [], []
    tokenizer = ContractTokenizer()

    def forbidden(*args, **kwargs):
        pytest.fail('CPU contract attempted network, accelerator, or real model IO')

    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    for name in ('is_available', 'device_count', 'set_device', 'get_device_properties',
                 'manual_seed_all', '_lazy_init'):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: forbidden(*a, **kw))
    from transformers import AutoModelForCausalLM
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)

    class CannedEnvironment:
        def __init__(self, *, environment_hash):
            self.environment_hash = environment_hash
            self.closed = False
            self.commands = []
            # Every role has one success and one failure; every episode requires
            # a continuation after the batched first action (the broken seam).
            self.won = len(environments) % 2 == 0
            environments.append(self)

        def reset(self, request):
            self.request = request
            assert request['environment_hash'] == self.environment_hash
            assert json.loads(resets[request['task_id']]['task_json']) == request
            row, = json.loads(resets[request['task_id']]['history_json'])
            return 0, Observation(0, row['content'], tuple(row['admissible']),
                                  request['world_hash'], done=row['done'], won=False)

        def step(self, cursor, command):
            assert cursor == len(self.commands)
            self.commands.append(command)
            index = len(self.commands)
            assert index <= 3
            return index, Observation(index, f'Canned observation {index}.', ('look',),
                self.request['world_hash'], done=index == 3, won=index == 3 and self.won)

        def close(self):
            self.closed = True

    monkeypatch.setattr(alfworld_support, 'RealStepper', CannedEnvironment)
    monkeypatch.setattr(runtime, 'load_tokenizer', lambda manifest: tokenizer)

    def load_backend(config, manifest, journal):
        execution, _, settings = runtime.backend_config(config)
        backend = runtime.HFGenerateBackend(ContractLM().eval(), tokenizer,
            base_checkpoint_hash='fake-cpu-policy', harness_hash='real-alfworld-contract',
            tokenizer_hash='bank-character-mod3', journal=journal, **settings)
        backend.student_config = execution
        backend.action_limit = MethodType(registry.get_benchmark(config).action_limit, backend)
        assert backend.max_action_tokens == 512 and backend.action_caps == {'agent_action': 256}
        return backend

    monkeypatch.setattr(runtime, 'load_backend', load_backend)

    def manifest(config, arm, audit, *, smoke=False, hardware=None):
        # The production manifest hashes GPU/model resources; supply only those
        # identities here. The CLI still audits the real sealed bank and caps.
        return dict(config=config, config_hash=digest(config), arm=arm, smoke=smoke,
            bank_path=audit['bank_path'], budget_ceilings=audit['budget_ceilings'],
            bank_public_cap_sum=audit['bank_public_cap_sum'], hardware_hash='fake-cpu',
            hardware={'hard': {}, 'metadata': {}}, model_path='fake-cpu-policy')

    monkeypatch.setattr(cli, 'make_manifest', manifest)
    monkeypatch.setattr(cli, 'hardware_identity', forbidden)
    monkeypatch.setattr(cli, 'instance', lambda hardware: {'uuid': 'cpu-contract'})
    monkeypatch.setattr(cli, 'device_class', lambda hardware: {'gpu': 'CPU', 'memory': 0})
    monkeypatch.setattr(cli, 'ComputeJournal', lambda path, **kw: ComputeJournal(path, cuda=False,
        deadline=kw.get('deadline'), deadline_seconds=kw.get('deadline_seconds')))
    original_run = P1Experiment.run

    def run(self, **kwargs):
        runs.append(self)
        return original_run(self, **kwargs)

    monkeypatch.setattr(P1Experiment, 'run', run)
    monkeypatch.setenv('AW_DISTILL', 'rtd_sft_kl')
    filename = 'unified_alfworld_gemma4_d0.yaml' if arm == 'D0' else 'unified_alfworld_gemma4.yaml'
    args = ['smoke', '--config', str(ROOT/'configs/rtd'/filename),
            '--run-dir', str(tmp_path/arm), '--smoke-deadline-seconds', '120']
    if arm == 'D0':
        from tools.rtd_unified_baseline import main
        assert main(args) == 0
    else:
        script = str(ROOT/'tools/rtd_experiment.py')
        monkeypatch.setattr(sys, 'argv', [script, *args, '--arm', arm])
        with pytest.raises(SystemExit) as finished:
            runpy.run_path(script, run_name='__main__')
        assert finished.value.code == 0

    e, = runs
    assert e.state['phase'] == 'complete'
    assert (e.state['round'], e.state['step']) == (1, 1)
    audit = json.loads((e.directory/'audit.json').read_text())
    assert audit['passed'] and audit['steps'] == audit['decision_windows'] == 1
    step, = e.state['steps']
    assert step['slots'] == 2 and len(step['selected']) == len(e.ledger.owned_ids) == 1
    assert e.ledger.spent > 0 and e.ledger.spent <= e.ledger.budget
    assert not e.state['posterior'].observations
    assert len(e.state['checkpoints']) == 1 and (e.directory/'round-1/checkpoint.json').is_file()
    trajectory = json.loads((e.directory/'trajectory.json').read_text())
    assert len(trajectory['steps']) == len(trajectory['checkpoints']) == 1
    assert e.backend._pending_generation is None and e.backend.max_action_tokens == 512
    assert all(p.device.type == 'cpu' for p in lora_parameters(e.backend.model).values())
    assert any(not torch.equal(e.state['initial'][n], p) for n, p in e.state['parameters'].items())
    assert all(c['max_new_tokens'] == 256 for c in e.backend.model.generation_settings)

    events = e.journal.events
    roles = {'acquisition_reference_feedback', 'same_batch_reference_feedback', 'post_commit_feedback'}
    feedback = [r for r in events if r['kind'] == 'feedback_rollout']
    assert Counter(r['role'] for r in feedback) == {role: 2 for role in roles}
    assert len(environments) == 6 and all(env.closed and len(env.commands) == 3 for env in environments)
    episodes = [r for r in events if r['kind'] == 'alfworld_episode']
    assert len(episodes) == 6 and all(r['failure'] is None and r['from_task_start'] for r in episodes)
    for role in roles:
        assert sorted(r['rollout']['reward'] for r in feedback if r['role'] == role) == [0., 1.]
    for row in feedback:
        assert row['parent_hash'] in e.state['feedback'] and row['parent_hash'] not in e.state['inner']
        actions = row['rollout']['actions']
        assert len(actions) == 3
        assert [a['generation_metadata']['batch_size'] for a in actions] == [2, 1, 1]
        for a in actions:
            m = a['generation_metadata']
            assert (m['temperature'], m['top_p'], m['top_k'], m['max_action_tokens']) == (1., 1., 0, 256)
            assert m['effective_action_limit'] == min(256, e.backend.max_context_tokens - len(a['prompt_ids']))
            assert len(a['generation_token_logprobs']) == len(a['action_ids'])
            assert a['policy_id'] == row['rollout']['policy_id']
    scores = [r for r in events if r['kind'] == 'score_consistency' and r.get('role') in roles]
    assert len(scores) == 18
    assert Counter(r['role'] for r in scores) == {role: 6 for role in roles}
    pairs = [r for r in events if r['kind'] == 'alpha_d_source_pair']
    assert Counter(r['role'] for r in pairs) == {'commit': 2, 'virtual_reference': 2}
    commits = [r for r in events if r['kind'] in {'p1_commit', 'alpha_d_student_commit'}]
    commit, = commits
    assert commit['round'] == commit['step'] == 1
    if arm == 'D1':
        assert commit['kind'] == 'alpha_d_student_commit'
        assert commit['alpha_stop_gradient'] and commit['d_stop_gradient']
        assert len([r for r in events if r['kind'] == 'alpha_d_solver']) == 1
    else:
        commit, = [r for r in events if r['kind'] == 'p1_commit']
        assert commit['backbone_updates'] == 1 and commit['rl_updates'] == 0
        if arm != 'D0':
            assert commit['solver']['status'] == 'local_convex_QP_KKT_verified'
            assert commit['solver']['stationarity'] <= e.unified.qp_tolerance
            assert commit['increment_error'] < 1e-10
