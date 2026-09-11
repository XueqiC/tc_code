"""D16 live bridge, ledger, isolation, and official 500-session contract on CPU."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from test_webshop_adapter import StubBridge, StubClient, StubTokenizer
from bfas.adapters.webshop import WebShopAdapter
from bfas.rtd.benchmarks.webshop_support import freeze_support, reset_state, WebShopSupport
from bfas.rtd.benchmarks.webshop_rollout import WebShopFeedbackContext, webshop_task_rollout
from bfas.rtd.benchmarks.webshop_bank import build_webshop_bank
from bfas.rtd.benchmarks.webshop_evaluation import validate_records
from bfas.rtd.bank_build import validate_state_certificate
from bfas.rtd.return_gradient import ActionTrace


class Tokenizer(StubTokenizer):
    eos_token_id = 0
    def encode(self, text, **kwargs):
        return [b+1 for b in text.encode()]


def adapter_with_bridge(reward=1., done_after=2):
    adapter = WebShopAdapter()
    adapter._tokenizer = Tokenizer()
    adapter._bridge = StubBridge(rewards=(reward,), done_after=done_after)
    return adapter


class Backend:
    tokenizer = Tokenizer()
    backend_id = 'stub'
    action_caps = {'agent_action': 128}
    max_action_tokens = 128
    max_context_tokens = 32768
    def __init__(self, text='click[Buy Now]'):
        self.text = text
        self.calls = []
    def identity(self, parameters):
        return 'policy'
    def sample_action(self, prompt, parameters, generator, **kwargs):
        self.calls.append((prompt, kwargs, self.max_action_tokens))
        ids = tuple(self.tokenizer.encode(self.text))+(0,)
        return ActionTrace(tuple(self.tokenizer.encode(prompt)), ids, 0, self.text, 0., self.backend_id, 'policy', (0.,)*len(ids))


def feedback_context(state, adapter):
    return WebShopFeedbackContext(SimpleNamespace(states={state.parent_hash: state}),
        2 - int(state.parent_hash, 16) % 2, None, lambda: adapter)


def test_adapter_split_excludes_eval_and_calibration():
    a = adapter_with_bridge()
    try:
        support = freeze_support(a)
        assert len(support['parents']) == 200
        assert set(support['split']['demand']).isdisjoint(support['split']['calibration'])
        assert min(map(int, support['split']['support'])) >= 500
        assert max(map(int, support['split']['support'])) <= 6909
    finally:
        a.close()


@pytest.mark.parametrize('reward, text, success, count', [(1., 'click[Buy Now]', 1., 2),
    (.7, 'click[Buy Now]', 0., 2), (1., 'invalid', 0., 3)])
def test_feedback_uses_live_task_start_and_unwarped_policy(reward, text, success, count):
    a = adapter_with_bridge(reward)
    state = reset_state(a, '500')
    backend = Backend(text)
    context = feedback_context(state, a)
    rollout = webshop_task_rollout(state, 'agent_action', [], backend, {}, torch.Generator(), checker=context)
    assert rollout.reward == success and rollout.from_task_start and len(rollout.actions) == count
    assert all(k == {'temperature': 1., 'top_p': 1.} and cap == 128 for _, k, cap in backend.calls)
    assert backend.calls[0][0] == state.prompt
    assert rollout.malformed is (text == 'invalid')
    assert a._bridge is None


def test_bridge_errors_propagate():
    a = adapter_with_bridge()
    state = reset_state(a, '500')
    def broken(*args, **kw):
        raise RuntimeError('bridge down')
    a._bridge.step = broken
    with pytest.raises(RuntimeError, match='bridge down'):
        webshop_task_rollout(state, 'agent_action', [], Backend(), {}, torch.Generator(),
                            checker=feedback_context(state, a))


def test_feedback_rejects_inner_fold_and_teacher_prefix_before_bridge_access():
    from dataclasses import replace
    a = adapter_with_bridge()
    state = reset_state(a, '500')
    context = feedback_context(state, a)
    try:
        with pytest.raises(PermissionError, match='opposite parent fold'):
            webshop_task_rollout(state, 'agent_action', [], Backend(), {}, torch.Generator(),
                checker=replace(context, round_number=3-context.round_number))
        with pytest.raises(ValueError, match='frozen support reset'):
            webshop_task_rollout(replace(state, prompt=state.prompt+'prefix'), 'agent_action', [],
                Backend(), {}, torch.Generator(), checker=context)
    finally:
        a.close()


def test_teacher_demo_bank_cost_prefixes_and_public_certificate(tmp_path, monkeypatch):
    from bfas import ledger as gateway
    from bfas.adapters import webshop
    from bfas.rtd.cli import load_config
    from bfas.rtd.broker import SealedReplayBroker
    from bfas.rtd.ledger import Ledger
    from bfas.rtd.selector import StudentSnapshot
    monkeypatch.setattr(gateway, 'LEDGER_ROOT', tmp_path/'ledger')
    monkeypatch.setenv('BFAS_TEACHER_MIN_INTERVAL_S', '0')
    class Teacher:
        def __init__(self):
            self.response_texts, self.tokens_spent = [], 0
            self.config = SimpleNamespace(name='gpt-5.4')
            self.usage = {}
        def generate_reply(self, messages, temperature):
            self.tokens_spent += 17
            self.response_texts.append('I found the requested bottle.\nclick[Buy Now]')
            return self.response_texts[-1]
    monkeypatch.setattr(webshop, '_TeacherSession', Teacher)
    a = adapter_with_bridge()
    tid = freeze_support(a)['split']['demand'][0]
    demos = a.teacher_demo([tid], attempts=3)
    pool = tmp_path/'pool.json'
    pool.write_text(json.dumps({t: asdict(d) for t,d in demos.items()}))
    ledger = tmp_path/'ledger/webshop.jsonl'
    config = load_config(Path(__file__).resolve().parents[1]/'configs/rtd/v1_1_webshop.yaml')
    out = tmp_path/'bank'
    summary = build_webshop_bank(tmp_path, out, pool=pool, ledger=ledger, config=config, adapter=a, tokenizer=Tokenizer())
    assert summary['recorded_bank_usage'] == 34 and summary['available_packages'] == 2
    cert = validate_state_certificate(out, benchmark='webshop', student=config['student'])
    assert cert['core']['class_caps'] == {'webshop_demo_state': 32}
    config.update(replay_bank_path=str(out), support_manifest=str(out/'public/support.json'))
    support = WebShopSupport(tmp_path, config)
    assert len(support.states) == 200  # includes parents with no teacher success
    h = next(h for h,t in support.parents.items() if t == tid)
    account = Ledger(64)
    broker = SealedReplayBroker(out, account, inner_parent_hashes={h})
    from bfas.rtd.benchmarks.webshop_caps import affordability
    assert affordability(broker._records.values(), certificate=cert) == [3, 9]
    candidates = broker.list_candidates(StudentSnapshot('s', frozenset({h})), (), 64)
    assert len(candidates) == 1
    first = broker.acquire(candidates[0].query_id)
    assert len(first.behaviors) == 1 and account.spent == 17
    candidates = broker.list_candidates(StudentSnapshot('s', frozenset({h})), account.owned_ids, account.remaining)
    assert len(candidates) == 1
    assert broker.acquire(candidates[0].query_id).dependencies == (first.query_id,)
    assert account.spent == 34
    original = Path.read_text
    def no_payload(path, *args, **kwargs):
        if path.parent.name == 'sealed' and len(path.stem) == 64:
            pytest.fail('certificate must not open an unpurchased payload')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', no_payload)
    validate_state_certificate(out)
    with pytest.raises(ValueError, match='binding'):
        validate_state_certificate(out, student='different')


def test_official_adapter_evaluation_is_500_greedy_sessions(tmp_path, monkeypatch):
    a = adapter_with_bridge(done_after=1)
    bridge = a._bridge
    client = StubClient()
    monkeypatch.setattr(a, 'prepare_renderer', lambda _: None)
    monkeypatch.setattr(a, '_client', lambda: client)
    metrics = a.evaluate('configured-student', tmp_path)
    assert bridge.sessions == list(range(500)) and metrics['success_rate'] == 1.
    assert all(c['temperature'] == 0 and c['max_tokens'] == 128 for c in client.calls)
    assert validate_records('webshop', tmp_path, metrics, {})['n'] == 500
    rows = (tmp_path/'records.jsonl').read_text().splitlines()
    (tmp_path/'records.jsonl').write_text('\n'.join(rows[:-1])+'\n')
    with pytest.raises(ValueError, match='every session'):
        validate_records('webshop', tmp_path, metrics, {})


def test_evaluation_result_binds_checkpoint_and_spend(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from bfas.rtd.benchmarks import webshop_evaluation as campaign
    from bfas.rtd import cli, identity, hardware, evaluation
    from bfas import run
    from bfas.rtd.persistence import digest, tree_hash
    root, directory, model = tmp_path, tmp_path/'run', tmp_path/'base'
    directory.mkdir(); model.mkdir(); (model/'config.json').write_text('{}')
    config = dict(benchmark='webshop', student='configured-student')
    hard = dict(gpu='stub')
    meta = dict(round=1, actual_spend=34, authorized_budget=100)
    manifest = dict(config=config, config_hash=digest(config), bank_path='bank', data_hash='data',
        model_path=str(model), base_checkpoint_hash=tree_hash(model), tokenizer_hash='tok', arm='V2')
    (directory/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(identity, 'verified_checkpoint', lambda *args: meta)
    monkeypatch.setattr(identity, 'guard_harness', lambda *args: dict(evaluation_harness={'expected': {}}, harness_hash='h'))
    monkeypatch.setattr(identity, 'record_code_drift', lambda *args, **kw: {})
    monkeypatch.setattr(cli, 'data_identity', lambda *args: 'data')
    monkeypatch.setattr(cli, 'hardware_identity', lambda: {'hard':hard})
    monkeypatch.setattr(hardware, 'guard_hardware', lambda *args, **kw: {'hard':hard})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    def export(manifest, checkpoint, out):
        out.mkdir(); (out/'model.safetensors').write_text('stub export')
    monkeypatch.setattr(evaluation, '_flatten_adapter', export)
    a = adapter_with_bridge(done_after=1)
    monkeypatch.setattr(a, 'prepare_renderer', lambda _: None)
    monkeypatch.setattr(a, '_client', lambda: StubClient())
    import bfas.adapters.webshop as adapter_module
    monkeypatch.setattr(adapter_module, 'WebShopAdapter', lambda **kw: a)
    @contextmanager
    def serve(*args):
        yield
    monkeypatch.setattr(run, 'serving_lane', serve)
    result = campaign.evaluate_adapter(root, directory, 1)
    assert result['overall_accuracy_percent'] == 100. and result['checkpoint_spend'] == 34
    assert result['identity'] == result['campaign_identity']
    assert result['identity']['checkpoint'] == meta
    assert json.loads((directory/'evaluation-1.json').read_text()) == result
    assert campaign.evaluate_adapter(root, directory, 1) == result
