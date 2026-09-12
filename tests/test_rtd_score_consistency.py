"""C25b/C25q: cached BF16 vs native CE and recorded outlier checks, on CPU."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.rtd.cli import load_config, resume_config
from bfas.rtd.functional_step import gradients, lora_parameters, snapshot
from bfas.rtd.persistence import ComputeJournal, digest
from bfas.rtd.return_gradient import ActionTrace, TaskRollout, reinforce_gradient
from bfas.rtd.runtime import HFGenerateBackend
from bfas.rtd.scoring import ScoreTolerance, enforce_score_diagnostic, score_diagnostic
from bfas.rtd.transport import Behavior, FullState, SourceSample


class Tokenizer:
    eos_token_id = 3
    bos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        # Match the failed smoke's 573-token prefix; traverse several delta-rule
        # chunks. The final prefix already includes the disabled-thinking text.
        assert text.endswith('<think>\n\n</think>\n\n')
        return [0, 1, 2]*191

    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(map(str, ids))


PROMPT = 'prompt<|im_start|>assistant\n<think>\n\n</think>\n\n'


def diagnostic_for_deltas(deltas, tolerance=ScoreTolerance(), **kwargs):
    rescored = torch.full((len(deltas),), -2., dtype=torch.float64)
    generated = tuple((rescored - torch.tensor(deltas, dtype=torch.float64)).tolist())
    action = ActionTrace((0, 1), (1,)*(len(deltas)-1)+(3,), 3, 'toy',
        sum(generated), 'test-backend', 'test-policy', generation_token_logprobs=generated)
    return score_diagnostic(action, rescored, rescored.sum(), {}, tolerance, **kwargs)


@pytest.mark.parametrize('deltas,passed', [
    ([.16, .48], True),  # D15: the two-token BF16 campaign failure.
    ([.16, 1.], True),
    ([.16, 1.01], False),  # No outlier allowance for short actions.
    ([.48], True),
    ([.16, .48, .16], True),
    ([.06]*7, True),
    ([.06]*8, False),  # The mean criterion starts at the exact threshold.
    ([.06]*20, False),
    ([.023]*20, True),
])
def test_short_action_score_consistency(deltas, passed, tmp_path):
    diagnostic = diagnostic_for_deltas(deltas)
    journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=False)
    journal.append('score_consistency', **diagnostic)
    saved = json.loads(journal.path.read_text())
    assert saved['passed'] is passed
    assert saved['mean_criterion'] == ('short_action_max_abs' if len(deltas) < 8 else 'mean_abs')
    assert saved['tolerance'] == dict(mean_abs=.05, max_abs=1., max_abs_outlier_tokens=2,
                                    max_abs_hard=8., min_tokens_for_mean=8)
    assert saved['version'] == 'rtd-score-consistency-v1.1'
    assert saved['mean_abs_difference'] == pytest.approx(sum(deltas)/len(deltas))
    assert saved['max_abs_difference'] == pytest.approx(max(deltas))
    if passed:
        enforce_score_diagnostic(saved)
    else:
        with pytest.raises(ValueError, match='likelihood differs'):
            enforce_score_diagnostic(saved)


@pytest.mark.parametrize('minimum,passed', [(0, False), (2, False), (3, True)])
def test_short_action_mean_threshold_override(minimum, passed):
    tolerance = ScoreTolerance(min_tokens_for_mean=minimum)
    diagnostic = diagnostic_for_deltas([.16, .48], tolerance)
    assert diagnostic['passed'] is passed
    assert diagnostic['tolerance']['min_tokens_for_mean'] == minimum
    assert diagnostic['mean_criterion'] == ('short_action_max_abs' if passed else 'mean_abs')


@pytest.mark.parametrize('deltas,passed', [
    ([.06]*43, True),
    ([.081]*43, False),
    ([1.1]*2+[0.]*254, True),
    ([1.1]*3+[0.]*253, False),
    ([8.01]+[0.]*255, False),
    ([1.01, 0.], False),
])
def test_d15_luna_mean_override_keeps_max_outlier_and_short_action_rules(deltas, passed):
    diagnostic = diagnostic_for_deltas(deltas, ScoreTolerance(mean_abs=.08))
    assert diagnostic['passed'] is passed
    if deltas == [.06]*43:
        assert not diagnostic_for_deltas(deltas)['passed']  # Default remains .05.


def test_short_action_preserves_hard_max_sequence_and_structural_checks():
    assert not diagnostic_for_deltas([.16, .48], ScoreTolerance(max_abs_hard=.4))['passed']
    assert not diagnostic_for_deltas([.16, .48], score_atol=.1)['passed']
    diagnostic = diagnostic_for_deltas([.16, .48], expected_prompt_ids=(0,))
    assert not diagnostic['passed']
    assert diagnostic['structural_errors'] == ['prompt token boundary/template mismatch']


def qwen_backend(journal=None):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    config = load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    torch.manual_seed(8)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(vocab_size=16, hidden_size=32,
        intermediate_size=64, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=2, max_position_embeddings=1024, eos_token_id=3, bos_token_id=0,
        attn_implementation='eager')).to(device='cpu', dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'],
        lora_dropout=0., task_type='CAUSAL_LM', target_modules=config['lora_target_modules'])).eval()
    for p in lora_parameters(model).values():
        p.data = p.data.float()
    return HFGenerateBackend(model, Tokenizer(), base_checkpoint_hash='random-tiny-qwen35',
        harness_hash='c25b-cpu', tokenizer_hash='c25b-tiny', max_action_tokens=128,
        max_context_tokens=1024, score_tolerance=ScoreTolerance.from_config(config), journal=journal)


@pytest.fixture(scope='module')
def sampled():
    b = qwen_backend()
    params = snapshot(lora_parameters(b.model))
    rng = torch.Generator(device='cpu').manual_seed(5)
    return b, params, b.sample_action(PROMPT, params, rng)


def test_c25_bf16_qwen_cache_discrepancy_is_recorded_and_reinforce_uses_same_ce(sampled, tmp_path):
    b, params, action = sampled
    journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=False)
    record = lambda d: journal.append('score_consistency', **d)
    score, diagnostic = b.checked_score_action(action, params, record=record,
        expected_prompt_ids=b.tokenizer.encode(PROMPT))
    assert diagnostic['passed']
    # This actual cached BF16 execution failed the old sequence equality check.
    assert diagnostic['sequence_abs_difference'] > 2e-4 + 2e-5*abs(action.generation_logprob)
    assert 0 < diagnostic['mean_abs_difference'] <= .05
    assert 0 < diagnostic['max_abs_difference'] <= 1.
    assert diagnostic['outlier_token_count'] == 0
    assert diagnostic['outlier_positions'] == [] and diagnostic['outlier_fraction'] == 0.
    assert action.generation_metadata['logits_dtypes'] == ['torch.bfloat16']
    assert action.generation_metadata['scores_dtypes'] == ['torch.float32']
    assert diagnostic['scoring_backend']['logits_dtype'] == 'torch.bfloat16'
    assert diagnostic['scoring_backend']['reduction_dtype'] == 'torch.float32'
    saved = json.loads(journal.path.read_text().splitlines()[-1])
    assert saved['action_ids'] == list(action.action_ids)
    assert len(saved['generation_token_logprobs']) == len(saved['teacher_forced_token_logprobs']) == len(action.action_ids)
    assert saved['masks']['teacher_forced_action'] == [False]*573 + [True]*len(action.action_ids)
    assert saved['eos']['included_in_both_scores'] and saved['action_ids'][-1] == 3
    state = FullState.create({'question': 'toy'}, [{'role': 'user', 'content': 'toy'}], PROMPT, 'inner')
    source = SourceSample(Behavior(state, action.text), action.policy_id, action.action_ids, 3, action.generation_logprob)
    torch.testing.assert_close(score, b.score_source(source, params), rtol=0, atol=0)
    expected = gradients(score, params)
    result = reinforce_gradient([TaskRollout('toy', (action,), 1., action.policy_id)], b, params,
        baseline='smoke_zero', diagnostic_record=lambda r, i, d: record(d))
    for name in params:
        torch.testing.assert_close(result.gradient[name], expected[name], rtol=0, atol=0)
    assert result.metadata['score_consistency'][0]['mean_abs_difference'] == diagnostic['mean_abs_difference']


@pytest.mark.parametrize('kind', ['strict', 'mean', 'max', 'prompt', 'nonfinite', 'coverage', 'total'])
def test_failed_checks_are_durable_and_tolerance_cannot_hide_structural_errors(sampled, tmp_path, monkeypatch, kind):
    b, params, action = sampled
    expected_prompt = action.prompt_ids
    if kind == 'strict':
        monkeypatch.setattr(b, 'score_tolerance', ScoreTolerance(1e-9, 1e-9))
    elif kind in {'mean', 'max'}:
        # Offset the recorded sampling path to exercise each bound separately.
        values = list(action.generation_token_logprobs)
        if kind == 'mean':
            values = [v-.06 for v in values]
        else:
            values[-1] -= 1.1
            monkeypatch.setattr(b, 'score_tolerance', ScoreTolerance(
                mean_abs=2., max_abs=1., max_abs_outlier_tokens=0))
        action = replace(action, generation_token_logprobs=tuple(values), generation_logprob=sum(values))
    elif kind == 'prompt':
        expected_prompt = action.prompt_ids[:-1]
    elif kind == 'nonfinite':
        action = replace(action, generation_token_logprobs=(float('nan'),)+action.generation_token_logprobs[1:])
    elif kind == 'coverage':
        action = replace(action, generation_token_logprobs=action.generation_token_logprobs[:-1])
    else:
        action = replace(action, generation_logprob=action.generation_logprob-1.)
    journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=False)
    with pytest.raises(ValueError, match='likelihood differs'):
        b.checked_score_action(action, params, expected_prompt_ids=expected_prompt,
            record=lambda d: journal.append('score_consistency', **d))
    saved = json.loads(journal.path.read_text().splitlines()[-1])
    assert saved['passed'] is False and saved['action_ids'] == list(action.action_ids)
    if kind in {'prompt', 'nonfinite', 'coverage', 'total'}:
        assert saved['structural_errors']


@pytest.mark.parametrize('value', [None, {'mean_abs': .001, 'max_abs': .02},
    {'max_abs_outlier_tokens': 0, 'max_abs_hard': 3.},
    {'min_tokens_for_mean': 0}, {'min_tokens_for_mean': 16},
    {'mean_abs': .1, 'max_abs': .5, 'max_abs_outlier_tokens': 4, 'max_abs_hard': 6.}])
def test_tolerance_config_defaults_and_overrides(tmp_path, value):
    import yaml
    config = load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    path = tmp_path/'config.yaml'
    config.pop('score_consistency_tolerance')
    if value is not None:
        config['score_consistency_tolerance'] = value
    path.write_text(yaml.safe_dump(config))
    expected = dict(mean_abs=.05, max_abs=1., max_abs_outlier_tokens=2,
                    max_abs_hard=8., min_tokens_for_mean=8) | (value or {})
    assert load_config(path)['score_consistency_tolerance'] == expected
    assert vars(ScoreTolerance.from_config(config)) == expected


@pytest.mark.parametrize('key', ['mean_abs', 'max_abs', 'max_abs_hard'])
@pytest.mark.parametrize('value', [-1., float('inf'), float('nan')])
def test_tolerance_config_rejects_invalid_numeric_bounds(tmp_path, key, value):
    import yaml
    config = load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    config['score_consistency_tolerance'][key] = value
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='finite and nonnegative'):
        load_config(path)


@pytest.mark.parametrize('key', ['max_abs_outlier_tokens', 'min_tokens_for_mean'])
@pytest.mark.parametrize('value', [-1, 1.5, 2., True, False, '8', None, float('inf'), float('nan')])
def test_tolerance_config_requires_nonnegative_integer_token_counts(tmp_path, key, value):
    import yaml
    config = load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    config['score_consistency_tolerance'][key] = value
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='nonnegative integer'):
        load_config(path)


@pytest.mark.parametrize('config_source', ['manifest', 'saved_yaml'])
@pytest.mark.parametrize('legacy_tolerance', [None, {'mean_abs': .05, 'max_abs': 1.}])
def test_resume_uses_new_tolerance_defaults_without_mutating_saved_config(tmp_path, config_source, legacy_tolerance):
    import yaml
    config = load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    config.pop('score_consistency_tolerance')
    if legacy_tolerance is not None:
        config['score_consistency_tolerance'] = legacy_tolerance
    saved = dict(config=config, config_hash=digest(config))
    original = json.dumps(saved, sort_keys=True)
    path = tmp_path/'saved.yaml'
    path.write_text(yaml.safe_dump(config))
    saved_bytes = path.read_bytes()
    resumed = resume_config(path if config_source == 'saved_yaml' else None, saved)
    assert vars(ScoreTolerance.from_config(resumed)) == dict(
        mean_abs=.05, max_abs=1., max_abs_outlier_tokens=2, max_abs_hard=8., min_tokens_for_mean=8)
    assert resumed is config and digest(resumed) == saved['config_hash']
    assert json.dumps(saved, sort_keys=True) == original and path.read_bytes() == saved_bytes


@pytest.mark.parametrize('kind,positions,magnitude,passed', [
    ('one', [44], 1.43, True),
    ('two_including_eos', [44, 255], 1.43, True),
    ('three', [0, 44, 255], 1.43, False),
    ('hard_max', [44], 9., False),
    ('hard_max_boundary', [44], 8., True),
    ('outlier_boundary', [0, 44, 255], 1., True),
    ('mean', list(range(256)), .06, False),
    ('structural', [44], 1.43, False),
])
def test_outlier_decision_records_all_tokens_and_preserves_native_score_and_gradient(
        tmp_path, capsys, kind, positions, magnitude, passed):
    from test_rtd_return_gradient import backend
    b = backend()
    b.max_context_tokens = 512
    # Match an old saved config: new limits must come from code defaults.
    b.score_tolerance = ScoreTolerance.from_config({'score_consistency_tolerance': {'mean_abs': .05, 'max_abs': 1.}})
    params = lora_parameters(b.model)
    action = ActionTrace((0, 1), (1,)*255+(3,), 3, 'toy', -256., b.backend_id, b.identity(params))
    native, values, _ = b.score_action(action, params, return_details=True)
    generated = values.detach().clone()
    generated[positions] -= magnitude
    action = replace(action, generation_token_logprobs=tuple(generated.tolist()),
                     generation_logprob=float(generated.sum()))
    expected_prompt = action.prompt_ids[:-1] if kind == 'structural' else action.prompt_ids
    journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=False)
    kwargs = dict(expected_prompt_ids=expected_prompt, record=lambda d: journal.append('score_consistency', **d))
    if passed:
        score, diagnostic = b.checked_score_action(action, params, **kwargs)
        torch.testing.assert_close(score, native, rtol=0, atol=0)
        actual_gradient, native_gradient = gradients(score, params), gradients(native, params)
        for name in params:
            torch.testing.assert_close(actual_gradient[name], native_gradient[name], rtol=0, atol=0)
    else:
        with pytest.raises(ValueError, match='likelihood differs'):
            b.checked_score_action(action, params, **kwargs)
    records = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert len(records) == 1
    saved = records[0]
    assert saved['passed'] is passed and saved['protocol_version'] == '1.0.6'
    assert saved['n_tokens'] == 256
    assert saved['mean_criterion'] == 'mean_abs'
    # Disabling D15 must leave every long-action diagnostic byte-identical,
    # apart from the explicitly recorded threshold itself.
    legacy = score_diagnostic(action, values, native, saved['scoring_backend'],
        replace(b.score_tolerance, min_tokens_for_mean=0), expected_prompt_ids=expected_prompt)
    current = score_diagnostic(action, values, native, saved['scoring_backend'],
        b.score_tolerance, expected_prompt_ids=expected_prompt)
    legacy['tolerance'] = asdict(b.score_tolerance)
    assert json.dumps(current) == json.dumps(legacy)
    outliers = positions if magnitude > 1. else []
    assert saved['outlier_positions'] == outliers
    assert saved['outlier_token_count'] == len(outliers)
    assert saved['outlier_fraction'] == pytest.approx(len(outliers)/256)
    assert saved['max_abs_difference'] == pytest.approx(magnitude)
    assert saved['mean_abs_difference'] == pytest.approx(len(positions)*magnitude/256)
    assert (saved['mean_abs_difference'] > .05) is (kind == 'mean')
    assert bool(saved['structural_errors']) is (kind == 'structural')
    assert saved['generation_token_logprobs'] == list(action.generation_token_logprobs)
    assert saved['teacher_forced_token_logprobs'] == values.detach().tolist()
    assert saved['eos']['included_in_both_scores'] and saved['eos']['action_positions'] == [255]
    assert saved['masks']['teacher_forced_action'] == [False]*2+[True]*256
    lines = capsys.readouterr().out.splitlines()
    if passed and outliers:
        assert len(lines) == 1 and lines[0].startswith('[rtd] score-consistency outlier ')
        for field in (f"state_hash={saved['state_hash']}", 'n_tokens=256',
                      f'outlier_token_count={len(outliers)}', f'max_abs_difference={magnitude:g}'):
            assert field in lines[0]
    else:
        assert lines == []


def test_source_sample_records_failing_state_before_raising(sampled, tmp_path, monkeypatch):
    from bfas.rtd.experiment import RTDExperiment
    b, params, action = sampled
    state = FullState.create({'question': 'toy'}, [{'role': 'user', 'content': 'toy'}], PROMPT, 'inner')
    e = RTDExperiment.__new__(RTDExperiment)
    e.backend, e.sampling_rng = b, torch.Generator().manual_seed(5)
    from types import SimpleNamespace
    e.support = SimpleNamespace(parents={'inner': 'toy'}, categories={'toy': 'simple_python'})
    e.journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=False)
    e.state = dict(inner={'inner'}, source=params, source_id=b.identity(params), source_cache={},
        projection_cache={state.state_hash: torch.zeros(4)}, round=1, step=1)
    monkeypatch.setattr(b, 'sample_action', lambda *a: action)
    with monkeypatch.context() as strict:
        strict.setattr(b, 'score_tolerance', ScoreTolerance(1e-9, 1e-9))
        with pytest.raises(ValueError, match='likelihood differs'):
            e.sample_state(state)
    saved = json.loads(e.journal.path.read_text().splitlines()[-1])
    assert saved['state_hash'] == state.state_hash and saved['parent_hash'] == 'inner'
    assert saved['round'] == saved['step'] == 1 and saved['sample_index'] == 0
    assert not saved['passed'] and not e.state['source_cache']
    assert len(e.sample_state(state)) == 2
    checks = [json.loads(line) for line in e.journal.path.read_text().splitlines()
              if json.loads(line)['kind'] == 'score_consistency']
    assert [d['passed'] for d in checks] == [False, True, True]
