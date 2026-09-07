"""C25b: real HF cached BF16 Qwen3.5 vs full native CE, entirely on CPU."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.rtd.cli import load_config
from bfas.rtd.functional_step import gradients, lora_parameters, snapshot
from bfas.rtd.persistence import ComputeJournal
from bfas.rtd.return_gradient import TaskRollout, reinforce_gradient
from bfas.rtd.runtime import HFGenerateBackend
from bfas.rtd.scoring import ScoreTolerance
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
            monkeypatch.setattr(b, 'score_tolerance', ScoreTolerance(mean_abs=2., max_abs=1.))
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


def test_tolerance_config_defaults_overrides_and_invalid_values(tmp_path):
    import yaml
    config = load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    path = tmp_path/'config.yaml'
    for value in (None, {'mean_abs': .001, 'max_abs': .02}, {'mean_abs': -1}, {'max_abs': float('inf')}):
        config.pop('score_consistency_tolerance', None)
        if value is not None:
            config['score_consistency_tolerance'] = value
        path.write_text(yaml.safe_dump(config))
        if value and any(v < 0 or not torch.isfinite(torch.tensor(v)) for v in value.values()):
            with pytest.raises(ValueError, match='finite and nonnegative'):
                load_config(path)
        else:
            expected = value or {'mean_abs': .05, 'max_abs': 1.}
            assert load_config(path)['score_consistency_tolerance'] == expected


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
