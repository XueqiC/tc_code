"""C25p malformed policy samples: official BFCL harness and CPU likelihoods."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from bfas.adapters.bfcl import BFCLAdapter
from bfas.rtd.functional_step import gradients, lora_parameters
from bfas.rtd.return_gradient import bfcl_task_rollout, reinforce_gradient
from test_rtd_return_gradient import ScriptedBackend, backend


@pytest.fixture(scope='module')
def official_checker():
    from tools.behavior_atom.checker_bridge import CheckerBridge
    with CheckerBridge() as checker:
        yield checker


def tool(body):
    return '<tool_call>\n' + body + '\n</tool_call>'


CALL = tool('{"name":"add","arguments":{"a":2,"b":3}}')
MALFORMED = [
    (tool('{"name":"add"}'), 'KeyError'),
    (tool('{"arguments":{}}'), 'KeyError'),
    (tool('{"name":null,"arguments":{}}'), 'TypeError'),
    (tool('{"name":[],"arguments":{}}'), 'TypeError'),
    *[(tool('{"name":"add","arguments":' + value + '}'), 'TypeError')
      for value in ('null', '[]', '42', '"{}"')],
    (tool('null'), 'TypeError'),
    (tool('[]'), 'TypeError'),
    (tool('not JSON'), 'JSONDecodeError'),
    (tool(''), 'JSONDecodeError'),
    ('<tool_call>\n{"name":', 'ValueError'),
    ('', 'ValueError'),
    (CALL + '\n' + tool('not JSON'), 'JSONDecodeError'),
]


def single_task():
    return dict(id='simple_python_999996',
        question=[[dict(role='user', content='Add 2 and 3.')]],
        function=[dict(name='add', description='sum', parameters=dict(type='dict',
            properties=dict(a=dict(type='integer'), b=dict(type='integer')), required=['a', 'b']))])


@pytest.mark.parametrize('text,exception', MALFORMED + [('not JSON', 'ValueError')])
def test_malformed_single_turn_fails_without_retry_or_token_changes(text, exception, official_checker):
    b = ScriptedBackend([text])
    rollout = bfcl_task_rollout(single_task(), 'simple_python', [{'add': {'a': [2], 'b': [3]}}],
                              b, {}, None, checker=official_checker)
    assert rollout.reward == 0 and rollout.malformed
    assert rollout.malformed_exception_types == (exception,)
    assert len(b.prompts) == len(rollout.actions) == 1
    action = rollout.actions[0]
    assert action.text == text and action.action_ids == (1, 3)
    assert action.generation_logprob == -.5 and action.malformed_exception_type == exception
    serialized = json.loads(json.dumps(asdict(rollout)))
    assert serialized['malformed'] is True
    assert serialized['actions'][0]['malformed_exception_type'] == exception


@pytest.mark.parametrize('text,exception', MALFORMED)
def test_malformed_multi_turn_history_survives_and_all_samples_remain(text, exception, official_checker):
    task = dict(id='multi_turn_base_999996', initial_config={}, involved_classes=['MathAPI'],
        question=[[dict(role='user', content='Add 2 and 3.')],
                  [dict(role='user', content='Now add 2 and 3 again.')]])
    b = ScriptedBackend([text, CALL, 'Done.'])
    rollout = bfcl_task_rollout(task, 'multi_turn_base', [['add(a=2, b=3)']]*2,
                              b, {}, None, checker=official_checker)
    assert rollout.reward == 0 and rollout.malformed
    assert rollout.malformed_exception_types == (exception,)
    assert len(b.prompts) == len(rollout.actions) == 3
    assert [a.text for a in rollout.actions] == [text, CALL, 'Done.']
    assert [a.malformed for a in rollout.actions] == [True, False, False]
    assert text in b.prompts[1] and 'Now add 2 and 3 again.' in b.prompts[1]
    assert '<tool_response>' not in b.prompts[1]  # no partial execution of mixed calls
    assert all(a.action_ids == (1, 3) for a in rollout.actions)
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    assert not any('_rtd_' in k and k.endswith('_instance') for k in vars(multi_turn_utils))


@pytest.mark.parametrize('category', ['live_relevance', 'live_irrelevance'])
def test_malformed_relevance_call_is_failed_and_official_handler_is_unchanged(category, official_checker):
    task = dict(single_task(), id=category + '_999996')
    text = tool('{"name":"add"}')
    rollout = bfcl_task_rollout(task, category, None, ScriptedBackend([text]), {}, None, checker=official_checker)
    assert rollout.reward == 0 and rollout.malformed_exception_types == ('KeyError',)
    # RTD patches its fresh instance only, never the campaign's class or parser.
    handler = BFCLAdapter()._handler()
    assert handler._extract_tool_calls(tool('not JSON')) == []
    assert handler._extract_tool_calls(text) == [{'name': 'add'}]
    with pytest.raises(KeyError, match='arguments'):
        handler._format_prompt([dict(role='assistant', content='', tool_calls=[{'name': 'add'}])], [])


def test_tool_delimiters_inside_valid_json_strings_are_not_malformed(official_checker):
    text = tool(json.dumps(dict(name='add', arguments=dict(a='<tool_call>', b='</tool_call>'))))
    task = dict(single_task(), id='live_relevance_999996')
    rollout = bfcl_task_rollout(task, 'live_relevance', None, ScriptedBackend([text]), {}, None,
                              checker=official_checker)
    assert rollout.reward == 1 and not rollout.malformed


@pytest.mark.parametrize('stage', ['sample', 'checker'])
@pytest.mark.parametrize('error', [KeyError('infrastructure'), TypeError('infrastructure'),
                                 json.JSONDecodeError('infrastructure', '', 0)])
def test_infrastructure_decode_shaped_errors_are_not_policy_failures(stage, error):
    class BrokenBackend(ScriptedBackend):
        def sample_action(self, *args, **kwargs):
            if stage == 'sample':
                raise error
            return super().sample_action(*args, **kwargs)

    class BrokenChecker:
        def check(self, *args):
            raise error

    with pytest.raises(type(error), match='infrastructure'):
        bfcl_task_rollout(single_task(), 'simple_python', [{'add': {'a': [2], 'b': [3]}}],
                          BrokenBackend([CALL]), {}, None, checker=BrokenChecker())


@pytest.mark.parametrize('capped', [False, True])
def test_malformed_reinforce_keeps_sampled_likelihood_and_loo_denominator(capped, official_checker):
    b = backend()
    params = lora_parameters(b.model)
    if capped:
        b.max_action_tokens = 2
        params['lora_transition'].data[:, 3] = -10000.
    rng = torch.Generator().manual_seed(10)
    original_sample = b.sample_action
    sampled = []
    texts = iter([tool('{"name":"add"}'), CALL])

    def sample(*args, **kwargs):
        action = replace(original_sample(*args, **kwargs), text=next(texts))
        sampled.append(action)
        return action

    b.sample_action = sample
    truth = [{'add': {'a': [2], 'b': [3]}}]
    rollouts = [bfcl_task_rollout(single_task(), 'simple_python', truth, b, params, rng,
                                checker=official_checker) for _ in range(2)]
    result = reinforce_gradient(rollouts, b, params)
    expected = gradients((b.score_action(sampled[1], params) - b.score_action(sampled[0], params)) / 2, params)
    torch.testing.assert_close(result.gradient['lora_transition'], expected['lora_transition'])
    assert result.metadata['rollouts'] == 2 and result.metadata['rewards'] == [0., 1.]
    assert result.metadata['baselines'] == [1., 0.]
    assert result.metadata['malformed_rollouts'] == result.metadata['malformed_actions'] == 1
    assert result.metadata['malformed_exception_types'] == {'KeyError': 1}
    assert result.metadata['action_tokens'] == sum(len(a.action_ids) for a in sampled)
    assert all(r.actions[0].action_ids == a.action_ids and r.truncated == capped for r, a in zip(rollouts, sampled))
