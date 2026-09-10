"""Complete temperature-one episodes through the deployment adapter bridge."""
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

from ...adapters.webshop import WebShopAdapter
from ..bank_build import privileged
from ..return_gradient import TaskRollout
from .webshop_caps import action_limit


@dataclass(frozen=True)
class WebShopFeedbackContext:
    support: object
    round_number: int
    journal: object
    adapter_factory: object = WebShopAdapter
    close_adapter: bool = True

    def check_syntax(self, text):
        from ...adapters.webshop import webshop_eval
        return dict(valid=webshop_eval.parse_action(text) is not None)


def webshop_task_rollout(entry, category, truth, backend, parameters, generator, *, checker=None):
    privileged()
    if category != 'agent_action' or truth != [] or not isinstance(checker, WebShopFeedbackContext):
        raise ValueError('WebShop requires agent_action, no labels, and bound feedback context')
    if checker.support is None or checker.support.states.get(entry.parent_hash) != entry:
        raise ValueError('WebShop feedback requires a frozen support reset, never a teacher prefix')
    if type(checker.round_number) is not int or checker.round_number not in (1, 2):
        raise ValueError('WebShop feedback requires a configured v1.1 round')
    if (int(entry.parent_hash, 16) % 2 == (checker.round_number-1) % 2
            and not getattr(backend, 'diagnostic_only', False)):
        raise PermissionError('feedback requires the opposite parent fold')
    task = json.loads(entry.task_json)
    adapter = checker.adapter_factory()
    adapter._tokenizer = backend.tokenizer
    traces = []
    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
        def create(self, **kwargs):
            prompt = adapter._render(kwargs['messages'])
            if not traces and prompt != entry.prompt:
                raise ValueError('live WebShop reset differs from frozen deployment prompt')
            with action_limit(backend, category):
                action = backend.sample_action(prompt, parameters, generator, temperature=1., top_p=1.)
            if not checker.check_syntax(action.text)['valid']:
                action = replace(action, malformed=True, malformed_exception_type='InvalidWebShopAction',
                                 malformed_stage='parse_action')
            traces.append(action)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=action.text))])
    try:
        rollout = adapter._episode(str(task['session']), 1., client=Client())
        if rollout.raw.get('error'):
            raise RuntimeError('incomplete WebShop feedback: ' + rollout.raw['error'])
        # Reward=1 is the official success criterion; partial reward is diagnostic.
        result = TaskRollout(rollout.task_id, tuple(traces), float(rollout.verified), backend.identity(parameters))
        if checker.journal is not None:
            checker.journal.append('webshop_episode', round=checker.round_number,
                task_id=rollout.task_id, success=result.reward, score=100*rollout.raw['reward'],
                steps=rollout.raw['steps'], from_task_start=True, temperature=1.)
        return result
    finally:
        if checker.close_adapter:
            adapter.close()
