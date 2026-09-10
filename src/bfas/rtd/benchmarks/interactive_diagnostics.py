"""Greedy training-support diagnostics, isolated from stochastic feedback fitting."""
from copy import deepcopy
import json
from .alfworld_state import _observed
from .alfworld_rollout import parse_action
from ..return_gradient import TaskRollout


def alfworld_greedy(state, backend, parameters, generator, context):
    if not getattr(backend, 'diagnostic_only', False):
        raise PermissionError('diagnostic greedy backend required')
    request = json.loads(state.task_json)
    env = context.env_factory()
    history, actions = [], []
    try:
        cursor, observation = env.reset(request)
        history.append(_observed(observation))
        for _ in range(40):
            prompt = context.renderer(request, deepcopy(history))
            with backend.action_limit('agent_action'):
                action = backend.sample_action(prompt, parameters, generator)
            actions.append(action)
            command, _ = parse_action(action.text, observation.admissible)
            cursor, observation = env.step(cursor, command)
            history.extend([dict(role='assistant', content=command), _observed(observation)])
            if observation.done:
                break
        return TaskRollout(request['task_id'], tuple(actions), float(observation.won), backend.identity(parameters))
    finally:
        env.close()
