"""Greedy training-support diagnostics, isolated from stochastic feedback fitting."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from .alfworld_state import _observed
from .alfworld_rollout import _run_serial, parse_action
from ..generation_batch import run_episode_streams_lockstep
from ..return_gradient import TaskRollout


def alfworld_greedy(state, backend, parameters, generator, context):
    with backend.action_limit('agent_action'):
        return _run_serial(_alfworld_greedy_stream(state, backend, parameters, generator, context),
                           backend, parameters)


def diagnostic_episode_limit():
    name = ('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES' if 'BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES' in os.environ
            else 'BFAS_FEEDBACK_LOCKSTEP_EPISODES')
    try:
        limit = int(os.environ.get(name, '32'))
    except ValueError:
        raise ValueError(f'{name} must be a positive integer') from None
    if limit < 1:
        raise ValueError(f'{name} must be a positive integer')
    return limit


def alfworld_greedy_batch(states, backend, parameters, generator, context):
    """Same diagnostic episodes, bounded cohorts and parallel environment RPCs.

    Diagnostics keep their original exception behavior (no feedback retries or
    fold guards) and do not consume RNG tickets or registry episode seeds.
    """
    limit = diagnostic_episode_limit()
    states = tuple(states)
    for offset in range(0, len(states), limit):
        cohort = states[offset:offset+limit]
        executor = ThreadPoolExecutor(max_workers=len(cohort), thread_name_prefix='alfworld-diagnostic')
        streams = [_alfworld_greedy_stream(state, backend, parameters, generator, context, executor)
                   for state in cohort]
        with backend.action_limit('agent_action'):
            results = run_episode_streams_lockstep(streams, backend, parameters, executor)
        yield from results


def _alfworld_greedy_stream(state, backend, parameters, generator, context, executor=None):
    if not getattr(backend, 'diagnostic_only', False):
        raise PermissionError('diagnostic greedy backend required')
    request = json.loads(state.task_json)
    env = context.env_factory()
    history, actions = [], []
    try:
        cursor, observation = (env.reset(request) if executor is None else
                               (yield executor.submit(env.reset, request)))
        history.append(_observed(observation))
        for _ in range(40):
            prompt = context.renderer(request, deepcopy(history))
            action = yield (prompt, generator)
            actions.append(action)
            command, _ = parse_action(action.text, observation.admissible)
            cursor, observation = (env.step(cursor, command) if executor is None else
                                   (yield executor.submit(env.step, cursor, command)))
            history.extend([dict(role='assistant', content=command), _observed(observation)])
            if observation.done:
                break
        return TaskRollout(request['task_id'], tuple(actions), float(observation.won), backend.identity(parameters))
    finally:
        env.close()
