"""Shared ReAct episodes with caller-thread sampling and cached wiki cohorts."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os

from ... import hotpotqa as hp
from ..bank_build import privileged
from ..feedback_rng import FeedbackRNG, FEEDBACK_RNG_VERSION
from ..generation_batch import run_episode_streams_lockstep
from ..return_gradient import TaskRollout
from .hotpotqa_caps import action_limit
from .hotpotqa_support import task_request


@dataclass(frozen=True)
class HotpotQAFeedbackContext:
    support: object
    round_number: int
    renderer: object
    journal: object
    wiki_factory: object
    question_loader: object = hp.load_questions

    def check_syntax(self, text):
        return dict(valid=hp.parse_action(text) is not None)


class EpisodeBackend:
    """Use D16's normalized V2 action metadata for serial and cohort draws."""
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def sample_feedback_actions(self, requests, parameters, **settings):
        return self.backend.sample_feedback_actions(requests, parameters,
            feedback_rng_version=FEEDBACK_RNG_VERSION, **settings)


def _episode(state, backend, parameters, generator, context, executor=None):
    privileged()
    if (not isinstance(context, HotpotQAFeedbackContext) or context.support is None
            or context.support.states.get(state.parent_hash) != state
            or type(context.round_number) is not int or context.round_number not in (1, 2)):
        raise ValueError('HotpotQA feedback needs a bound support reset and round')
    diagnostic = getattr(backend, 'diagnostic_only', False)
    if not diagnostic and int(state.parent_hash, 16) % 2 == (context.round_number-1) % 2:
        raise PermissionError('feedback requires the opposite parent fold')
    questions = {q['_id']: q for q in context.question_loader('train')}
    task = json.loads(state.task_json)
    question = questions[task['task_id']]
    if task != task_request(question):
        raise ValueError('HotpotQA question differs from frozen reset')
    wiki = context.wiki_factory()  # every episode owns its lookup cursor
    stream = hp.episode_stream(question, wiki, temperature=0. if diagnostic else 1.)
    actions = []
    try:
        event = next(stream)
        while True:
            try:
                if event[0] == 'generate':
                    _, messages, stops, temperature = event
                    prompt = context.renderer(messages)
                    if not actions and prompt != state.prompt:
                        raise ValueError('live HotpotQA reset differs from frozen prompt')
                    action = yield (prompt, generator)
                    if (action.policy_id != backend.identity(parameters) or action.backend_id != backend.backend_id
                            or action.prompt_ids != tuple(backend.tokenizer.encode(prompt, add_special_tokens=False))):
                        raise ValueError('HotpotQA generation policy/prompt mismatch')
                    # Keep every sampled ID/logprob (including termination) for
                    # feedback scoring. Stop strings affect the environment only.
                    text = action.text
                    for stop in stops:
                        text = text.split(stop, 1)[0]
                    # ReAct's action-only retry can recover a format error.
                    # BFCL's malformed=True means terminal reward zero, so it
                    # cannot represent these recoverable protocol failures.
                    actions.append(action)
                    value = text
                else:
                    fn = getattr(wiki, event[1])
                    value = fn(event[2]) if executor is None else (yield executor.submit(fn, event[2]))
            except Exception as exc:
                event = stream.throw(exc)
            else:
                event = stream.send(value)
    except StopIteration as done:
        record = done.value
        if record['error']:
            raise RuntimeError('incomplete HotpotQA feedback: ' + record['error'])
        result = TaskRollout(record['task_id'], tuple(actions), float(record['em']), backend.identity(parameters))
        if context.journal is not None:
            context.journal.append('hotpotqa_episode', round=context.round_number,
                task_id=result.task_id, em=record['em'], f1=record['f1'], steps=record['steps'],
                model_calls=record['model_calls'], format_failures=record['format_failures'], temperature=0. if diagnostic else 1.,
                feedback_rng_version=FEEDBACK_RNG_VERSION, diagnostic_only=diagnostic,
                from_task_start=True, wiki_queries=record['wiki_queries'])
        return result
    finally:
        stream.close()
        if hasattr(wiki, 'close'):
            wiki.close()


def _serial(stream, backend, parameters):
    try:
        request = next(stream)
        while True:
            prompt, generator = request
            try:
                if getattr(backend, 'generation_batch', None) is not None and not getattr(backend, 'diagnostic_only', False):
                    action = backend.sample_feedback_actions([(prompt, generator)], parameters)[0]
                else:
                    action = backend.sample_action(prompt, parameters, generator, temperature=1., top_p=1.)
            except BaseException as exc:
                request = stream.throw(exc)
            else:
                request = stream.send(action)
    except StopIteration as done:
        return done.value
    finally:
        stream.close()


def hotpotqa_task_rollout(entry, category, truth, backend, parameters, generator, *, checker=None):
    if category != 'agent_action' or truth != []:
        raise ValueError('HotpotQA requires agent_action without labels')
    with action_limit(backend, category):
        wrapped = EpisodeBackend(backend)
        return _serial(_episode(entry, wrapped, parameters, generator, checker), wrapped, parameters)


def feedback_streams(support, tasks, backend, parameters, rng_context, checker, *, lockstep):
    if (not isinstance(checker, HotpotQAFeedbackContext) or checker.support is not support
            or not isinstance(rng_context, FeedbackRNG) or rng_context.round != checker.round_number):
        raise ValueError('HotpotQA feedback requires its V2 seed/round/step/role context')
    natural = support.config['meta_tasks_per_feedback'] * support.config['rollouts_per_meta_task']
    limit = int(os.environ.get('BFAS_FEEDBACK_LOCKSTEP_EPISODES', str(natural))) if lockstep else 1
    if limit < 1:
        raise ValueError('BFAS_FEEDBACK_LOCKSTEP_EPISODES must be a positive integer')
    device = next(iter(parameters.values())).device
    indices, entries = defaultdict(int), []
    for parent, count in tasks:
        if type(count) is not int or count < 1:
            raise ValueError('positive HotpotQA rollout count required')
        tid = support.parents[parent]
        for _ in range(count):
            entries.append((parent, rng_context.generator(tid, indices[tid], device=device)))
            indices[tid] += 1
    wrapped = EpisodeBackend(backend)
    for offset in range(0, len(entries), limit):
        cohort = entries[offset:offset+limit]
        with action_limit(backend, 'agent_action'):
            if lockstep:
                executor = ThreadPoolExecutor(max_workers=len(cohort), thread_name_prefix='hotpotqa-wiki')
                streams = [_episode(support.states[p], wrapped, parameters, g, checker, executor) for p, g in cohort]
                results = run_episode_streams_lockstep(streams, wrapped, parameters, executor)
            else:
                results = [_serial(_episode(support.states[p], wrapped, parameters, g, checker), wrapped, parameters)
                           for p, g in cohort]
        yield from ((p, result) for (p, _), result in zip(cohort, results, strict=True))


def diagnostic_batch(support, parents, backend, parameters, generator, checker):
    from .interactive_diagnostics import diagnostic_episode_limit
    if not getattr(backend, 'diagnostic_only', False):
        raise PermissionError('diagnostic greedy backend required')
    parents, limit = tuple(parents), diagnostic_episode_limit()
    for offset in range(0, len(parents), limit):
        cohort = parents[offset:offset+limit]
        executor = ThreadPoolExecutor(max_workers=len(cohort), thread_name_prefix='hotpotqa-diagnostic')
        streams = [_episode(support.states[p], backend, parameters, generator, checker, executor) for p in cohort]
        with action_limit(backend, 'agent_action'):
            results = run_episode_streams_lockstep(streams, backend, parameters, executor)
        yield from zip(cohort, results, strict=True)
