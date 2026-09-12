"""Batch BFCL model queries while its synchronous Python harness steps episodes."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from queue import Queue
from threading import Lock

from ..generation_batch import action_cap
from ..feedback_rng import FEEDBACK_RNG_VERSION
from ..return_gradient import TaskRollout, bfcl_task_rollout


class BFCLFeedbackBackend:
    """Singleton logical draws and stable action records under RNG V2.

    Physical batch sizes/timings remain in compute records. Logical action
    metadata and generated-token hashes agree with the serial stream driver.
    """
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def sample_action(self, prompt, parameters, generator, **settings):
        if getattr(self.backend, 'generation_batch', None) is None:
            return self.backend.sample_action(prompt, parameters, generator, **settings)
        if settings != dict(temperature=1., top_p=1.):
            raise ValueError('BFCL requires temperature=1/top_p=1')
        return self.sample_feedback_actions([(prompt, generator)], parameters)[0]

    def sample_feedback_actions(self, requests, parameters, **settings):
        return self.backend.sample_feedback_actions(requests, parameters,
            feedback_rng_version=FEEDBACK_RNG_VERSION, **settings)


class _Cancelled(BaseException):
    """Unwind even vendor handlers that catch Exception around tool execution."""


@dataclass
class _Query:
    prompt: str
    response: Future


class _EpisodeBackend:
    """Only the harness runs on a worker; no model access or shared cap mutation."""
    def __init__(self, backend, identity, category):
        self.student_config = getattr(backend, 'student_config', {})
        self.tokenizer = getattr(backend, 'tokenizer', None)
        self._identity, self.category = identity, category
        self.events, self.lock = Queue(), Lock()
        self.pending, self.cancelled = None, False

    def identity(self, parameters):
        return self._identity

    def action_limit(self, category):
        if category != self.category:
            raise AssertionError('BFCL episode action category changed')
        return nullcontext()

    def sample_action(self, prompt, parameters, generator, **settings):
        if settings != dict(temperature=1., top_p=1.):
            raise ValueError('BFCL requires temperature=1/top_p=1')
        with self.lock:
            if self.cancelled:
                raise _Cancelled()
            response = self.pending = Future()
            self.events.put(_Query(prompt, response))
        return response.result()

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.pending is not None and not self.pending.done():
                self.pending.set_exception(_Cancelled())


def bfcl_task_rollouts_lockstep(requests, backend, parameters, generators, *, checker=None,
                               first_actions=None):
    """Run aligned (entry, category, truth) episodes with caller-owned RNGs.

    Reuse bfcl_task_rollout verbatim, including malformed-action guards, unique
    simulator namespaces, terminal checking and cleanup. Independent episode
    streams are required for stochastic continuation batching. Greedy queries
    never touch the supplied generators. Cohort sizing belongs to the provider.
    """
    requests, generators = tuple(requests), tuple(generators)
    if len(requests) != len(generators) or (first_actions is not None and len(first_actions) != len(requests)):
        raise ValueError('aligned BFCL episodes, generators and task starts required')
    if not requests:
        return ()
    if not getattr(backend, 'diagnostic_only', False) and len({id(g) for g in generators}) != len(generators):
        raise ValueError('BFCL lockstep requires independent episode RNG streams')
    identity = backend.identity(parameters)
    caps = [action_cap(backend, category) for _, category, _ in requests]
    episodes = [_EpisodeBackend(backend, identity, category) for _, category, _ in requests]
    results, live = [None] * len(requests), list(range(len(requests)))
    executor = ThreadPoolExecutor(max_workers=len(requests), thread_name_prefix='bfcl-rollout')

    def run(i):
        entry, category, truth = requests[i]
        try:
            result = bfcl_task_rollout(entry, category, truth, episodes[i], parameters,
                                       generators[i], checker=checker)
        except BaseException as exc:
            result = exc
        episodes[i].events.put(result)

    try:
        for i in live:
            executor.submit(run, i)
        first = True
        while live:
            queries = {}
            for i in live:
                event = episodes[i].events.get()
                if isinstance(event, BaseException):
                    raise event
                if isinstance(event, TaskRollout):
                    if first and first_actions is not None:
                        raise AssertionError('feedback did not consume its task-start action')
                    results[i] = event
                else:
                    queries[i] = event
            live = list(queries)
            if first and first_actions is not None:
                for i, query in queries.items():
                    action = first_actions[i]
                    if backend._prompt_ids(query.prompt) != action.prompt_ids or action.policy_id != identity:
                        raise AssertionError('batched task-start prompt/policy mismatch')
                    query.response.set_result(action)
            else:
                # Mixed single/multi-turn categories retain their original cap;
                # D12 splits further by effective context cap and token budget.
                groups = {}
                for i in live:
                    groups.setdefault(caps[i], []).append(i)
                for indices in groups.values():
                    def dispatch(batch_indices, actions):
                        for index, action in zip(batch_indices, actions, strict=True):
                            queries[indices[index]].response.set_result(action)
                    with backend.action_limit(requests[indices[0]][1]):
                        actions = backend.sample_feedback_actions(
                            [(queries[i].prompt, generators[i]) for i in indices], parameters,
                            prompts_per_batch=len(requests), on_batch=dispatch)
                    if len(actions) != len(indices):
                        raise AssertionError('unaligned BFCL action batch')
            first = False
        return tuple(results)
    finally:
        # Release blocked callbacks BEFORE joining, including a failure in a
        # later token-budget sub-batch. Each original rollout cleans its cache.
        for episode in episodes:
            episode.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
