"""Opt-in HF batching, with ordered RNG tickets and ephemeral draw queues.

One int64 ticket is consumed from the durable sampling generator per action,
in caller order, before length sorting. HF's batch stream is seeded by the
ordered tickets in that batch. The batch layout is part of backend identity;
changing it changes realizations, never the temperature-1 categorical law.
No tickets, generated actions or KV caches survive a sampling scope.
"""
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from itertools import groupby
import os

from torch.overrides import TorchFunctionMode

import torch

from .persistence import digest
from .return_gradient import ActionTrace, IncompleteRolloutError


RNG_RULE = 'ordered-int64-tickets-hf-batch-sha256-v1'


@dataclass(frozen=True)
class GenerationBatch:
    prompts_per_batch: int = 8
    max_batch_tokens: int = 16384
    forward_prompts_per_batch: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in self.sampling_config().values()):
            raise ValueError('generation_batch requires positive integer limits')
        if type(self.forward_prompts_per_batch) is not int or self.forward_prompts_per_batch < 0:
            raise ValueError('forward_prompts_per_batch must be a nonnegative integer (0 disables)')

    def sampling_config(self):
        # Forward layout must never change D12 policy IDs, tickets or samples.
        return dict(prompts_per_batch=self.prompts_per_batch, max_batch_tokens=self.max_batch_tokens)

    def config(self):
        return self.sampling_config() | (dict(forward_prompts_per_batch=self.forward_prompts_per_batch)
                                       if self.forward_prompts_per_batch else {})

    @classmethod
    def from_config(cls, config):
        if 'generation_batch' not in config:
            return None
        if config.get('protocol_version') != '1.1.0':
            raise ValueError('generation_batch requires protocol_version 1.1.0')
        if not isinstance(config['generation_batch'], dict):
            raise ValueError('generation_batch must be a mapping')
        return cls(**config['generation_batch'])


def length_bucketed_groups(rows, *, prompts_per_batch, budget):
    """D12 planner shared by generation and D13 full-sequence forwards.

    Forward rows use ids=prompt+action and limit=0. Stable original indices
    restore caller order; an oversized row runs alone without truncation.
    """
    ordered = sorted(rows, key=lambda r: (r['limit'], len(r['ids']), r['index']))
    groups, group, blocks = [], [], []
    for _, members in groupby(ordered, key=lambda r: r['prompt_index']):
        members = list(members)
        capacity = max(1, budget//(len(members[0]['ids'])+members[0]['limit']))
        # Generation members have equal lengths. Forward actions can differ:
        # split before adding a row that would exceed the padded token budget.
        block = []
        for row in members:
            if block and ((len(block)+1)*(len(row['ids'])+row['limit']) > budget or
                          len(row['ids']) > 2*len(block[0]['ids'])):
                blocks.append(block)
                block = []
            block.append(row)
            if len(block) == capacity:
                blocks.append(block)
                block = []
        if block:
            blocks.append(block)
    for block in blocks:
        row = block[0]
        candidate = group + block
        width = max(len(r['ids']) for r in candidate)
        if group and (row['limit'] != group[0]['limit'] or
                len({r['prompt_index'] for r in candidate}) > prompts_per_batch or
                len(candidate)*(width+row['limit']) > budget or
                width > 2*len(group[0]['ids'])):
            groups.append(group)
            group = []
        group.extend(block)
    if group:
        groups.append(group)
    return groups


def ticket(generator):
    before = digest(generator.get_state().tolist())
    value = int(torch.randint(0, 2**63-1, (), device=generator.device, generator=generator))
    return dict(seed=value, rng_before=before, rng_after=digest(generator.get_state().tolist()),
                draw_id=digest([RNG_RULE, before, value]))


def sample_actions(backend, prompt, n, parameters, generator, **settings):
    """Keep the legacy call/score interleaving lazy when batching is disabled."""
    if getattr(backend, 'generation_batch', None) is not None:
        yield from backend.sample_actions(prompt, n, parameters, generator, **settings)
    else:
        for _ in range(n):
            yield backend.sample_action(prompt, parameters, generator, **settings)


def action_cap(backend, category):
    # Plan with the same registered benchmark policy used when consuming draws.
    # Interactive agent_action caps differ from BFCL's single/multi-turn caps.
    with backend.action_limit(category):
        return backend.max_action_tokens


def feedback_generation_groups(rows, policy):
    """Cap physical feedback batches without splitting a legacy RNG group.

    A single oversized prompt (or pre-existing task-start RNG group) runs
    alone, as in D12. Effective action limits must agree within a generate call.
    """
    batches, batch = [], []
    for _, members in groupby(rows, key=lambda row: id(row['sampling_group'])):
        members = list(members)
        candidate = batch + members
        width = max(len(row['ids']) for row in candidate)
        if batch and (len(candidate) > policy.prompts_per_batch or
                len(candidate) * (width + members[0]['limit']) > policy.max_batch_tokens or
                batch[0]['limit'] != members[0]['limit']):
            batches.append(batch)
            batch = []
        batch.extend(members)
    if batch:
        batches.append(batch)
    return batches


class _FeedbackMultinomial(TorchFunctionMode):
    """Route HF's categorical draws to the original logical-batch generators.

    This thread-local dispatch mode changes only torch.multinomial inside the
    generate call; logits, scores, stopping and model code remain HF's own.
    Singleton continuation streams match serial D12 generation. Task starts
    retain D12's original multi-row draws, including sampling finished rows.
    """
    def __init__(self, rows, device):
        self.rows, self.groups, self.after = rows, [], {}
        for _, members in groupby(enumerate(rows), key=lambda item: id(item[1]['sampling_group'])):
            members = list(members)
            spec = members[0][1]['sampling_group']
            self.groups.append((slice(members[0][0], members[-1][0]+1), spec,
                                torch.Generator(device=device).manual_seed(spec['seed']), []))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is not torch.multinomial:
            return func(*args, **kwargs)
        probabilities = args[0]
        if (probabilities.ndim != 2 or probabilities.shape[0] != len(self.rows) or
                kwargs.get('generator') is not None):
            raise ValueError('unexpected HF feedback categorical sampling call')
        samples = []
        for indices, _, generator, states in self.groups:
            samples.append(func(probabilities[indices], *args[1:], **(kwargs | dict(generator=generator))))
            states.append(generator.get_state())
        return torch.cat(samples, dim=0)

    def finish(self, sequences, stops):
        for indices, spec, _, states in self.groups:
            length = max(next((i+1 for i, token in enumerate(sequence) if token in stops), len(sequence))
                         for sequence in sequences[indices])
            if len(states) != len(sequences[0]):
                raise ValueError('HF did not use the feedback categorical RNG on every decode step')
            # A shorter logical group would have returned at this decode step.
            # Later physical-batch padding draws must not enter its RNG record.
            self.after[id(spec)] = digest(states[length-1].tolist())

    def rng_after(self, row):
        return self.after[id(row['sampling_group'])]


class HFGenerationBatchMixin:
    def _prompt_ids(self, prompt):
        ids = tuple(self.tokenizer.encode(prompt, add_special_tokens=False)) if isinstance(prompt, str) else tuple(prompt)
        if not ids or self.max_context_tokens-len(ids) < 1:
            raise IncompleteRolloutError('complete prompt exceeds context; no truncation')
        return ids

    def greedy_actions(self, prompts, parameters, *, prompts_per_batch, on_batch=None):
        """Diagnostic argmax with D12 padding/budgets, without sampling tickets.

        Keep the serial diagnostic ActionTrace fields exactly; only physical
        compute counts change. Finished rows are stripped at their first stop.
        """
        if self.model.training:
            raise ValueError('frozen eval policy required')
        rows = []
        for prompt in prompts:
            ids = self._prompt_ids(prompt)
            rows.append(dict(index=len(rows), prompt_index=len(rows), ids=ids,
                             limit=min(self.max_action_tokens, self.max_context_tokens-len(ids))))
        groups = length_bucketed_groups(rows, prompts_per_batch=prompts_per_batch,
            budget=self.generation_batch.max_batch_tokens)
        result = [None] * len(rows)
        for group in groups:
            actions = self._generate_greedy_group(group, parameters)
            if len(actions) != len(group):
                raise AssertionError('greedy backend returned an unaligned action batch')
            for row, action in zip(group, actions):
                result[row['index']] = action
            if on_batch is not None:
                on_batch(tuple(row['index'] for row in group), actions)
        return tuple(result)

    def _generate_greedy_group(self, rows, parameters):
        from transformers import GenerationConfig
        from .runtime import installed_parameters
        from .student import termination_ids
        eos, stops = self.tokenizer.eos_token_id, termination_ids(self)
        device = next(iter(parameters.values())).device
        width, size = max(len(r['ids']) for r in rows), len(rows)
        ids = torch.full((size, width), eos, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, -len(row['ids']):] = torch.tensor(row['ids'], device=device)
            mask[i, -len(row['ids']):] = 1
        settings = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=rows[0]['limit'],
            eos_token_id=list(stops), pad_token_id=eos, bos_token_id=self.tokenizer.bos_token_id,
            repetition_penalty=1., use_cache=True, return_dict_in_generate=True, output_scores=True)
        with self.measured('diagnostic_greedy_generation', sequences=size,
                prompt_tokens=sum(len(r['ids']) for r in rows), padded_prompt_tokens=size*width), \
                installed_parameters(self.model, parameters), torch.no_grad():
            output = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=settings)
            tokens = output.sequences[:, width:]
            if not output.scores or len(output.scores) != tokens.shape[1]:
                raise ValueError('HF generation scores do not cover greedy actions')
            values = torch.stack([s.float().log_softmax(-1).gather(1, tokens[:, t:t+1]).squeeze(1)
                                  for t, s in enumerate(output.scores)], dim=1).cpu().tolist()
            sequences = tokens.cpu().tolist()
        identity, result = self.identity(parameters), []
        for row, sequence, logps in zip(rows, sequences, values):
            length = next((i+1 for i, token in enumerate(sequence) if token in stops), len(sequence))
            sequence, logps = sequence[:length], logps[:length]
            truncated = sequence[-1] not in stops
            if truncated and len(sequence) != row['limit']:
                raise ValueError('malformed generation: expected first EOS or action limit')
            result.append(ActionTrace(row['ids'], tuple(sequence), eos if truncated else sequence[-1],
                self.tokenizer.decode(sequence if truncated else sequence[:-1], skip_special_tokens=False),
                sum(logps), self.backend_id, identity, tuple(logps),
                dict(temperature=0., do_sample=False, diagnostic_only=True), truncated=truncated))
        return tuple(result)

    def sample_actions(self, prompt_ids, n, parameters, generator, *, temperature=1., top_p=1.):
        if type(n) is not int or n < 1:
            raise ValueError('positive sample count required')
        if self.model.training or temperature != 1 or top_p != 1:
            raise ValueError('frozen eval policy with temperature=1/top_p=1 required')
        ids = self._prompt_ids(prompt_ids)
        pending = getattr(self, '_pending_generation', None)
        if pending is not None:
            result = []
            for _ in range(n):
                if not pending:
                    raise AssertionError('prefetched draw queue exhausted')
                action = pending.popleft()
                if (action.prompt_ids != ids or action.policy_id != self.identity(parameters) or
                        action.generation_metadata['max_action_tokens'] != self.max_action_tokens):
                    raise AssertionError('prefetched draw prompt/policy/cap mismatch')
                if ticket(generator) != action.generation_metadata['rng_ticket']:
                    raise AssertionError('prefetched draw RNG order mismatch')
                result.append(action)
            return tuple(result)
        return self.sample_actions_batch([ids], n, parameters, generator,
            max_batch_tokens=(self.generation_batch or GenerationBatch()).max_batch_tokens)[0]

    def sample_actions_batch(self, list_of_prompt_ids, n_per_prompt, parameters, generator, *, max_batch_tokens=None):
        """Return prompt-major tuples; token budget includes padding AND action caps.

        Equal effective limits are grouped so HF stops each row at its original
        cap without forced EOS. Length sorting and <=2x padding bound prefill.
        An oversized single row runs alone, without changing its context/cap.
        """
        if type(n_per_prompt) is not int or n_per_prompt < 1:
            raise ValueError('positive sample count required')
        requests = [(p, n_per_prompt, self.max_action_tokens) for p in list_of_prompt_ids]
        flat = self._sample_requests(requests, parameters, generator, max_batch_tokens=max_batch_tokens)
        return tuple(tuple(flat[i:i+n_per_prompt]) for i in range(0, len(flat), n_per_prompt))

    def _sample_requests(self, requests, parameters, generator, *, max_batch_tokens=None):
        if self.model.training:
            raise ValueError('frozen eval policy required')
        policy = self.generation_batch or GenerationBatch()
        budget = policy.max_batch_tokens if max_batch_tokens is None else max_batch_tokens
        if type(budget) is not int or budget < 1:
            raise ValueError('positive max_batch_tokens required')
        rows = self._request_rows(requests, generator)
        groups = length_bucketed_groups(rows, prompts_per_batch=policy.prompts_per_batch, budget=budget)
        result = [None]*len(rows)
        for group in groups:
            for row, action in zip(group, self._generate_group(group, parameters)):
                result[row['index']] = action
        return tuple(result)

    def _request_rows(self, requests, generator):
        rows = []
        # Validate everything before advancing the caller's RNG.
        for prompt_index, (prompt, n, cap) in enumerate(requests):
            ids = self._prompt_ids(prompt)
            if type(n) is not int or n < 1 or type(cap) is not int or cap < 1:
                raise ValueError('positive sample count/action cap required')
            for _ in range(n):
                rows.append(dict(index=len(rows), prompt_index=prompt_index, ids=ids, cap=cap,
                                 limit=min(cap, self.max_context_tokens-len(ids))))
        for row in rows:
            row['ticket'] = ticket(generator)
        return rows

    def feedback_start_groups(self, prompt, count, generator):
        """Freeze exactly the task-start tickets and logical batches used by D12."""
        rows = self._request_rows([(prompt, count, self.max_action_tokens)], generator)
        return length_bucketed_groups(rows, prompts_per_batch=self.generation_batch.prompts_per_batch,
                                      budget=self.generation_batch.max_batch_tokens)

    def sample_feedback_actions(self, requests, parameters, *, prompts_per_batch=None, on_batch=None):
        """One independent serial-equivalent RNG stream per live continuation."""
        groups = [self._request_rows([(prompt, 1, self.max_action_tokens)], generator)
                  for prompt, generator in requests]
        return self.generate_feedback_groups(groups, parameters,
            prompts_per_batch=prompts_per_batch, on_batch=on_batch)

    def generate_feedback_groups(self, groups, parameters, *, prompts_per_batch=None, on_batch=None):
        """Keep logical RNG groups fixed while independently sizing physical calls.

        on_batch receives caller indices/actions after each physical call, so
        environment RPCs can overlap the remaining sub-batches on this thread.
        """
        if self.model.training:
            raise ValueError('frozen eval policy required')
        policy = self.generation_batch
        if prompts_per_batch is not None:
            policy = replace(policy, prompts_per_batch=prompts_per_batch)
        rows = []
        for group in groups:
            # Sampling layout stays frozen even when several logical batches
            # share a physical generate call. Continuations are singleton groups.
            sampling = dict(seed=int(digest([RNG_RULE, [r['ticket'] for r in group]])[:15], 16),
                            size=len(group), width=max(len(r['ids']) for r in group))
            offset = len(rows)
            rows.extend(dict(row, index=offset+i, sampling_group=sampling)
                        for i, row in enumerate(group))
        result = [None] * len(rows)
        # Keep each original RNG group together and preserve its row ordering.
        for batch in feedback_generation_groups(rows, policy):
            actions = self._generate_group(batch, parameters)
            if len(actions) != len(batch):
                raise AssertionError('feedback backend returned an unaligned action batch')
            for row, action in zip(batch, actions):
                result[row['index']] = action
            if on_batch is not None:
                on_batch(tuple(row['index'] for row in batch), actions)
        return tuple(result)

    @contextmanager
    def prefetch_actions(self, requests, parameters, generator, *, score=False):
        """Fresh source draws, consumed once in original order with D7 checks.

        Use a clone to generate ahead, then advance the durable stream one
        ticket per consumed action. Failure discards the entire ephemeral queue;
        the existing durable phase checkpoint determines recovery.
        """
        if getattr(self, '_pending_generation', None) is not None:
            raise AssertionError('nested generation prefetch')
        cloned = torch.Generator(device=generator.device)
        cloned.set_state(generator.get_state())
        actions = self._sample_requests(requests, parameters, cloned)
        self._pending_generation = deque(actions)
        try:
            with self.prefetch_scores(actions, parameters) if score else nullcontext():
                yield
                if self._pending_generation:
                    raise AssertionError('prefetched draws not fully consumed')
        finally:
            self._pending_generation = None

    def _generate_group(self, rows, parameters):
        from transformers import GenerationConfig
        import transformers
        from .runtime import installed_parameters
        from .scoring import attention_implementation
        eos = self.tokenizer.eos_token_id
        from .student import termination_ids
        stops = termination_ids(self)
        if type(eos) is not int:
            raise ValueError('one configured EOS token required')
        device = next(iter(parameters.values())).device
        width, size = max(len(r['ids']) for r in rows), len(rows)
        ids = torch.full((size, width), eos, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, -len(row['ids']):] = torch.tensor(row['ids'], device=device)
            mask[i, -len(row['ids']):] = 1
        settings = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
            typical_p=1., repetition_penalty=1., num_beams=1, num_return_sequences=1,
            max_new_tokens=rows[0]['limit'], eos_token_id=list(stops), pad_token_id=eos,
            bos_token_id=self.tokenizer.bos_token_id, use_cache=True,
            return_dict_in_generate=True, output_scores=True)
        seed = int(digest([RNG_RULE, [r['ticket'] for r in rows]])[:15], 16)
        independent = _FeedbackMultinomial(rows, device) if 'sampling_group' in rows[0] else None
        devices = [device.index or 0] if device.type == 'cuda' else []
        identity = self.identity(parameters)
        generation_model = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        dtypes = set()
        def observe(module, inputs, output):
            dtypes.add(str(output.logits.dtype))
        with self.measured('generation', prompt_tokens=sum(len(r['ids']) for r in rows),
                sequences=size, padded_prompt_tokens=size*width, batch_seed=seed), \
                installed_parameters(self.model, parameters), torch.random.fork_rng(devices=devices), torch.no_grad():
            batch_rng = torch.Generator(device=device).manual_seed(seed)
            if devices:
                torch.cuda.set_rng_state(batch_rng.get_state(), device)
            else:
                torch.set_rng_state(batch_rng.get_state())
            hook = generation_model.register_forward_hook(observe)
            try:
                with independent if independent is not None else nullcontext():
                    output = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=settings)
            finally:
                hook.remove()
            rng_after = digest((torch.cuda.get_rng_state(device) if devices else torch.get_rng_state()).tolist())
            if len(output.scores) != output.sequences.shape[1]-width:
                raise ValueError('HF generation scores do not cover sampled actions')
            # One vector operation per decode step, not one host sync per token.
            tokens = output.sequences[:, width:]
            values = torch.stack([s.float().log_softmax(-1).gather(1, tokens[:, t:t+1]).squeeze(1)
                                  for t, s in enumerate(output.scores)], dim=1).cpu().tolist()
            sequences = tokens.cpu().tolist()
            if independent is not None:
                independent.finish(sequences, stops)
            cache = getattr(output, 'past_key_values', None)
            result = []
            for row, sequence, logps in zip(rows, sequences, values):
                terminal = next((i for i, t in enumerate(sequence) if t in stops), None)
                if terminal is not None:
                    length = terminal+1
                    sequence, logps = sequence[:length], logps[:length]
                truncated = sequence[-1] not in stops
                action_eos = eos if truncated else sequence[-1]
                if truncated and len(sequence) != row['limit']:
                    raise ValueError('malformed generation: expected first EOS or action limit')
                action_seed = row['sampling_group']['seed'] if independent is not None else seed
                action_rng_after = independent.rng_after(row) if independent is not None else rng_after
                metadata = dict(implementation='hf-generate-kv-batched-categorical-v1', use_cache=True,
                    logits_dtypes=sorted(dtypes), scores_dtypes=sorted({str(s.dtype) for s in output.scores}),
                    logprob_dtype='torch.float32', reduction_dtype='python.float',
                    parameter_dtypes=sorted({str(p.dtype) for p in parameters.values()}),
                    model_class=type(generation_model).__name__, torch_version=torch.__version__,
                    transformers_version=transformers.__version__, attention=attention_implementation(self.model),
                    cache_type=type(cache).__name__ if cache is not None else None, temperature=1., top_p=1.,
                    top_k=0, repetition_penalty=1., max_action_tokens=row['cap'], effective_action_limit=row['limit'],
                    rng_rule=RNG_RULE, rng_ticket=row['ticket'], batch_seed=action_seed, batch_rng_after=action_rng_after,
                    batch_size=size, padded_prompt_tokens=width, padding_side='left')
                if independent is not None:
                    metadata.update(sampling_batch_size=row['sampling_group']['size'],
                                    sampling_prompt_width=row['sampling_group']['width'])
                action = ActionTrace(row['ids'], tuple(sequence), action_eos,
                    self.tokenizer.decode(sequence if truncated else sequence[:-1], skip_special_tokens=False),
                    sum(logps), self.backend_id, identity, tuple(logps), metadata, truncated=truncated)
                result.append(action)
                if self.journal:
                    self.journal.append('generated_tokens', context=self.context, action_tokens=len(sequence),
                        complete=not truncated, truncated=truncated, policy_id=identity,
                        action_hash=digest(action.action_ids), sample_hash=digest(asdict(action)),
                        **row['ticket'], rng_rule=RNG_RULE, batch_seed=action_seed, batch_rng_after=action_rng_after)
        return tuple(result)


class FirstActionBackend:
    """Consume a newly sampled task-start action, then run its environment normally."""
    def __init__(self, backend, action):
        self.backend, self.action = backend, action
        self.used = False

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def sample_action(self, prompt, parameters, generator, **settings):
        if self.used:
            return self.backend.sample_action(prompt, parameters, generator, **settings)
        if (self.backend._prompt_ids(prompt) != self.action.prompt_ids or
                self.backend.identity(parameters) != self.action.policy_id):
            raise AssertionError('batched task-start prompt/policy mismatch')
        self.used = True
        return self.action


def feedback_rollouts(support, parent, count, backend, parameters, generator, checker):
    if getattr(backend, 'generation_batch', None) is None:
        for _ in range(count):
            yield support.feedback(parent, backend, parameters, generator, checker)
        return
    category = support.categories[support.parents[parent]]
    with backend.action_limit(category):
        actions = backend.sample_actions(support.states[parent].prompt, count, parameters, generator)
    for action in actions:
        proxy = FirstActionBackend(backend, action)
        # A rollout provider can set the cap on the proxy itself. Continuation
        # sampling delegates to the real backend, so it needs the same category
        # scope as the batched task starts. Restore it before yielding/scoring.
        with backend.action_limit(category):
            rollout = support.feedback(parent, proxy, parameters, generator, checker)
        if not proxy.used:
            raise AssertionError('feedback did not consume its task-start action')
        yield rollout


def feedback_rollout_tasks(support, tasks, backend, parameters, generator, checker):
    """Collect across parents so the natural feedback block shares each decode."""
    if (os.environ.get('BFAS_FEEDBACK_LOCKSTEP', '1') != '0' and
            getattr(backend, 'generation_batch', None) is not None and
            not getattr(backend, 'diagnostic_only', False) and hasattr(support, 'feedback_batch')):
        yield from support.feedback_batch(tasks, backend, parameters, generator, checker)
    else:
        for parent, count in tasks:
            for rollout in feedback_rollouts(support, parent, count, backend, parameters, generator, checker):
                yield parent, rollout


def run_episode_streams_lockstep(streams, backend, parameters, executor, *, first_actions=None):
    """Share the feedback sampling barrier and RPC cleanup with diagnostics.

    Streams yield either (prompt, generator) or an environment RPC Future and
    return their original episode record. All policy and stream work stays on
    the caller thread. This driver owns the executor and closes every stream.
    """
    starts = first_actions
    results, live = [None] * len(streams), {}

    def advance(i, method, value):
        try:
            live[i] = method(value)
        except StopIteration as completed:
            results[i] = completed.value
            live.pop(i, None)

    def settle():
        # Submit every ready step before waiting. Retries may replay several
        # cached steps; drain them to the next unsampled prompt at this barrier.
        while any(isinstance(request, Future) for request in live.values()):
            for i, request in tuple(live.items()):
                if isinstance(request, Future):
                    try:
                        response = request.result()
                    except BaseException as exc:
                        advance(i, streams[i].throw, exc)
                    else:
                        advance(i, streams[i].send, response)

    try:
        # Exit joins every RPC before throwing into/closing suspended streams,
        # so worker cleanup cannot race a pipe read on error or cancellation.
        with executor:
            for i, stream in enumerate(streams):
                try:
                    live[i] = next(stream)
                except StopIteration as completed:
                    results[i] = completed.value
            settle()
            first = True
            while live:
                order = tuple(live)

                def dispatch(batch_indices, actions):
                    for index, action in zip(batch_indices, actions):
                        i = order[index]
                        advance(i, streams[i].send, action)

                if first and starts is not None:
                    actions = []
                    for i, (prompt, _) in live.items():
                        action = starts[i]
                        if (backend._prompt_ids(prompt) != action.prompt_ids or
                                backend.identity(parameters) != action.policy_id):
                            raise AssertionError('batched task-start prompt/policy mismatch')
                        actions.append(action)
                    dispatch(range(len(order)), actions)
                    first = False
                else:
                    actions = backend.sample_feedback_actions(tuple(live.values()), parameters,
                        prompts_per_batch=len(streams), on_batch=dispatch)
                if len(actions) != len(order):
                    raise AssertionError('lockstep backend returned an unaligned action batch')
                settle()
        return tuple(results)
    except BaseException as exc:
        # Journal policy failures against every suspended episode and close all
        # owned workers even if one worker/renderer/validator aborts the batch.
        for i in tuple(live):
            try:
                streams[i].throw(exc)
            except BaseException:
                pass
        raise
    finally:
        for stream in streams:
            stream.close()
