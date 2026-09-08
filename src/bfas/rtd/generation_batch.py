"""Opt-in HF batching, with ordered RNG tickets and ephemeral draw queues.

One int64 ticket is consumed from the durable sampling generator per action,
in caller order, before length sorting. HF's batch stream is seeded by the
ordered tickets in that batch. The batch layout is part of backend identity;
changing it changes realizations, never the temperature-1 categorical law.
No tickets, generated actions or KV caches survive a sampling scope.
"""
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from itertools import groupby

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
    return backend.action_caps.get('multi_turn' if category.startswith('multi_turn') else 'single_turn',
                                   backend.max_action_tokens)


class HFGenerationBatchMixin:
    def _prompt_ids(self, prompt):
        ids = tuple(self.tokenizer.encode(prompt, add_special_tokens=False)) if isinstance(prompt, str) else tuple(prompt)
        if not ids or self.max_context_tokens-len(ids) < 1:
            raise IncompleteRolloutError('complete prompt exceeds context; no truncation')
        return ids

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
        groups = length_bucketed_groups(rows, prompts_per_batch=policy.prompts_per_batch, budget=budget)
        result = [None]*len(rows)
        for group in groups:
            for row, action in zip(group, self._generate_group(group, parameters)):
                result[row['index']] = action
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
        eos = self.tokenizer.eos_token_id
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
            max_new_tokens=rows[0]['limit'], eos_token_id=eos, pad_token_id=eos,
            bos_token_id=self.tokenizer.bos_token_id, use_cache=True,
            return_dict_in_generate=True, output_scores=True)
        seed = int(digest([RNG_RULE, [r['ticket'] for r in rows]])[:15], 16)
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
            result = []
            for row, sequence, logps in zip(rows, sequences, values):
                if eos in sequence:
                    length = sequence.index(eos)+1
                    sequence, logps = sequence[:length], logps[:length]
                truncated = sequence[-1] != eos
                if truncated and len(sequence) != row['limit']:
                    raise ValueError('malformed generation: expected first EOS or action limit')
                metadata = dict(implementation='hf-generate-kv-batched-categorical-v1', use_cache=True,
                    logits_dtypes=sorted(dtypes), scores_dtypes=sorted({str(s.dtype) for s in output.scores}),
                    logprob_dtype='torch.float32', reduction_dtype='python.float',
                    parameter_dtypes=sorted({str(p.dtype) for p in parameters.values()}),
                    model_class=type(generation_model).__name__, torch_version=torch.__version__,
                    transformers_version=transformers.__version__, attention='eager', temperature=1., top_p=1.,
                    top_k=0, repetition_penalty=1., max_action_tokens=row['cap'], effective_action_limit=row['limit'],
                    rng_rule=RNG_RULE, rng_ticket=row['ticket'], batch_seed=seed, batch_rng_after=rng_after,
                    batch_size=size, padded_prompt_tokens=width, padding_side='left')
                action = ActionTrace(row['ids'], tuple(sequence), eos,
                    self.tokenizer.decode(sequence if truncated else sequence[:-1], skip_special_tokens=False),
                    sum(logps), self.backend_id, identity, tuple(logps), metadata, truncated=truncated)
                result.append(action)
                if self.journal:
                    self.journal.append('generated_tokens', context=self.context, action_tokens=len(sequence),
                        complete=not truncated, truncated=truncated, policy_id=identity,
                        action_hash=digest(action.action_ids), sample_hash=digest(asdict(action)),
                        **row['ticket'], rng_rule=RNG_RULE, batch_seed=seed, batch_rng_after=rng_after)
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
        rollout = support.feedback(parent, proxy, parameters, generator, checker)
        if not proxy.used:
            raise AssertionError('feedback did not consume its task-start action')
        yield rollout
