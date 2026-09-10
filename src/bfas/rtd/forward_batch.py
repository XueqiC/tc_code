"""Opt-in no-grad teacher forcing; no live batched gradient graphs or KV cache.

Right padding leaves every real token at its original position. Only action
prediction hidden states reach the vocabulary head, in bounded position chunks.
Scores are consumed through the original score/diagnostic checks in caller order.
"""
from collections import defaultdict, deque
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch.func import functional_call
from torch.nn import functional as F

from .functional_step import _matching, lora_parameters
from .generation_batch import length_bucketed_groups
from .return_gradient import IncompleteRolloutError
from .source_scoring import _HeadInput, sampled_prefix_positions


# At vocab=248320, 32 positions need ~32 MB per FP32 vocabulary tensor.
# This is a workspace bound, independent of the action/token reduction law.
POSITION_CHUNK = 32


def forward_enabled(backend):
    return bool(getattr(getattr(backend, 'generation_batch', None), 'forward_prompts_per_batch', 0))


def token_row(prompt_ids, action_ids, eos_token_id, truncated):
    return SimpleNamespace(prompt_ids=tuple(prompt_ids), action_ids=tuple(action_ids),
                           eos_token_id=eos_token_id, truncated=truncated)


def positions(action):
    return sampled_prefix_positions(SimpleNamespace(token_ids=action.action_ids,
        eos_token_id=action.eos_token_id, truncated=action.truncated), action.prompt_ids)


def score_key(action):
    return (tuple(action.prompt_ids), tuple(action.action_ids), action.eos_token_id, action.truncated)


def plan_forwards(actions, policy, max_context_tokens):
    rows, prompt_index, previous = [], -1, None
    for i, action in enumerate(actions):
        selected = positions(action)  # Original sampled EOS/cap validation.
        ids = tuple(action.prompt_ids) + tuple(action.action_ids)
        if len(ids) > max_context_tokens:
            raise IncompleteRolloutError('context limit exceeded; no semantic truncation allowed')
        prompt = tuple(action.prompt_ids)
        if prompt != previous:
            prompt_index += 1
        previous = prompt
        rows.append(dict(index=i, prompt_index=prompt_index, ids=ids, limit=0,
                         positions=selected, action=action))
    return length_bucketed_groups(rows, prompts_per_batch=policy.forward_prompts_per_batch,
                                  budget=policy.max_batch_tokens)


@contextmanager
def source_score_scope(backend, actions, parameters):
    """Local source pairs, or the enclosing D12 cross-state prefetch scope."""
    if forward_enabled(backend) and getattr(backend, '_pending_forward_scores', None) is None:
        actions = tuple(actions)
        with backend.prefetch_scores(actions, parameters):
            yield actions
    else:
        yield actions


class HFForwardBatchMixin:
    def _forward_head(self):
        head = self.model.get_output_embeddings()
        if head is None:
            raise ValueError('batched scoring requires a positionwise output head without post-head transforms')
        prefix = next(n for n, module in self.model.named_modules() if module is head)
        return head, prefix + '.' if prefix else ''

    def _forward_hidden(self, rows, parameters, head):
        _matching(lora_parameters(self.model), parameters)
        if self.model.training:
            raise ValueError('dropout/training mode changes the sampling policy')
        device = next(iter(parameters.values())).device
        width = max(len(r['ids']) for r in rows)
        ids = torch.full((len(rows), width), self.tokenizer.eos_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            length = len(row['ids'])
            ids[i, :length] = torch.tensor(row['ids'], device=device)
            mask[i, :length] = 1
        def capture(module, args):
            raise _HeadInput(args[0])
        hook = head.register_forward_pre_hook(capture)
        try:
            # Right padding needs no shifted RoPE positions; the real tokens
            # have exactly the batch-1 positions. No recurrent/KV state escapes.
            functional_call(self.model, parameters, (), dict(input_ids=ids, attention_mask=mask, use_cache=False))
        except _HeadInput as captured:
            # Copy only scored positions, releasing padded/prompt hidden states.
            return tuple(captured.hidden[i, list(row['positions'])].detach().clone()
                         for i, row in enumerate(rows))
        finally:
            hook.remove()
        raise ValueError('teacher-forced model did not call its output embedding head')

    def _project(self, head, prefix, parameters, hidden):
        bound = {n[len(prefix):]: p for n, p in parameters.items() if n.startswith(prefix)}
        from .source_scoring import cap_logits, logit_softcap
        return cap_logits(functional_call(head, bound, (hidden,)), logit_softcap(self.model))

    def _score_group(self, rows, parameters):
        head, prefix = self._forward_head()
        with self.measured('teacher_forced_forward', sequences=len(rows), forward_passes=1,
                prompt_tokens=sum(len(r['action'].prompt_ids) for r in rows),
                action_tokens=sum(len(r['action'].action_ids) for r in rows),
                padded_sequence_tokens=len(rows)*max(len(r['ids']) for r in rows),
                padding_side='right', position_chunk=POSITION_CHUNK):
            hidden = self._forward_hidden(rows, parameters, head)
            result = []
            for row, h in zip(rows, hidden):
                labels = torch.tensor(row['action'].action_ids, device=h.device)
                chunks = []
                for start in range(0, len(h), POSITION_CHUNK):
                    logits = self._project(head, prefix, parameters, h[start:start+POSITION_CHUNK])
                    logits_dtype = str(logits.dtype)
                    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
                    chunks.append(-F.cross_entropy(logits.to(dtype), labels[start:start+POSITION_CHUNK], reduction='none'))
                    del logits
                values = torch.cat(chunks)
                score = values.sum()  # Same per-action ordered native CE reduction.
                result.append((score, values, dict(implementation='torch-functional-teacher-forced-native-ce-v1',
                    use_cache=False, logits_dtype=logits_dtype, logprob_dtype=str(values.dtype),
                    reduction_dtype=str(score.dtype), parameter_dtypes=sorted({str(p.dtype) for p in parameters.values()}),
                    model_class=type(self.model).__name__, torch_version=torch.__version__)))
            return result

    def _score_token_rows(self, actions, parameters):
        if torch.is_grad_enabled():
            raise ValueError('batched scoring is no-grad only; per-unit gradients require the serial path')
        groups = plan_forwards(actions, self.generation_batch, self.max_context_tokens)
        results = [None]*len(actions)
        for rows in groups:
            for row, result in zip(rows, self._score_group(rows, parameters)):
                results[row['index']] = result
        return tuple(results)

    def score_actions_batch(self, actions, parameters, *, verify_policy=True, return_details=False):
        """No-grad API, prompt/action-major results with all existing ID checks."""
        if torch.is_grad_enabled():
            raise ValueError('batched scoring is no-grad only')
        actions = tuple(actions)
        policy_id = self.identity(parameters)
        for action in actions:
            if action.backend_id != self.backend_id or (verify_policy and action.policy_id != policy_id):
                raise ValueError('generation/scoring backend or policy mismatch')
        if not forward_enabled(self):
            return tuple(self.score_action(a, parameters, verify_policy=verify_policy,
                                          return_details=return_details) for a in actions)
        results = self._score_token_rows(actions, parameters)
        return results if return_details else tuple(r[0] for r in results)

    @contextmanager
    def prefetch_scores(self, actions, parameters):
        """Ephemeral, single-use scores; original checked_score_action journals them.

        This scope never disables gradients in its caller. A gradient-carrying
        score cannot consume detached prefetched values, even at the same policy.
        """
        if getattr(self, '_pending_forward_scores', None) is not None:
            raise AssertionError('nested forward score prefetch')
        if not forward_enabled(self):
            yield
            return
        actions = tuple(actions)
        with torch.no_grad():
            results = self.score_actions_batch(actions, parameters, return_details=True)
        pending = defaultdict(deque)
        for action, result in zip(actions, results):
            pending[score_key(action)].append(result)
        self._pending_forward_scores = self.identity(parameters), pending
        try:
            yield
            if any(pending.values()):
                raise AssertionError('prefetched scores not fully consumed')
        finally:
            self._pending_forward_scores = None

    def _prefetched_score(self, action, parameters):
        identity, pending = self._pending_forward_scores
        if identity != self.identity(parameters) or not pending.get(score_key(action)):
            raise AssertionError('prefetched score prompt/action/policy mismatch or already consumed')
        return pending[score_key(action)].popleft()

    def _source_kl_batch(self, actions, source_parameters, updated_parameters):
        if not actions:
            raise ValueError('frozen train-side source actions required')
        identity = self.identity(source_parameters)
        for action in actions:
            if action.policy_id != identity or action.backend_id != self.backend_id:
                raise ValueError('pilot source policy mismatch')
        _matching(lora_parameters(self.model), updated_parameters)
        groups = plan_forwards(actions, self.generation_batch, self.max_context_tokens)
        head, prefix = self._forward_head()
        values = [None]*len(actions)
        with self.measured('pilot_conditional_kl', forward_passes=2*len(groups), sequences=len(actions),
                prompt_tokens=2*sum(len(a.prompt_ids) for a in actions),
                action_tokens=2*sum(len(a.action_ids) for a in actions)):
            for rows in groups:
                old = self._forward_hidden(rows, source_parameters, head)
                new = self._forward_hidden(rows, updated_parameters, head)
                for row, hp, hq in zip(rows, old, new):
                    chunks = []
                    for start in range(0, len(hp), POSITION_CHUNK):
                        chunks.append(self._kl_chunk(head, prefix, source_parameters, updated_parameters,
                            hp[start:start+POSITION_CHUNK], hq[start:start+POSITION_CHUNK]))
                    values[row['index']] = torch.stack(chunks).sum()
                del old, new
        return torch.stack(values).mean()  # Restore the original action order before averaging.

    def _kl_chunk(self, head, prefix, source, updated, hp, hq):
        old = self._project(head, prefix, source, hp)
        new = self._project(head, prefix, updated, hq)
        dtype = torch.float64 if old.dtype == torch.float64 else torch.float32
        logp, logq = old.to(dtype).log_softmax(-1), new.to(dtype).log_softmax(-1)
        delta = logq-logp
        # Preserve the pilot's cancellation-safe conditional KL expression.
        return (logp.exp()*(torch.expm1(delta)-delta).clamp_min(0)).sum()
