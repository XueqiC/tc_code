"""Stochastic BFCL feedback (including capped actions) and RTD gate VJPs (4)--(6).

The local torch backend generates AND recomputes scores. No vLLM/torch score
substitution, text re-tokenization of sampled actions, or observation scoring.
Model/backend construction is explicit; importing this module launches nothing.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
import copy
import math
import uuid
from types import SimpleNamespace

import torch
from torch.func import functional_call

from ..behavior.deltas import canonical_hash, tensor_state_hash
from ..cc_pairs import thinking_off
from .functional_step import FrozenStep, _matching, gradients, lora_parameters, policy_identity, snapshot
from .transport import Behavior, SourceSample, complete_token_logprobs, positive_mixture_loss, validate_sampled_action
from .scoring import ScoreTolerance, attention_implementation, enforce_score_diagnostic, score_diagnostic
from .bfcl_decode import DECODE_ERRORS, guard_rtd_decoding


class IncompleteRolloutError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionTrace:
    prompt_ids: tuple[int, ...]
    action_ids: tuple[int, ...]
    eos_token_id: int
    text: str
    generation_logprob: float
    backend_id: str
    policy_id: str
    generation_token_logprobs: tuple[float, ...] = ()
    generation_metadata: dict = field(default_factory=dict)
    truncated: bool = False
    malformed: bool = False
    malformed_exception_type: str | None = None
    malformed_stage: str | None = None

    def __post_init__(self):
        if not self.prompt_ids:
            raise ValueError("complete prompt/state required")
        validate_sampled_action(self.action_ids, self.eos_token_id, self.truncated)
        if not math.isfinite(self.generation_logprob) or self.generation_logprob > 0:
            raise ValueError("invalid generation log probability")
        if self.malformed != bool(self.malformed_exception_type):
            raise ValueError('malformed actions require an exception type')


@dataclass(frozen=True)
class TaskRollout:
    task_id: str
    actions: tuple[ActionTrace, ...]
    reward: float
    policy_id: str
    from_task_start: bool = True
    truncated: bool = field(init=False)
    malformed: bool = field(init=False)
    malformed_exception_types: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        if not self.from_task_start or not self.actions or not 0 <= self.reward <= 1:
            raise ValueError("full task-start rollout and official terminal reward required")
        if any(a.policy_id != self.policy_id for a in self.actions):
            raise ValueError("mixed policies in one rollout")
        object.__setattr__(self, 'truncated', any(a.truncated for a in self.actions))
        object.__setattr__(self, 'malformed', any(a.malformed for a in self.actions))
        object.__setattr__(self, 'malformed_exception_types', tuple(sorted({
            a.malformed_exception_type for a in self.actions if a.malformed_exception_type})))
        if self.malformed and self.reward != 0:
            raise ValueError('malformed actions require a failed rollout (reward zero)')


class TorchPolicyBackend:
    """Unwarped categorical autoregressive sampling with differentiable scoring.

    Use an already-loaded eval-mode causal LM and tokenizer. All model weights
    outside LoRA are frozen. The modest token-at-a-time path is also a CPU oracle;
    a future optimized backend must retain this sampling/likelihood contract.
    """
    def __init__(self, model, tokenizer, *, base_checkpoint_hash, harness_hash,
                 tokenizer_hash, max_action_tokens=4096, max_context_tokens=32768,
                 score_tolerance=None, action_caps=None):
        lora_parameters(model)
        if model.training or not tokenizer_hash or max_action_tokens < 1 or max_context_tokens < 2:
            raise ValueError("eval-mode model, tokenizer hash and positive limits required")
        self.model, self.tokenizer = model, tokenizer
        self.base_checkpoint_hash, self.harness_hash = base_checkpoint_hash, harness_hash
        self.max_action_tokens, self.max_context_tokens = max_action_tokens, max_context_tokens
        self.action_caps = dict(action_caps or {})
        if any(type(v) is not int or v < 1 for v in self.action_caps.values()):
            raise ValueError('positive action token caps required')
        self.score_tolerance = score_tolerance or ScoreTolerance(mean_abs=2e-6, max_abs=2e-5)
        self.backend_id = canonical_hash(dict(backend="rtd-torch-categorical-v1", base=base_checkpoint_hash,
            harness=harness_hash, tokenizer=tokenizer_hash, temperature=1., top_p=1.,
            max_action_tokens=max_action_tokens, max_context_tokens=max_context_tokens, action_caps=self.action_caps))

    @contextmanager
    def action_limit(self, category):
        previous = self.max_action_tokens
        self.max_action_tokens = self.action_caps.get(
            'multi_turn' if category.startswith('multi_turn') else 'single_turn', previous)
        try:
            yield
        finally:
            self.max_action_tokens = previous

    def identity(self, parameters):
        return policy_identity(parameters, base_checkpoint_hash=self.base_checkpoint_hash,
                               harness_hash=self.backend_id)

    def _logits(self, parameters, ids):
        if self.model.training:
            raise ValueError("dropout/training mode changes the sampling policy")
        if ids.shape[1] > self.max_context_tokens:
            raise IncompleteRolloutError("context limit exceeded; no semantic truncation allowed")
        return functional_call(self.model, parameters, (), dict(input_ids=ids,
            attention_mask=torch.ones_like(ids), use_cache=False)).logits

    def sample_action(self, prompt, parameters, generator, *, temperature=1., top_p=1.):
        if temperature != 1 or top_p != 1:
            raise ValueError("v1 scores require temperature=1 and top_p=1")
        _matching(lora_parameters(self.model), parameters)
        device = next(iter(parameters.values())).device
        prompt_ids = tuple(self.tokenizer.encode(prompt, add_special_tokens=False))
        if not prompt_ids:
            raise ValueError("empty prompt tokens")
        eos = self.tokenizer.eos_token_id
        if type(eos) is not int:
            raise ValueError("one configured EOS token required")
        from .student import termination_ids
        stops = termination_ids(self)
        action, token_logprobs = [], []
        room = self.max_context_tokens - len(prompt_ids)
        if room < 1:
            raise IncompleteRolloutError("complete prompt exceeds context; no semantic truncation allowed")
        with torch.no_grad():
            for _ in range(min(room, self.max_action_tokens)):
                ids = torch.tensor([prompt_ids + tuple(action)], device=device)
                logits = self._logits(parameters, ids)[0, -1]
                dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
                logps = logits.to(dtype).log_softmax(-1)
                token = int(torch.multinomial(logps.exp(), 1, generator=generator))
                action.append(token)
                token_logprobs.append(float(logps[token]))
                if token in stops:
                    break
        truncated = action[-1] not in stops
        eos = eos if truncated else action[-1]
        return ActionTrace(prompt_ids, tuple(action), eos,
            self.tokenizer.decode(action if truncated else action[:-1], skip_special_tokens=False),
            sum(token_logprobs), self.backend_id, self.identity(parameters), tuple(token_logprobs),
            dict(implementation='torch-functional-autoregressive-categorical-v1', use_cache=False,
                 logits_dtype=str(logits.dtype), logprob_dtype=str(logps.dtype),
                 temperature=temperature, top_p=top_p, top_k=0,
                 max_action_tokens=self.max_action_tokens, effective_action_limit=min(room, self.max_action_tokens)),
            truncated=truncated)

    def score_action(self, action, parameters, *, verify_policy=True, return_details=False):
        if action.backend_id != self.backend_id:
            raise ValueError('generation/scoring backend or policy mismatch: backend_id '
                f'generated={action.backend_id} scoring={self.backend_id}')
        if verify_policy:
            policy_id = self.identity(parameters)
            if action.policy_id != policy_id:
                raise ValueError('generation/scoring backend or policy mismatch: policy_id '
                    f'generated={action.policy_id} scoring={policy_id}')
        return self.score_tokens(action.prompt_ids, action.action_ids, parameters,
                                 eos_token_id=action.eos_token_id, return_details=return_details,
                                 truncated=action.truncated)

    def score_tokens(self, prompt_ids, action_ids, parameters, *, eos_token_id, return_details=False, truncated=False):
        """Teacher/source likelihood at current theta; action IDs are constants."""
        _matching(lora_parameters(self.model), parameters)
        if not prompt_ids:
            raise ValueError('complete prompt/state required')
        validate_sampled_action(action_ids, eos_token_id, truncated)
        device = next(iter(parameters.values())).device
        ids = torch.tensor([tuple(prompt_ids) + tuple(action_ids)], device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, len(prompt_ids):] = True
        logits = self._logits(parameters, ids)
        values = complete_token_logprobs(logits, ids, mask, eos_token_id=eos_token_id, truncated=truncated)
        score = values.sum()
        if return_details:
            return score, values, dict(implementation='torch-functional-teacher-forced-native-ce-v1',
                use_cache=False, cache_type=None, attention=attention_implementation(self.model),
                logits_dtype=str(logits.dtype), logprob_dtype=str(values.dtype),
                reduction_dtype=str(score.dtype), parameter_dtypes=sorted({str(p.dtype) for p in parameters.values()}),
                model_class=type(self.model).__name__, torch_version=torch.__version__)
        return score

    def checked_score_action(self, action, parameters, *, expected_prompt_ids=None, record=None,
                             score_atol=None, score_rtol=None):
        # This exact tensor, from score_tokens' native CE, also supplies REINFORCE.
        score, values, metadata = self.score_action(action, parameters, return_details=True)
        diagnostic = score_diagnostic(action, values, score, metadata, self.score_tolerance,
            expected_prompt_ids=expected_prompt_ids, score_atol=score_atol, score_rtol=score_rtol)
        if record is not None:
            record(diagnostic)
        elif getattr(self, 'journal', None) is not None:
            self.journal.append('score_consistency', context=self.context, **diagnostic)
        enforce_score_diagnostic(diagnostic)
        return score, diagnostic

    def score_source(self, source: SourceSample, parameters):
        prompt = self.tokenizer.encode(source.behavior.state.prompt, add_special_tokens=False)
        return self.score_tokens(prompt, source.token_ids, parameters, eos_token_id=source.eos_token_id,
                                 truncated=source.truncated)

    def score_behavior(self, behavior: Behavior, parameters):
        prompt = self.tokenizer.encode(behavior.state.prompt, add_special_tokens=False)
        action = list(self.tokenizer.encode(behavior.text, add_special_tokens=False))
        eos = self.tokenizer.eos_token_id
        from .student import termination_ids
        # Authored native targets already include their turn/handoff terminator.
        if action and action[-1] in termination_ids(self):
            eos = action[-1]
        if not action or action[-1] != eos:
            action.append(eos)
        return self.score_tokens(prompt, action, parameters, eos_token_id=eos)

    def initial_hidden(self, prompt, initial_parameters, *, initial_snapshot_id):
        if self.identity(initial_parameters) != initial_snapshot_id or self.model.training:
            raise ValueError("hidden projection requires the frozen initial eval policy")
        ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=False)],
                           device=next(iter(initial_parameters.values())).device)
        if not ids.numel() or ids.shape[1] > self.max_context_tokens:
            raise IncompleteRolloutError("initial hidden state needs the complete prompt")
        with torch.no_grad():
            output = functional_call(self.model, initial_parameters, (), dict(input_ids=ids,
                attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True))
        return output.hidden_states[-1][0, -1].detach().clone()

    def source_sampler(self, frozen_parameters):
        frozen = snapshot(frozen_parameters)
        identity = self.identity(frozen)
        def sample(request, generator):
            from .generation_batch import sample_actions
            if request.frozen_snapshot_id != identity:
                raise ValueError("source snapshot mismatch")
            result = []
            for action in sample_actions(self, request.state.prompt, request.samples, frozen, generator):
                result.append(SourceSample(Behavior(request.state, action.text), identity,
                    action.action_ids, action.eos_token_id, action.generation_logprob, truncated=action.truncated))
            return tuple(result)
        return sample

    def source_kl(self, source_actions, source_parameters, updated_parameters):
        """Train-source conditional forward KL, summed over sampled action tokens."""
        if not source_actions:
            raise ValueError("frozen train-side source actions required")
        values = []
        for action in source_actions:
            values.append(self._action_kl(action, source_parameters, updated_parameters))
        return torch.stack(values).mean()

    def _action_kl(self, action, source_parameters, updated_parameters):
        # Scope vocabulary tensors to one action. The pilot runs under no_grad,
        # retaining only scalar values for the identical ordered stack/mean.
        if action.policy_id != self.identity(source_parameters) or action.backend_id != self.backend_id:
            raise ValueError("pilot source policy mismatch")
        ids = torch.tensor([action.prompt_ids + action.action_ids],
                           device=next(iter(source_parameters.values())).device)
        sl = slice(len(action.prompt_ids) - 1, ids.shape[1] - 1)
        old = self._logits(source_parameters, ids)[0, sl].detach()
        new = self._logits(updated_parameters, ids)[0, sl]
        dtype = torch.float64 if old.dtype == torch.float64 else torch.float32
        logp, logq = old.to(dtype).log_softmax(-1), new.to(dtype).log_softmax(-1)
        # E_p[exp(d)-1-d] equals KL(p||q); expm1 avoids cancellation.
        delta = logq - logp
        return (logp.exp() * (torch.expm1(delta) - delta).clamp_min(0)).sum()


def bfcl_task_rollout(entry, category, truth, backend, parameters, generator, *, checker=None):
    """Run the configured student BFCL harness from a fresh copy of the task start.

    The model query uses the local scoreable backend; RTD-only decoding guards
    preserve malformed samples as failed actions. Multi-turn observations and
    execution stay in BFCL, as does the official campaign's handling.
    Missing environments/checkers raise; infrastructure errors are never rewards.
    """
    from ..adapters.bfcl import BFCLAdapter
    # _handler adds the repository's vendored BFCL path. It does not start a server.
    config = getattr(backend, 'student_config', {})
    from .bfcl_decode import student_handler
    handler = student_handler(config, getattr(backend, 'tokenizer', None))
    from bfcl_eval.utils import contain_multi_turn_interaction, populate_test_cases_with_predefined_functions
    from bfcl_eval.constants.enums import ReturnFormat
    from types import MethodType

    if category.startswith(("memory", "web_search")):
        raise ValueError("BFCL agentic memory/web-search requires its own initial-state/checker adapter")
    task = populate_test_cases_with_predefined_functions([copy.deepcopy(entry)])[0]
    actions = []
    def record_malformed(exc, stage):
        if not actions:
            raise exc  # no sampled action: this is not a policy failure
        if not actions[-1].malformed:
            actions[-1] = replace(actions[-1], malformed=True,
                malformed_exception_type=type(exc).__name__, malformed_stage=stage)
    guard_rtd_decoding(handler, record_malformed, student_call_format=config.get('student_call_format', 'qwen'))
    def query(_handler, inference_data):
        prompt = _handler._format_prompt(inference_data["message"], inference_data["function"])
        if config.get("student_call_format", "qwen") == "qwen":
            prompt = thinking_off(prompt)
        with backend.action_limit(category) if hasattr(backend, 'action_limit') else nullcontext():
            action = backend.sample_action(prompt, parameters, generator, temperature=1., top_p=1.)
        actions.append(action)
        inference_data["inference_input_log"] = {"formatted_prompt": prompt}
        response = SimpleNamespace(choices=[SimpleNamespace(text=action.text)],
            usage=SimpleNamespace(prompt_tokens=len(action.prompt_ids), completion_tokens=len(action.action_ids)))
        return response, 0.
    handler.temperature = 1.
    handler._query_prompting = MethodType(query, handler)
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    from tools.behavior_atom.checker_bridge import CheckerBridge
    # BFCL caches simulators in module globals. A copied task alone does not
    # reset the environment; unique run identity plus cleanup does.
    handler.model_name_underline_replaced += "_rtd_" + uuid.uuid4().hex
    prefix = handler.model_name_underline_replaced + "_"
    try:
        result, _ = handler.inference(copy.deepcopy(task), include_input_log=True, exclude_state_log=True)
    finally:
        for key in list(vars(multi_turn_utils)):
            if key.startswith(prefix) and key.endswith("_instance"):
                delattr(multi_turn_utils, key)
    owns_checker = checker is None
    checker = checker or (CheckerBridge(config['student']+'-FC',
        student_call_format=config.get('student_call_format', 'qwen')) if config.get('student') else CheckerBridge())
    try:
        if contain_multi_turn_interaction(task["id"]):
            verdict = checker.check_multi_turn(task, result, truth, category)
        elif "relevance" in category:
            if 'irrelevance' not in category:
                try:
                    if not handler.decode_ast(result, language=ReturnFormat.PYTHON, has_tool_call_tag=False):
                        raise ValueError('model returned no call for a required action')
                except DECODE_ERRORS as exc:
                    record_malformed(exc, 'decode_ast')
            verdict = checker.check_relevance(task, result, category)
        else:
            language = ReturnFormat.JAVA if "java" in category and "javascript" not in category else (
                ReturnFormat.JAVASCRIPT if "javascript" in category else ReturnFormat.PYTHON)
            try:
                calls = handler.decode_ast(result, language=language, has_tool_call_tag=False)
                if not calls and truth:
                    raise ValueError('model returned no call for a required action')
            except DECODE_ERRORS as exc:
                record_malformed(exc, 'decode_ast')
                calls = []  # same official invalid-action decoding behavior
            verdict = checker.check(dict(function=task["function"], truth=truth, category=category,
                expected="abstain" if not truth else "call"), calls)
    finally:
        if owns_checker:
            checker.close()
    if type(verdict.get("valid")) is not bool:
        raise RuntimeError("official checker failed to return a terminal verdict")
    reward = 0. if any(a.malformed for a in actions) else float(verdict['valid'])
    return TaskRollout(task["id"], tuple(actions), reward, backend.identity(parameters))


def collect_feedback(tasks, rollout, *, feedback_parent_hashes, rollouts_per_task=2):
    """The runner chooses up to four legal feedback parents before this call."""
    if rollouts_per_task < 2:
        raise ValueError("leave-one-out needs at least two same-task rollouts")
    results = []
    seen = set()
    for task_id, parent in tasks:
        if parent not in feedback_parent_hashes or task_id in seen:
            raise ValueError("illegal or duplicate feedback task")
        seen.add(task_id)
        for _ in range(rollouts_per_task):
            result = rollout(task_id)
            if result.task_id != task_id:
                raise ValueError("feedback task mismatch")
            results.append(result)
    return tuple(results)


def leave_one_out(task_ids, rewards):
    rewards = torch.as_tensor(rewards).detach()
    if rewards.ndim != 1 or len(task_ids) != len(rewards) or not torch.isfinite(rewards).all():
        raise ValueError("finite task-aligned rewards required")
    counts = Counter(task_ids)
    if any(n < 2 for n in counts.values()):
        raise ValueError("each task needs two independent rollouts for LOO")
    return torch.stack([(sum(rewards[j] for j, t in enumerate(task_ids) if t == task) - rewards[i]) /
                        (counts[task] - 1) for i, task in enumerate(task_ids)]).detach()


@dataclass(frozen=True)
class ReturnGradient:
    gradient: dict[str, torch.Tensor]
    parameter_hash: str
    metadata: dict

    def __post_init__(self):
        if not self.parameter_hash or any(not torch.isfinite(g).all() for g in self.gradient.values()):
            raise ValueError("finite return gradient and model hash required")
        object.__setattr__(self, "gradient", {n: g.detach().clone() for n, g in self.gradient.items()})


def reinforce_gradient(rollouts, backend, parameters, *, score_atol=None, score_rtol=None,
                       baseline='leave_one_out_same_task', diagnostic_record=None, trajectory_scores=None):
    if not rollouts:
        raise ValueError("complete feedback rollouts required")
    rewards = next(iter(parameters.values())).new_tensor([r.reward for r in rollouts]).detach()
    if baseline not in {'leave_one_out_same_task', 'smoke_zero'}:
        raise ValueError('unknown feedback baseline')
    baselines = (torch.zeros_like(rewards) if baseline == 'smoke_zero' else
                 leave_one_out([r.task_id for r in rollouts], rewards))
    advantages = (rewards - baselines).detach()
    estimate = {n: torch.zeros_like(p) for n, p in parameters.items()}
    discrepancies, diagnostics, tokens = [], [], 0
    # One action graph at a time; both trajectory actions and observations are
    # fixed. Weight is per ROLLOUT, never per action/token or success subset.
    for rollout, advantage in zip(rollouts, advantages):
        if trajectory_scores is not None:
            trajectory_score = {n: torch.zeros_like(p, device='cpu') for n, p in parameters.items()}
        for action_index, action in enumerate(rollout.actions):
            record = (None if diagnostic_record is None else
                      lambda d: diagnostic_record(rollout, action_index, d))
            score, diagnostic = backend.checked_score_action(action, parameters, record=record,
                score_atol=score_atol, score_rtol=score_rtol)
            discrepancies.append(diagnostic['sequence_abs_difference'])
            diagnostics.append({k: diagnostic[k] for k in ('state_hash', 'mean_abs_difference',
                'max_abs_difference', 'tolerance', 'passed', 'generation_backend', 'scoring_backend')})
            gradient = gradients(score, parameters)
            if trajectory_scores is not None:
                for n in trajectory_score:
                    trajectory_score[n].add_(gradient[n].detach().cpu())
            for n in estimate:
                estimate[n].add_(gradient[n].detach() * advantage / len(rollouts))
            del score, gradient  # release this action before scoring the next
            tokens += len(action.action_ids)
        if trajectory_scores is not None:
            trajectory_scores.append(trajectory_score)
    return ReturnGradient(estimate, tensor_state_hash(parameters), dict(
        rollouts=len(rollouts), tasks=len(set(r.task_id for r in rollouts)), action_tokens=tokens,
        truncated_rollouts=sum(r.truncated for r in rollouts),
        truncated_actions=sum(a.truncated for r in rollouts for a in r.actions),
        malformed_rollouts=sum(r.malformed for r in rollouts),
        malformed_actions=sum(a.malformed for r in rollouts for a in r.actions),
        malformed_exception_types=dict(Counter(a.malformed_exception_type
            for r in rollouts for a in r.actions if a.malformed)),
        rewards=rewards.tolist(), baselines=baselines.tolist(), baseline=baseline,
        score_backend=backend.backend_id, generation_backend=backend.backend_id,
        score_consistency=diagnostics,
        max_score_discrepancy=max(discrepancies), identifiable=bool(advantages.ne(0).any())))


def gate_vjp(loss, parameters, phi, step: FrozenStep, return_gradient):
    """(d theta+/d phi)^T stop(gJ), without demonstrations-by-parameters storage."""
    _matching(parameters, return_gradient)
    _matching(parameters, step.diagonal)
    inner = gradients(loss, parameters, create_graph=True)
    contraction = sum((g * step.diagonal[n].detach() * return_gradient[n].detach()).sum()
                      for n, g in inner.items())
    if not contraction.requires_grad:
        return torch.zeros_like(phi)
    value = torch.autograd.grad(contraction, phi, allow_unused=True)[0]
    return torch.zeros_like(phi) if value is None else (-step.eta * value).detach()


def actual_gate_vjp(loss, parameters, phi, step, updated, feedback: ReturnGradient):
    if feedback.parameter_hash != tensor_state_hash(updated):
        raise ValueError("gate requires return gradient at the actual updated model")
    return gate_vjp(loss, parameters, phi, step, feedback.gradient)


def source_rollout_gradients(rollouts, backend, source_parameters, *, parent_by_task, inner_parent_hashes):
    """Unrewarded train-source score gradients for the frozen round RMS P."""
    for rollout in rollouts:
        parent = parent_by_task[rollout.task_id]
        if parent not in inner_parent_hashes:
            raise ValueError("source rollout outside the inner fold")
        loss = -sum(backend.score_action(a, source_parameters) for a in rollout.actions)
        yield parent, {n: g.detach() for n, g in gradients(loss, source_parameters).items()}


def freeze_slot_targets(slots, generator):
    """Choose cached teacher texts once, uniformly, before differentiating a block."""
    return tuple((slot.source, slot.sample_teacher(generator)) for slot in slots)


def loss_on_slots(targets, chi, phi, backend, parameters):
    """Current-theta positive mixture; frozen source IDs/features/teacher draw."""
    if not targets or chi.shape != (len(targets), phi.numel()):
        raise ValueError("one frozen feature vector per complete source slot required")
    source_scores, teacher_scores, available = [], [], []
    for source, teacher in targets:
        source_scores.append(backend.score_source(source, parameters))
        if teacher is not None:
            source.behavior.state.assert_matches(teacher.state)
        teacher_scores.append(backend.score_behavior(teacher, parameters) if teacher is not None
                              else source_scores[-1].new_zeros(()))
        available.append(teacher is not None)
    scores = torch.stack(source_scores)
    return positive_mixture_loss(scores, torch.stack(teacher_scores), (chi.detach() @ phi).sigmoid(),
        teacher_available=torch.tensor(available, device=scores.device, dtype=torch.bool))


class GateController:
    """Ascent (6); running RMS is computed from PAST feedback blocks only."""
    def __init__(self, *, learning_rate=0.1, ridge=1., numerical_epsilon=1e-8):
        if learning_rate <= 0 or ridge < 0 or numerical_epsilon <= 0:
            raise ValueError("invalid gate update settings")
        self.learning_rate, self.ridge, self.epsilon = learning_rate, ridge, numerical_epsilon
        self.past_sum_squares, self.past_coordinates = 0., 0

    def update(self, phi, vjp):
        if phi.shape != vjp.shape or not torch.isfinite(vjp).all():
            raise ValueError("finite gate VJP required")
        scale = max(math.sqrt(self.past_sum_squares / self.past_coordinates)
                    if self.past_coordinates else 0., self.epsilon)
        normalized = vjp.detach() / scale  # frozen scalar normalization
        updated = phi.detach() + self.learning_rate * (normalized - self.ridge * phi.detach())
        self.past_sum_squares += float(vjp.detach().double().square().sum())
        self.past_coordinates += vjp.numel()
        return updated.requires_grad_(True), dict(past_rms_scale=scale)
