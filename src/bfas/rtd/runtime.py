"""Opt-in local HF generation and streamed complete-action RTD gradients."""
from contextlib import contextmanager
import time

import torch

from ..behavior.deltas import tensor_state_hash
from .functional_step import _matching, gradients, lora_parameters, snapshot
from .checkpointing import enable_gradient_checkpointing
from .return_gradient import ActionTrace, IncompleteRolloutError, TorchPolicyBackend
from .transport import positive_mixture_loss
from .persistence import digest
from .scoring import ScoreTolerance, attention_implementation
from .memory import MemoryPolicy, memory_batches
from .generation_batch import GenerationBatch, HFGenerationBatchMixin, RNG_RULE
from .forward_batch import HFForwardBatchMixin, forward_enabled, token_row


@contextmanager
def installed_parameters(model, parameters):
    """HF generate uses module parameters; restore the actual student even on error.

    Only used between gradient graphs. Functional scoring never uses this context.
    """
    actual = lora_parameters(model)
    saved = snapshot(actual)
    if tuple(actual) != tuple(parameters):
        raise ValueError('generation parameter layout mismatch')
    with torch.no_grad():
        for n, p in actual.items():
            p.copy_(parameters[n])
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in actual.items():
                p.copy_(saved[n])


class HFGenerateBackend(HFForwardBatchMixin, HFGenerationBatchMixin, TorchPolicyBackend):
    """HF generate with KV cache, no warpers, and CE over all sampled tokens.

    BF16 logits are sampled in FP32 by HF. Return scores are recorded during
    generation and independently checked by teacher forcing. Single GPU only.
    """
    def __init__(self, *args, journal=None, memory_policy=None, generation_batch=None, **kwargs):
        kwargs.setdefault('score_tolerance', ScoreTolerance())
        super().__init__(*args, **kwargs)
        self.backend_id = digest(dict(parent=self.backend_id, implementation='hf-generate-kv-v1',
                                      top_k=0, repetition_penalty=1., attention=attention_implementation(self.model)))
        self.journal = journal
        self.context = 'unspecified'
        self.memory_policy = memory_policy or MemoryPolicy()
        self.generation_batch = generation_batch
        if generation_batch is not None:
            import transformers
            self.backend_id = digest(dict(parent=self.backend_id, generation_batch=generation_batch.sampling_config(),
                rng_rule=RNG_RULE, transformers_version=transformers.__version__, torch_version=torch.__version__))

    def batches(self, items, operation):
        device = next(iter(lora_parameters(self.model).values())).device
        return memory_batches(items, device, self.memory_policy,
            record=(lambda row: self.journal.append('memory_batch', operation=operation,
                    context=self.context, **row)) if self.journal else None)

    @contextmanager
    def measured(self, operation, **counts):
        if self.journal is None:
            yield
        else:
            with self.journal.measure_phase(operation, context=self.context, **counts):
                yield

    def sample_action(self, prompt, parameters, generator, *, temperature=1., top_p=1.):
        if self.generation_batch is not None:
            return self.sample_actions(prompt, 1, parameters, generator, temperature=temperature, top_p=top_p)[0]
        from transformers import GenerationConfig
        if self.model.training or temperature != 1 or top_p != 1:
            raise ValueError('frozen eval policy with temperature=1/top_p=1 required')
        device = next(iter(parameters.values())).device
        prompt_ids = tuple(self.tokenizer.encode(prompt, add_special_tokens=False))
        room = self.max_context_tokens - len(prompt_ids)
        if not prompt_ids or room < 1:
            raise IncompleteRolloutError('complete prompt exceeds context; no truncation')
        eos = self.tokenizer.eos_token_id
        from .student import termination_ids
        stops = termination_ids(self)
        if type(eos) is not int:
            raise ValueError('one configured EOS token required')
        settings = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
            typical_p=1., repetition_penalty=1., num_beams=1, num_return_sequences=1,
            max_new_tokens=min(room, self.max_action_tokens), eos_token_id=list(stops),
            pad_token_id=eos, bos_token_id=self.tokenizer.bos_token_id,
            use_cache=True, return_dict_in_generate=True, output_scores=True)
        devices = [device.index or 0] if device.type == 'cuda' else []
        identity = self.identity(parameters)
        logits_dtypes = set()
        # Observe dtype only: retaining all raw vocabulary logits would double
        # generation memory. PEFT.generate delegates to the base causal LM.
        generation_model = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        def observe_logits(module, inputs, output):
            logits_dtypes.add(str(output.logits.dtype))
        with self.measured('generation', prompt_tokens=len(prompt_ids)), installed_parameters(self.model, parameters):
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.set_rng_state(generator.get_state().cpu(), device)
                else:
                    torch.set_rng_state(generator.get_state().cpu())
                hook = generation_model.register_forward_hook(observe_logits)
                try:
                    output = self.model.generate(input_ids=torch.tensor([prompt_ids], device=device),
                        attention_mask=torch.ones((1, len(prompt_ids)), device=device, dtype=torch.long),
                        generation_config=settings)
                finally:
                    hook.remove()
                generator.set_state((torch.cuda.get_rng_state(device) if devices else torch.get_rng_state()).cpu())
            ids = tuple(output.sequences[0, len(prompt_ids):].tolist())
            truncated = bool(ids and ids[-1] not in stops)
            eos = eos if truncated or not ids else ids[-1]
            if self.journal:
                self.journal.append('generated_tokens', context=self.context, action_tokens=len(ids),
                                    complete=bool(ids and ids[-1] == eos), truncated=truncated, policy_id=identity)
            if not ids or eos in ids[:-1] or (truncated and len(ids) != settings.max_new_tokens):
                raise ValueError('malformed generation: expected first EOS or configured action limit')
            if len(output.scores) != len(ids):
                raise ValueError('HF generation scores do not cover complete sampled action')
            token_logprobs = tuple(float(scores[0].to(torch.float64 if scores.dtype == torch.float64 else torch.float32)
                                .log_softmax(-1)[token]) for token, scores in zip(ids, output.scores))
        import transformers
        cache = getattr(output, 'past_key_values', None)
        return ActionTrace(prompt_ids, ids, eos, self.tokenizer.decode(ids if truncated else ids[:-1], skip_special_tokens=False),
            sum(token_logprobs), self.backend_id, identity, token_logprobs,
            dict(implementation='hf-generate-kv-categorical-v1', use_cache=True,
                 logits_dtypes=sorted(logits_dtypes), scores_dtypes=sorted({str(s.dtype) for s in output.scores}),
                 logprob_dtype='torch.float64' if output.scores[0].dtype == torch.float64 else 'torch.float32',
                 reduction_dtype='python.float', parameter_dtypes=sorted({str(p.dtype) for p in parameters.values()}),
                 model_class=type(generation_model).__name__, torch_version=torch.__version__,
                 transformers_version=transformers.__version__, attention=attention_implementation(self.model),
                 cache_type=type(cache).__name__ if cache is not None else None,
                 temperature=temperature, top_p=top_p, top_k=0, repetition_penalty=1.,
                 max_action_tokens=self.max_action_tokens, effective_action_limit=settings.max_new_tokens),
            truncated=truncated)

    def score_tokens(self, prompt_ids, action_ids, parameters, *, eos_token_id, return_details=False, truncated=False):
        if forward_enabled(self) and not torch.is_grad_enabled():
            _matching(lora_parameters(self.model), parameters)
            if self.model.training:
                raise ValueError('dropout/training mode changes the sampling policy')
            action = token_row(prompt_ids, action_ids, eos_token_id, truncated)
            result = (self._prefetched_score(action, parameters)
                      if getattr(self, '_pending_forward_scores', None) is not None else
                      self._score_token_rows((action,), parameters)[0])
            return result if return_details else result[0]
        with self.measured('teacher_forced_forward', prompt_tokens=len(prompt_ids), action_tokens=len(action_ids)):
            return super().score_tokens(prompt_ids, action_ids, parameters,
                                        eos_token_id=eos_token_id, return_details=return_details, truncated=truncated)

    def initial_hidden(self, prompt, initial_parameters, *, initial_snapshot_id):
        with self.measured('initial_hidden', prompt_tokens=len(self.tokenizer.encode(prompt, add_special_tokens=False))):
            return super().initial_hidden(prompt, initial_parameters, initial_snapshot_id=initial_snapshot_id)

    def source_kl(self, source_actions, source_parameters, updated_parameters):
        if forward_enabled(self) and not torch.is_grad_enabled():
            return self._source_kl_batch(tuple(source_actions), source_parameters, updated_parameters)
        with self.measured('pilot_conditional_kl', forward_passes=2*len(source_actions),
            prompt_tokens=2*sum(len(a.prompt_ids) for a in source_actions),
            action_tokens=2*sum(len(a.action_ids) for a in source_actions)):
            def actions():
                for batch in self.batches(source_actions, 'pilot_conditional_kl'):
                    yield from batch
            return super().source_kl(actions(), source_parameters, updated_parameters)


def slot_loss(target, chi, phi, backend, parameters, *, gate='linear_sigmoid'):
    source, teacher = target
    a = (chi.detach() @ phi).sigmoid()
    if gate == 'fixed_half':
        a = a * 0 + .5
    elif gate == 'teacher_only':
        a = a * 0 + 1.
    if teacher is not None:
        source.behavior.state.assert_matches(teacher.state)
    source_score = backend.score_source(source, parameters).reshape(1)
    teacher_score = None if teacher is None else backend.score_behavior(teacher, parameters).reshape(1)
    return positive_mixture_loss(source_score, teacher_score, a.reshape(1))


def streamed_gradient(targets, chi, phi, backend, parameters, *, gate='linear_sigmoid',
                      source_estimator='hard2', source_parameters=None, cv_cs_mode='loo',
                      source_controls=None, diagnostic_gradients=None):
    """One complete-action graph at a time, including the two sides of a slot.

    Retain only detached LoRA gradients. Weight each side before backward (as
    in slot_loss), then accumulate the slot mean in parameter precision.
    """
    if source_estimator != 'hard2':
        return _estimated_gradient(targets, chi, phi, backend, parameters, gate=gate,
            source_estimator=source_estimator, source_parameters=source_parameters,
            cv_cs_mode=cv_cs_mode, source_controls=source_controls, diagnostic_gradients=diagnostic_gradients)
    if len(targets) != len(chi) or not targets:
        raise ValueError('aligned nonempty slots required')
    _matching(lora_parameters(backend.model), parameters)
    result = {n: torch.zeros_like(p) for n, p in parameters.items()}
    for (source, teacher), features in zip(targets, chi):
        a = (features.detach() @ phi.detach()).sigmoid()
        if gate == 'fixed_half':
            a = a * 0 + .5
        elif gate == 'teacher_only':
            a = a * 0 + 1.
        if teacher is not None:
            source.behavior.state.assert_matches(teacher.state)
        # Function boundaries release scores and their saved activations before
        # starting the next side. There is no create_graph/retain_graph here.
        g = _side_gradient(backend.score_source(source, parameters), parameters,
                           1-a if teacher is not None else a*0+1)
        if teacher is not None:
            teacher_g = _side_gradient(backend.score_behavior(teacher, parameters), parameters, a)
            for n in g:
                g[n].add_(teacher_g[n])
            del teacher_g
        if getattr(backend, 'journal', None):
            backend.journal.append('gradient_evaluation', context=backend.context, slots=1,
                                   reduction_weight=1/len(targets), create_graph=False)
        for n in result:
            result[n].add_(g[n] / len(targets))
        del g
    return result


def _side_gradient(score, parameters, weight):
    # Keep the same full-action likelihood and validation as the loss oracle.
    loss = positive_mixture_loss(score.reshape(1), None, weight.reshape(1)) * weight
    return {n: g.detach() for n, g in gradients(loss, parameters).items()}


def streamed_gate_vjp(targets, chi, phi, backend, parameters, step, feedback, *, gate='linear_sigmoid',
                      source_estimator='hard2', source_parameters=None, cv_cs_mode='loo', source_controls=None):
    """Exact equation (5) contracted with the frozen sigmoid feature Jacobian.

    Phi only weights the loss, never the LM. Thus d_phi g_theta is exactly
    a(1-a)*chi*(gT-gS). First-order LM gradients suffice; no backward through
    fused attention/linear-attention backward kernels is needed. This remains
    matrix-free and agrees with the general autograd gate_vjp oracle.
    """
    if source_estimator != 'hard2':
        return _estimated_gate_vjp(targets, chi, phi, backend, parameters, step, feedback, gate=gate,
            source_estimator=source_estimator, source_parameters=source_parameters,
            cv_cs_mode=cv_cs_mode, source_controls=source_controls)
    value = torch.zeros_like(phi)
    if len(targets) != len(chi) or not targets:
        raise ValueError('aligned nonempty slots required')
    _matching(lora_parameters(backend.model), parameters)
    if gate not in {'linear_sigmoid', 'scalar_sigmoid'}:
        return value
    for target, features in zip(targets, chi):
        source, teacher = target
        if teacher is None:
            continue
        source.behavior.state.assert_matches(teacher.state)
        one = phi.new_ones(())
        source_g = _side_gradient(backend.score_source(source, parameters), parameters, one)
        difference = _side_gradient(backend.score_behavior(teacher, parameters), parameters, one)
        for n in difference:
            difference[n].sub_(source_g[n])  # gT - gS for negative log likelihood
        del source_g
        contraction = sum((difference[n]*step.diagonal[n].detach()*feedback.gradient[n].detach()).sum()
                          for n in parameters)
        del difference
        a = (features.detach() @ phi.detach()).sigmoid()
        value.add_(-step.eta*contraction*a*(1-a)*features.detach()/len(targets))
        if getattr(backend, 'journal', None):
            backend.journal.append('gate_vjp_evaluation', context=backend.context, slots=1,
                                   reduction_weight=1/len(targets), create_graph=False,
                                   implementation='exact frozen-sigmoid Jacobian contraction')
    return value.detach()


def _estimator_sides(targets, backend, parameters, source_parameters):
    from .source_scoring import source_gradient_pair
    if source_parameters is None:
        raise ValueError('frozen source_parameters required for soft/CV scoring')
    for source, teacher in targets:
        hard, soft, metadata = source_gradient_pair(backend, source, parameters, source_parameters)
        teacher_g = ({n: torch.zeros_like(p) for n, p in parameters.items()} if teacher is None else
                     _side_gradient(backend.score_behavior(teacher, parameters), parameters,
                                    next(iter(parameters.values())).new_ones(())))
        yield hard, soft, teacher_g, metadata


def _estimated_gradient(targets, chi, phi, backend, parameters, *, gate, source_estimator,
                        source_parameters, cv_cs_mode, source_controls, diagnostic_gradients):
    from .source_estimator import estimator_coefficients, gradient_norm
    weights, cs = estimator_coefficients(targets, chi, phi.detach(), gate=gate,
        source_estimator=source_estimator, cs_mode=cv_cs_mode, controls=source_controls)
    result = {n: torch.zeros_like(p) for n, p in parameters.items()}
    totals = ({key: {n: torch.zeros_like(p) for n, p in parameters.items()} for key in ('hard2', 'soft', 'cv')}
              if diagnostic_gradients is not None else None)
    for i, (hard, soft, teacher, metadata) in enumerate(_estimator_sides(targets, backend, parameters, source_parameters)):
        h, c, a = weights[i]
        actual = {n: h*hard[n] + c*soft[n] + a*teacher[n] for n in parameters}
        comparisons = {'hard2': {n: (1-a)*hard[n]+a*teacher[n] for n in parameters},
                       'soft': {n: (1-a)*soft[n]+a*teacher[n] for n in parameters}, 'cv': actual}
        for n in parameters:
            result[n].add_(actual[n] / len(targets))
        if totals is not None:
            for key in totals:
                for n in parameters:
                    totals[key][n].add_(comparisons[key][n] / len(targets))
        if getattr(backend, 'journal', None):
            backend.journal.append('source_estimator_slot', context=backend.context, slot_index=i,
                source_estimator=source_estimator, cv_cs_mode=cv_cs_mode, c_s=float(cs[i]),
                state_hash=targets[i][0].behavior.state.state_hash, source_id=targets[i][0].frozen_snapshot_id,
                hard_gradient_norm=gradient_norm(comparisons['hard2']),
                soft_gradient_norm=gradient_norm(comparisons['soft']), cv_gradient_norm=gradient_norm(actual),
                hard_source_gradient_norm=gradient_norm(hard), soft_source_gradient_norm=gradient_norm(soft),
                reduction_weight=1/len(targets), **metadata)
    if diagnostic_gradients is not None:
        diagnostic_gradients.append(totals)
    return result


def _estimated_gate_vjp(targets, chi, phi, backend, parameters, step, feedback, *, gate,
                       source_estimator, source_parameters, cv_cs_mode, source_controls):
    """Differentiate the NEW update weights, including auxiliary c_s gates.

    LM-side gradients are independent of phi. Contract them first, then use
    autograd on the small coefficient graph; no LM second derivatives needed.
    This also includes cross-draw LOO terms absent from the v1 VJP.
    """
    from .source_estimator import estimator_coefficients
    control = phi.detach().clone().requires_grad_(True)
    weights, _ = estimator_coefficients(targets, chi, control, gate=gate,
        source_estimator=source_estimator, cs_mode=cv_cs_mode, controls=source_controls)
    contractions = []
    for hard, soft, teacher, _ in _estimator_sides(targets, backend, parameters, source_parameters):
        # Match FrozenStep.update's eta*P multiplication before promotion to
        # the gradient dtype (also matters for mixed-precision CPU oracles).
        contractions.append(torch.stack([sum((g[n]*(step.eta*step.diagonal[n].detach())*
                                              feedback.gradient[n].detach()).sum()
                                              for n in parameters) for g in (hard, soft, teacher)]))
    outer = -(weights * torch.stack(contractions).detach()).sum() / len(targets)
    value, = torch.autograd.grad(outer, control)
    if getattr(backend, 'journal', None):
        backend.journal.append('gate_vjp_evaluation', context=backend.context, slots=len(targets),
            source_estimator=source_estimator, cv_cs_mode=cv_cs_mode, create_graph=False,
            implementation='v1.1 autograd through estimator weights including c_s')
    return value.detach()


def backend_config(config):
    """Construct the exact CPU-side backend settings before touching CUDA.

    Unified manifests retain their P0/P1 identity; D12--D16 execute with the
    frozen v1.1 adapter, just like P1Experiment and make_manifest.
    """
    from peft import LoraConfig
    if config.get('method') == 'rtd_unified':
        from .unified.config import runtime_config
        config = runtime_config(config)
    settings = dict(score_tolerance=ScoreTolerance.from_config(config),
        max_action_tokens=config.get('max_action_tokens', 512), max_context_tokens=config['max_context_tokens'],
        action_caps=config.get('max_action_tokens_by_benchmark', {}).get(config['benchmark'],
            {'single_turn': 512, 'multi_turn': 1024}
            if config['benchmark'] == 'bfcl' and 'max_action_tokens' not in config else {}),
        memory_policy=MemoryPolicy.from_config(config), generation_batch=GenerationBatch.from_config(config))
    if settings['max_context_tokens'] < 2:
        raise ValueError('backend requires max_context_tokens >= 2')
    lora = LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'],
        lora_dropout=0., bias='none', task_type='CAUSAL_LM', target_modules=config['lora_target_modules'])
    return config, lora, settings


def load_tokenizer(manifest):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True, trust_remote_code=False)


def load_backend(config, manifest, journal):
    """Called only by run/smoke/resume, never by CPU audit/report/tests."""
    config, lora, settings = backend_config(config)
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('set CUDA_VISIBLE_DEVICES to exactly one available GPU')
    device = torch.device('cuda:0')  # relative to the verbatim inherited visibility
    torch.cuda.set_device(device)
    torch.manual_seed(config['training_seed'])
    torch.cuda.manual_seed_all(config['training_seed'])
    with journal.measure('model_load'):
        tokenizer = load_tokenizer(manifest)
        model = AutoModelForCausalLM.from_pretrained(manifest['model_path'], local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='eager').to(device)
        model = get_peft_model(model, lora)
        # Keep update coordinates in FP32; base matmuls remain BF16. Eager
        # attention gives one explicit scoring implementation on both machines.
        for p in lora_parameters(model).values():
            p.data = p.data.float()
        model.eval()
        checkpoint_layers = enable_gradient_checkpointing(model)
        journal.append('gradient_memory_policy', logical_device=str(device),
            gradient_checkpointing='non_reentrant_eval_functional_lora', checkpoint_layers=checkpoint_layers,
            max_live_action_graphs=1, parameter_space='lora_trainables',
            trainable_numel=sum(p.numel() for p in lora_parameters(model).values()))
    backend = HFGenerateBackend(model, tokenizer, base_checkpoint_hash=manifest['base_checkpoint_hash'],
        harness_hash=manifest['harness_hash'], tokenizer_hash=manifest['tokenizer_hash'], journal=journal,
        **settings)

    backend.student_config = dict(config)
    if config['benchmark'] != 'bfcl':
        from types import MethodType
        from .benchmarks.registry import get_benchmark
        backend.action_limit = MethodType(get_benchmark(config).action_limit, backend)
    return backend
