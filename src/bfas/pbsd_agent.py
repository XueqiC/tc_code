"""PBSD — agent adaptation. Opt-in; no SFT, verifier, or external teacher.

One PEFT model holds immutable base weights and trainable LoRA. Teacher work
finishes before student autograd graphs are built, including checkpoint replay.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import copy
import json
import math
import os
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from .pbsd_evidence import EvidenceUnit, digest, load_units, select_units


def configuration(environ=None):
    env = os.environ if environ is None else environ

    def number(key, default, cast=float, zero=False):
        # Public manifest keys are pbsd_*, shell knobs are AW_PBSD_*.
        name = 'AW_' + key.upper()
        value = cast(env.get(name, default))
        if not math.isfinite(value) or (value < 0 if zero else value <= 0):
            raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
        return value

    return dict(mode='pbsd_agent',
        pbsd_beta=number('pbsd_beta', .1),
        pbsd_positive_temperature=number('pbsd_positive_temperature', 1.),
        pbsd_negative_refresh_steps=number('pbsd_negative_refresh_steps', 1, int),
        pbsd_positive_refresh_steps=number('pbsd_positive_refresh_steps', 1, int, zero=True),
        pbsd_max_context_tokens=number('pbsd_max_context_tokens', 32768, int))


@contextmanager
def offline_only():
    """Fail closed on attempted IP networking, including DNS and UDP.

    Local UNIX sockets (e.g. torch workers) are allowed. Model loads additionally
    require local_files_only; no subprocess or teacher API client is used here.
    """
    attempts = []

    def denied(*args, **kwargs):
        attempts.append('network attempted')
        raise RuntimeError('pbsd_agent forbids network use; pre-cache the model')

    def socket_guard(original):
        def guarded(sock, *args, **kwargs):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                return denied()
            return original(sock, *args, **kwargs)
        return guarded

    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1', 'DO_NOT_TRACK': '1'}))
        for method in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
            if hasattr(socket.socket, method):
                stack.enter_context(patch.object(socket.socket, method,
                    socket_guard(getattr(socket.socket, method))))
        for method in ('create_connection', 'getaddrinfo', 'gethostbyname',
                       'gethostbyname_ex', 'gethostbyaddr'):
            stack.enter_context(patch.object(socket, method, denied))
        yield attempts
        if attempts:
            raise RuntimeError('pbsd_agent observed a swallowed network attempt')


def evidence_context(unit):
    return ('\n<PBSD_EVIDENCE_' + unit.key + '>\n'
            'Purchased demonstrations for this exact task-start state:\n' +
            '\n\n'.join(r['response'] for r in unit.rows) +
            '\n</PBSD_EVIDENCE_' + unit.key + '>\n'
            'Now produce your own next assistant response for the task.\n')


def contextual_state(unit):
    state = copy.deepcopy(unit.state)
    context = evidence_context(unit)
    if state.get('messages'):
        users = [m for m in state['messages'] if m['role'] == 'user']
        if not users or not isinstance(users[-1]['content'], str):
            raise ValueError('PBSD needs a textual task-start user message')
        users[-1]['content'] += context
    else:
        prompt = state['prompt']
        # Preserve the existing agent serialization and final generation marker.
        markers = ['<|assistant|>', '<|im_start|>assistant']
        position = max(prompt.rfind(m) for m in markers)
        state['prompt'] = (prompt[:position] + context + prompt[position:]
                           if position >= 0 else prompt + context)
    return state


@dataclass(frozen=True)
class Contexts:
    unit: EvidenceUnit
    student: tuple[int, ...]
    teacher: tuple[int, ...]

    def audit(self):
        return dict(student_prompt_sha256=digest(self.student),
                    teacher_prompt_sha256=digest(self.teacher),
                    evidence_sha256=digest(evidence_context(self.unit)))


def prepare_contexts(trainer, tokenizer, units, config):
    contexts = []
    explicit_cap = trainer.prompt_truncation_config()[0] if 'AW_MAX_PROMPT_TOKENS' in os.environ else None
    for unit in units:
        student = tuple(trainer.prompt_token_ids(tokenizer, unit.state))
        teacher = tuple(trainer.prompt_token_ids(tokenizer, contextual_state(unit)))
        if not student or student == teacher:
            raise ValueError('empty state or contextual teacher missing evidence')
        if explicit_cap is not None and len(student) > explicit_cap:
            raise ValueError(f'{unit.task_id}: complete state exceeds AW_MAX_PROMPT_TOKENS; increase cap')
        if len(teacher) + trainer.MAX_RESPONSE_TOKENS > config['pbsd_max_context_tokens']:
            raise ValueError(f'{unit.task_id}: complete (s,c) exceeds context cap; no evidence truncation allowed')
        contexts.append(Contexts(unit, student, teacher))
    return contexts


def encode_action(prompt, action, device):
    """Same response-only mask as trainer.encode, on exact generated tokens.

    No decode/re-encode, synthetic EOS, prompt truncation, or environment step.
    EOS is supervised iff sampled. A cap-truncated action remains a prefix.
    """
    if not prompt or not action:
        raise ValueError('nonempty state and sampled action required')
    ids = torch.tensor([tuple(prompt) + tuple(action)], device=device)
    labels = ids.clone()
    labels[:, :len(prompt)] = -100
    return ids, labels


def preference_loss(student_positive, teacher_positive, student_negative, teacher_negative, beta=.1):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('PBSD beta must be finite and positive')
    return -F.logsigmoid(beta * (student_positive - teacher_positive.detach()
                                 - student_negative + teacher_negative.detach()))


@dataclass(frozen=True)
class Sample:
    ids: tuple[int, ...]
    step: int
    generation_id: int
    truncated: bool


@dataclass(frozen=True)
class Pair:
    context: Contexts
    positive: Sample
    negative: Sample
    teacher_positive: torch.Tensor
    teacher_negative: torch.Tensor


class Engine:
    def __init__(self, trainer, model, tokenizer, contexts, config, *, audit=False):
        self.trainer, self.model, self.tokenizer = trainer, model, tokenizer
        self.contexts, self.config, self.audit_enabled = contexts, config, audit
        self.device = next(model.parameters()).device
        self.counts = Counter()
        self.cache = {}
        self.generation_id = 0
        self.forward_audit = []
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n for n, _ in trainable):
            raise ValueError('only student LoRA parameters may be trainable')
        self.trainable = [p for _, p in trainable]

    @contextmanager
    def branch(self, context, teacher, *, generation=False):
        model = self.model
        was_training, was_cache = model.training, model.config.use_cache
        prompt = context.teacher if teacher else context.student
        # The baseline prompt helper has no route to the evidence rows.
        expected = tuple(self.trainer.prompt_token_ids(self.tokenizer, context.unit.state))
        if context.student != expected:
            raise AssertionError('student conditioning differs from normal state')
        seen = []
        handles = []
        if self.audit_enabled:
            def hook(_model, args, kwargs):
                ids = kwargs.get('input_ids', args[0] if args else None)
                if ids is None:
                    raise AssertionError('audit requires explicit input IDs')
                actual = tuple(ids[0].tolist())
                cached = kwargs.get('past_key_values') is not None
                # Cached decode has the prefill in KV/recurrent state. Track its
                # ancestry; never pass a cache between branches or generate calls.
                if len(actual) >= len(prompt) and actual[:len(prompt)] == prompt:
                    pass
                elif not (generation and cached and seen):
                    raise AssertionError('forward lost its branch conditioning')
                seen.append(len(actual))
                self.forward_audit.append(dict(teacher=teacher, generation=generation,
                    cached=cached, input_sha256=digest(actual),
                    conditioning_sha256=digest(prompt), **context.audit()))
            # PEFT.generate bypasses PeftModel.__call__, while PEFT scoring
            # may call the wrapped LM's .forward directly (bypassing its
            # __call__ hooks). Audit both entrances; versions may visit both.
            for target in (model, model.get_base_model()):
                handles.append(target.register_forward_pre_hook(hook, with_kwargs=True))
        try:
            model.config.use_cache = generation
            model.eval() if teacher or generation else model.train()
            with model.disable_adapter() if teacher else nullcontext():
                with torch.no_grad() if teacher or generation else nullcontext():
                    yield prompt
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
            model.config.use_cache = was_cache

    def sample(self, context, teacher, step):
        from transformers import GenerationConfig
        side = 'teacher' if teacher else 'student'
        eos = getattr(self.model.generation_config, 'eos_token_id', None) or self.tokenizer.eos_token_id
        temperature = self.config['pbsd_positive_temperature'] if teacher else 1.
        with self.branch(context, teacher, generation=True) as prompt:
            ids = torch.tensor([prompt], device=self.device)
            # A fresh config avoids inherited top-k/min-p/repetition processors.
            generation = GenerationConfig(do_sample=True, temperature=temperature,
                top_p=1., top_k=0, min_p=None, repetition_penalty=1.,
                max_new_tokens=self.trainer.MAX_RESPONSE_TOKENS,
                eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True, bos_token_id=self.tokenizer.bos_token_id)
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         generation_config=generation)
        action = tuple(output[0, len(prompt):].tolist())
        if not action or len(action) > self.trainer.MAX_RESPONSE_TOKENS:
            raise AssertionError('invalid sampled action length')
        eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
        self.generation_id += 1
        self.counts[f'{side}_generation_calls'] += 1
        self.counts[f'{side}_generation_prompt_tokens'] += len(prompt)
        self.counts[f'{side}_generation_output_tokens'] += len(action)
        return Sample(action, step, self.generation_id, action[-1] not in eos_ids)

    def score(self, context, action, teacher, *, diagnostic=False):
        side = ('diagnostic_' if diagnostic else '') + ('teacher' if teacher else 'student')
        with self.branch(context, teacher) as prompt:
            ids, labels = encode_action(prompt, action, self.device)
            value = self.trainer.completion_log_prob(self.model, ids, labels)
        if not torch.isfinite(value):
            raise FloatingPointError('nonfinite PBSD log probability')
        self.counts[f'{side}_score_input_tokens'] += ids.numel()
        self.counts[f'{side}_score_output_tokens'] += len(action)
        return value.detach() if teacher else value

    def pair(self, context, step):
        samples, scores = [], []
        for teacher in (True, False):
            interval = self.config['pbsd_positive_refresh_steps' if teacher else 'pbsd_negative_refresh_steps']
            bucket = step // interval if interval else 0
            key = (context.unit.key, teacher)
            cached = self.cache.get(key)
            if cached is None or cached[0] != bucket:
                sample = self.sample(context, teacher, step)
                reference = self.score(context, sample.ids, True)
                cached = (bucket, sample, reference)
                self.cache[key] = cached
            samples.append(cached[1])
            scores.append(cached[2])
        return Pair(context, *samples, *scores)

    def loss(self, pair, *, diagnostic=False):
        # No teacher toggling between these graphs and backward (checkpoint safe).
        positive = self.score(pair.context, pair.positive.ids, False, diagnostic=diagnostic)
        negative = self.score(pair.context, pair.negative.ids, False, diagnostic=diagnostic)
        loss = preference_loss(positive, pair.teacher_positive, negative,
                               pair.teacher_negative, self.config['pbsd_beta'])
        record = dict(task_id=pair.context.unit.task_id, unit_id=pair.context.unit.key,
            student_positive_logp=float(positive.detach()), teacher_positive_logp=float(pair.teacher_positive),
            student_negative_logp=float(negative.detach()), teacher_negative_logp=float(pair.teacher_negative),
            positive_action_sha256=digest(pair.positive.ids), negative_action_sha256=digest(pair.negative.ids),
            positive_generation_id=pair.positive.generation_id, negative_generation_id=pair.negative.generation_id,
            positive_sample_step=pair.positive.step, negative_sample_step=pair.negative.step,
            positive_tokens=len(pair.positive.ids), negative_tokens=len(pair.negative.ids),
            positive_truncated=pair.positive.truncated, negative_truncated=pair.negative.truncated,
            loss=float(loss.detach()), **pair.context.audit())
        return loss, record

    def check_gradients(self):
        if any(p.grad is not None for n, p in self.model.named_parameters() if 'lora_' not in n):
            raise AssertionError('gradient leaked to frozen base weights')
        grads = [p.grad for p in self.trainable if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError('missing or nonfinite student LoRA gradients')


def load_local_model(trainer, student, seed):
    """Baseline LoRA setup, with explicit offline HF loading."""
    trainer.seed_everything(seed)
    tokenizer = trainer.AutoTokenizer.from_pretrained(student, trust_remote_code=False, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError('student tokenizer has no EOS')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = trainer.AutoModelForCausalLM.from_pretrained(student, torch_dtype=trainer.DTYPE,
        trust_remote_code=False, local_files_only=True).to(trainer.DEVICE)
    model = trainer.get_peft_model(base, trainer.lora_config())
    if os.environ.get('AW_GRAD_CKPT') == '1':
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model, tokenizer


def train(engine, *, seed, epochs, learning_rate, batch_size, journal):
    optimizer = torch.optim.AdamW(engine.trainable, lr=learning_rate)
    rng = np.random.default_rng(seed)
    steps = 0
    for epoch in range(epochs):
        order = rng.permutation(len(engine.contexts))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            records = []
            # All reference work precedes autograd; shared base remains frozen.
            pairs = [engine.pair(engine.contexts[int(i)], steps) for i in indices]
            for pair in pairs:
                loss, record = engine.loss(pair)
                (loss / len(pairs)).backward()
                records.append(record)
            engine.check_gradients()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            entry = dict(step=steps, epoch=epoch + 1, pairs=records,
                         loss=sum(r['loss'] for r in records) / len(records),
                         compute=dict(engine.counts), evidence_output_tokens=sum(c.unit.cost for c in engine.contexts),
                         teacher_api_calls=0)
            journal.write(json.dumps(entry, allow_nan=False) + '\n')
            journal.flush()
            print(f'[train][pbsd_agent] step={steps} loss={entry["loss"]:.6f}', flush=True)
    return dict(optimizer_steps=steps, compute=dict(engine.counts))


def run(args, trainer):
    config = configuration()
    output = trainer.OUTPUT_ROOT / args.tag
    if output.exists():
        raise FileExistsError(f'use a fresh PBSD tag: {output}')
    with offline_only() as attempts:
        units = load_units(trainer, trainer.POOL_PATH, os.environ.get('AW_PBSD_COST_RECORDS', ''))
        selected = select_units(units, args.selection, args.budget, args.seed)
        model, tokenizer = load_local_model(trainer, args.student, args.seed)
        contexts = prepare_contexts(trainer, tokenizer, selected, config)
        model_context = getattr(getattr(model.config, 'text_config', model.config), 'max_position_embeddings', None)
        if model_context and max(len(c.teacher) for c in contexts) + trainer.MAX_RESPONSE_TOKENS > model_context:
            raise ValueError('teacher state plus action cap exceeds model context window')
        engine = Engine(trainer, model, tokenizer, contexts, config)
        manifest = dict(mode='pbsd_agent', algorithm='PBSD — agent adaptation', config=config,
            student=args.student, seed=args.seed, selection=args.selection, budget=args.budget,
            pool=str(trainer.POOL_PATH), pool_sha256=digest(trainer.POOL_PATH.read_text()),
            evidence_output_tokens=sum(u.cost for u in selected),
            selection_stats=dict(selected_rows=sum(len(u.rows) for u in selected), selected_tasks=len(selected)),
            evidence=[u.manifest() for u in selected], context_hashes=[c.audit() for c in contexts],
            max_action_tokens=trainer.MAX_RESPONSE_TOKENS, epochs=trainer.training_epochs(),
            learning_rate=trainer.training_lr(), gradient_accumulation=trainer.GRADIENT_ACCUMULATION,
            teacher='frozen initial base; adapter disabled; state + same-task evidence',
            teacher_api_calls=0, network_attempts=0, status='training')
        output.mkdir(parents=True)
        path = output / 'selection_manifest.json'
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        with (output / 'pbsd_agent_journal.jsonl').open('w') as journal:
            result = train(engine, seed=args.seed, epochs=manifest['epochs'],
                learning_rate=manifest['learning_rate'], batch_size=trainer.GRADIENT_ACCUMULATION, journal=journal)
        if attempts:
            raise AssertionError('network use observed')
        model.config.use_cache = True
        merged = model.merge_and_unload()
        adapter = output / 'adapter'
        merged.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        manifest.update(result, status='complete')
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        print(f'SUMMARY mode=pbsd_agent evidence_tokens={manifest["evidence_output_tokens"]} '
              f'optimizer_steps={result["optimizer_steps"]} saved={adapter}', flush=True)
