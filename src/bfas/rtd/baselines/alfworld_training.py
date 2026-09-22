"""Additive PaperTrainer path: shared encoding/losses, pi1 AdamW, frozen rows."""
from dataclasses import asdict
import inspect
from pathlib import Path
import time

import torch

from ..functional_step import gradients
from ..persistence import atomic_json, digest, tree_hash
from ..source_scoring import require_device
from .alfworld_curriculum import exposure_schedule
from .alfworld_selection import exclusive_json
from .paper_losses import span_ce, token_kinds
from .paper_progress import progress
from .paper_train import PaperTrainer


def encoded_spans(tokenizer, row, max_context_tokens):
    # Import the shared owner. Never construct, append, or repair labels here.
    from .pi1 import encode_teacher_turn
    encoded = encode_teacher_turn(tokenizer, row, max_context_tokens)
    authored, kinds = token_kinds(tokenizer, row.target, benchmark=row.benchmark,
                                  final_step=row.final_step, prompt=row.prompt)
    target = tuple(encoded['target_ids'])
    if target[:len(authored)] != authored:
        raise ValueError('shared encoder and span tokenizer disagree on authored tokens')
    # Only classify extra tokens already supplied by the shared encoder. The
    # native boundary inherits the last supervised span; no fixed ID is used.
    suffix = target[len(authored):]
    if suffix and any(i not in tokenizer.all_special_ids for i in suffix):
        raise ValueError('shared encoder appended an unclassified non-special token')
    kinds.extend([next((k for k in reversed(kinds) if k != 'observation'), 'final')]*len(suffix))
    return encoded, kinds


def encoder_identity():
    from .pi1 import encode_teacher_turn
    return dict(callable='bfas.rtd.baselines.pi1.encode_teacher_turn',
                implementation_sha256=digest(inspect.getsource(encode_teacher_turn)),
                module_sha256=__import__('hashlib').sha256(
                    Path(inspect.getfile(encode_teacher_turn)).read_bytes()).hexdigest())


def training_hyperparameters(config, seed, method, *, sad_variant=None):
    from .pi1 import plain_ce_hyperparameters
    if sad_variant is not None and (method != 'sad' or sad_variant not in {'sad_sum', 'sad_mean'}):
        raise ValueError('sad_variant requires method sad and must be sad_sum or sad_mean')
    hp = plain_ce_hyperparameters(config, seed)
    hp.update(method=method, loss=f'span_ce({method if method != "ce" else "sft"})',
        loss_normalization='shared per-row span loss, weighted by supervised row tokens per update',
        target='pi1.encode_teacher_turn (including its native boundary contract)',
        observations_masked=True)
    if method == 'sad':
        variant = 'sad_sum' if sad_variant is None else sad_variant
        hp.update(sad_variant=variant, loss=f'span_ce({variant})',
            loss_normalization=(
                'equal-weight reason + action/final token sum / generated tokens per row; '
                'rows weighted by generated tokens / total generated tokens per update'
                if variant == 'sad_sum' else
                'mean of present reason and action/final group means; '
                'rows weighted by generated tokens / total generated tokens per update'))
    return hp


def prepare_training(output, rows, config, seed, method, identity, costs, *, sad_variant=None):
    output = Path(output)
    if output.is_symlink() or output.exists():
        raise FileExistsError(f'refusing existing output directory: {output}')
    plan = exposure_schedule(rows, costs, config, seed, method)
    hp = training_hyperparameters(config, seed, method, sad_variant=sad_variant)
    values = [asdict(r) for r in rows]
    identity = dict(identity, method=method, seed=seed, config=config, hyperparameters=hp,
        rows_hash=digest(values), exposure_schedule_hash=digest(plan))
    manifest = dict(version='alfworld-k32-hard-label-v1', method=method, seed=seed,
        identity=identity, identity_hash=digest(identity), hyperparameters=hp,
        teacher_data_cost=identity['teacher_data_cost'], curriculum=plan['curriculum'],
        deviations=identity.get('deviations', []), bank=identity['bank'],
        status='prepared', exposure_passes=[3, 10], endpoints=[],
        supervised_token_count=0, optimizer_step_count=0)
    if method == 'sad':
        manifest['sad_variant'] = hp['sad_variant']
    output.mkdir(parents=True, exist_ok=False)
    exclusive_json(output/'training_rows.json', values)
    exclusive_json(output/'exposure_schedule.json', plan)
    exclusive_json(output/'manifest.json', manifest)
    return manifest, plan


class K32PaperTrainer(PaperTrainer):
    """Shared PaperTrainer scoring/parameters, with an additive offline loop.

    The legacy train() does online-in-loop selection and fixed preconditioning;
    this path consumes frozen rows and implements the pi1 AdamW recipe. It calls
    span_ce with the loss variant recorded in the manifest. No generation or
    development evaluation.
    """
    def __init__(self, backend, rows, config, method, directory, journal, *, manifest, plan):
        super().__init__(backend, rows, config, method, directory, journal, manifest=manifest)
        self.hp, self.plan = manifest['hyperparameters'], plan
        self.loss_method = (self.hp['sad_variant'] if method == 'sad' else
                            'sft' if method in {'ce', 'kang'} else method)
        if (digest([asdict(r) for r in rows]) != manifest['identity']['rows_hash']
                or digest(plan) != manifest['identity']['exposure_schedule_hash']):
            raise ValueError('training inputs differ from manifest')

    def encode(self, row):
        if row not in self.encoded_rows:
            encoded, kinds = encoded_spans(self.backend.tokenizer, row, self.config['max_context_tokens'])
            self.encoded_rows[row] = (encoded['prompt_ids'], encoded['target_ids'], kinds,
                                      self.backend.tokenizer.eos_token_id)
        return self.encoded_rows[row]

    def logprobs(self, row):
        prompt, ids, kinds, eos = self.encode(row)
        _, values, _ = self.backend.score_tokens(prompt, ids, self.parameters,
            eos_token_id=eos, truncated=True, return_details=True)
        require_device(self.device, token_logprobs=values)
        self.scored_tokens += sum(k != 'observation' for k in kinds)
        return values, kinds

    def train(self):
        costs = [sum(k != 'observation' for k in self.encode(r)[2]) for r in self.rows]
        if costs != self.plan['costs']:
            raise ValueError('encoded supervision differs from frozen schedule')
        hp = self.hp
        optimizer = torch.optim.AdamW(list(self.parameters.values()), lr=hp['learning_rate'],
            betas=tuple(hp['adam_betas']), eps=hp['adam_epsilon'], weight_decay=hp['weight_decay'],
            amsgrad=False, foreach=False, fused=False)
        trace = []
        for step, batch in enumerate(self.plan['batches'], 1):
            before, started = self.scored_tokens, time.monotonic()
            gradient = {n: torch.zeros_like(p) for n, p in self.parameters.items()}
            total = 0.
            with self.journal.measure('student_step', step=step):
                for index in batch['indices']:
                    values, kinds = self.logprobs(self.rows[index])
                    loss = span_ce(values, kinds, self.loss_method)
                    g = gradients(loss, self.parameters)
                    weight = costs[index]/batch['supervised_tokens']
                    for n in gradient:
                        gradient[n].add_(g[n], alpha=weight)
                    total += float(loss.detach())*weight
                optimizer.zero_grad(set_to_none=True)
                for n, p in self.parameters.items():
                    p.grad = gradient[n]
                norm = float(torch.nn.utils.clip_grad_norm_(list(self.parameters.values()),
                             hp['gradient_clip'], error_if_nonfinite=True))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (self.scored_tokens-before != batch['supervised_tokens']
                    or self.scored_tokens != batch['cumulative_tokens']):
                raise ValueError('realized supervision differs from schedule')
            trace.append(dict(optimizer_step=step, loss=total, supervised_tokens=self.scored_tokens-before,
                cumulative_supervised_tokens=self.scored_tokens, gradient_norm_before_clip=norm))
            self.manifest.update(status='training', supervised_token_count=self.scored_tokens,
                                 optimizer_step_count=step)
            if batch['endpoint'] is not None:
                self.publish_endpoint(batch['endpoint'], trace)
                self.manifest['endpoints'].append(batch['endpoint'])
            atomic_json(self.directory/'loss_trace.json', trace)
            atomic_json(self.directory/'manifest.json', self.manifest)
            progress('training', 'commit', step=step, loss=total, tokens=self.scored_tokens-before,
                     tokens_per_second=(self.scored_tokens-before)/(time.monotonic()-started))
        self.manifest['status'] = 'trained'
        atomic_json(self.directory/'manifest.json', self.manifest)
        return dict(student_commits=len(trace), supervised_tokens=self.scored_tokens,
                    endpoints=self.manifest['endpoints'])

    def publish_endpoint(self, passes, trace):
        directory = self.directory/f'pass-{passes}'
        directory.mkdir(exist_ok=False)
        if self.scored_tokens != passes*self.plan['bank_supervised_tokens']:
            raise ValueError('exposure endpoint missed')
        self.backend.model.save_pretrained(directory/'lora', safe_serialization=True)
        self.backend.tokenizer.save_pretrained(directory/'lora')
        exclusive_json(directory/'loss_trace.json', trace)
        exclusive_json(directory/'manifest.json', dict(self.manifest,
            status='trained_endpoint', exposure_endpoint=passes,
            target_supervised_tokens=self.scored_tokens, adapter='lora',
            adapter_hash=tree_hash(directory/'lora'), loss_trace='loss_trace.json',
            exposure_check=dict(matched=True, target_tokens=self.scored_tokens,
                                actual_tokens=self.scored_tokens, relative_tolerance=0.)))
