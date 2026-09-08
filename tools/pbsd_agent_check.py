#!/usr/bin/env python3
"""GPU pass-through: four purchased BFCL support tasks, two real LoRA updates.

No download, teacher API, environment execution, verification filter, retry,
or loss-based sample selection. Any failed assertion exits nonzero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import appworld_train as trainer
from bfas import pbsd_agent as p
from bfas.pbsd_evidence import digest, load_units, select_units


# Fixed before any sampling; both function-calling and abstention examples.
DEFAULT_TASKS = ['irrelevance_16', 'live_multiple_207-91-1', 'parallel_80', 'simple_python_31']


def check(engine, *, learning_rate, journal):
    if len(engine.contexts) != 4 or engine.config['pbsd_negative_refresh_steps'] != 1:
        raise ValueError('check requires four tasks and refresh every optimizer step')
    optimizer = torch.optim.AdamW(engine.trainable, lr=learning_rate)
    histories, online_losses, fixed_losses = [], [], []
    audit_start = len(engine.forward_audit)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        pairs = [engine.pair(c, step) for c in engine.contexts]
        if step == 0:
            fixed_pairs = pairs
            with torch.no_grad():
                fixed_losses.append(sum(float(engine.loss(pair, diagnostic=True)[0]) for pair in fixed_pairs) / 4)
        records = []
        for pair in pairs:
            loss, record = engine.loss(pair)
            (loss / 4).backward()
            records.append(record)
            print(json.dumps(dict(step=step + 1, **record), sort_keys=True), flush=True)
        engine.check_gradients()
        if not any(x.grad is not None and x.grad.abs().sum() > 0 for x in engine.trainable):
            raise AssertionError('all student gradients are zero')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for pair in pairs:
                for sample, reference in ((pair.positive, pair.teacher_positive), (pair.negative, pair.teacher_negative)):
                    if reference.requires_grad or reference.grad_fn is not None:
                        raise AssertionError('teacher score retained autograd')
                    recomputed = engine.score(pair.context, sample.ids, True, diagnostic=True)
                    if not torch.equal(reference, recomputed):
                        raise AssertionError(f'teacher changed after update: {reference.item()} != {recomputed.item()}')
            # Compare exactly the same first-step pairs before/after both updates.
            fixed_losses.append(sum(float(engine.loss(pair, diagnostic=True)[0]) for pair in fixed_pairs) / 4)
        histories.append(records)
        online_losses.append(sum(r['loss'] for r in records) / 4)
        entry = dict(step=step + 1, pairs=records, loss=online_losses[-1],
            fixed_pair_loss_after_update=fixed_losses[-1], compute=dict(engine.counts),
            evidence_output_tokens=sum(c.unit.cost for c in engine.contexts), teacher_api_calls=0)
        journal.write(json.dumps(entry, allow_nan=False) + '\n')
        journal.flush()

    # Include the old first-step negative after the SECOND update too.
    with torch.no_grad():
        for pair in fixed_pairs:
            for sample, ref in ((pair.positive, pair.teacher_positive), (pair.negative, pair.teacher_negative)):
                if not torch.equal(engine.score(pair.context, sample.ids, True, diagnostic=True), ref):
                    raise AssertionError('initial contextual reference drifted across two steps')
    audits = engine.forward_audit[audit_start:]
    if not audits or not any(a['teacher'] for a in audits) or not any(not a['teacher'] for a in audits):
        raise AssertionError('missing actual forward audits')
    for teacher in (True, False):
        for generation in (True, False):
            if not any(a['teacher'] == teacher and a['generation'] == generation for a in audits):
                raise AssertionError('sampling or scoring branch bypassed the forward audit')
    for a in audits:
        expected = a['teacher_prompt_sha256'] if a['teacher'] else a['student_prompt_sha256']
        if a['conditioning_sha256'] != expected or a['student_prompt_sha256'] == a['teacher_prompt_sha256']:
            raise AssertionError('context isolation hash check failed')
    failures = []
    for old, new in zip(*histories):
        if new['negative_generation_id'] == old['negative_generation_id'] or new['negative_sample_step'] != 1:
            failures.append(f'{new["task_id"]}: negative was not refreshed from current student')
        if new['negative_action_sha256'] == old['negative_action_sha256']:
            failures.append(f'{new["task_id"]}: refreshed negatives have identical action hashes')
        interval = engine.config['pbsd_positive_refresh_steps']
        should_refresh = interval == 1
        if (new['positive_generation_id'] != old['positive_generation_id']) != should_refresh:
            failures.append(f'{new["task_id"]}: positive generation schedule mismatch')
        if not should_refresh and new['positive_action_sha256'] != old['positive_action_sha256']:
            failures.append(f'{new["task_id"]}: cached positive changed')
    if not online_losses[1] < online_losses[0]:
        failures.append('online mean pair loss did not decrease from step 1 to step 2')
    if not fixed_losses[2] < fixed_losses[0]:
        failures.append('fixed-pair mean loss did not decrease over two optimizer updates')
    evidence_cost = sum(c.unit.cost for c in engine.contexts)
    recorded_cost = sum(sum(c.unit.costs) for c in engine.contexts)
    if evidence_cost != recorded_cost:
        failures.append('evidence budget differs from recorded used-row costs')
    return dict(passed=not failures, failures=failures, online_losses=online_losses,
        fixed_pair_losses=fixed_losses, evidence_output_tokens=evidence_cost,
        recorded_used_row_costs=recorded_cost, compute=dict(engine.counts),
        context_isolation=True, audited_forwards=len(audits), teacher_recompute_bitwise_equal=True,
        teacher_stop_gradient=True, teacher_api_calls=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--student', default='Qwen/Qwen3.5-4B')
    ap.add_argument('--pool', type=Path, default=ROOT / 'data/bfcl_sft/pool_bfcl_ds_sft.jsonl')
    ap.add_argument('--cost-records', default='')
    ap.add_argument('--tasks', nargs=4, default=DEFAULT_TASKS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--budget', type=trainer.positive_int, default=4768)
    ap.add_argument('--learning-rate', type=float, default=trainer.training_lr())
    ap.add_argument('--positive-refresh-steps', type=int, default=0,
        help='check default caches positives; 1 regenerates each step (training default)')
    ap.add_argument('--tag', type=trainer.tag_name, default='pbsd_agent_check_s0')
    args = ap.parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('this pass-through check requires a BF16 GPU; run CPU pytest separately')
    if len(set(args.tasks)) != 4:
        raise ValueError('four distinct tasks required')
    if not 0 < args.learning_rate < float('inf'):
        raise ValueError('learning rate must be finite and positive')
    output = trainer.OUTPUT_ROOT / args.tag
    if output.exists():
        raise FileExistsError(f'use a fresh check tag: {output}')
    with p.offline_only() as attempts:
        config = p.configuration(dict(p.os.environ,
            AW_PBSD_POSITIVE_REFRESH_STEPS=str(args.positive_refresh_steps), AW_PBSD_NEGATIVE_REFRESH_STEPS='1'))
        all_units = load_units(trainer, args.pool, args.cost_records)
        by_id = {u.task_id: u for u in all_units}
        chosen = [by_id[tid] for tid in args.tasks]
        selected = select_units(chosen, 'full', args.budget, args.seed)
        if len(selected) != 4:
            raise ValueError('budget does not cover all four chosen task evidence units')
        # The supplied BFCL demo pool was built from these verified support IDs.
        verified = set(json.loads((ROOT / 'data/bfcl_demos_ds_verified.json').read_text()))
        if not set(args.tasks) <= verified:
            raise ValueError('check tasks must belong to the purchased verified BFCL support pool')
        split = json.loads((ROOT / 'configs/bfcl_support_split.json').read_text())
        if not set(args.tasks) <= set(split['demand']):
            raise ValueError('check tasks must be training demand tasks from the BFCL support split')
        model, tokenizer = p.load_local_model(trainer, args.student, args.seed)
        contexts = p.prepare_contexts(trainer, tokenizer, selected, config)
        engine = p.Engine(trainer, model, tokenizer, contexts, config, audit=True)
        output.mkdir(parents=True)
        manifest = dict(config=config, student=args.student, seed=args.seed,
            budget=args.budget, max_action_tokens=trainer.MAX_RESPONSE_TOKENS,
            learning_rate=args.learning_rate, evidence=[u.manifest() for u in selected],
            evidence_output_tokens=sum(u.cost for u in selected))
        (output / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        with (output / 'pbsd_agent_journal.jsonl').open('w') as journal:
            report = check(engine, learning_rate=args.learning_rate, journal=journal)
        report['network_attempts'] = len(attempts)
        if attempts:
            report['passed'] = False
            report['failures'].append('network attempt observed')
        (output / 'check_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
