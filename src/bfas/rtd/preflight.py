"""Read-only CPU startup checks shared by the command line and GPU launches.

No model, environment episode, teacher request, or training ledger is created.
The provisional CPU manifest is never persisted in a training run directory.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def prepare_renderer(config, support, tokenizer, journal):
    """Prepare the real feedback renderer and tokenize every frozen reset."""
    from .benchmarks.registry import get_benchmark
    from .student import termination_ids
    from .runtime import backend_config
    _, _, settings = backend_config(config)
    backend = SimpleNamespace(tokenizer=tokenizer, student_config=config,
        max_action_tokens=settings['max_action_tokens'], action_caps=settings['action_caps'])
    termination_ids(backend)
    provider = get_benchmark(config)
    categories = set(support.categories.values())
    for category in sorted(categories):
        with provider.action_limit(backend, category):
            pass
    context = support.feedback_context(1, backend, journal) if hasattr(support, 'feedback_context') else None
    count = 0
    for state in support.states.values():
        if config['benchmark'] == 'hotpotqa':
            rendered = context.renderer(json.loads(state.history_json))
        elif config['benchmark'] == 'alfworld':
            rendered = context.renderer(json.loads(state.task_json), json.loads(state.history_json))
        elif config['benchmark'] == 'webshop':
            rendered = context.adapter_factory()._render(json.loads(state.history_json))
        else:
            # BFCLSupport already builds every prompt with bfcl_prompt.
            rendered = state.prompt
        if rendered != state.prompt:
            raise ValueError('feedback renderer differs from frozen support prompt: ' + state.state_hash)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if not ids or len(ids) >= config['max_context_tokens']:
            raise ValueError('support prompt leaves no room for an action: ' + state.state_hash)
        count += 1
    return count


def acquisition_preflight(manifest, support):
    """Buy a frozen prefix in an in-memory ledger, stopping before overflow.

    Attempt-ledger banks buy from the whole bank exactly as the paper baseline;
    legacy banks use legal support parents. No student features or window quotas
    affect these receipts. Training folds still govern exposure after purchase.
    """
    from .experiment import prepare_ledger
    from .persistence import digest
    from .selector import StudentSnapshot
    ledger, broker = prepare_ledger(manifest, support)
    # Attempt banks use the baseline seed-zero order. Legacy request
    # banks retain dependency-first ID order. Never filter this order by cost.
    remaining = {q for q, r in broker._records.items() if broker._legal(r)}
    order, ordered = [], set()
    if broker.attempt_packages:
        order = [q for q in broker.purchase_order if q in remaining]
        remaining.clear()
    while remaining:
        ready = sorted(q for q in remaining if set(broker._records[q].dependencies) <= ordered)
        if not ready:
            raise ValueError('acquisition preflight has unavailable/cyclic dependencies')
        order.extend(ready)
        ordered.update(ready)
        remaining.difference_update(ready)
    view = StudentSnapshot('cpu-acquisition-preflight', broker.inner_parent_hashes)
    position, checkpoints = 0, []
    for cap in manifest['budget_ceilings']:
        ledger.authorize(cap)
        while position < len(order):
            q = order[position]
            offered = {s.query_id for s in broker.list_candidates(view, ledger.owned_ids, ledger.remaining)}
            if q not in offered:
                if broker._records[q].spec.cost_upper_bound <= ledger.remaining:
                    raise ValueError('frozen acquisition prefix became unavailable')
                break
            broker.acquire(q)
            position += 1
        next_id = order[position] if position < len(order) else None
        if ledger.reservations or ledger.spent > cap:
            raise ValueError('acquisition preflight left an invalid ledger')
        checkpoints.append(dict(budget_tokens=cap, purchase_count=len(ledger.owned_ids),
            tasks_purchased=(len({broker.attempt_packages[q]['task_id'] for q in ledger.owned_ids})
                             if broker.attempt_packages else None),
            usable_packages=sum(broker._records[q].unavailable_reason is None for q in ledger.owned_ids),
            attempts_purchased=len(ledger.owned_ids),
            purchased_query_ids=order[:position],
            usable_query_ids=[q for q in order[:position] if broker._records[q].unavailable_reason is None],
            spent_tokens=ledger.spent, remaining_tokens=ledger.remaining,
            purchased_query_ids_sha256=digest(order[:position]), next_query_id=next_id,
            next_reservation_tokens=(broker._records[next_id].spec.cost_upper_bound if next_id else None),
            stop_reason='first_overflow' if next_id else 'inventory_exhausted',
            individually_affordable_at_zero_spend=sum(broker._records[q].spec.cost_upper_bound <= cap for q in order)))
    return dict(reservation_basis=broker.reservation_basis,
        order=('sorted_attempt_ids_random_seed_0' if broker.attempt_packages else 'query_id_ascending_dependency_first'),
        order_sha256=digest(order), inventory_tasks=(len({a['task_id'] for a in broker.attempt_packages.values()})
                         if broker.attempt_packages else None),
        available_packages=sum(broker._records[q].unavailable_reason is None for q in order),
        scope=('all bank attempts; fold-independent cumulative prefix without training window quotas'
               if broker.attempt_packages else 'all legal support parents; cumulative prefix without training window quotas'),
        checkpoints=checkpoints)


def preflight_config(config, arm, *, smoke=True):
    """Raise the first startup exception; return a summary on CPU success."""
    from . import cli
    from .runtime import backend_config, load_tokenizer
    from .experiment import executor_config, prepare_support, prepare_ledger
    from .persistence import ComputeJournal
    print(f'[rtd-preflight] checking arm={arm} mode={"smoke" if smoke else "run"}', flush=True)
    audit = cli.bank_audit(config)
    # Hardware probing initializes CUDA. It is the only manifest input replaced
    # here; the real launch constructs and binds its own hardware identity.
    hardware = dict(version='rtd-cpu-preflight-only', hard={'device': 'cpu'}, metadata={})
    manifest = cli.make_manifest(config, arm, audit, smoke=smoke, hardware=hardware)
    runtime, _, settings = backend_config(config)
    executor = executor_config(config, arm)
    if runtime != executor:
        raise ValueError('backend and executor runtime configurations differ')
    support = prepare_support(cli.ROOT, executor, manifest)
    try:
        ledger, _ = prepare_ledger(manifest, support)
        acquisition = acquisition_preflight(manifest, support)
        deferred = []
        with TemporaryDirectory(prefix='rtd-preflight-') as scratch:
            journal = ComputeJournal(Path(scratch)/'compute.jsonl', cuda=False)
            if config.get('replay_schedule'):
                from .conventions import load_schedule
                if config.get('replay_mode') == 'streaming':
                    from .streaming_replay import prepare_manifest, StreamingSchedule
                    holder = SimpleNamespace(config=runtime, manifest=manifest, support=support,
                        directory=Path(scratch), journal=journal)
                    prepare_manifest(holder)
                    reader = StreamingSchedule(holder, smoke=smoke)
                    try:
                        reader.snapshot(wait=False, check_initial_parameters=False)
                    except FileNotFoundError:
                        deferred.append('streaming V0 schedule not yet published; executor waits for it')
                else:
                    load_schedule(runtime, manifest, support, smoke=smoke, check_initial_parameters=False)
            tokenizer = load_tokenizer(manifest)
            rendered = prepare_renderer(runtime, support, tokenizer, journal)
            with (cli._checker_context(runtime) if runtime['benchmark'] == 'bfcl' else nullcontext()):
                pass
        summary = dict(status='OK', arm=arm, protocol_version=runtime['protocol_version'],
            config_protocol_version=config['protocol_version'], bank_path=manifest['bank_path'],
            available_packages=audit['available_packages'], support_states=len(support.states),
            rendered_states=rendered, tokenizer='local cache', ledger_spent=ledger.spent,
            acquisition=acquisition,
            deferred=deferred,
            generation_batch=None if settings['generation_batch'] is None else settings['generation_batch'].config(),
            gpu_required=['hardware binding', 'model/LoRA initialization and replay parameter identity',
                          'generation/scoring numerical consistency, gradients and peak memory'])
        print('[rtd-preflight] OK ' + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if hasattr(support, 'close'):
            support.close()


def main(argv=None):
    from .cli import startup_config, positive_seconds, bank_audit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--arm')
    parser.add_argument('--mode', choices=['smoke', 'run'], default='smoke')
    parser.add_argument('--acquisition-only', action='store_true',
        help='check certified bank purchase prefixes only; omit harness/model/tokenizer startup checks')
    parser.add_argument('--replay-schedule', type=Path)
    parser.add_argument('--replay-mode', choices=['complete', 'streaming'])
    parser.add_argument('--replay-poll-seconds', type=float)
    parser.add_argument('--replay-timeout-seconds', type=float)
    parser.add_argument('--smoke-deadline-seconds', type=positive_seconds, default=900)
    args = parser.parse_args(argv)
    try:
        replay_options = {k: getattr(args, k) for k in ('replay_mode', 'replay_poll_seconds', 'replay_timeout_seconds')
                          if getattr(args, k) is not None}
        config, arm = startup_config(args.config, args.arm, args.replay_schedule,
            smoke=args.mode == 'smoke', smoke_deadline_seconds=args.smoke_deadline_seconds, **replay_options)
        if args.acquisition_only:
            audit = bank_audit(config)
            rows = json.loads((Path(audit['bank_path'])/'public/requests.json').read_text())
            support = SimpleNamespace(parents={r['parent_hash'] for r in rows})
            summary = dict(status='OK', arm=arm, mode='acquisition_only', bank_path=audit['bank_path'],
                cap_certificate_sha256=audit.get('cap_certificate_sha256'),
                acquisition=acquisition_preflight(audit, support))
            print('[rtd-preflight] OK ' + json.dumps(summary, sort_keys=True), flush=True)
        else:
            preflight_config(config, arm, smoke=args.mode == 'smoke')
    except Exception as exc:
        print(f'[rtd-preflight] FAIL {type(exc).__name__}: {exc}', flush=True)
        return 1
    return 0
