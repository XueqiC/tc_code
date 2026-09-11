"""C26-E dispatch contracts. Importing/resolving providers starts no experiment.

BFCL providers are the original objects, imported lazily. ALFWorld adapters
translate arguments/results only; the sole training state machine remains
experiment.RTDExperiment and the sole return estimator is reinforce_gradient.
The old entrypoints do not consult this registry until C26-F.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib import import_module
import json
from pathlib import Path
from types import MappingProxyType

from ...cc_pairs import render_prompt
from .. import selector


def _privileged():
    if selector._active.get() is not None:
        raise PermissionError("selector cannot resolve privileged benchmark providers")


@dataclass(frozen=True)
class Providers:
    bank_builder: object
    broker_builder: object
    support_protocol: object
    action_limit: object
    feedback_rollout: object
    official_evaluation: object
    harness_identity: object
    cap_policy: object
    affordability: object
    scoring_projection: object


@dataclass(frozen=True)
class ALFWorldFeedbackContext:
    support: object
    round_number: int
    renderer: object
    env_factory: object
    journal: object

    def check_syntax(self, text):
        from ...adapters.alfworld import ALFWorldAdapter
        return dict(valid=bool(ALFWorldAdapter._teacher_command_text(text).strip()))


def alfworld_bank_builder(root, directory, *, entries=None, renderer=render_prompt):
    """Explicit C26-A inventory builder, never passed off as a verified B bank.

    A fresh inventory still needs C26-B replay/seal_verified_bank. Audit/run
    always require that verified bank; no automatic environment replay here.
    BFCL-specific ingestion overrides cannot silently change ALFWorld inputs.
    """
    _privileged()
    if entries is not None or renderer is not render_prompt:
        raise ValueError("ALFWorld archive builder does not accept BFCL entries/renderer")
    from .alfworld_bank import build_alfworld_bank
    return build_alfworld_bank(root, directory)


class ALFWorldExperimentSupport:
    """BFCLSupport-shaped view of C26-B's frozen parent/reset protocol."""
    def __init__(self, root, config):
        _privileged()
        if config.get('method') == 'rtd_unified':
            from ..unified.config import runtime_config
            config = runtime_config(config)
        from .alfworld_config import bank_audit
        from .alfworld_support import ALFWorldSupport
        from ..transport import FullState
        if config.get('protocol_version') == '1.1.0':
            from ..bank_build import validate_state_certificate
            bank_path = Path(root)/config['replay_bank_path']
            validate_state_certificate(bank_path, benchmark='alfworld', student=config['student'])
            if json.loads((Path(root)/config['support_manifest']).read_text()) != json.loads((bank_path/'public/support.json').read_text()):
                raise ValueError('external support differs from ALFWorld bank')
            audit = {'bank_path': str(bank_path)}
        else:
            audit = bank_audit(root, config)
        bank = Path(audit['bank_path'])
        self.config = dict(config)
        self.protocol = ALFWorldSupport(json.loads((bank / 'public/support.json').read_text()))
        manifest = self.protocol.manifest
        self.parents = {h: p['selected_task_id'] for h, p in manifest['parents'].items()}
        self.entries = {tid: manifest['tasks'][tid]['request'] for tid in self.parents.values()}
        # Explicit registered action class; no BFCL category/label lookup.
        self.categories = {tid: 'agent_action' for tid in self.parents.values()}
        resets = json.loads((bank / 'public/reset_states.json').read_text())
        self.states, self.unavailable = {}, {}
        for h, tid in self.parents.items():
            if tid not in resets:
                # Excluding unsuccessful/unowned parents would bias feedback.
                raise ValueError('selected support trial missing audited reset: ' + tid)
            state = FullState(**resets[tid])
            self.protocol.guard_states([state], 1 if int(h, 16) % 2 == 0 else 2)
            if len(json.loads(state.history_json)) != 1:
                raise ValueError('support must start at a full reset')
            self.states[h] = state

    def syntax_success(self, text, state):
        from .alfworld_rollout import parse_action
        return not parse_action(text, json.loads(state.history_json)[-1]['admissible'])[1]

    def feedback_context(self, round_number, backend, journal):
        """C26-F feedback dispatch supplies a fresh explicit per-window context."""
        _privileged()
        from .alfworld_support import RealStepper, prompt_messages
        from ...adapters.alfworld import ALFWorldAdapter
        from ...cc_pairs import thinking_off
        adapter = ALFWorldAdapter()
        adapter._tokenizer = backend.tokenizer
        def renderer(request, history):
            prompt = adapter._render(prompt_messages(request, history))
            return prompt if self.config.get('protocol_version') == '1.1.0' else thinking_off(prompt)
        environment_hash = self.protocol.manifest['environment']['environment_hash']
        return ALFWorldFeedbackContext(self.protocol, round_number, renderer,
            lambda: RealStepper(environment_hash=environment_hash), journal)

    def feedback(self, parent, backend, parameters, generator, checker):
        _privileged()
        if not isinstance(checker, ALFWorldFeedbackContext) or checker.support is not self.protocol:
            raise ValueError('ALFWorld feedback needs its bound support/window context')
        if getattr(backend, 'diagnostic_only', False):
            from .interactive_diagnostics import alfworld_greedy
            return alfworld_greedy(self.states[parent], backend, parameters, generator, checker)
        tid = self.parents[parent]
        return alfworld_feedback_rollout(self.states[parent], self.categories[tid], [],
            backend, parameters, generator, checker=checker)

    def feedback_batch(self, tasks, backend, parameters, generator, checker):
        """Keep serial task/ticket/seed order; co-schedule bounded episode cohorts."""
        _privileged()
        if not isinstance(checker, ALFWorldFeedbackContext) or checker.support is not self.protocol:
            raise ValueError('ALFWorld feedback needs its bound support/window context')
        import torch
        from ..generation_batch import feedback_generation_groups
        from .alfworld_rollout import alfworld_task_rollouts_lockstep
        tasks = tuple(tasks)
        for parent, _ in tasks:
            _alfworld_feedback_request(self.states[parent], checker)
        # RTD v1.1 can have more selected episodes than the configured natural
        # batch. Preserve all selected tasks while bounding the live workers.
        natural = self.config['meta_tasks_per_feedback'] * self.config['rollouts_per_meta_task']
        policy = replace(backend.generation_batch,
                         prompts_per_batch=min(natural, backend.generation_batch.prompts_per_batch))
        entries, groups = [], []
        with alfworld_action_limit(backend, 'agent_action'):
            for parent, count in tasks:
                offset = len(entries)
                planned = backend.feedback_start_groups(self.states[parent].prompt, count, generator)
                for group in planned:
                    marker = object()
                    groups.extend(dict(row, episode_index=offset+row['index'], sampling_group=marker)
                                  for row in group)
                # Exactly the durable draws made by registry.feedback after the
                # original parent's batched task starts. Episode continuations
                # and retry replay never consume this experiment generator.
                for _ in range(count):
                    seed = int(torch.randint(0, 2**63-1, (), generator=generator, device=generator.device))
                    entries.append((parent, self.states[parent], seed))
            for cohort in feedback_generation_groups(groups, policy):
                # A legacy RNG group cannot be split without changing its
                # multinomial realization. Normal ALFWorld groups contain 2/4
                # starts, below the configured worker bound.
                from itertools import groupby
                logical = [list(rows) for _, rows in groupby(cohort, key=lambda r: id(r['sampling_group']))]
                starts = backend.generate_feedback_groups(logical, parameters)
                selected = [entries[row['episode_index']] for row in cohort]
                # D12 counts distinct prompts, so a legacy same-prompt start
                # group can exceed prompts_per_batch. Keep its sampling intact
                # but still enforce the tighter bound on live env workers.
                for offset in range(0, len(selected), policy.prompts_per_batch):
                    live = selected[offset:offset+policy.prompts_per_batch]
                    episodes = alfworld_task_rollouts_lockstep([entry[1] for entry in live], backend,
                        parameters, env_factory=checker.env_factory, renderer=checker.renderer,
                        journal=checker.journal, rollout_indices=[entry[2] for entry in live],
                        first_actions=starts[offset:offset+len(live)])
                    # Restore parent-major order even when an episode
                    # terminates before its neighbours.
                    for (parent, _, _), episode in zip(live, episodes):
                        yield parent, episode.as_task_rollout()


@contextmanager
def alfworld_action_limit(self, category):
    """Unbound backend policy, same (self, category) contract as BFCL."""
    _privileged()
    if category != 'agent_action':
        raise ValueError('unregistered ALFWorld action class')
    cap = self.action_caps.get('agent_action')
    if type(cap) is not int or cap != 256:
        raise ValueError('ALFWorld agent_action=256 required')
    previous = self.max_action_tokens
    self.max_action_tokens = cap
    try:
        yield
    finally:
        self.max_action_tokens = previous


def alfworld_feedback_rollout(entry, category, truth, backend, parameters, generator, *, checker=None):
    """Adapt C26-C complete episodes to the existing TaskRollout consumer.

    One seed draw from the experiment's saved generator supplies an independent
    C26-C episode stream. Its derived seed is journaled by C26-C; resume restores
    the pre-phase generator, so an interrupted phase repeats identical draws.
    RPC failures retry once with that same stream and fresh reset. Exhausted
    infrastructure errors propagate through as_task_rollout; no zero imputation.
    """
    _privileged()
    if category != 'agent_action' or truth != [] or not isinstance(checker, ALFWorldFeedbackContext):
        raise ValueError('ALFWorld feedback requires agent_action, no truth labels, and bound context')
    task_ref = entry
    _alfworld_feedback_request(entry, checker)
    import torch
    from .alfworld_rollout import alfworld_task_rollout_with_retry
    seed = int(torch.randint(0, 2**63 - 1, (), generator=generator, device=generator.device).item())
    with alfworld_action_limit(backend, category):
        episode = alfworld_task_rollout_with_retry(task_ref, backend, parameters, env_factory=checker.env_factory,
            renderer=checker.renderer, journal=checker.journal, rollout_index=seed, base_seed=0)
    return episode.as_task_rollout()


def _alfworld_feedback_request(entry, checker):
    from ..transport import FullState
    entry = json.loads(entry.task_json) if isinstance(entry, FullState) else entry
    checker.support.guard_tasks([entry['task_id']], checker.round_number, use='feedback')
    manifest = checker.support.manifest
    from .alfworld_state import parent_hash
    if (entry != manifest['tasks'][entry['task_id']]['request'] or
            manifest['parents'][parent_hash(entry['task_id'])]['selected_task_id'] != entry['task_id']):
        raise ValueError('feedback differs from frozen selected trial')


def alfworld_cap_policy(request_class, *, limits=None, evidence=()):
    _privileged()
    from .alfworld_caps import CapConfiguration, public_cap
    if limits is not None and not isinstance(limits, CapConfiguration):
        raise ValueError('ALFWorld caps require CapConfiguration, never BFCL RequestLimits')
    return public_cap(request_class, configuration=limits or CapConfiguration(), evidence=evidence)


def alfworld_harness_identity(root, config):
    _privileged()
    if config.get('method') == 'rtd_unified' or config.get('p1'):
        from .webshop_identity import evaluation_harness_identity as identity
        return identity(root, config)
    from .alfworld_config import model_directory, validate_config
    from .alfworld_identity import evaluation_harness_identity
    config = validate_config(config)
    if config.get('protocol_version') == '1.1.0':
        from .webshop_identity import evaluation_harness_identity as identity
        return identity(root, config)
    model = model_directory(root, config)
    return evaluation_harness_identity(root, config,
        data_root=Path(root) / config['alfworld_data_root'], model_path=model, tokenizer_path=model,
        environment_root=Path(root) / config['alfworld_environment_root'])


def alfworld_evaluate(root, directory, round_number, *, port=None, base_evaluation=None,
                      lock_timeout=None, lock_log_interval=None):
    """C26-D campaign adapter; C26-F common identity guards remain mandatory.

    TextWorld uses subprocess pipes, so no HTTP port is opened. Common campaign
    call arguments are preserved; the native ALF campaign owns the tag lease.
    """
    _privileged()
    from ..persistence import atomic_json, digest
    from ..hardware import hardware_identity, guard_hardware
    from ..identity import guard_harness, record_code_drift
    from .alfworld_config import data_identity, validate_config
    from .alfworld_identity import make_manifest
    from .alfworld_evaluation import evaluate
    root, directory = Path(root), Path(directory)
    saved = json.loads((directory / 'manifest.json').read_text())
    if saved['config'].get('protocol_version') == '1.1.0' or saved['config'].get('method') == 'rtd_unified':
        from .webshop_evaluation import evaluate_adapter
        return evaluate_adapter(root, directory, round_number, port=port, base_evaluation=base_evaluation,
            lock_timeout=lock_timeout, lock_log_interval=lock_log_interval)
    config = validate_config(saved['config'])
    if digest(config) != saved['config_hash']:
        raise ValueError('training config identity mismatch')
    if data_identity(root, config, saved['bank_path']) != saved['data_hash']:
        raise ValueError('evaluation data differs from training')
    hardware = guard_hardware(root, directory, saved, hardware_identity(), context=f'evaluation-{round_number}')
    identities = guard_harness(root, directory, saved)  # requires C26-F ALF dispatch
    record_code_drift(root, directory, saved, context=f'evaluation-{round_number}', identities=identities)
    binding = make_manifest(root, config, data_root=root / config['alfworld_data_root'],
        model_path=saved['model_path'], hardware=hardware, run_directory=directory,
        round_number=round_number, environment_root=root / config['alfworld_environment_root'])
    # Separate sibling output, never a child of the training/input directory.
    result = evaluate(root, binding, output_root=directory.parent / (directory.name + '-alfworld-evaluations'),
        tag=f'round-{round_number}', hardware=hardware, base_evaluation=base_evaluation,
        lock_timeout=lock_timeout)
    atomic_json(directory / f'evaluation-{round_number}.json', result)
    return result


# Declarative lazy imports make BFCL pointer identity testable. Resolving BFCL
# does not import any ALF provider; resolving ALF does not import experiment.py.
REGISTRY = MappingProxyType({
    'bfcl': MappingProxyType(dict(
        bank_builder=('bfas.rtd.bank', 'build_bfcl_bank'),
        broker_builder=('bfas.rtd.broker', 'SealedReplayBroker'),
        support_protocol=('bfas.rtd.experiment', 'BFCLSupport'),
        action_limit=('bfas.rtd.return_gradient', 'TorchPolicyBackend.action_limit'),
        feedback_rollout=('bfas.rtd.return_gradient', 'bfcl_task_rollout'),
        official_evaluation=('bfas.rtd.evaluation', 'evaluate'),
        harness_identity=('bfas.rtd.identity', 'evaluation_harness_identity'),
        cap_policy=('bfas.rtd.caps', 'public_cap'),
        affordability=('bfas.rtd.caps', 'affordability'),
        scoring_projection=('bfas.rtd.scoring_scope', 'scoring_projection'))),
    'webshop': MappingProxyType(dict(
        bank_builder=('bfas.rtd.benchmarks.webshop_bank', 'build_webshop_bank'),
        broker_builder=('bfas.rtd.broker', 'SealedReplayBroker'),
        support_protocol=('bfas.rtd.benchmarks.webshop_support', 'WebShopSupport'),
        action_limit=('bfas.rtd.benchmarks.webshop_caps', 'action_limit'),
        feedback_rollout=('bfas.rtd.benchmarks.webshop_rollout', 'webshop_task_rollout'),
        official_evaluation=('bfas.rtd.benchmarks.webshop_evaluation', 'evaluate'),
        harness_identity=('bfas.rtd.benchmarks.webshop_identity', 'evaluation_harness_identity'),
        cap_policy=('bfas.rtd.benchmarks.webshop_caps', 'public_cap'),
        affordability=('bfas.rtd.benchmarks.webshop_caps', 'affordability'),
        scoring_projection=('bfas.rtd.benchmarks.webshop_identity', 'scoring_projection'))),
    'alfworld': MappingProxyType(dict(
        bank_builder=(__name__, 'alfworld_bank_builder'),
        broker_builder=('bfas.rtd.broker', 'SealedReplayBroker'),
        support_protocol=(__name__, 'ALFWorldExperimentSupport'),
        action_limit=(__name__, 'alfworld_action_limit'),
        feedback_rollout=(__name__, 'alfworld_feedback_rollout'),
        official_evaluation=(__name__, 'alfworld_evaluate'),
        harness_identity=(__name__, 'alfworld_harness_identity'),
        cap_policy=(__name__, 'alfworld_cap_policy'),
        affordability=('bfas.rtd.benchmarks.alfworld_caps', 'affordability'),
        scoring_projection=('bfas.rtd.benchmarks.alfworld_identity', 'scoring_projection'))),
})


def get_benchmark(config):
    _privileged()
    key = config.get('benchmark', 'bfcl')
    if not isinstance(key, str) or key not in REGISTRY:
        raise ValueError('unknown RTD benchmark: ' + str(key))
    values = {}
    for role, (module, path) in REGISTRY[key].items():
        value = import_module(module)
        for part in path.split('.'):
            value = getattr(value, part)
        values[role] = value
    return Providers(**values)
