"""Frozen 200-question RTD support, with hash-parity folds and no public gold."""
import json
from pathlib import Path

from ... import hotpotqa as hp
from ...adapters.hotpotqa import HotpotQAAdapter
from ..bank_build import privileged, validate_state_certificate
from ..persistence import digest
from ..transport import FullState


def parent_hash(task_id):
    if not isinstance(task_id, str) or not task_id:
        raise ValueError('HotpotQA requires a nonempty question ID')
    return digest(dict(benchmark='hotpotqa', task_id=task_id))


def task_request(question):
    return dict(benchmark='hotpotqa', task_id=question['_id'], question=question['question'],
                category=question['type'], prompt_version=hp.PROMPT_VERSION)


def freeze_support():
    split, evaluation = hp.load_manifest('train'), hp.load_manifest('dev')
    ids = split['ids']
    if (len(ids) != 200 or len(set(ids)) != 200 or len(evaluation['ids']) != 500
            or set(ids) & set(evaluation['ids'])):
        raise ValueError('HotpotQA requires 200 train / 500 disjoint dev questions')
    # RTD uses the whole requested support. BFAS's separate 160/40 partition is
    # retained as provenance, not accidentally applied as an RTD exclusion.
    return dict(version='rtd-hotpotqa-support-v1', split=split, evaluation=evaluation,
        support_scope='all_200_support_questions',
        parents=[dict(official_id=t, parent_hash=parent_hash(t), fold=int(parent_hash(t), 16) % 2) for t in ids])


def reset_state(adapter, question):
    messages = hp.build_messages(question['question'])
    return FullState.create(task_request(question), messages, adapter._render(messages), parent_hash(question['_id']))


class HotpotQASupport:
    def __init__(self, root, config):
        privileged()
        self.config = dict(config)
        bank = Path(root)/config['replay_bank_path']
        validate_state_certificate(bank, benchmark='hotpotqa', student=config['student'])
        self.manifest = json.loads((bank/'public/support.json').read_text())
        if (self.manifest != freeze_support() or
                json.loads((Path(root)/config['support_manifest']).read_text()) != self.manifest):
            raise ValueError('HotpotQA support differs from the frozen split')
        self.parents = {p['parent_hash']: p['official_id'] for p in self.manifest['parents']}
        resets = json.loads((bank/'public/reset_states.json').read_text())
        if set(resets) != set(self.parents.values()):
            raise ValueError('HotpotQA requires every support reset, including unverified tasks')
        self.states = {h: FullState(**resets[t]).validate() for h, t in self.parents.items()}
        self.entries = {t: json.loads(self.states[h].task_json) for h, t in self.parents.items()}
        self.categories = {t: 'agent_action' for t in self.parents.values()}
        self.unavailable = {}
        for h, state in self.states.items():
            task = json.loads(state.task_json)
            if (state.parent_hash != h or task.get('task_id') != self.parents[h]
                    or set(task) != {'benchmark', 'task_id', 'question', 'category', 'prompt_version'}
                    or task['benchmark'] != 'hotpotqa' or task['prompt_version'] != hp.PROMPT_VERSION
                    or json.loads(state.history_json) != hp.build_messages(task['question'])):
                raise ValueError('HotpotQA reset identity mismatch')

    def feedback_context(self, round_number, backend, journal):
        from .hotpotqa_rollout import HotpotQAFeedbackContext
        adapter = HotpotQAAdapter(offline=True)
        adapter._tokenizer = backend.tokenizer
        return HotpotQAFeedbackContext(self, round_number, adapter._render, journal,
            lambda: hp.Wikipedia(cache_dir=Path(hp.ROOT)/self.config['hotpotqa_cache_root'], offline=True),
            lambda split: adapter.questions(split).values())

    def feedback(self, parent, backend, parameters, generator, checker):
        from .hotpotqa_rollout import hotpotqa_task_rollout
        if checker.support is not self:
            raise ValueError('HotpotQA feedback context belongs to another support')
        return hotpotqa_task_rollout(self.states[parent], 'agent_action', [], backend,
                                    parameters, generator, checker=checker)

    def feedback_streams(self, tasks, backend, parameters, rng_context, checker, *, lockstep):
        from .hotpotqa_rollout import feedback_streams
        yield from feedback_streams(self, tasks, backend, parameters, rng_context, checker, lockstep=lockstep)

    def diagnostic_batch(self, parents, backend, parameters, generator, checker):
        from .hotpotqa_rollout import diagnostic_batch
        yield from diagnostic_batch(self, parents, backend, parameters, generator, checker)
