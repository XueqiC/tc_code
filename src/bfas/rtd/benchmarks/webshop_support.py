"""Adapter S_d/S_c split, reset states, and rotating RTD feedback folds."""
import json
from pathlib import Path

from ...adapters.webshop import WebShopAdapter, SUPPORT_SESSIONS, webshop_eval
from ..bank_build import privileged
from ..persistence import digest
from ..transport import FullState


def parent_hash(task_id):
    if not isinstance(task_id, str) or str(int(task_id)) != task_id or int(task_id) not in SUPPORT_SESSIONS:
        raise ValueError('WebShop support requires train sessions 500..6909')
    return digest(dict(benchmark='webshop', session=int(task_id)))


def freeze_support(adapter):
    privileged()
    split = adapter.support_split().as_dict()
    if (len(split['support']), len(split['demand']), len(split['calibration'])) != (250, 200, 50):
        raise ValueError('WebShop support must be N=250, S_d=200, S_c=50')
    parents = [dict(official_id=t, parent_hash=parent_hash(t), fold=int(parent_hash(t), 16) % 2)
               for t in split['demand']]
    return validate_support(dict(version='rtd-webshop-support-v1', split=split, parents=parents,
                                 pool=[500, 6910], evaluation=[0, 500]))


def validate_support(value):
    split = value['split']
    s, d, c = (split[k] for k in ('support', 'demand', 'calibration'))
    if (len(s), len(d), len(c)) != (250, 200, 50) or len(set(s)) != 250 or set(d) & set(c) or set(s) != set(d) | set(c):
        raise ValueError('invalid WebShop S_d/S_c split')
    for tid in s:
        parent_hash(tid)
    expected = [dict(official_id=t, parent_hash=parent_hash(t), fold=int(parent_hash(t), 16) % 2) for t in d]
    if value['parents'] != expected or value['pool'] != [500, 6910] or value['evaluation'] != [0, 500]:
        raise ValueError('WebShop parent/evaluation protocol changed')
    return value


def reset_state(adapter, task_id):
    observation = adapter._environment().reset(session=int(task_id))
    messages = webshop_eval.build_messages([], observation, webshop_eval.OBS_CHARS)
    return FullState.create(dict(benchmark='webshop', session=int(task_id)), messages,
                            adapter._render(messages), parent_hash(task_id))


class WebShopSupport:
    def __init__(self, root, config):
        privileged()
        from ..bank_build import validate_state_certificate
        bank = Path(root)/config['replay_bank_path']
        validate_state_certificate(bank, benchmark='webshop', student=config['student'])
        self.config = dict(config)
        self._feedback_adapter = None
        self.manifest = validate_support(json.loads((bank/'public/support.json').read_text()))
        external = Path(root)/config['support_manifest']
        if json.loads(external.read_text()) != self.manifest:
            raise ValueError('support differs from sealed WebShop split')
        self.parents = {p['parent_hash']: p['official_id'] for p in self.manifest['parents']}
        resets = json.loads((bank/'public/reset_states.json').read_text())
        self.states = {h: FullState(**resets[t]) for h, t in self.parents.items()}
        self.entries = {t: dict(benchmark='webshop', session=int(t)) for t in self.parents.values()}
        self.categories = {t: 'agent_action' for t in self.parents.values()}
        self.unavailable = {}
        for h, state in self.states.items():
            if state.parent_hash != h or json.loads(state.task_json) != self.entries[self.parents[h]]:
                raise ValueError('WebShop reset identity mismatch')

    def feedback_context(self, round_number, backend, journal):
        from .webshop_rollout import WebShopFeedbackContext
        if self._feedback_adapter is None:
            self._feedback_adapter = WebShopAdapter()
        self._feedback_adapter._tokenizer = backend.tokenizer
        return WebShopFeedbackContext(self, round_number, journal, lambda: self._feedback_adapter, False)

    def close(self):
        if self._feedback_adapter is not None:
            self._feedback_adapter.close()
            self._feedback_adapter = None

    def feedback(self, parent, backend, parameters, generator, checker):
        from .webshop_rollout import WebShopFeedbackContext, webshop_task_rollout
        if not isinstance(checker, WebShopFeedbackContext) or checker.support is not self:
            raise ValueError('WebShop feedback needs its bound round context')
        if int(parent, 16) % 2 == (checker.round_number-1) % 2:
            # Frozen diagnostics can explicitly inspect both folds, never fitting.
            if not getattr(backend, 'diagnostic_only', False):
                raise PermissionError('feedback requires the opposite parent fold')
        return webshop_task_rollout(self.states[parent], 'agent_action', [], backend, parameters,
                                    generator, checker=checker)
