"""Frozen, resumable SmartAD selection. No environment evaluation or purchasing."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

from .paper_data import TeacherRow
from .alfworld_cost import read_json
from ..persistence import digest, file_hash


SELECTION_RULE = dict(statistic='trajectory_supervised_token_mean_nll',
    formula='sum(turn total NLL) / sum(turn supervised tokens)',
    tie_break='lexicographic candidate_id', student='initial base snapshot; no updates',
    source='docs/PAPER_BASELINES.md, SmartAD paragraph',
    justification='Local description explicitly says generated-token mean NLL; avoids length bias.',
    deviation='Original paper normalization not locally recoverable; per-token mean chosen explicitly. '
              'Not total NLL and not an unweighted mean of turn means.')


def exclusive_json(path, value):
    """Publish complete JSON atomically without replacing an existing artifact."""
    import tempfile
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.publish-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)  # fails if destination exists, including races
    finally:
        os.unlink(name)


def load_candidates(bank, renderer, cost, *, candidate_sets=None):
    """Read verified C26 rows using the shared pi1 loader and frozen renderer."""
    from .pi1 import load_bank
    bank = Path(bank).resolve()
    support = read_json(bank/'public/support.json')
    records = read_json(bank/'public/requests.json')
    usable = [r for r in records if r['unavailable_reason'] is None]
    payloads = {r['spec']['query_id']: read_json(bank/'sealed'/f"{r['spec']['query_id']}.json")
                for r in usable}
    cfg = dict(sealed_manifest_sha256=file_hash(bank/'sealed/manifest.json'),
        support_size=32, demonstrations=len(usable),
        supervised_turns=sum(len(p.get('teacher_react_turns', [])) for p in payloads.values()))
    rows, identity = load_bank(bank, cfg, renderer)
    candidates = []
    for qid, payload in sorted(payloads.items()):
        tid, attempt = payload['provenance']['task_id'], payload['provenance']['attempt_index']
        candidates.append(dict(candidate_id=payload.get('collection', {}).get('candidate_id', qid),
            package_id=qid, task_id=tid, attempt_index=attempt, verified=True,
            cost=cost['per_attempt'][tid][str(attempt)],
            rows=[asdict(r) for r in rows if r.package_id == qid]))
    if candidate_sets is not None:
        groups = read_json(candidate_sets)
        if (groups['method'] != 'smartad' or groups['bank_manifest_sha256'] != cfg['sealed_manifest_sha256']
                or groups['ledger_sha256'] not in cost['sources'].values()
                or groups['usage_sha256'] not in cost['sources'].values()):
            raise ValueError('candidate sets differ from frozen bank/ledgers')
        expected = {(tid, a['candidate_id'], a['query_id']) for tid, task in groups['tasks'].items()
                    for a in task['attempts'] if a['status'] == 'usable'}
        if expected != {(c['task_id'], c['candidate_id'], c['package_id']) for c in candidates}:
            raise ValueError('candidate set inventory differs from verified bank')
        identity['candidate_sets_sha256'] = file_hash(candidate_sets)
    return candidates, sorted(support['training_task_ids']), identity


def select_candidates(candidates, task_ids, scorer, output, identity, data_cost, *, resume=False):
    """Scorer returns (unweighted total NLL, supervised token count) per row.

    Save one immutable score per whole trajectory. An interruption redoes at
    most that trajectory. Selection is identical for both training seeds.
    """
    candidates = sorted(candidates, key=lambda c: (c['task_id'], c['candidate_id']))
    if len(task_ids) != len(set(task_ids)):
        raise ValueError('duplicate training task ID')
    if len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate ID')
    if not candidates or any(not c['verified'] or not c['rows'] or c['task_id'] not in task_ids
                             for c in candidates):
        raise ValueError('only nonempty verified training candidates are allowed')
    for c in candidates:
        if (any(r['task_id'] != c['task_id'] or r['package_id'] != c['package_id'] for r in c['rows'])
                or [r['index'] for r in c['rows']] != list(range(len(c['rows'])))):
            raise ValueError('candidate rows have different identities')
    output = Path(output)
    frozen = dict(version='alfworld-smartad-selection-v1', identity=identity,
        candidates_hash=digest(candidates), task_ids=sorted(task_ids), rule=SELECTION_RULE,
        teacher_data_cost=data_cost)
    if resume:
        if output.is_symlink() or read_json(output/'manifest.json') != frozen:
            raise ValueError('selection resume identity differs')
    else:
        output.mkdir(parents=True, exist_ok=False)
        exclusive_json(output/'manifest.json', frozen)
        (output/'scores').mkdir()
    from ..evaluation_lock import evaluation_lock
    with evaluation_lock(output/'.selection.lock', tag='smartad-selection', timeout=0):
        scores = {}
        for c in candidates:
            key = digest(c)
            path = output/'scores'/f'{key}.json'
            if path.exists():
                score = read_json(path)
                if score.get('candidate_hash') != key:
                    raise ValueError('saved candidate score identity differs')
            else:
                pieces = [scorer(TeacherRow(**r)) for r in c['rows']]
                if any(not math.isfinite(nll) or nll < 0 or type(n) is not int or n < 1
                       for nll, n in pieces):
                    raise ValueError('finite nonnegative NLL and positive token count required')
                nll, count = math.fsum(p[0] for p in pieces), sum(p[1] for p in pieces)
                score = dict(candidate_hash=key, candidate_id=c['candidate_id'],
                    total_nll=nll, supervised_tokens=count, mean_nll=nll/count)
                exclusive_json(path, score)
            if (not math.isfinite(score['total_nll']) or score['total_nll'] < 0
                    or type(score['supervised_tokens']) is not int or score['supervised_tokens'] < 1
                    or score['mean_nll'] != score['total_nll']/score['supervised_tokens']
                    or score['candidate_id'] != c['candidate_id']):
                raise ValueError('invalid saved candidate score')
            scores[c['candidate_id']] = score
        tasks, selected_rows = [], []
        for tid in sorted(task_ids):
            choices = [dict(c, **scores[c['candidate_id']]) for c in candidates if c['task_id'] == tid]
            chosen = min(choices, key=lambda c: (c['mean_nll'], c['candidate_id'])) if choices else None
            if chosen:
                selected_rows.extend(chosen['rows'])
            tasks.append(dict(task_id=tid, candidate_count=len(choices), requested_candidates=3,
                shortfall=max(0, 3-len(choices)), fewer_than_requested=len(choices) < 3,
                candidates=[{k: v for k, v in c.items() if k != 'rows'} for c in choices],
                chosen_candidate_id=chosen['candidate_id'] if chosen else None,
                reason=('no verified candidate; excluded from training' if not choices else
                        'single candidate, no choice' if len(choices) == 1 else
                        'minimum trajectory supervised-token mean NLL; candidate ID tie break'),
                purchase_cost=data_cost['per_task'][tid]))
        artifact = dict(**frozen, tasks=tasks, training_rows=selected_rows,
                        training_rows_hash=digest(selected_rows))
        artifact['artifact_hash'] = digest(artifact)
        destination = output/'selection.json'
        if destination.exists():
            if read_json(destination) != artifact:
                raise ValueError('frozen selection differs; refusing overwrite')
        else:
            exclusive_json(destination, artifact)
        return artifact


def read_selection(path, student_identity):
    artifact = read_json(path)
    unsigned = {k: v for k, v in artifact.items() if k != 'artifact_hash'}
    if (digest(unsigned) != artifact['artifact_hash']
            or digest(artifact['training_rows']) != artifact['training_rows_hash']
            or artifact['identity']['student'] != student_identity):
        raise ValueError('selection integrity or base/tokenizer/encoding identity differs')
    expected_rows = []
    for task in artifact['tasks']:
        candidates = task['candidates']
        if not candidates:
            if task['chosen_candidate_id'] is not None:
                raise ValueError('empty task has a selected candidate')
            continue
        chosen = min(candidates, key=lambda c: (c['mean_nll'], c['candidate_id']))
        if chosen['candidate_id'] != task['chosen_candidate_id']:
            raise ValueError('selected candidate is not the recorded argmin')
        rows = [r for r in artifact['training_rows'] if r['package_id'] == chosen['package_id']]
        original = {k: chosen[k] for k in ('candidate_id', 'package_id', 'task_id',
                                          'attempt_index', 'verified', 'cost')}
        if not rows or digest(dict(original, rows=rows)) != chosen['candidate_hash']:
            raise ValueError('selected rows differ from the scored candidate')
        expected_rows.extend(rows)
    if expected_rows != artifact['training_rows']:
        raise ValueError('training rows include unselected or reordered candidates')
    return [TeacherRow(**r) for r in artifact['training_rows']], artifact
