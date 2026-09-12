"""Official answer EM/F1 validation for the shared adapter campaign."""
import json
import math
from pathlib import Path
from ... import hotpotqa as hp
from ..persistence import digest


def validate_records(out, metrics, expected):
    rows = [json.loads(line) for line in (Path(out)/'records.jsonl').read_text().splitlines() if line.strip()]
    ids = [r['task_id'] for r in rows]
    if ids != expected['task_ids'] or len(set(ids)) != len(ids):
        raise ValueError('HotpotQA evaluation must cover all dev IDs once in frozen order')
    if not rows or any(r.get('error') for r in rows):
        raise ValueError('incomplete official HotpotQA evaluation')
    questions = [dict(_id=r['task_id'], question=r['question'], answer=r['gold'], type=r['category']) for r in rows]
    if expected.get('questions_hash', digest(questions)) != digest(questions):
        raise ValueError('HotpotQA evaluated questions/gold differ from frozen dev data')
    for row in rows:
        if (type(row['steps']) is not int or not 0 < row['steps'] <= 7
                or type(row['model_calls']) is not int or not row['steps'] <= row['model_calls'] <= 2*row['steps']
                or row['prompt_version'] != hp.PROMPT_VERSION or type(row['finished']) is not bool):
            raise ValueError('HotpotQA evaluation horizon/prompt differs')
        scores = hp.answer_metrics(row['prediction'], row['gold']) if row['finished'] else {'em': 0., 'f1': 0.}
        if any(not math.isclose(row[k], scores[k], abs_tol=1e-12) for k in scores):
            raise ValueError('HotpotQA answer metrics disagree with records')
    if (metrics.get('complete') is not True or metrics['n'] != len(rows)
            or metrics['requested_n'] != len(rows)
            or any(not math.isclose(metrics[k], sum(r[k] for r in rows)/len(rows), abs_tol=1e-12) for k in ('em', 'f1'))
            or metrics['headline'] != metrics['em']):
        raise ValueError('HotpotQA official EM/F1 summary differs from records')
    cfg = metrics['config']
    if (cfg['temperature'] != 0. or cfg['max_steps'] != 7 or cfg['max_tokens'] != 100
            or cfg['task_ids'] != ids or cfg['prompt_version'] != hp.PROMPT_VERSION
            or cfg['wiki_version'] != hp.WIKI_VERSION or not metrics.get('offline')):
        raise ValueError('HotpotQA official decoding/cache protocol differs')
    return dict(complete=True, verdicts={r['task_id']: r['em'] == 1. for r in rows},
                n=len(rows), em=metrics['em'], f1=metrics['f1'])
