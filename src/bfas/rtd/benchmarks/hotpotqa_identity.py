"""Frozen ReAct prompt, splits, local dataset and cached Wikipedia identities."""
from pathlib import Path
from ... import hotpotqa as hp
from ..persistence import digest, file_hash, tree_hash
from .config import benchmark_protocol


def scoring_projection(name, source):
    return dict(path=name, source=source)


def evaluation_harness_identity(root, config):
    if config.get('method') == 'rtd_unified':
        from ..unified.config import runtime_config
        config = runtime_config(config)
    root = Path(root)
    protocol = benchmark_protocol('hotpotqa')
    for key, value in protocol.items():
        if config.get(key) != value:
            raise ValueError('frozen HotpotQA evaluation protocol differs: ' + key)
    offline = hp.retrieval_offline()
    cache = root/config['hotpotqa_cache_root']
    if offline and not cache.is_dir():
        raise FileNotFoundError(cache)
    split = hp.load_manifest('dev')
    if len(split['ids']) != 500 or len(set(split['ids'])) != 500:
        raise ValueError('HotpotQA requires the frozen 500-dev inventory')
    # load_questions verifies the full source ID order before selecting dev IDs.
    questions = hp.load_questions('dev', data_dir=root/config['hotpotqa_data_root'])
    names = ['src/bfas/adapters/hotpotqa.py', 'src/bfas/hotpotqa.py', 'tools/hotpotqa_eval.py',
        'src/bfas/run.py', 'src/bfas/rtd/benchmarks/adapter_evaluation.py',
        'src/bfas/rtd/benchmarks/hotpotqa_evaluation.py', 'src/bfas/rtd/benchmarks/hotpotqa_identity.py',
        'prompts/hotpotqa_react_6shot.txt', 'configs/hotpotqa_support_split.json', 'configs/hotpotqa_eval_split.json']
    # Live evaluation grows the cache. Only explicit replay freezes snapshots.
    snapshots = {p.name: file_hash(p) for p in sorted(cache.glob('*.json'))} if offline else {}
    return dict(version='rtd-hotpotqa-adapter-scoring-v1', benchmark='hotpotqa',
        files={n: file_hash(root/n) for n in names},
        expected=dict(split='dev_distractor_first500', task_ids=split['ids'], steps=7,
            model_calls=14, max_action_tokens=100, metric='em', secondary_metric='f1',
            questions_hash=digest(questions), data_hash=tree_hash(root/config['hotpotqa_data_root'])),
        environment=dict(wiki_version=hp.WIKI_VERSION, snapshots=snapshots, offline=offline),
        prompt_version=hp.PROMPT_VERSION, student=config['student'], protocol=protocol,
        evaluation_temperature=0., backend='vllm',
        serving_python=file_hash(root/'envs/vllm-serve/.venv/bin/python'),
        max_context_tokens=config['max_context_tokens'])
