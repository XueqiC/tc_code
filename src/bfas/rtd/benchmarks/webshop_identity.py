"""Content-bound v1.1 adapter evaluation identities (no GPU or environment boot)."""
from pathlib import Path
from ..persistence import digest, file_hash, tree_hash


def scoring_projection(name, source):
    # These small adapter coordinators deliberately bind all implementation bytes.
    return dict(path=name, source=source)


def evaluation_harness_identity(root, config):
    root = Path(root)
    benchmark = config['benchmark']
    from .config import benchmark_protocol
    protocol = benchmark_protocol(benchmark)
    for k, v in protocol.items():
        if config.get(k) != v:
            raise ValueError('frozen evaluation protocol differs: ' + k)
    names = [f'src/bfas/adapters/{benchmark}.py', 'src/bfas/run.py',
             'src/bfas/rtd/benchmarks/webshop_evaluation.py',
             'src/bfas/rtd/benchmarks/webshop_identity.py']
    names += ['tools/webshop_eval.py'] if benchmark == 'webshop' else ['src/alfworld_eval.py']
    if benchmark == 'alfworld':
        from .alfworld_identity import official_expectations, environment_identity
        expected = official_expectations(root/config['alfworld_data_root'])
        environment = environment_identity(root/config['alfworld_environment_root'])
    else:
        expected = dict(split='test', sessions=list(range(500)), steps=15,
                        metric='success_rate', index='full', data_hash=tree_hash(root/config['webshop_data_root']))
        env = root/config['webshop_environment_root']
        environment = dict(engine=tree_hash(env/'repo/web_agent_site'),
                           python=file_hash(env/'venv/bin/python'))
        if not (env/'repo/search_engine').is_dir():
            raise FileNotFoundError('full WebShop search index unavailable')
        environment['search_index'] = tree_hash(env/'repo/search_engine')
    return dict(version='rtd-v11-adapter-scoring-v1', benchmark=benchmark,
        files={n: file_hash(root/n) for n in names}, expected=expected, environment=environment,
        student=config['student'], protocol=protocol, evaluation_temperature=0., backend='vllm',
        serving_python=file_hash(root/'envs/vllm-serve/.venv/bin/python'),
        max_context_tokens=config['max_context_tokens'])
