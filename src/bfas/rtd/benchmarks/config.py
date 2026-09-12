"""Benchmark-specific frozen values; all RTD math stays in the shared validator."""
import json
from pathlib import Path
from ..persistence import digest, file_hash, tree_hash


def benchmark_protocol(benchmark):
    common = dict(benchmark=benchmark, replay_public_cap_output_tokens_by_class='certificate',
                  meta_tasks_per_feedback=4, rollouts_per_meta_task=2,
                  meta_tasks_multi_turn=4, rollouts_multi_turn=2,
                  max_action_tokens_by_benchmark={benchmark: {'agent_action': 256 if benchmark == 'alfworld' else 128}})
    if benchmark == 'alfworld':
        return common | dict(support_parent_tasks_m=135, alfworld_train_split='train',
            alfworld_evaluation_split='valid_seen', alfworld_expected_eval_tasks=140,
            alfworld_max_episode_steps=40, alfworld_student_react=True,
            alfworld_data_root='envs/alfworld/data/json_2.1.1', alfworld_environment_root='envs/alfworld')
    if benchmark == 'webshop':
        return common | dict(support_parent_tasks_m=200, webshop_evaluation_split='test',
            webshop_expected_eval_tasks=500, webshop_max_episode_steps=15,
            webshop_obs_chars=2500, webshop_history_obs_chars=600, webshop_max_prompt_chars=60000,
            webshop_data_root='envs/webshop/repo/data', webshop_environment_root='envs/webshop')
    if benchmark == 'hotpotqa':
        return common | dict(support_parent_tasks_m=200, hotpotqa_evaluation_split='dev_distractor_first500',
            hotpotqa_expected_eval_tasks=500, hotpotqa_max_episode_steps=7,
            hotpotqa_data_root='envs/hotpotqa/data', hotpotqa_cache_root='envs/hotpotqa/cache',
            hotpotqa_offline=True, max_action_tokens_by_benchmark={'hotpotqa': {'agent_action': 100}})
    raise ValueError('unknown interactive benchmark')


def data_identity(root, config, bank):
    root, bank = Path(root), Path(bank)
    benchmark = config['benchmark']
    from ..bank_build import validate_state_certificate
    cert = validate_state_certificate(bank, benchmark=benchmark, student=config['student'])
    support = json.loads((root/config['support_manifest']).read_text())
    if support != json.loads((bank/'public/support.json').read_text()):
        raise ValueError('external support manifest differs from bank')
    data = root/config[benchmark+'_data_root']
    if not data.is_dir():
        raise FileNotFoundError(data)
    return digest(dict(certificate=cert, support=file_hash(root/config['support_manifest']), official=tree_hash(data)))
