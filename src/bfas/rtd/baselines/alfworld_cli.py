"""Offline K=32 baseline commands. Heavy imports follow GPU visibility setup."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]


def use_pi1_root(root):
    """Explicit read-only dependency for worktrees awaiting the pi1 merge.

    Import the owner under its canonical package name; never copy its encoder.
    A merged local pi1 always wins. Record the actual module bytes in identity.
    """
    if root is not None:
        root = Path(root).resolve()
        if not (root/'src/bfas/rtd/baselines/pi1.py').is_file():
            raise ValueError('--pi1-root must contain the shared pi1 module')
        import bfas.rtd.baselines
        import tools
        for package, directory in ((bfas.rtd.baselines, root/'src/bfas/rtd/baselines'),
                                   (tools, root/'tools')):
            # Freeze a list: a namespace _NamespacePath otherwise discards
            # manually appended entries when another CLI modifies sys.path.
            package.__path__ = list(dict.fromkeys([*package.__path__, str(directory)]))
    try:
        pi1 = importlib.import_module('bfas.rtd.baselines.pi1')
        paths = list(sys.path)
        try:
            # The owner's executable adds its own checkout to sys.path at
            # import time. Use its guards without changing our other imports.
            importlib.import_module('tools.alf_pi1_train')
        finally:
            sys.path[:] = paths
        return pi1
    except ModuleNotFoundError as exc:
        if exc.name == 'bfas.rtd.baselines.pi1':
            raise RuntimeError('pi1 dependency missing: merge pi1 or supply --pi1-root; encoder will not be copied') from exc
        raise


def safe_output(output, protected):
    output = Path(output)
    if output.is_symlink():
        raise ValueError('output directory must not be a symlink')
    resolved = output.resolve()
    roots = [ROOT/'data', ROOT/'artifacts', *[Path(p).resolve() for p in protected if p]]
    if any(resolved == p or resolved.is_relative_to(p) for p in roots):
        raise ValueError('output must be outside data, artifacts, source banks, and model snapshot')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('select', 'train', 'cost'):
        q = sub.add_parser(command)
        q.add_argument('--output', type=Path, required=True)
        q.add_argument('--bank', type=Path)
        q.add_argument('--collection', type=Path)
        if command != 'cost':
            q.add_argument('--pi1-root', type=Path)
            q.add_argument('--config', type=Path, required=True, help='the registered pi1 YAML')
            q.add_argument('--model-path', type=Path, required=True, help='local base snapshot, no adapter')
            q.add_argument('--gpu-uuid', required=True, help='full GPU UUID; pi1 PCI_BUS_ID guard')
        if command == 'select':
            q.add_argument('--candidate-sets', type=Path)
            q.add_argument('--resume', action='store_true')
        if command == 'train':
            q.add_argument('--method', choices=('smartad', 'sad'), required=True)
            q.add_argument('--seed', type=int, choices=(0, 1), required=True)
            q.add_argument('--selection', type=Path)
        if command == 'cost':
            q.add_argument('--method', choices=('smartad', 'sad', 'ce', 'kang'), required=True)
            q.add_argument('--run-manifest', type=Path,
                           help='copy an existing run receipt into the new directory with its bank cost')
    return p


def source_identity(pi1):
    from ..benchmarks.alfworld_identity import SCOPES
    from ..persistence import file_hash
    from tools import alf_pi1_train
    sources = [ROOT/name for name in SCOPES]
    sources += list((ROOT/'src/bfas/rtd/baselines').glob('alfworld_*.py'))
    sources += [Path(pi1.__file__), Path(alf_pi1_train.__file__), ROOT/'tools/alf_baseline.py']
    sources += [ROOT/'src/bfas/rtd'/name for name in (
        'baselines/paper_train.py', 'baselines/paper_losses.py', 'baselines/paper_data.py',
        'baselines/paper_seeds.py', 'runtime.py', 'source_scoring.py', 'return_gradient.py',
        'checkpointing.py', 'functional_step.py', 'persistence.py', 'scoring.py', 'student.py')]
    return {str(p.resolve()): file_hash(p) for p in sources}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists() and not getattr(args, 'resume', False):
        raise FileExistsError(f'refusing existing output directory: {args.output}')
    safe_output(args.output, [args.bank, args.collection, getattr(args, 'model_path', None)])
    if args.command == 'select':
        if args.bank is None or not (args.bank/'sealed/manifest.json').is_file():
            raise ValueError('selection requires a frozen --bank with sealed/manifest.json')
        sets = args.candidate_sets or Path(args.collection or str(args.bank)+'.collection')/'candidate_sets.json'
        if not sets.is_file():
            raise ValueError('selection requires the frozen candidate_sets.json export')
    if args.command == 'train':
        if args.method == 'smartad' and (args.selection is None or not args.selection.is_file()
                                         or args.bank is not None or args.collection is not None):
            raise ValueError('SmartAD training requires ONLY a frozen --selection artifact')
        if args.method == 'sad' and (args.bank is None or args.selection is not None):
            raise ValueError('SAD requires the plain D0 --bank, not a selection')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[name] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.command == 'cost' else args.gpu_uuid
    from .alfworld_cost import collection_cost
    from .alfworld_selection import exclusive_json, load_candidates, read_selection, select_candidates, SELECTION_RULE
    from ..persistence import ComputeJournal, digest, file_hash, tree_hash
    if args.command == 'cost':
        if args.bank is None:
            raise ValueError('--bank is required for cost accounting')
        cost = collection_cost(args.bank, method=args.method, collection=args.collection)
        receipt = dict(method=args.method, status='cost_only', bank=str(args.bank.resolve()), new_teacher_calls=0)
        if args.run_manifest is not None:
            receipt = json.loads(args.run_manifest.read_text())
            bank_identity = receipt.get('bank') or receipt.get('identity', {}).get('bank')
            if (not isinstance(bank_identity, dict) or bank_identity.get('sealed_manifest_sha256')
                    != file_hash(args.bank/'sealed/manifest.json')):
                raise ValueError('run manifest and accounting bank differ')
            if 'teacher_data_cost' in receipt and receipt['teacher_data_cost'] != cost:
                raise ValueError('existing run cost differs from its ledger')
            receipt = dict(receipt, cost_annotation=dict(
                source_run_manifest=str(args.run_manifest.resolve()),
                source_run_manifest_sha256=file_hash(args.run_manifest),
                original_run_directory=str(args.run_manifest.resolve().parent),
                relative_artifact_paths_resolve_against='original_run_directory',
                reporting_method=args.method))
        args.output.mkdir(parents=True, exist_ok=False)
        exclusive_json(args.output/'manifest.json', dict(receipt, teacher_data_cost=cost))
        print(json.dumps(dict(output=str(args.output), method=args.method, tokens=cost['tokens'],
                              estimated_usd=cost['estimated_usd'])))
        return 0

    pi1 = use_pi1_root(args.pi1_root)
    from tools.alf_pi1_train import select_device, verify_cuda_device
    from .alfworld_training import (K32PaperTrainer, encoded_spans, encoder_identity, prepare_training)
    from .alfworld_curriculum import SAD_CURRICULUM
    from .paper_seeds import seed_training
    from ..benchmarks.alfworld_support import FrozenRenderer
    from ..benchmarks.alfworld_identity import tokenizer_identity
    config = pi1.load_config(args.config)
    seed = 0 if args.command == 'select' else args.seed
    config.update(training_seed=seed, training_device='cuda:0')
    model = args.model_path.resolve()
    if not model.is_dir() or (model/'adapter_config.json').exists():
        raise ValueError('--model-path must be a local base snapshot, not an adapter')
    sources = source_identity(pi1)
    student = dict(model_path=str(model), base_checkpoint_hash=tree_hash(model),
        tokenizer_hash=tokenizer_identity(model)['hash'], encoder=encoder_identity(),
        max_context_tokens=config['max_context_tokens'],
        score_position_chunk_size=config['score_position_chunk_size'],
        versions={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'numpy')},
        scoring_source_hashes={k: v for k, v in sources.items() if k.endswith((
            'paper_losses.py', 'alfworld_training.py', 'runtime.py', 'source_scoring.py',
            'return_gradient.py', 'checkpointing.py', 'functional_step.py', 'student.py', 'scoring.py'))})
    selected = select_device(args.gpu_uuid)  # shared UUID/PCI guard before CUDA/model allocation
    renderer = FrozenRenderer(model)
    tokenizer = renderer.adapter._tokenizer
    if args.command == 'select':
        cost = collection_cost(args.bank, method='smartad', collection=args.collection)
        candidates, task_ids, bank_identity = load_candidates(args.bank, renderer, cost, candidate_sets=sets)
        identity = dict(student=student, bank=bank_identity, source_hashes=sources, device=selected,
                        config_sha256=file_hash(args.config), initial_student=True, selection_seed=0)
    else:
        if args.method == 'smartad':
            rows, selection = read_selection(args.selection, student)
            cost, bank_identity = selection['teacher_data_cost'], selection['identity']['bank']
            deviations = [SELECTION_RULE['deviation']]
            artifact_ref = dict(path=str(args.selection.resolve()), sha256=file_hash(args.selection),
                                artifact_hash=selection['artifact_hash'])
        else:
            if file_hash(args.bank/'sealed/manifest.json') != config['sealed_manifest_sha256']:
                raise ValueError('SAD bank must be the pi1 registered plain D0 bank')
            cost = collection_cost(args.bank, method='sad', collection=args.collection)
            rows, bank_identity = pi1.load_bank(args.bank, config, renderer)
            deviations, artifact_ref = [SAD_CURRICULUM['deviation']], None
        costs = [sum(k != 'observation' for k in encoded_spans(tokenizer, r,
                 config['max_context_tokens'])[1]) for r in rows]
        identity = dict(student=student, bank=bank_identity, source_hashes=sources, device=selected,
            teacher_data_cost=cost, selection=artifact_ref, deviations=deviations,
            config_sha256=file_hash(args.config), new_teacher_calls=0)
        manifest, plan = prepare_training(args.output, rows, config, seed, args.method, identity, costs)

    backend = None
    def get_backend():
        nonlocal backend
        if backend is None:
            import torch
            from ..runtime import load_backend
            device = verify_cuda_device(selected, config)
            torch.set_num_threads(config['cpu_threads'])
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            seed_training(seed)
            journal = ComputeJournal(args.output/'compute.jsonl', cuda=True, release_phase_cache=False)
            journal.append('baseline_device_verified', **device)
            backend = load_backend(config, dict(student, harness_hash=digest(sources)), journal)
        return backend

    if args.command == 'select':
        def score(row):
            import torch
            from ..functional_step import lora_parameters
            b = get_backend()
            encoded, kinds = encoded_spans(b.tokenizer, row, config['max_context_tokens'])
            with torch.no_grad():
                _, values, _ = b.score_tokens(encoded['prompt_ids'], encoded['target_ids'],
                    lora_parameters(b.model), eos_token_id=b.tokenizer.eos_token_id,
                    truncated=True, return_details=True)
                mask = values.new_tensor([k != 'observation' for k in kinds], dtype=torch.bool)
                return float(-values[mask].double().sum()), int(mask.sum())
        artifact = select_candidates(candidates, task_ids, score, args.output, identity, cost, resume=args.resume)
        print(json.dumps(dict(output=str(args.output/'selection.json'),
            selected_tasks=sum(t['chosen_candidate_id'] is not None for t in artifact['tasks']),
            candidate_counts=[t['candidate_count'] for t in artifact['tasks']])))
    else:
        b = get_backend()
        trainer = K32PaperTrainer(b, rows, config, args.method, args.output, b.journal,
                                  manifest=manifest, plan=plan)
        print(json.dumps(dict(output=str(args.output), **trainer.train())))
    return 0
