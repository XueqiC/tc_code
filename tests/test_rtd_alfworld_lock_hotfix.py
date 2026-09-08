"""C26-I: real flock isolation and pre-hotfix receipts, with CPU-only fixtures."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Barrier

import pytest

from bfas.rtd import hardware, identity as common_identity
from bfas.rtd.benchmarks import alfworld_evaluation as evaluation
from bfas.rtd.benchmarks import alfworld_identity as identity, registry
from bfas.rtd.evaluation_lock import evaluation_lock, tag_lock_path
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash, tree_hash
from rtd_alfworld_evaluation_fixtures import ROOT, FakeBackend, FakeEnv, campaign, put, run
from test_rtd_alfworld_config import sealed_campaign
from test_rtd_alfworld_resume import integrated


EVIDENCE = json.loads((ROOT / 'configs/rtd/identity_evidence/c26i.json').read_text())


def previous_evaluate():
    # Execute the archived pre-hotfix coordinator, including its original
    # tag-only lock, to produce historical receipts without requiring git.
    namespace = dict(vars(evaluation))
    namespace['tag_lock_path'] = lambda root, tag: (
        Path(root) / 'results/alfworld_std/.locks' / digest(tag) / '.lock')
    source = next(row['previous_coordinator'] for row in EVIDENCE['files']
                  if 'previous_coordinator' in row)
    exec(compile(source, '<reviewed pre-C26-I coordinator>', 'exec'), namespace)
    return namespace['evaluate']


def test_reviewed_source_and_complete_scoring_inventory(campaign):
    for row in EVIDENCE['files']:
        assert file_hash(ROOT / row['path']) == row['after_sha256']
        assert row['before_sha256'] != row['after_sha256']
        if 'before_scoring_hash' in row:
            assert row['before_scoring_hash'] == row['after_scoring_hash']
    assert campaign.manifest['evaluation_harness']['scoring_files'] == EVIDENCE['previous_scoring_files']


@pytest.mark.parametrize('same_basename', [False, True])
def test_two_campaigns_enter_backend_concurrently_without_wait(campaign, same_basename, capsys):
    c = campaign
    barrier = Barrier(2)
    outputs = [c.root / 'results/c26f' / (name + '-alfworld-evaluations')
               for name in ('hpg-R0', 'hpg-R1s')]
    if same_basename:
        outputs = [c.root / parent / 'hpg-R0-alfworld-evaluations' for parent in ('one', 'two')]
    def evaluate(output):
        def backend(_):
            # Both backends must be inside their own campaign lease at once.
            barrier.wait(timeout=15)
            return FakeBackend()
        return run(c, output_root=output, tag='round-1', backend_factory=backend,
                   on_lock_wait=lambda **_: pytest.fail('different campaigns waited'))
    with ThreadPoolExecutor(max_workers=2) as workers:
        jobs = [workers.submit(evaluate, output) for output in outputs]
        results = [job.result(timeout=30) for job in jobs]
    assert all(result['aggregate']['tasks'] == 140 for result in results)
    assert 'waiting for evaluation lock' not in capsys.readouterr().out


def test_resolved_directory_aliases_contend_and_inode_is_permanent(campaign, monkeypatch):
    c = campaign
    output = c.root / 'results/hpg-R0-alfworld-evaluations'
    output.mkdir(parents=True)
    alias = c.root / 'alias'
    alias.symlink_to(output, target_is_directory=True)
    monkeypatch.chdir(c.root)
    paths = [evaluation.tag_lock_path(c.root, 'round-1', output_root=p)
             for p in (output, alias, Path('results/hpg-R0-alfworld-evaluations'))]
    expected = hashlib.sha256(str((output / 'round-1').resolve()).encode()).hexdigest()
    assert all(p == c.root / 'results/alfworld_std/.locks' / expected / '.lock' for p in paths)
    with evaluation_lock(paths[0], tag='alfworld/hpg-R0/round-1', timeout=0):
        inode, before = paths[0].stat().st_ino, paths[0].read_bytes()
        for alias_output in (output, alias, Path('results/hpg-R0-alfworld-evaluations')):
            with pytest.raises(TimeoutError, match='alfworld/hpg-R0/round-1'):
                run(c, output_root=alias_output, tag='round-1')
        assert paths[0].read_bytes() == before
    with evaluation_lock(paths[0], tag='resumed', timeout=0):
        assert paths[0].stat().st_ino == inode


def test_distinct_run_process_enters_while_same_run_process_is_refused(campaign):
    c = campaign
    outputs = [c.root / 'results' / (arm + '-alfworld-evaluations') for arm in ('hpg-R0', 'hpg-R1')]
    code = '''
import sys
from bfas.rtd.evaluation_lock import evaluation_lock, tag_lock_path
path = tag_lock_path(sys.argv[1], 'round-1', benchmark='alfworld', output_root=sys.argv[2])
try:
    with evaluation_lock(path, tag='child', timeout=0):
        print('acquired')
except TimeoutError:
    raise SystemExit(3)
'''
    lock = evaluation.tag_lock_path(c.root, 'round-1', output_root=outputs[0])
    with evaluation_lock(lock, tag='alfworld/hpg-R0/round-1', timeout=0):
        for index, output in enumerate(outputs):
            child = subprocess.run([sys.executable, '-c', code, str(c.root), str(output)],
                                   capture_output=True, text=True, timeout=15)
            assert child.returncode == (3 if index == 0 else 0), child.stderr
            assert ('waiting for evaluation lock' in child.stdout) == (index == 0)


def test_alfworld_requires_output_and_bfcl_keeps_original_key(tmp_path):
    with pytest.raises(ValueError, match='output_root'):
        tag_lock_path(tmp_path, 'round-1', benchmark='alfworld')
    assert tag_lock_path(tmp_path, 'round-1') == (
        tmp_path / 'results/bfcl_std/.locks' / hashlib.sha256(b'round-1').hexdigest() / '.lock')


def test_previous_coordinator_campaign_reuses_and_journals_unchanged_hash(campaign):
    c = campaign
    old_manifest = deepcopy(c.manifest)
    old_manifest['evaluation_harness']['scoring_files'] = EVIDENCE['previous_scoring_files']
    old_manifest['harness_hash'] = digest(old_manifest['evaluation_harness'])
    first = previous_evaluate()(c.root, old_manifest, **c.kwargs)
    before = (c.directory / 'campaign.json').read_bytes()
    current, hashes = identity.guard_manifest(c.root, old_manifest, hardware=c.hardware)
    assert hashes == [old_manifest['harness_hash']] == [current['harness_hash']]
    assert run(c, backend_factory=lambda _: pytest.fail('reuse loaded a model')) == first
    assert (c.directory / 'campaign.json').read_bytes() == before
    assert tree_hash(c.directory / 'artifacts') == first['artifacts_hash']
    event, = [e for e in ComputeJournal(c.directory / 'audit.jsonl').events if e['kind'] == 'identity_audit']
    assert event['reason'] == EVIDENCE['reason']
    assert event['previous_harness_hash'] == event['current_harness_hash'] == hashes[0]
    assert event['audited_hashes'] == hashes
    assert event['gpu_seconds'] == event['gpu_reserved_seconds'] == 0


def test_training_receipt_and_resume_keep_previous_identity(integrated, monkeypatch):
    c = integrated
    directory = c.engine('hpg-R0').directory
    saved = json.loads((directory / 'manifest.json').read_text())
    assert saved['evaluation_harness']['scoring_files'] == EVIDENCE['previous_scoring_files']
    checkpoint = directory / 'round-1'
    put(checkpoint / 'lora/adapter_config.json', dict(peft_type='LORA'))
    put(checkpoint / 'lora/adapter_model.safetensors', 'fixture adapter bytes')
    put(checkpoint / 'round_state.pt', 'fixture round state')
    put(checkpoint / 'checkpoint.json', dict(round=1, manifest_hash=digest(saved),
        config_hash=saved['config_hash'], adapter_hash=tree_hash(checkpoint / 'lora'),
        round_state_hash=file_hash(checkpoint / 'round_state.pt')))
    binding = identity.make_manifest(c.root, c.config, data_root=c.data, model_path=c.model,
        hardware=c.hardware, run_directory=directory, round_number=1)
    output = directory.with_name(directory.name + '-alfworld-evaluations')
    result = previous_evaluate()(c.root, binding, output_root=output, tag='round-1',
        hardware=c.hardware, backend_factory=lambda _: FakeBackend(), env_factory=FakeEnv)
    receipt = directory / 'evaluation-1.json'
    atomic_json(receipt, result)
    immutable = {p: p.read_bytes() for p in (directory / 'manifest.json', receipt,
        output / 'round-1/campaign.json', checkpoint / 'checkpoint.json')}
    monkeypatch.setattr(hardware, 'hardware_identity', lambda: c.hardware)
    monkeypatch.setattr(evaluation, 'HFBackend', lambda _: pytest.fail('reuse allocated model'))
    common_identity.validate_resume(c.root, directory, saved, deepcopy(saved))
    assert registry.alfworld_evaluate(c.root, directory, 1, lock_timeout=0) == result
    assert all(p.read_bytes() == content for p, content in immutable.items())
    lock = evaluation.tag_lock_path(c.root, 'round-1', output_root=output)
    assert json.loads(lock.read_text()) == dict(pid=os.getpid(), tag='alfworld/hpg-R0/round-1',
                                               hostname=socket.gethostname())
    # An unknown scoring identity is refused even if both receipts agree.
    result['identity']['evaluation_harness_hash'] = 'f' * 64
    put(receipt, result)
    put(output / 'round-1/campaign.json', result)
    with pytest.raises(ValueError, match='identity'):
        registry.alfworld_evaluate(c.root, directory, 1, lock_timeout=0)
