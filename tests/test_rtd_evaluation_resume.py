"""C25e: CPU-only provenance guards and interrupted endpoint campaigns."""
import copy
import csv
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from bfas.rtd import cli, evaluation, identity
from bfas.rtd.persistence import atomic_json, digest, file_hash, tree_hash
from rtd_identity_fixtures import put_tools, tool_content
from rtd_identity_fixtures import hardware_fixture
from bfcl_fake_socket import install_fake_socket


def put(path, text='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def checkpoint(directory, manifest, number):
    p = directory/f'round-{number}'
    put(p/'lora/adapter.safetensors', f'weights-{number}')
    put(p/'round_state.pt', f'state-{number}')
    meta = dict(round=number, manifest_hash=digest(manifest), config_hash=manifest['config_hash'],
                adapter_hash=tree_hash(p/'lora'), round_state_hash=file_hash(p/'round_state.pt'))
    atomic_json(p/'checkpoint.json', meta)
    return meta


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    install_fake_socket(monkeypatch)
    root, directory = tmp_path, tmp_path/'run'
    put(root/identity.LEADERBOARD/'bfcl_eval/utils.py')
    put(root/identity.LEADERBOARD/'bfcl_eval/data/tasks.json')
    put_tools(root)
    put(root/'src/bfas/rtd/experiment.py', 'original RTD')
    executable = put(root/'envs/bfcl/.venv/bin/bfcl')
    executable.chmod(0o755)
    put(root/'envs/bfcl/.venv/lib/python3.11/site-packages/bfcl_eval-1.dist-info/METADATA')
    put(root/'base/model.safetensors')
    hardware = hardware_fixture()
    monkeypatch.setattr(cli, 'ROOT', root)
    monkeypatch.setattr(cli, 'hardware_identity', lambda: hardware)
    monkeypatch.setattr(cli, 'data_identity', lambda *a: 'data')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'test-only-never-launched')
    expected = dict(generation={'simple_python': ['t1', 't2']}, scoring={'simple_python': ['t1', 't2']})
    monkeypatch.setattr(evaluation, 'official_expectations', lambda root: expected)
    calls, order = [], []
    incomplete, after_campaign, campaign_returncode = [], [], []

    def launch(command, **kwargs):
        if command[0] == 'git':
            return SimpleNamespace(stdout='fixture-revision\n')
        calls.append((command, kwargs))
        if command[0] == 'bash':
            tag = command[-1]
            number = int(tag.rsplit('_r', 1)[1])
            order.append(('evaluate', number))
            out = root/'results/bfcl_std'/tag
            put(out/'resultdir/BFCL_v4_simple_python_result.json',
                '{"id":"t1"}\n' + ('' if incomplete else '{"id":"t2"}\n'))
            put(out/'scoredir/BFCL_v4_simple_python_score.json',
                '{"correct_count":1,"total_count":2}\n{"id":"t2","valid":false}\n')
            put(out/'data_overall.csv', 'Overall Acc\n50.0\n')
            kwargs['stdout'].write(f'[bfclstd] {tag} OVERALL=50.0\n')
            for callback in after_campaign:
                callback()
            if campaign_returncode:
                return SimpleNamespace(returncode=campaign_returncode[-1])
        elif command[1].endswith('bfcl_hub_merge_export.py'):
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            assert '--verify' in command
            put(Path(command[command.index('--out')+1])/'model.safetensors.index.json')
        elif command[1].endswith('rtd_experiment.py'):
            number = int(command[command.index('--through-round')+1])
            assert command[2] == 'resume'
            assert '--acknowledge-code-drift' in command
            assert '--config' not in command  # manifest config crosses worker boundary
            order.append(('train', number))
            checkpoint(directory, json.loads((directory/'manifest.json').read_text()), number)
        else:
            pytest.fail(f'unexpected subprocess: {command}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(identity.subprocess, 'run', launch)
    monkeypatch.setattr(evaluation, '_flatten_adapter',
                        lambda m, c, dest: put(dest/'model.safetensors'))
    config = dict(evaluation_temperature=0., output_root='unused', evaluate_after_round=True,
                  max_action_tokens=4096, mode='sealed_replay')
    harness = identity.evaluation_harness_identity(root, config)
    manifest = dict(config=config, config_hash=digest(config), arm='R0', smoke=False,
                    hardware=hardware, hardware_hash=digest(hardware['hard']), data_hash='data', bank_path='bank',
                    model_path=str(root/'base'), base_checkpoint_hash=tree_hash(root/'base'),
                    harness_hash=digest(harness), evaluation_harness=harness,
                    rtd_source=identity.source_identity(root), initial_parameter_hash='initial')
    atomic_json(directory/'manifest.json', manifest)
    checkpoint(directory, manifest, 1)
    monkeypatch.setattr(cli, 'bank_audit', lambda config: {})
    monkeypatch.setattr(cli, 'make_manifest', lambda *a, **kw: {
        **{k: v for k, v in manifest.items() if k != 'initial_parameter_hash'},
        'evaluation_harness': identity.evaluation_harness_identity(root, config),
        'rtd_source': identity.source_identity(root)})
    monkeypatch.setattr(evaluation, 'report', lambda *a: None)
    return SimpleNamespace(root=root, directory=directory, manifest=manifest, calls=calls,
                           order=order, incomplete=incomplete, after_campaign=after_campaign,
                           campaign_returncode=campaign_returncode)


def drift(c):
    put(c.root/'src/bfas/rtd/experiment.py', 'changed RTD')


def harness_drift(c, path):
    if path == 'tools/behavior_atom/checker_bridge.py':
        # C25r projects the bridge AST: keep valid Python while changing a
        # scoring selector, so this tests drift refusal rather than parsing.
        content = (c.root/path).read_text()
        before = 'handler = QwenFCHandler(model, 1., model, True)'
        assert before in content
        put(c.root/path, content.replace(before, 'handler = QwenFCHandler(model, 0., model, True)', 1))
    else:
        put(c.root/path, 'changed harness')


def test_percentage_parser_with_real_format_csv_row():
    # Full BFCL export layout; synthetic aggregate reproduces job 41264649.
    row = next(csv.DictReader(io.StringIO((ROOT/'tests/fixtures/bfcl/data_overall.csv').read_text())))
    expected = {'Overall Acc': 46.18, 'Non-Live AST Acc': 75.42, 'Live Acc': 76.71,
                'Multi Turn Acc': 50., 'Memory Acc': 0., 'Irrelevance Detection': 72.5,
                'Relevance Detection': 69.23, 'Web Search Acc': None}
    for axis, value in expected.items():
        assert evaluation._parse_percentage(row[axis]) == value
        assert evaluation._parse_percentage(' \t' + row[axis].replace('%', ' %') + ' \n') == value
        assert evaluation._parse_percentage(row[axis].removesuffix('%')) == value


@pytest.mark.parametrize('cell,score', [('46.18%', 46.18), (' 46.18 % ', 46.18), (' N/A ', None)])
@pytest.mark.parametrize('reuse', [False, True])
def test_real_format_csv_publishes_campaign_score(campaign, monkeypatch, cell, score, reuse):
    c = campaign
    contents = (ROOT/'tests/fixtures/bfcl/data_overall.csv').read_text().replace('46.18%', cell)
    def write_csv():
        out, = (c.root/'results/bfcl_std').glob('rtd_*')
        put(out/'data_overall.csv', contents)
    c.after_campaign.append(write_csv)
    if reuse:
        c.campaign_returncode.append(1)
        with pytest.raises(RuntimeError, match='official campaign failed'):
            evaluation.evaluate(c.root, c.directory, 1, port=0)
        def forbidden(*args, **kwargs):
            pytest.fail('completed CSV must be read without launching work')
        monkeypatch.setattr(evaluation, 'reserve_port', forbidden)
        monkeypatch.setattr(evaluation, '_flatten_adapter', forbidden)
        monkeypatch.setattr(evaluation.subprocess, 'run', forbidden)
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    result = evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert result['overall_accuracy_percent'] == score
    assert result['validation']['complete'] and result['reused_campaign'] == reuse
    assert result['identity']['evaluation_harness_hash'] == c.manifest['harness_hash']
    assert json.loads((c.directory/'evaluation-1.json').read_text()) == result


@pytest.mark.parametrize('cell', ['nan%', 'inf%', '-0.01%', '100.01%', 'invalid'])
def test_percentage_aggregate_still_rejects_invalid_scores(campaign, cell):
    c = campaign
    def write_csv():
        out, = (c.root/'results/bfcl_std').glob('rtd_*')
        put(out/'data_overall.csv', f'Overall Acc\n{cell}\n')
    c.after_campaign.append(write_csv)
    with pytest.raises(ValueError):
        evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert not (c.directory/'evaluation-1.json').exists()


def test_source_drift_evaluates_without_ack_and_reuses_bound_score(campaign):
    c = campaign
    original = (c.directory/'manifest.json').read_bytes()
    drift(c)
    result = evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert result['validation']['complete'] and result['overall_accuracy_percent'] == 50
    event = result['code_drift'][-1]
    assert event['kind'] == 'code_drift' and event['old_hash'] != event['new_hash']
    assert event['files'] == ['src/bfas/rtd/experiment.py'] and not event['acknowledged']
    assert (c.directory/'manifest.json').read_bytes() == original
    command, options = next(row for row in c.calls if row[0][0] == 'bash')
    assert command[1] == str(c.root/'tools/bfcl_std_campaign.sh')
    assert options['env']['BFCLSTD_PRESERVE_GENERATION'] == '1'
    assert options['env']['BFCLSTD_TEMPERATURE'] == '0.0'
    assert 'BFCLSTD_PREMERGED' in options['env']
    first_calls = len(c.calls)
    put(c.root/'src/bfas/rtd/new.py', 'another source update')
    reused = evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert reused['identity'] == result['identity'] and len(c.calls) == first_calls
    assert len(reused['code_drift']) == 2
    put(Path(result['output_directory'])/'data_overall.csv', 'Overall Acc\n99\n')
    with pytest.raises(ValueError, match='identity/artifacts'):
        evaluation.evaluate(c.root, c.directory, 1, port=0)


@pytest.mark.parametrize('path', [identity.EVALUATION_TOOLS[0], 'tools/bfcl_hub_merge_export.py',
    identity.LEADERBOARD+'/bfcl_eval/utils.py', identity.LEADERBOARD+'/bfcl_eval/data/tasks.json'])
def test_harness_drift_is_hard_even_with_training_ack(campaign, path):
    c = campaign
    harness_drift(c, path)
    with pytest.raises(ValueError, match='evaluation harness'):
        evaluation.evaluate(c.root, c.directory, 1)
    current = cli.make_manifest(None)
    with pytest.raises(ValueError, match='evaluation harness'):
        identity.validate_resume(c.root, c.directory, c.manifest, current, acknowledge=True)
    assert not c.calls


@pytest.mark.parametrize('part', ['base', 'adapter', 'round_state', 'config', 'hardware', 'data', 'round'])
def test_evaluation_rejects_wrong_model_environment_or_checkpoint(campaign, monkeypatch, part):
    c = campaign
    if part in {'base', 'adapter', 'round_state'}:
        path = {'base': c.root/'base/model.safetensors',
                'adapter': c.directory/'round-1/lora/adapter.safetensors',
                'round_state': c.directory/'round-1/round_state.pt'}[part]
        put(path, 'corruption')
    elif part == 'hardware':
        monkeypatch.setattr(cli, 'hardware_identity', lambda: {'uuid': 'other'})
    elif part == 'data':
        monkeypatch.setattr(cli, 'data_identity', lambda *a: 'other')
    elif part == 'round':
        p = c.directory/'round-1/checkpoint.json'
        meta = json.loads(p.read_text()); meta['round'] = 2
        atomic_json(p, meta)
    else:
        changed = copy.deepcopy(c.manifest)
        changed['config']['evaluation_temperature'] = 1.
        atomic_json(c.directory/'manifest.json', changed)
    with pytest.raises(ValueError):
        evaluation.evaluate(c.root, c.directory, 1)
    assert not c.calls


def test_incomplete_campaign_never_publishes_evaluation(campaign):
    c = campaign
    c.incomplete.append(True)
    with pytest.raises(ValueError, match='incomplete evaluation generation'):
        evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert not (c.directory/'evaluation-1.json').exists()


def test_reuses_completed_campaign_after_cleanup_failure(campaign, monkeypatch, capsys):
    from bfas.rtd.persistence import ComputeJournal
    c = campaign
    c.campaign_returncode.append(1)
    with pytest.raises(RuntimeError, match='official campaign failed'):
        evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert not (c.directory/'evaluation-1.json').exists()
    out, = (c.root/'results/bfcl_std').glob('rtd_*')
    stage = c.root/'results/appworld_students'/out.name
    before = {p: (file_hash(p), p.stat().st_mtime_ns)
              for directory in (stage, out, c.directory/'round-1')
              for p in directory.rglob('*') if p.is_file()}
    events = ComputeJournal(c.directory/'compute.jsonl').events
    calls = len(c.calls)
    def forbidden(*args, **kwargs):
        pytest.fail('completed campaign must not reserve a port, export, or launch work')
    monkeypatch.setattr(evaluation, 'reserve_port', forbidden)
    monkeypatch.setattr(evaluation, '_flatten_adapter', forbidden)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    result = evaluation.evaluate(c.root, c.directory, 1, port=43891)
    assert 'reusing completed campaign' in capsys.readouterr().out
    assert result['reused_campaign'] and result['overall_accuracy_percent'] == 50
    assert result['validation']['complete'] and result['campaign_seconds'] == 0
    assert result['port'] is None and len(c.calls) == calls
    assert result['identity'] == json.loads((stage/'rtd_binding.json').read_text())
    assert result['artifacts_hash'] == tree_hash(out)
    assert json.loads((c.directory/'evaluation-1.json').read_text()) == result
    assert all((file_hash(p), p.stat().st_mtime_ns) == stamp for p, stamp in before.items())
    assert not list(out.parent.glob('*.incomplete-*'))
    after = ComputeJournal(c.directory/'compute.jsonl').events
    assert after[:-1] == events  # retain original failed attempt's time/accounting
    assert after[-1]['kind'] == 'evaluation_reused'
    assert after[-1]['gpu_seconds'] == after[-1]['gpu_reserved_seconds'] == 0


@pytest.mark.parametrize('part', ['binding', 'missing_binding', 'missing_export', 'export',
                                 'merged', 'adapter', 'checkpoint', 'base', 'harness', 'aggregate'])
def test_completed_campaign_reuse_keeps_identity_and_export_guards(campaign, part):
    c = campaign
    first = evaluation.evaluate(c.root, c.directory, 1, port=0)
    (c.directory/'evaluation-1.json').unlink()
    out = Path(first['output_directory'])
    stage = c.root/'results/appworld_students'/out.name
    if part.startswith('missing_'):
        (stage/('rtd_binding.json' if part == 'missing_binding' else 'export.json')).unlink()
    elif part in {'binding', 'export'}:
        path = stage/('rtd_binding.json' if part == 'binding' else 'export.json')
        value = json.loads(path.read_text())
        value['config_hash' if part == 'binding' else 'merged_hash'] = 'wrong'
        atomic_json(path, value)
    else:
        path = {'merged': stage/'hub_merged/model.safetensors.index.json',
                'adapter': stage/'adapter/model.safetensors',
                'checkpoint': c.directory/'round-1/lora/adapter.safetensors',
                'base': c.root/'base/model.safetensors',
                'harness': c.root/identity.EVALUATION_TOOLS[0],
                'aggregate': out/'data_overall.csv'}[part]
        put(path, 'Overall Acc\nnan\n' if part == 'aggregate' else 'changed')
    before, calls = tree_hash(out), len(c.calls)
    with pytest.raises(ValueError):
        evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert not (c.directory/'evaluation-1.json').exists()
    assert tree_hash(out) == before and len(c.calls) == calls
    assert not list(out.parent.glob('*.incomplete-*'))


def test_completed_campaign_reuse_still_requires_tag_lease(campaign):
    from bfas.rtd.evaluation_lock import evaluation_lock, tag_lock_path
    c = campaign
    first = evaluation.evaluate(c.root, c.directory, 1, port=0)
    (c.directory/'evaluation-1.json').unlink()
    tag = Path(first['output_directory']).name
    calls = len(c.calls)
    with evaluation_lock(tag_lock_path(c.root, tag), tag='another-owner'):
        with pytest.raises(TimeoutError):
            evaluation.evaluate(c.root, c.directory, 1, lock_timeout=0)
    assert len(c.calls) == calls and not (c.directory/'evaluation-1.json').exists()


@pytest.fixture
def historical_campaign(campaign):
    """Rai R0 shape: v1 campaign, C25g migration, then C25j scoring audit."""
    c = campaign
    current = c.manifest.pop('evaluation_harness')
    c.manifest.pop('rtd_source')
    c.manifest['harness_hash'] = 'legacy-combined-hash'
    atomic_json(c.directory/'manifest.json', c.manifest)
    checkpoint(c.directory, c.manifest, 1)
    old = dict(version='bfcl-evaluation-harness-v1', fixture='historical')
    content = dict(version='bfcl-evaluation-harness-content-v2', fixture='historical')
    supplement = dict(version='rtd-c25j-audited-identity-v1', manifest_hash=digest(c.manifest),
        legacy_harness_hash=c.manifest['harness_hash'], evaluation_harness=current,
        harness_hash=digest(current), rtd_source=dict(kind='legacy-mixed-harness', hash='legacy', files=None),
        model_identity=dict(base_checkpoint_hash=c.manifest['base_checkpoint_hash'], tokenizer_hash=None, student=None),
        identity_updates=[dict(manifest_hash=digest(c.manifest), new_harness_hash=digest(current),
            previous_identity=dict(harness_hash=digest(content), evaluation_harness=content))],
        identity_migrations=[dict(previous_harness_hash=digest(old), previous_evaluation_harness=old,
            content_harness_hash=digest(content), verified_manifest_data_hash=c.manifest['data_hash'])])
    c.supplement = c.root/'configs/rtd/legacy_identities'/f'{digest(c.manifest)}.json'
    atomic_json(c.supplement, supplement)
    first = evaluation.evaluate(c.root, c.directory, 1, port=0)
    (c.directory/'evaluation-1.json').unlink()
    bound = {k: v for k, v in first['identity'].items() if k not in {'base_checkpoint_hash', 'tokenizer_hash'}}
    bound['evaluation_harness_hash'] = digest(old)
    tag = f"rtd_R0_{digest(bound)[:16]}_r1"
    out = Path(first['output_directory'])
    c.historical_out = out.with_name(tag)
    out.rename(c.historical_out)
    stage = c.root/'results/appworld_students'/out.name
    stage.rename(stage.with_name(tag))
    c.historical_stage = stage.with_name(tag)
    atomic_json(c.historical_stage/'rtd_binding.json', bound)
    c.order.clear()
    return c


def test_resume_reuses_audited_historical_tag_without_rebinding(historical_campaign):
    c = historical_campaign
    stage_hash, output_hash = tree_hash(c.historical_stage), tree_hash(c.historical_out)
    args = SimpleNamespace(command='resume', run_dir=c.directory, arm='R0', port=0,
                           config=None, acknowledge_code_drift=True)
    cli.run_campaign(args, c.manifest['config'])
    assert c.order == [('train', 2), ('evaluate', 2), ('train', 3), ('evaluate', 3)]
    result = json.loads((c.directory/'evaluation-1.json').read_text())
    assert result['reused_campaign'] and result['output_directory'] == str(c.historical_out)
    assert result['campaign_identity'] == json.loads((c.historical_stage/'rtd_binding.json').read_text())
    assert result['campaign_identity'] != result['identity']
    assert tree_hash(c.historical_stage) == stage_hash and tree_hash(c.historical_out) == output_hash
    calls = len(c.calls)
    cached = evaluation.evaluate(c.root, c.directory, 1)
    assert {k: v for k, v in cached.items() if k != 'code_drift'} == {
        k: v for k, v in result.items() if k != 'code_drift'}
    assert len(c.calls) == calls


@pytest.mark.parametrize('part', ['chain', 'old_hash', 'data', 'binding', 'incomplete'])
def test_historical_reuse_refuses_bad_audit_or_artifacts(historical_campaign, part):
    c = historical_campaign
    if part == 'binding':
        atomic_json(c.historical_stage/'rtd_binding.json', {'checkpoint': 'different'})
    elif part == 'incomplete':
        (c.historical_out/'resultdir/BFCL_v4_simple_python_result.json').unlink()
    else:
        supplement = json.loads(c.supplement.read_text())
        key = {'chain': 'content_harness_hash', 'old_hash': 'previous_harness_hash',
               'data': 'verified_manifest_data_hash'}[part]
        supplement['identity_migrations'][0][key] = 'wrong'
        atomic_json(c.supplement, supplement)
    before, calls = tree_hash(c.historical_out), len(c.calls)
    with pytest.raises(ValueError):
        evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert len(c.calls) == calls and tree_hash(c.historical_out) == before
    assert not (c.directory/'evaluation-1.json').exists()


@pytest.mark.parametrize('resource', ['tag', 'port'])
def test_evaluation_lock_wait_is_idle_and_excluded_from_campaign_time(campaign, resource, monkeypatch):
    import threading
    from bfas.rtd.evaluation_lock import evaluation_lock, port_lock_path, tag_lock_path
    from bfas.rtd.persistence import ComputeJournal
    c = campaign
    first = evaluation.evaluate(c.root, c.directory, 1, port=43891)
    (c.directory/'evaluation-1.json').unlink()
    # Exercise a new campaign's waits; a complete result no longer needs a port.
    (Path(first['output_directory'])/'data_overall.csv').unlink()
    tag = Path(first['output_directory']).name
    path = tag_lock_path(c.root, tag) if resource == 'tag' else port_lock_path(c.root, 43891)
    ready, release = threading.Event(), threading.Event()
    def holder():
        with evaluation_lock(path, tag='other-fake-run'):
            ready.set()
            assert release.wait(5)
    worker = threading.Thread(target=holder)
    worker.start()
    assert ready.wait(5)
    try:
        calls = len(c.calls)
        with pytest.raises(TimeoutError):
            evaluation.evaluate(c.root, c.directory, 1, port=43891,
                                lock_timeout=.08, lock_log_interval=.02)
        assert len(c.calls) == calls  # no CPU export or GPU campaign on timeout
        waits = [e for e in ComputeJournal(c.directory/'compute.jsonl').events
                 if e['kind'] == 'evaluation_lock_wait']
        assert waits[-1]['status'] == 'timeout'
        assert all(e['gpu_seconds'] == e['gpu_reserved_seconds'] == 0
                   and e['accounting'] == 'idle' and e['round'] == 1 for e in waits)
        assert sum(e['idle_seconds'] for e in waits) >= .07
        # Start the release delay on observed contention, after identity hashing.
        # Source inventory growth must not shorten the measured lock interval.
        timer = threading.Timer(.1, release.set)
        original_append = ComputeJournal.append
        started = []
        def append(journal, kind, **event):
            if kind == 'evaluation_lock_wait' and event['status'] == 'waiting' and not started:
                started.append(True)
                timer.start()
            return original_append(journal, kind, **event)
        monkeypatch.setattr(ComputeJournal, 'append', append)
        try:
            result = evaluation.evaluate(c.root, c.directory, 1, port=43891,
                                         lock_timeout=2, lock_log_interval=.02)
        finally:
            timer.cancel()
        assert result['evaluation_lock_idle_seconds'] >= .08
        events = ComputeJournal(c.directory/'compute.jsonl').events
        last_wait = max(e['sequence'] for e in events if e['kind'] == 'evaluation_lock_wait')
        last_begin = max(e['sequence'] for e in events if e['kind'] == 'evaluation_begin')
        assert last_wait < last_begin  # idle never enters the campaign timer
        end = next(e for e in reversed(events) if e['kind'] == 'evaluation_end')
        assert end['gpu_reserved_seconds'] == result['campaign_seconds']
    finally:
        release.set()
        worker.join(5)


def test_another_arm_evaluates_while_first_tag_is_locked(campaign):
    from bfas.rtd.evaluation_lock import evaluation_lock, tag_lock_path
    c = campaign
    first = evaluation.evaluate(c.root, c.directory, 1)
    tag = Path(first['output_directory']).name
    other = c.root/'other-run'
    manifest = dict(c.manifest, arm='R1')
    atomic_json(other/'manifest.json', manifest)
    checkpoint(other, manifest, 1)
    with evaluation_lock(tag_lock_path(c.root, tag), tag=tag):
        result = evaluation.evaluate(c.root, other, 1, lock_timeout=0)
    assert result['validation']['complete']
    assert result['output_directory'] != first['output_directory']


@pytest.mark.parametrize('source_only', [True, False])
def test_changes_during_campaign_are_recorded_or_rejected(campaign, source_only):
    c = campaign
    c.after_campaign.append(lambda: drift(c) if source_only else
                            harness_drift(c, 'tools/behavior_atom/checker_bridge.py'))
    if source_only:
        result = evaluation.evaluate(c.root, c.directory, 1, port=0)
        assert result['code_drift'][-1]['files'] == ['src/bfas/rtd/experiment.py']
    else:
        with pytest.raises(ValueError, match='evaluation harness'):
            evaluation.evaluate(c.root, c.directory, 1, port=0)
        assert not (c.directory/'evaluation-1.json').exists()


def test_fake_round1_resume_evaluates_before_round2_without_round1_worker(campaign):
    c = campaign
    drift(c)
    args = SimpleNamespace(command='resume', run_dir=c.directory, arm='R0', port=0,
                           config=None, acknowledge_code_drift=False)
    with pytest.raises(ValueError, match='--acknowledge-code-drift'):
        cli.run_campaign(args, c.manifest['config'])
    assert not c.calls
    args.acknowledge_code_drift = True
    c.incomplete.append(True)
    with pytest.raises(ValueError, match='incomplete evaluation generation'):
        cli.run_campaign(args, c.manifest['config'])
    assert c.order == [('evaluate', 1)]  # failure still cannot advance training
    c.incomplete.clear(); c.order.clear()
    before = tree_hash(c.directory/'round-1')
    cli.run_campaign(args, c.manifest['config'])
    assert c.order == [('evaluate', 1), ('train', 2), ('evaluate', 2), ('train', 3), ('evaluate', 3)]
    assert tree_hash(c.directory/'round-1') == before
    c.order.clear()
    args.acknowledge_code_drift = False  # all training complete: evaluation only
    cli.run_campaign(args, c.manifest['config'])
    assert c.order == []


@pytest.mark.parametrize('arm', ['R0', 'R1'])
def test_resume_reuses_scored_round1_before_round2(campaign, arm, capsys):
    c = campaign
    c.manifest['arm'] = arm
    atomic_json(c.directory/'manifest.json', c.manifest)
    checkpoint(c.directory, c.manifest, 1)
    c.campaign_returncode.append(1)
    with pytest.raises(RuntimeError, match='official campaign failed'):
        evaluation.evaluate(c.root, c.directory, 1, port=0)
    c.campaign_returncode.clear()
    c.order.clear()
    drift(c)
    before = tree_hash(c.directory/'round-1')
    args = SimpleNamespace(command='resume', run_dir=c.directory, arm=arm, port=0,
                           config=None, acknowledge_code_drift=True)
    cli.run_campaign(args, c.manifest['config'])
    assert c.order == [('train', 2), ('evaluate', 2), ('train', 3), ('evaluate', 3)]
    assert tree_hash(c.directory/'round-1') == before
    assert 'reusing completed campaign' in capsys.readouterr().out
    result = json.loads((c.directory/'evaluation-1.json').read_text())
    assert result['reused_campaign'] and result['validation']['complete']
    assert result['identity']['config_hash'] == c.manifest['config_hash']


def test_worker_requires_ack_before_model_load_and_keeps_original_manifest(campaign, monkeypatch):
    from bfas.rtd import runtime
    c = campaign
    drift(c)
    args = SimpleNamespace(command='resume', config=None, run_dir=c.directory, arm='R0',
                           training_worker=True, acknowledge_code_drift=False)
    def model_load(config, manifest, journal):
        assert manifest == c.manifest and config == c.manifest['config']
        raise RuntimeError('reached model boundary')
    monkeypatch.setattr(runtime, 'load_backend', model_load)
    with pytest.raises(ValueError, match='--acknowledge-code-drift'):
        cli.run_command(args)
    args.acknowledge_code_drift = True
    with pytest.raises(RuntimeError, match='reached model boundary'):
        cli.run_command(args)


def test_legacy_supplement_is_bound_to_full_manifest_and_does_not_invent_old_source(campaign):
    c = campaign
    legacy = {k: v for k, v in c.manifest.items() if k not in {'evaluation_harness', 'rtd_source'}}
    legacy['harness_hash'] = 'old-combined-hash'
    atomic_json(c.directory/'manifest.json', legacy)
    checkpoint(c.directory, legacy, 1)
    with pytest.raises(ValueError, match='audited, manifest-bound'):
        evaluation.evaluate(c.root, c.directory, 1)
    supplement = dict(manifest_hash=digest(legacy), legacy_harness_hash=legacy['harness_hash'],
                      evaluation_harness=c.manifest['evaluation_harness'], harness_hash=c.manifest['harness_hash'],
                      rtd_source=dict(kind='legacy-mixed-harness', hash=legacy['harness_hash'], files=None))
    atomic_json(c.root/'configs/rtd/legacy_identities'/f'{digest(legacy)}.json', supplement)
    result = evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert result['code_drift'][0]['old_hash'] == 'old-combined-hash'
    assert result['code_drift'][0]['files'] is None
    changed = dict(legacy, arm='R1')
    with pytest.raises(ValueError, match='audited, manifest-bound'):
        identity.saved_identities(c.root, c.directory, changed)


def test_resume_preserves_saved_config_and_rejects_current_defaults(tmp_path, monkeypatch):
    config = cli.load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    for k in ('max_action_tokens_by_benchmark', 'max_state_batch_size', 'memory_peak_budget_gb',
              'memory_reserve_gb', 'memory_state_estimate_gb'):
        config.pop(k)
    config['max_action_tokens'] = 4096
    saved = dict(config=config, config_hash=digest(config))
    assert cli.resume_config(None, saved) == config
    path = put(tmp_path/'saved.yaml', yaml.safe_dump(config))
    assert cli.resume_config(path, saved) == config
    with pytest.raises(ValueError, match='resume config changed'):
        cli.resume_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml', saved)
    received = []
    monkeypatch.setenv('RTD_CONFIG', 'must-not-override-saved-config')
    monkeypatch.setattr(cli, 'run_command', lambda args: received.append(args))
    cli.main(['resume', '--run-dir', str(tmp_path), '--acknowledge-code-drift'])
    assert received[0].config is None and received[0].acknowledge_code_drift


def test_flatten_adapter_places_base_and_lora_on_cpu(tmp_path, monkeypatch):
    import peft
    import torch
    import transformers
    calls = []
    model = SimpleNamespace(save_pretrained=lambda dest, **kw: put(dest/'model.safetensors'))
    def base(*a, **kw):
        calls.append(('base', kw)); return model
    def lora(*a, **kw):
        calls.append(('lora', kw))
        return SimpleNamespace(merge_and_unload=lambda **kw: model)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', base)
    monkeypatch.setattr(peft.PeftModel, 'from_pretrained', lora)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained',
                        lambda *a, **kw: SimpleNamespace(save_pretrained=lambda dest: None))
    evaluation._flatten_adapter({'model_path': 'fake'}, tmp_path/'round-1', tmp_path/'export')
    assert len(calls) == 2 and all(kw['device_map'] == 'cpu' and kw['local_files_only'] for _, kw in calls)
    assert calls[0][1]['torch_dtype'] == torch.bfloat16
    assert calls[1][1]['torch_device'] == 'cpu'


@pytest.mark.parametrize('cuda_available', [False, True])
def test_real_tiny_peft_export_never_initializes_cuda(tmp_path, monkeypatch, cuda_available):
    import peft.peft_model
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
    from safetensors.torch import load_file
    def forbidden():
        pytest.fail('CPU export must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    load_peft_weights = peft.peft_model.load_peft_weights
    def cpu_weights(*args, **kwargs):
        # Newer safetensors can initialize CUDA in native code, bypassing
        # _lazy_init. Reject that request before any real GPU allocation.
        assert kwargs['device'] == 'cpu'
        return load_peft_weights(*args, **kwargs)
    monkeypatch.setattr(peft.peft_model, 'load_peft_weights', cpu_weights)
    base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16))
    base.save_pretrained(tmp_path/'base')
    model = get_peft_model(base, LoraConfig(r=2, target_modules=['c_attn'], task_type='CAUSAL_LM'))
    model.save_pretrained(tmp_path/'round-1/lora')
    # Tokenizer copying is independent of tensor/device placement.
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained',
                        lambda *a, **kw: SimpleNamespace(save_pretrained=lambda dest: None))
    # Exercise PEFT's GPU-present branch even on CPU-only CI; _lazy_init
    # remains forbidden so this cannot create a CUDA context on GPU hosts.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    evaluation._flatten_adapter({'model_path': str(tmp_path/'base')}, tmp_path/'round-1', tmp_path/'export')
    weights = load_file(str(tmp_path/'export/model.safetensors'))
    assert weights and all(t.device.type == 'cpu' for t in weights.values())
    assert all(t.dtype == torch.bfloat16 for t in weights.values())


def test_harness_ignores_generated_outputs_and_supports_alternate_venv(campaign):
    c = campaign
    before = identity.evaluation_harness_identity(c.root, c.manifest['config'])
    put(c.root/identity.LEADERBOARD/'result_p9170/tasks.json')
    put(c.root/identity.LEADERBOARD/'bfcl_eval/__pycache__/utils.pyc')
    assert identity.evaluation_harness_identity(c.root, c.manifest['config']) == before
    (c.root/'envs/bfcl/.venv/bin/bfcl').unlink()
    put(c.root/'envs/bfcl-venv/bin/bfcl').chmod(0o755)
    after = identity.evaluation_harness_identity(c.root, c.manifest['config'])
    assert after == before
    put(c.root/'envs/bfcl/.venv/lib/python3.11/site-packages/bfcl_eval-1.dist-info/METADATA', 'local install')
    assert identity.evaluation_harness_identity(c.root, c.manifest['config']) == before


def test_git_metadata_is_optional_for_resume_and_cached_evaluation(campaign):
    c = campaign
    current = dict(cli.make_manifest(None), evaluation_harness_metadata={'checkout_revision': 'other'})
    identity.validate_resume(c.root, c.directory, c.manifest, current)
    result = evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert result['evaluation_harness_metadata'] == {}
    calls = len(c.calls)
    (c.root/identity.LEADERBOARD/'.git').mkdir()
    assert identity.evaluation_harness_metadata(c.root) == {'checkout_revision': 'fixture-revision'}
    assert identity.guard_harness(c.root, c.directory, c.manifest) == c.manifest
    assert evaluation.evaluate(c.root, c.directory, 1, port=0)['identity'] == result['identity']
    assert len(c.calls) == calls


@pytest.mark.parametrize('legacy', [True, False])
def test_backend_loader_retains_saved_scalar_cap_without_overriding_current_caps(tmp_path, monkeypatch, legacy):
    import peft
    import torch
    import transformers
    from bfas.rtd import runtime
    from bfas.rtd.persistence import ComputeJournal
    config = cli.load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    if legacy:
        config.pop('max_action_tokens_by_benchmark')
        config['max_action_tokens'] = 4096
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.ones(1))
        def to(self, device):
            assert str(device) == 'cuda:0'  # mock the training device boundary
            return self
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: None)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **kw: Model())
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: object())
    monkeypatch.setattr(peft, 'get_peft_model', lambda model, config: model)
    monkeypatch.setattr(runtime, 'enable_gradient_checkpointing', lambda model: 0)
    manifest = dict(model_path='fake', base_checkpoint_hash='base', harness_hash='harness', tokenizer_hash='tokens')
    backend = runtime.load_backend(config, manifest, ComputeJournal(tmp_path/'compute.jsonl'))
    for category, expected in [('simple_python', 512), ('multi_turn_base', 1024)]:
        with backend.action_limit(category):
            assert backend.max_action_tokens == (4096 if legacy else expected)
