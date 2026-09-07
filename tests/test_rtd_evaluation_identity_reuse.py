"""C25r-b: completed evaluation reuse across manifest-bound scoring audits."""
import copy
import json
from pathlib import Path

import pytest

from test_rtd_evaluation_resume import campaign, checkpoint, historical_campaign, put
from bfas.rtd import evaluation, identity
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash, tree_hash


@pytest.fixture
def audited_evaluation(campaign, monkeypatch):
    c = campaign
    current = c.manifest['evaluation_harness']
    previous = dict(current, version='bfcl-evaluation-harness-scoring-v3',
                    tools=dict(current['tools']))
    bridge = 'tools/behavior_atom/checker_bridge.py'
    previous['tools'][bridge] = file_hash(c.root/bridge)
    c.manifest.update(evaluation_harness=previous, harness_hash=digest(previous))
    atomic_json(c.directory/'manifest.json', c.manifest)
    checkpoint(c.directory, c.manifest, 1)
    with monkeypatch.context() as old_code:
        old_code.setattr(identity, 'evaluation_harness_identity', lambda *args: previous)
        c.result = evaluation.evaluate(c.root, c.directory, 1, port=0)
    c.supplement = c.root/'configs/rtd/legacy_identities'/f'{digest(c.manifest)}.json'
    atomic_json(c.supplement, dict(
        version='rtd-c25j-audited-identity-v1', manifest_hash=digest(c.manifest),
        legacy_harness_hash=c.manifest['harness_hash'], evaluation_harness=current,
        harness_hash=digest(current), rtd_source=c.manifest['rtd_source'],
        model_identity=dict(base_checkpoint_hash=c.manifest['base_checkpoint_hash'],
                            tokenizer_hash=None, student=None),
        identity_updates=[dict(manifest_hash=digest(c.manifest), new_harness_hash=digest(current),
            previous_identity=dict(harness_hash=digest(previous), evaluation_harness=previous))]))
    return c


def reuse_events(c):
    return [e for e in ComputeJournal(c.directory/'compute.jsonl').events
            if e['kind'] == 'evaluation_reuse_via_audited_identity']


def test_previous_audited_hash_reuses_completed_evaluation_without_rewrite(
        audited_evaluation, monkeypatch, capsys):
    c = audited_evaluation
    original = (c.directory/'evaluation-1.json').read_bytes()
    manifest, supplement = (c.directory/'manifest.json').read_bytes(), c.supplement.read_bytes()
    out = Path(c.result['output_directory'])
    artifacts, stage = tree_hash(out), tree_hash(c.root/'results/appworld_students'/out.name)
    def forbidden(*args, **kwargs):
        pytest.fail('cached evaluation must not launch work or reserve a GPU port')
    monkeypatch.setattr(evaluation, 'reserve_port', forbidden)
    monkeypatch.setattr(evaluation, '_flatten_adapter', forbidden)
    monkeypatch.setattr(evaluation.subprocess, 'run', forbidden)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    capsys.readouterr()
    for count in (1, 2):
        assert evaluation.evaluate(c.root, c.directory, 1) == c.result
        assert (c.directory/'evaluation-1.json').read_bytes() == original
        assert len(reuse_events(c)) == count  # one event per successful reuse
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1 and lines[0].startswith('[rtd] evaluation reuse via audited identity ')
    event = reuse_events(c)[0]
    assert event['stored_harness_hash'] == c.result['identity']['evaluation_harness_hash']
    assert event['current_harness_hash'] == json.loads(supplement)['harness_hash']
    assert event['supplement_path'] == str(c.supplement)
    assert event['round'] == 1 and event['gpu_seconds'] == event['gpu_reserved_seconds'] == 0
    assert (c.directory/'manifest.json').read_bytes() == manifest
    assert c.supplement.read_bytes() == supplement
    assert tree_hash(out) == artifacts
    assert tree_hash(c.root/'results/appworld_students'/out.name) == stage


@pytest.mark.parametrize('field', [
    'evaluation_harness_hash', 'checkpoint', 'config_hash', 'data_hash', 'expected_hash',
    'hardware_hash', 'tokenizer_hash', 'evaluation_temperature', 'base_checkpoint_hash',
    'missing_tokenizer_hash', 'extra_field', 'artifacts_hash', 'artifacts', 'incomplete'])
def test_audited_reuse_keeps_all_other_identity_and_artifact_guards(audited_evaluation, field):
    c = audited_evaluation
    result = copy.deepcopy(c.result)
    out = Path(result['output_directory'])
    if field == 'missing_tokenizer_hash':
        del result['identity']['tokenizer_hash']
    elif field == 'artifacts_hash':
        result['artifacts_hash'] = 'changed'
    elif field == 'artifacts':
        put(out/'data_overall.csv', 'Overall Acc\n99\n')
    elif field == 'incomplete':
        put(out/'resultdir/BFCL_v4_simple_python_result.json', '{"id":"t1"}\n')
        result['artifacts_hash'] = tree_hash(out)
    else:
        result['identity'][field] = 'unknown-to-this-run'
    atomic_json(c.directory/'evaluation-1.json', result)
    before, calls = (c.directory/'evaluation-1.json').read_bytes(), len(c.calls)
    with pytest.raises(ValueError, match='identity/artifacts|incomplete evaluation'):
        evaluation.evaluate(c.root, c.directory, 1)
    assert not reuse_events(c) and len(c.calls) == calls
    assert (c.directory/'evaluation-1.json').read_bytes() == before


def test_audited_reuse_preserves_existing_code_drift_refresh(audited_evaluation):
    c = audited_evaluation
    put(c.root/'src/bfas/rtd/new.py', 'new operational source')
    result = evaluation.evaluate(c.root, c.directory, 1)
    assert result['identity'] == c.result['identity']
    assert result['code_drift'] != c.result['code_drift']
    assert result == json.loads((c.directory/'evaluation-1.json').read_text())
    assert len(reuse_events(c)) == 1


def test_multiple_updates_and_read_only_check(audited_evaluation):
    from tools.rtd_check_evaluation_identity import check
    c = audited_evaluation
    saved = json.loads(c.supplement.read_text())
    intermediate = dict(saved['evaluation_harness'], fixture='intermediate audited identity')
    saved['identity_updates'][0]['new_harness_hash'] = digest(intermediate)
    saved['identity_updates'].append(dict(
        manifest_hash=digest(c.manifest), new_harness_hash=saved['harness_hash'],
        previous_identity=dict(harness_hash=digest(intermediate), evaluation_harness=intermediate)))
    atomic_json(c.supplement, saved)
    before, supplement = tree_hash(c.directory), c.supplement.read_bytes()
    row = check(c.root, c.directory)
    assert row['in_audited_chain']
    assert row['audited_harness_hashes'] == [
        saved['harness_hash'], digest(intermediate), c.manifest['harness_hash']]
    assert tree_hash(c.directory) == before and c.supplement.read_bytes() == supplement
    assert evaluation.evaluate(c.root, c.directory, 1) == c.result
    assert len(reuse_events(c)) == 1


def test_another_runs_supplement_cannot_authorize_a_hash(audited_evaluation):
    c = audited_evaluation
    other = dict(c.manifest, arm='another-arm')
    saved = json.loads(c.supplement.read_text())
    foreign = dict(saved['evaluation_harness'], fixture='another run only')
    saved.update(manifest_hash=digest(other), harness_hash=digest(foreign), evaluation_harness=foreign)
    saved['identity_updates'][0]['manifest_hash'] = digest(other)
    saved['identity_updates'][0]['new_harness_hash'] = digest(foreign)
    atomic_json(c.supplement.with_name(f'{digest(other)}.json'), saved)
    result = copy.deepcopy(c.result)
    result['identity']['evaluation_harness_hash'] = digest(foreign)
    atomic_json(c.directory/'evaluation-1.json', result)
    with pytest.raises(ValueError, match='identity/artifacts'):
        evaluation.evaluate(c.root, c.directory, 1)
    assert not reuse_events(c)


@pytest.mark.parametrize('stored', ['current', 'unknown', 'artifacts'])
def test_fresh_run_without_supplement_retains_exact_reuse(campaign, stored, capsys):
    c = campaign
    first = evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert identity.saved_identities(c.root, c.directory, c.manifest) == c.manifest
    assert identity.audited_harness_hashes(c.manifest, c.manifest) == [c.manifest['harness_hash']]
    if stored == 'unknown':
        first['identity']['evaluation_harness_hash'] = 'unaudited'
        atomic_json(c.directory/'evaluation-1.json', first)
    elif stored == 'artifacts':
        put(Path(first['output_directory'])/'data_overall.csv', 'Overall Acc\n99\n')
    original, calls = (c.directory/'evaluation-1.json').read_bytes(), len(c.calls)
    capsys.readouterr()
    if stored == 'current':
        assert evaluation.evaluate(c.root, c.directory, 1) == first
    else:
        with pytest.raises(ValueError, match='identity/artifacts'):
            evaluation.evaluate(c.root, c.directory, 1)
    assert (c.directory/'evaluation-1.json').read_bytes() == original
    assert not reuse_events(c) and len(c.calls) == calls
    assert 'reuse via audited identity' not in capsys.readouterr().out


@pytest.mark.parametrize('source', ['migration', 'manifest', 'update'])
def test_legacy_evaluation_reuses_each_connected_audited_hash(historical_campaign, source):
    c = historical_campaign
    result = evaluation.evaluate(c.root, c.directory, 1)
    saved = identity.saved_identities(c.root, c.directory, c.manifest)
    hashes = dict(migration=saved['identity_migrations'][0]['previous_harness_hash'],
                  manifest=c.manifest['harness_hash'],
                  update=saved['identity_updates'][0]['previous_identity']['harness_hash'])
    result['identity']['evaluation_harness_hash'] = hashes[source]
    atomic_json(c.directory/'evaluation-1.json', result)
    before = (c.directory/'evaluation-1.json').read_bytes()
    assert evaluation.evaluate(c.root, c.directory, 1) == result
    assert (c.directory/'evaluation-1.json').read_bytes() == before
    assert len(reuse_events(c)) == 1


@pytest.mark.parametrize('corruption', ['manifest', 'update_link', 'migration_link'])
def test_unbound_or_disconnected_audit_never_authorizes_reuse(historical_campaign, corruption):
    c = historical_campaign
    result = evaluation.evaluate(c.root, c.directory, 1)
    saved = json.loads(c.supplement.read_text())
    result['identity']['evaluation_harness_hash'] = saved['identity_migrations'][0]['previous_harness_hash']
    atomic_json(c.directory/'evaluation-1.json', result)
    if corruption == 'manifest':
        saved['manifest_hash'] = 'another-run'
    elif corruption == 'update_link':
        saved['identity_updates'][0]['new_harness_hash'] = 'disconnected'
    else:
        saved['identity_migrations'][0]['content_harness_hash'] = 'disconnected'
    atomic_json(c.supplement, saved)
    with pytest.raises(ValueError, match='binding mismatch|chain mismatch'):
        evaluation.evaluate(c.root, c.directory, 1)
    assert not reuse_events(c)
