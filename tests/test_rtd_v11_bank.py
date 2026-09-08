"""Offline certificates and config/manifest isolation, using only tmp banks."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import yaml

from bfas.cc_pairs import digest
from bfas.rtd import cli
from bfas.rtd.bank_v11 import build_v11_bank, validate_v11_certificate, CERTIFICATE
from bfas.rtd.broker import RequestRecord, seal_bank
from bfas.rtd.caps import public_cap, recorded_budget_ceilings, V11_CLASS_CAPS
from bfas.rtd.persistence import file_hash
from bfas.rtd.selector import PublicFeatures, PublicQuerySpec
from bfas.rtd.transport import FullState
from test_rtd_manifest_tolerance import manifest_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archived_bank(tmp_path):
    records, payloads = [], {}
    for i, (kind, cost, confidence, unavailable) in enumerate([
            ('demo_attempt', 1409, 'exact', None), ('generator_item', 371, 'estimated', None),
            ('demo_attempt', 5000, 'exact', 'excluded')]):
        q, parent = digest(['query', i]), digest(['parent', i])
        state = FullState.create({'question': str(i)}, [{'role': 'user', 'content': str(i)}], 'prompt', parent)
        cap, provenance = public_cap(kind)
        spec = PublicQuerySpec(q, state.state_hash, PublicFeatures(), 1, cap, confidence, provenance)
        records.append(RequestRecord(spec, parent, unavailable_reason=unavailable))
        payloads[q] = dict(cost=cost, cost_confidence=confidence, usage=dict(output_tokens=cost, input_tokens=None,
            reasoning_tokens=None, dollars=None), historical_response='sealed teacher', provenance={},
            behaviors=[dict(state=asdict(state), text='teacher')])
    bank = seal_bank(tmp_path/'archive', records, payloads)
    (bank/'sealed/audit.json').write_text(json.dumps(dict(bank_cost=1780, m=2)))
    return bank


def convert(source, destination):
    return build_v11_bank(source, destination, expected_total=1780, expected_source_hash=None)


def test_budget_half_up_fixed_denominator_and_two_rounds():
    assert recorded_budget_ceilings() == [5537, 13843, 27685]
    assert recorded_budget_ceilings(rounds=2) == [5537, 13843]
    assert recorded_budget_ceilings(55370)[1] - recorded_budget_ceilings(55370)[0] == 8306
    with pytest.raises(ValueError):
        recorded_budget_ceilings(rounds=1)


def test_build_certificate_preserves_all_payload_bytes_and_unavailable_records(archived_bank, tmp_path):
    source = archived_bank
    before = {str(p.relative_to(source)): (file_hash(p), p.stat().st_mtime_ns) for p in source.rglob('*') if p.is_file()}
    output = tmp_path/'v11'
    summary = convert(source, output)
    certificate = validate_v11_certificate(output, expected_total=1780)
    assert certificate['core']['class_caps'] == V11_CLASS_CAPS
    assert summary['budget_denominator'] == 1780 and summary['bank_public_cap_sum'] == 2560
    assert summary['available_cost_by_confidence'] == dict(exact=1409, estimated=371)
    assert 'not the full teacher bill' in summary['cost_scope']
    original = json.loads((source/'public/requests.json').read_text())
    new = json.loads((output/'public/requests.json').read_text())
    assert original[-1] == new[-1]
    for a, b in zip(original[:-1], new[:-1]):
        assert a['spec']['query_id'] == b['spec']['query_id']
        assert a['spec']['features'] == b['spec']['features']
        assert 'cost' not in b['spec']
        assert b['dependencies'] == a['dependencies']
        assert json.loads(b['spec']['cap_provenance'])['historical_cap_provenance'] == json.loads(a['spec']['cap_provenance'])
    for name, stamp in before.items():
        assert (file_hash(source/name), (source/name).stat().st_mtime_ns) == stamp
        if name.startswith('sealed/'):
            assert (source/name).read_bytes() == (output/name).read_bytes()
    with pytest.raises(ValueError, match='new isolated'):
        convert(source, source)
    with pytest.raises(ValueError, match='new isolated'):
        convert(source, output)


@pytest.mark.parametrize('change', ['payload', 'public', 'integrity', 'certificate', 'mean_cap'])
def test_bad_binding_and_usage_caps_are_rejected(archived_bank, tmp_path, change):
    source, output = archived_bank, tmp_path/'v11'
    if change == 'payload':
        q = json.loads((source/'public/requests.json').read_text())[0]['spec']['query_id']
        path = source/'sealed'/f'{q}.json'
        value = json.loads(path.read_text()); value['cost'] = 3000
        path.write_text(json.dumps(value))
        integrity = json.loads((source/'sealed/integrity.json').read_text()); integrity[q] = digest(value)
        (source/'sealed/integrity.json').write_text(json.dumps(integrity))
        with pytest.raises(ValueError, match='exceeds certified'):
            convert(source, output)
        assert not output.exists()
        return
    convert(source, output)
    path = {'public': output/'public/requests.json', 'integrity': output/'sealed/integrity.json',
            'certificate': output/CERTIFICATE, 'mean_cap': output/CERTIFICATE}[change]
    value = json.loads(path.read_text())
    if change == 'public':
        value[0]['spec']['cost_upper_bound'] = 512
    elif change == 'integrity':
        value[next(iter(value))] = 'tampered'
    elif change == 'certificate':
        value['core']['budget_denominator'] = 1
    else:
        value['core']['class_caps']['demo_attempt'] = 253
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='binding'):
        validate_v11_certificate(output, expected_total=1780)


def test_certificate_validation_never_reads_unpurchased_payloads(archived_bank, tmp_path, monkeypatch):
    output = tmp_path/'v11'; convert(archived_bank, output)
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        if path.parent.name == 'sealed' and len(path.stem) == 64:
            pytest.fail('runtime certificate check opened an unpurchased payload')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    validate_v11_certificate(output, expected_total=1780)


def test_v11_config_defaults_schedule_and_obsolete_key_rejection(tmp_path):
    raw = yaml.safe_load((ROOT/'configs/rtd/v1_1_bfcl.yaml').read_text())
    for key in ('exposure_slots_per_window', 'max_new_packages_per_window', 'slots_per_step'):
        raw.pop(key)
    raw.update(rounds=2, budget_checkpoints_bank_fraction=[.1, .25])
    path = tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(raw))
    config = cli.load_config(path)
    assert config['slots_per_step'] == config['exposure_slots_per_window'] == 40
    assert config['max_new_packages_per_window'] == 20 and config['rounds'] == 2
    for change in [dict(replay_prior_mass=.5), dict(max_new_packages_per_decision=1),
                   dict(rounds=1), dict(rounds=3), dict(slots_per_step=8),
                   dict(exposure_slots_per_window=4, max_new_packages_per_window=5)]:
        path.write_text(yaml.safe_dump(raw | change))
        with pytest.raises(ValueError):
            cli.load_config(path)
    legacy = yaml.safe_load((ROOT/'configs/rtd/v1_bfcl_c25.yaml').read_text())
    path.write_text(yaml.safe_dump(legacy | dict(exposure_slots_per_window=40)))
    with pytest.raises(ValueError, match='1.1.0'):
        cli.load_config(path)


def stable_manifest(manifest, root):
    # Paths vary across tmp fixtures. Code/hardware identity is controlled by
    # the fixture: real source hashes must, correctly, change after code edits.
    value = json.loads(json.dumps(manifest).replace(str(root), '<ROOT>'))
    value['config_hash'] = cli.digest(value['config'])
    return json.dumps(value, sort_keys=True, indent=2)+'\n'


def test_legacy_manifest_bytes_match_prechange_fixture(manifest_inputs, monkeypatch):
    c = manifest_inputs
    monkeypatch.setattr(cli, 'source_identity', lambda root: dict(hash='frozen-source'))
    monkeypatch.setattr(cli, 'evaluation_harness_identity', lambda *args: dict(hash='frozen-harness'))
    current = stable_manifest(cli.make_manifest(c.config, 'R1', c.audit), c.root)
    assert current == (ROOT/'tests/fixtures/rtd_v11/v10_manifest.json').read_text()


def test_new_manifest_labels_content_cost_and_two_round_schedule(manifest_inputs, monkeypatch):
    c = manifest_inputs
    config = yaml.safe_load((ROOT/'configs/rtd/v1_1_bfcl.yaml').read_text())
    config.update(student=c.config['student'], rounds=2, budget_checkpoints_bank_fraction=[.1, .25])
    audit = c.audit | dict(budget_denominator=55370, budget_ceilings=[5537, 13843],
        cap_certificate_sha256='certificate', cost_scope='cached-content cost; not the full teacher bill',
        public_cost_assumption='bank-dependent class caps')
    manifest = cli.make_manifest(config, 'R1', audit)
    assert manifest['version'] == 'rtd-v1.1.0-run'
    assert manifest['budget_ceilings'] == [5537, 13843] and manifest['budget_denominator'] == 55370
    assert manifest['cost_scope'] == audit['cost_scope'] and manifest['cap_certificate_sha256'] == 'certificate'
    assert 'rounds 1/2;' in manifest['checkpoint_schedule']
