"""C25r: live simulator diagnostics survive the actual worker protocol."""
import io
import json
from pathlib import Path
import sys
import textwrap

import pytest

from tools.behavior_atom import checker_bridge as bridge


WORKER_STUBS = textwrap.dedent('''\
    import sys
    from types import ModuleType

    class Directory:
        def __repr__(self):
            return "<Directory: /, Contents: {'café': []}>"

    def verdict():
        return {
            "valid": False,
            "error": ["state mismatch"],
            "error_type": "multi_turn:instance_state_mismatch",
            "error_message": "model state differs",
            "score": -0.0,
            "extra": [True, None, 3, 0.125, "Unicode café\\nnext line"],
            "details": {
                "model_instance_state": {"root": Directory()},
                "ground_truth_instance_state": {"nested": [Directory(), {"root": Directory()}]},
            },
        }

    def ast_checker(*args):
        return {"valid": True}

    def multi_turn(request):
        assert request["entry"] == {"id": "multi_turn_base_0"}
        assert request["truth"] == {"ground_truth": []}
        assert request["category"] == "multi_turn_base"
        return verdict() if request["result"] == [] else {"valid": True}

    enums = ModuleType("bfcl_eval.constants.enums")
    enums.Language = str
''')


@pytest.fixture
def worker_stubs(monkeypatch):
    stubs = {}
    exec(WORKER_STUBS, stubs)
    monkeypatch.setitem(sys.modules, 'bfcl_eval.constants.enums', stubs['enums'])
    monkeypatch.setattr(bridge, '_load_direct', lambda: (stubs['ast_checker'], None))
    monkeypatch.setattr(bridge, '_check_multi_turn', stubs['multi_turn'])
    monkeypatch.setattr(bridge, '_checker_version', lambda *args: 'worker-version')
    return stubs


def assert_mismatch(response):
    assert response == dict(
        valid=False, error=['state mismatch'], error_type='multi_turn:instance_state_mismatch',
        error_message='model state differs', score=-0.0,
        extra=[True, None, 3, 0.125, 'Unicode café\nnext line'],
        details=dict(model_instance_state=dict(root="<Directory: /, Contents: {'café': []}>"),
                     ground_truth_instance_state=dict(nested=[
                         "<Directory: /, Contents: {'café': []}>",
                         dict(root="<Directory: /, Contents: {'café': []}>")])),
        checker_version='worker-version')
    assert response['valid'] is False
    assert response['score'].hex() == (-0.0).hex()


def test_worker_serializes_live_objects_recursively_without_changing_verdict(worker_stubs, monkeypatch):
    request = dict(kind='multi_turn', checker_model='test-model', entry={'id': 'multi_turn_base_0'},
                   result=[], truth={'ground_truth': []}, category='multi_turn_base')
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(request) + '\n' +
                       json.dumps(dict(request, result=['success'])) + '\n'))
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', output)
    assert bridge.worker_main() == 0
    failure, success = map(json.loads, output.getvalue().splitlines())
    assert_mismatch(failure)
    assert success == dict(valid=True, error=None, checker_version='worker-version')
    # JSON-native values retain exactly the historical encoding, including -0.0.
    plain = worker_stubs['verdict']()
    del plain['details']
    assert json.dumps(plain, ensure_ascii=False, default=bridge._diagnostic_repr) == json.dumps(
        plain, ensure_ascii=False)


def test_real_bridge_round_trip_runs_worker_main_and_reuses_process(tmp_path, monkeypatch):
    python = tmp_path/'worker python'
    python.write_text(f'#!{sys.executable}\n' +
        f'import sys\nsys.path.insert(0, {str(Path(bridge.__file__).resolve().parents[2])!r})\n' +
        'from tools.behavior_atom import checker_bridge as bridge\n' + WORKER_STUBS +
        "sys.modules['bfcl_eval.constants.enums'] = enums\n"
        "bridge._load_direct = lambda: (ast_checker, None)\n"
        "bridge._checker_version = lambda *args: 'worker-version'\n"
        "bridge._check_multi_turn = multi_turn\n"
        'raise SystemExit(bridge.worker_main())\n')
    python.chmod(0o755)
    def unavailable():
        raise ImportError('force subprocess')
    monkeypatch.setattr(bridge, '_load_direct', unavailable)
    monkeypatch.setattr(bridge, '_checker_version', lambda *args: 'worker-version')
    with bridge.CheckerBridge(python=python) as checker:
        args = ({'id': 'multi_turn_base_0'}, [], {'ground_truth': []}, 'multi_turn_base')
        assert_mismatch(checker.check_multi_turn(*args))
        proc = checker._proc
        assert_mismatch(checker.check_multi_turn(*args))
        assert checker._proc is proc and proc.poll() is None
        assert checker.check_multi_turn(args[0], ['success'], *args[2:])['valid'] is True
    assert proc.poll() == 0


def test_diagnostic_strings_remove_addresses_and_survive_bad_repr():
    class Opaque:
        pass
    class Broken:
        def __repr__(self):
            raise ValueError('broken repr')
    assert bridge._diagnostic_repr(Opaque()) == bridge._diagnostic_repr(Opaque())
    assert '0x' not in bridge._diagnostic_repr(Opaque())
    assert 'unrepresentable' in bridge._diagnostic_repr(Broken())
    assert bridge._diagnostic_repr({'z', 'a'}) == "set('a', 'z')"


def test_real_gorilla_directory_diagnostics(monkeypatch):
    monkeypatch.syspath_prepend(str(bridge.BFCL))
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system import Directory
    root = Directory('/')
    root._add_directory('nested')
    root._add_file('café', 'text')
    payload = {'valid': False, 'details': {'model_instance_state': {'root': root}}}
    result = json.loads(json.dumps(payload, default=bridge._diagnostic_repr))
    assert result['valid'] is False
    assert result['details']['model_instance_state']['root'] == repr(root)
