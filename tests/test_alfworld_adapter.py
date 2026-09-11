"""CPU subprocess RPC tests; stub executables never import ALFWorld or a model."""
import json
import logging
import os
from pathlib import Path
import sys
import textwrap
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from bfas.adapters import alfworld as adapter
from bfas.rtd.benchmarks.alfworld_support import BoundedEnvBridge


def stub_worker(tmp_path, monkeypatch, body):
    script = tmp_path / 'stub-worker'
    script.write_text(f'#!{sys.executable}\n' + textwrap.dedent(body))
    script.chmod(0o700)
    monkeypatch.setattr(adapter, 'ENV_PYTHON', script)


READY = '''
import json, os, sys, time
print(json.dumps(dict(bfas_worker=True, op='ready')), flush=True)
print(json.dumps(dict(bfas_worker=True, op='state', observation='reset')), flush=True)
'''


@pytest.fixture(params=['adapter', 'bounded'])
def bridge_factory(request):
    if request.param == 'bounded':
        return lambda **kw: BoundedEnvBridge('game/trial', **kw)
    return lambda **kw: adapter._EnvBridge('train', 'game/trial', **kw)


@pytest.mark.parametrize('fault', ['exception', 'exit', 'timeout', 'id_mismatch'])
def test_alfworld_adapter_step_diagnostics(tmp_path, monkeypatch, caplog, bridge_factory, fault):
    body = READY + f'''
request = json.loads(sys.stdin.readline())
sys.stderr.write('x' * 10000 + '\\nworker stderr sentinel\\n')
sys.stderr.flush()
fault = {fault!r}
if fault == 'timeout':
    time.sleep(10)
elif fault == 'exception':
    print(json.dumps(dict(bfas_worker=True, op='error',
        worker_exception='RuntimeError: step exploded', worker_traceback='remote traceback sentinel')), flush=True)
    sys.exit(7)
elif fault == 'id_mismatch':
    print(json.dumps(dict(bfas_worker=True, op='state', id=request['id'] + 1)), flush=True)
    time.sleep(10)
else:
    sys.exit(9)
'''
    stub_worker(tmp_path, monkeypatch, body)
    bridge = bridge_factory(timeout=.2, reset_timeout=3.)
    process = bridge.process
    directory = Path(bridge._tmp.name) if hasattr(bridge, '_tmp') else None
    try:
        assert bridge._read()['observation'] == 'reset'
        with pytest.raises(adapter.ALFWorldRPCError) as caught:
            bridge.step('look')
        diagnostic = caught.value.diagnostics
        assert diagnostic['rpc_operation'] == 'step'
        assert diagnostic['worker_pid'] == process.pid
        assert 0 <= diagnostic['rpc_elapsed_seconds'] < 3
        assert diagnostic['timeout_seconds'] == .2
        assert diagnostic['stderr_tail'].endswith('worker stderr sentinel\n')
        assert len(diagnostic['stderr_tail']) <= 8192
        if fault == 'exception':
            assert diagnostic['worker_exception'] == 'RuntimeError: step exploded'
            assert diagnostic['worker_traceback'] == 'remote traceback sentinel'
            assert diagnostic['exit_code'] == 7
        elif fault == 'exit':
            assert diagnostic['exit_code'] == 9
        elif fault == 'timeout':
            assert diagnostic['exit_code'] is None  # alive when the timeout fired
            assert diagnostic['rpc_elapsed_seconds'] >= .2
            assert 'timeout' in diagnostic['worker_exception']
        else:
            assert 'id mismatch' in diagnostic['worker_exception']
        assert any(r.levelno == logging.ERROR and 'worker stderr sentinel' in r.message
                   and 'rpc_elapsed_seconds' in r.message and 'exit_code' in r.message for r in caplog.records)
    finally:
        bridge.close()
    assert process.poll() is not None
    assert directory is None or not directory.exists()
    # These remain useful after cleanup removes worker.stderr.
    assert caught.value.diagnostics == diagnostic


@pytest.mark.parametrize('fault', ['timeout', 'exit'])
def test_alfworld_adapter_reset_diagnostics_and_cleanup(tmp_path, monkeypatch, bridge_factory, fault):
    sentinel = tmp_path / 'worker.json'
    stub_worker(tmp_path, monkeypatch, f'''
        import json, os, sys, time
        with open({str(sentinel)!r}, 'w') as f:
            json.dump(dict(pid=os.getpid(), cwd=os.getcwd()), f)
        print('reset stderr sentinel', file=sys.stderr, flush=True)
        if {fault!r} == 'timeout':
            time.sleep(10)
        sys.exit(5)
    ''')
    with pytest.raises(adapter.ALFWorldRPCError) as caught:
        bridge_factory(timeout=.1, reset_timeout=.5)
    diagnostic = caught.value.diagnostics
    assert diagnostic['rpc_operation'] == 'reset'
    assert diagnostic['timeout_seconds'] == .5
    assert diagnostic['stderr_tail'] == 'reset stderr sentinel\n'
    assert diagnostic['exit_code'] == (5 if fault == 'exit' else None)
    assert diagnostic['rpc_elapsed_seconds'] >= (.5 if fault == 'timeout' else 0)
    values = json.loads(sentinel.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(values['pid'], 0)
    if values['cwd'] != str(adapter.ROOT):
        assert not Path(values['cwd']).exists()


@pytest.mark.parametrize('overrides,expected', [({}, (120., 300.)),
    ({'BFAS_ALFWORLD_STEP_TIMEOUT': '240', 'BFAS_ALFWORLD_RESET_TIMEOUT': '600'}, (240., 600.))])
def test_alfworld_adapter_timeout_config_and_policy_think_time(tmp_path, monkeypatch, bridge_factory,
                                                             overrides, expected):
    for key in ('BFAS_ALFWORLD_STEP_TIMEOUT', 'BFAS_ALFWORLD_RESET_TIMEOUT'):
        monkeypatch.delenv(key, raising=False)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    stub_worker(tmp_path, monkeypatch, READY + '''
request = json.loads(sys.stdin.readline())
print(json.dumps(dict(bfas_worker=True, op='state', id=request['id'], observation='step')), flush=True)
''')
    bridge = bridge_factory()
    try:
        assert (bridge.timeout, bridge.reset_timeout) == expected
        assert bridge._read()['observation'] == 'reset'
        # Simulate long policy generation without actually sleeping.
        monotonic = time.monotonic
        monkeypatch.setattr(adapter.time, 'monotonic', lambda: monotonic() + 1000)
        assert bridge.step('look')['observation'] == 'step'
    finally:
        bridge.close()


@pytest.mark.parametrize('name', ['BFAS_ALFWORLD_STEP_TIMEOUT', 'BFAS_ALFWORLD_RESET_TIMEOUT'])
@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'invalid'])
def test_alfworld_adapter_invalid_timeout_before_spawn(monkeypatch, bridge_factory, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(adapter.subprocess, 'Popen', lambda *a, **kw: pytest.fail('spawned invalid config'))
    with pytest.raises(ValueError):
        bridge_factory()


def test_alfworld_adapter_explicit_episode_deadline_still_supported(tmp_path, monkeypatch):
    stub_worker(tmp_path, monkeypatch, READY + 'time.sleep(10)\n')
    bridge = BoundedEnvBridge('game/trial', timeout=3., episode_timeout=30.)
    try:
        bridge._read()
        monotonic = time.monotonic
        monkeypatch.setattr(adapter.time, 'monotonic', lambda: monotonic() + 31)
        with pytest.raises(adapter.ALFWorldRPCError, match='episode_timeout=30.0') as caught:
            bridge.step('look')
        assert caught.value.diagnostics['episode_timeout_seconds'] == 30.
    finally:
        bridge.close()


def test_alfworld_adapter_cleanup_includes_owned_stdout_child(tmp_path, monkeypatch, bridge_factory):
    stub_worker(tmp_path, monkeypatch, READY + '''
import subprocess
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
print(json.dumps(dict(bfas_worker=True, op='child', pid=child.pid, pgid=os.getpgrp())), flush=True)
time.sleep(30)
''')
    bridge = bridge_factory(timeout=3.)
    try:
        bridge._read()
        child = bridge._read()
        assert child['pgid'] == bridge.process.pid
        assert child['pgid'] != os.getpgrp()
    finally:
        started = time.monotonic()
        bridge.close()
    assert time.monotonic() - started < 3
    assert not bridge._reader.is_alive()


def test_alfworld_adapter_worker_reports_exception_on_both_channels(monkeypatch, capsys):
    def broken(*args):
        raise RuntimeError('worker step sentinel')
    monkeypatch.setattr(adapter, '_worker_episode', broken)
    assert adapter._worker('train', 'game/trial') == 1
    output = capsys.readouterr()
    response = json.loads(output.out)
    assert response['op'] == 'error' and response['bfas_worker'] is True
    assert response['worker_exception'] == 'RuntimeError: worker step sentinel'
    assert 'RuntimeError: worker step sentinel' in response['worker_traceback']
    assert 'Traceback' in output.err and 'worker step sentinel' in output.err


@pytest.mark.parametrize('old,new', [
    ('bool(info["won"][0])', 'False'),
    ('value.get("bfas_worker") is True', 'value.get("bfas_worker") is False'),
])
def test_registry_identity_covers_worker_body_and_shared_rpc_reader(old, new):
    from bfas.rtd.benchmarks.alfworld_identity import scoring_projection
    name = 'src/bfas/adapters/alfworld.py'
    content = Path(adapter.__file__).read_text()
    assert old in content
    assert scoring_projection(name, content) != scoring_projection(name, content.replace(old, new))
