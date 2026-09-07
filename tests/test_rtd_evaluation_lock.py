"""C25h: real flock/campaign concurrency, simulated sockets and CPU-only BFCL."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from bfas.rtd.evaluation_lock import (evaluation_lock, inherited_lock, port_lock_path,
                                     preferred_port, reserve_port, tag_lock_path, wait_settings)
from bfas.rtd.persistence import ComputeJournal, exclusive_run
from bfcl_fake_socket import FakeSocket, fake_network_pythonpath, install_fake_socket


@pytest.fixture(autouse=True)
def fake_socket_boundary(monkeypatch):
    install_fake_socket(monkeypatch)


def wait_for(predicate, processes=(), timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        for process in processes:
            assert process.poll() is None, f'fake run exited early: {process.returncode}'
        if time.monotonic() >= deadline:
            pytest.fail('timed out waiting for fake run synchronization')
        time.sleep(.01)


def test_wait_defaults_and_environment(monkeypatch):
    monkeypatch.delenv('RTD_EVALUATION_LOCK_TIMEOUT_SECONDS', raising=False)
    monkeypatch.delenv('RTD_EVALUATION_LOCK_LOG_INTERVAL_SECONDS', raising=False)
    assert wait_settings() == (21600, 60)
    monkeypatch.setenv('RTD_EVALUATION_LOCK_TIMEOUT_SECONDS', '.25')
    monkeypatch.setenv('RTD_EVALUATION_LOCK_LOG_INTERVAL_SECONDS', '.05')
    assert wait_settings() == (.25, .05)
    assert wait_settings(3, 2) == (3, 2)
    for timeout, interval in [(-1, 1), (float('inf'), 1), (1, 0), (1, float('nan'))]:
        with pytest.raises(ValueError):
            wait_settings(timeout, interval)


def test_job_and_gpu_ports_skip_collisions_and_legacy_locks(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '41270793')
    expected = 20000 + 41270793 % 30000
    assert preferred_port('GPU-a') == expected
    # Legacy exclusive_run has no metadata and remains respected.
    with exclusive_run(port_lock_path(tmp_path, expected).parent):
        with reserve_port(tmp_path, tag='new', gpu_uuid='GPU-a') as (selected, _):
            assert selected != expected
    monkeypatch.delenv('SLURM_JOB_ID')
    assert preferred_port('GPU-a') != preferred_port('GPU-b')
    assert preferred_port('GPU-a') == preferred_port('GPU-a')
    with reserve_port(tmp_path, tag='one', gpu_uuid='GPU-a') as (one, _):
        with reserve_port(tmp_path, tag='two', gpu_uuid='GPU-a') as (two, _):
            assert one != two
    with FakeSocket() as server:
        server.bind(('0.0.0.0', 0))
        monkeypatch.setattr('bfas.rtd.evaluation_lock.preferred_port', lambda gpu_uuid: server.getsockname()[1])
        with reserve_port(tmp_path, tag='free') as (selected, _):
            assert selected != server.getsockname()[1]


def test_wait_timeout_is_periodic_and_idle_even_on_failure(tmp_path, capsys):
    path = tag_lock_path(tmp_path, 'same')
    journal = ComputeJournal(tmp_path / 'compute.jsonl')
    def record(**event):
        journal.append('evaluation_lock_wait', accounting='idle', idle_seconds=event['wall_seconds'],
                       gpu_seconds=0., gpu_reserved_seconds=0., **event)
    with evaluation_lock(path, tag='R0'):
        before = path.read_text()
        start = time.monotonic()
        with pytest.raises(TimeoutError, match='holder=.*R0'):
            with evaluation_lock(path, tag='R1', timeout=.15, log_interval=.04, on_wait=record):
                pytest.fail('contender must not enter')
        elapsed = time.monotonic() - start
        assert path.read_text() == before
    events = ComputeJournal(journal.path).events
    assert len(events) >= 3 and events[-1]['status'] == 'timeout'
    assert all(e['gpu_seconds'] == e['gpu_reserved_seconds'] == 0 and e['accounting'] == 'idle' for e in events)
    assert .14 <= sum(e['idle_seconds'] for e in events) <= elapsed
    assert capsys.readouterr().out.count(f'waiting for evaluation lock held by {os.getpid()}/R0') >= 3
    # Closing the owner releases the permanent inode; metadata is replaced only
    # after acquiring the lease, including after a failed protected operation.
    with pytest.raises(RuntimeError):
        with evaluation_lock(path, tag='R1', timeout=0):
            raise RuntimeError('fake campaign failed')
    with evaluation_lock(path, tag='R2', timeout=0):
        assert json.loads(path.read_text())['tag'] == 'R2'


def test_child_keeps_lease_when_parent_context_closes(tmp_path):
    path = tag_lock_path(tmp_path, 'child')
    with evaluation_lock(path, tag='parent') as fd:
        assert inherited_lock(fd, path) == fd
        process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                   pass_fds=(fd,), stdin=subprocess.PIPE)
    try:
        with pytest.raises(TimeoutError):
            with evaluation_lock(path, tag='contender', timeout=.05):
                pytest.fail('child still owns lease')
    finally:
        process.communicate(timeout=5)
    with evaluation_lock(path, tag='next', timeout=0):
        pass
    with path.open('a+') as stream:
        with pytest.raises(ValueError, match='not locked'):
            inherited_lock(stream.fileno(), path)


FAKE_BFCL = r'''
import json, os
from pathlib import Path
import sys, time
args = sys.argv[1:]
def option(key): return args[args.index(key) + 1]
root = Path(os.environ['TEST_CONTROL'])
label = os.environ['TEST_LABEL']
result = Path(option('--result-dir'))
model = result / 'Qwen_Qwen3.5-4B-FC'
if args[0] == 'generate':
    assert os.environ['BFCL_PROJECT_ROOT'] == str(Path.cwd())
    assert not list(result.iterdir()), 'must never reuse a previous result directory'
    model.mkdir()
    (model / 'BFCL_v4_simple_python_result.json').write_text(json.dumps({'id': label}) + '\n')
    (root / (label + '.started')).write_text(json.dumps(dict(port=os.environ['LOCAL_SERVER_PORT'],
        result=str(result), gpu=os.environ['CUDA_VISIBLE_DEVICES'], attempt=os.environ['BFCLSTD_RUN_ID'])))
    deadline = time.monotonic() + 10
    while not (root / (label + '.release')).exists():
        assert time.monotonic() < deadline, 'fake release timed out'
        time.sleep(.01)
else:
    assert args[0] == 'evaluate'
    assert json.loads((model / 'BFCL_v4_simple_python_result.json').read_text())['id'] == label
    score = Path(option('--score-dir'))
    assert not list(score.iterdir())
    (score / 'Qwen_Qwen3.5-4B-FC').mkdir()
    (score / 'Qwen_Qwen3.5-4B-FC' / 'owner').write_text(label)
    (score / 'data_overall.csv').write_text('Overall Acc\n50\n')
    (root / (label + '.scored')).write_text(str(score))
'''


@pytest.fixture
def fake_campaign(tmp_path):
    root = tmp_path / 'checkout with spaces'
    (root / 'tools').mkdir(parents=True)
    for name in ('bfcl_std_campaign.sh', 'bfcl_campaign_lock.py'):
        shutil.copy(ROOT / 'tools' / name, root / 'tools' / name)
    (root / 'tools/bfcl_generation_check.py').write_text('# Fake complete generation\n')
    package = root / 'src/bfas/rtd'
    package.mkdir(parents=True)
    shutil.copy(ROOT / 'src/bfas/rtd/evaluation_lock.py', package)
    (package / '__init__.py').touch()
    (package.parent / '__init__.py').touch()
    network_path = fake_network_pythonpath(root)
    leaderboard = root / 'envs/bfcl/gorilla/berkeley-function-call-leaderboard'
    leaderboard.mkdir(parents=True)
    old = leaderboard / 'result_p9170/other-run'
    old.mkdir(parents=True)
    (old / 'evidence').write_text('untouched')
    venv = root / 'envs/bfcl/.venv/bin'
    venv.mkdir(parents=True)
    (venv / 'python').symlink_to(sys.executable)
    (venv / 'bfcl').write_text(f'#!{sys.executable}\n' + FAKE_BFCL)
    (venv / 'bfcl').chmod(0o755)
    (venv / 'lsof').write_text('#!/bin/bash\nexit 1\n')
    (venv / 'lsof').chmod(0o755)
    (root / 'merged').mkdir()
    (root / 'merged/model.safetensors.index.json').write_text('{}')
    control = root / 'control'
    control.mkdir()
    processes = []
    @contextmanager
    def launch(label, tag, port='auto', inherited=None):
        log = control / (label + '.log')
        env = {k: v for k, v in os.environ.items() if not k.startswith(('BFCLSTD_', 'SLURM_', 'RTD_EVALUATION_LOCK_'))}
        env.update(TEST_CONTROL=str(control), TEST_LABEL=label, BFCLSTD_PREMERGED=str(root / 'merged'),
                   PYTHONPATH=network_path,
                   BFCLSTD_PRESERVE_GENERATION='1', PATH=f"{venv}:{env['PATH']}",
                   RTD_EVALUATION_LOCK_TIMEOUT_SECONDS='2', RTD_EVALUATION_LOCK_LOG_INTERVAL_SECONDS='.03')
        fds = ()
        if inherited:
            tag_fd, port_fd = inherited
            env.update(BFCLSTD_TAG_LOCK_FD=str(tag_fd), BFCLSTD_PORT_LOCK_FD=str(port_fd))
            fds = inherited
        with log.open('w') as stream:
            process = subprocess.Popen(['bash', str(root / 'tools/bfcl_std_campaign.sh'),
                'GPU-' + label, str(port), tag], env=env, stdout=stream, stderr=subprocess.STDOUT, pass_fds=fds)
            processes.append(process)
            try:
                yield process, log
            finally:
                (control / (label + '.release')).touch()
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                assert (old / 'evidence').read_text() == 'untouched'
    return root, control, launch


def test_two_tags_generate_concurrently_with_distinct_ports_and_outputs(fake_campaign):
    root, control, launch = fake_campaign
    with launch('A', 'R0') as (one, log1), launch('B', 'R1') as (two, log2):
        wait_for(lambda: all((control / (v + '.started')).exists() for v in ('A', 'B')), [one, two])
        a, b = [json.loads((control / (v + '.started')).read_text()) for v in ('A', 'B')]
        assert a['port'] != b['port'] and a['result'] != b['result'] and a['attempt'] != b['attempt']
        assert (a['gpu'], b['gpu']) == ('GPU-A', 'GPU-B')
        for label in ('A', 'B'):
            (control / (label + '.release')).touch()
        assert one.wait(timeout=5) == two.wait(timeout=5) == 0, log1.read_text() + log2.read_text()
    assert (root / 'results/bfcl_std/R0/scoredir/owner').read_text() == 'A'
    assert (root / 'results/bfcl_std/R1/scoredir/owner').read_text() == 'B'
    assert (control / 'A.scored').read_text() != (control / 'B.scored').read_text()


def test_same_tag_waits_before_generation_and_never_reuses_attempt(fake_campaign):
    root, control, launch = fake_campaign
    with launch('A', 'shared') as (one, log1):
        wait_for(lambda: (control / 'A.started').exists(), [one])
        with launch('B', 'shared') as (two, log2):
            wait_for(lambda: 'waiting for evaluation lock held by' in log2.read_text(), [two])
            assert not (control / 'B.started').exists()
            assert '/shared' in log2.read_text()
            (control / 'A.release').touch()
            assert one.wait(timeout=5) == 0, log1.read_text()
            wait_for(lambda: (control / 'B.started').exists(), [two])
            (control / 'B.release').touch()
            assert two.wait(timeout=5) == 0, log2.read_text()
    a, b = [json.loads((control / (v + '.started')).read_text()) for v in ('A', 'B')]
    assert a['result'] != b['result']
    assert (root / 'results/bfcl_std/shared/scoredir/owner').read_text() == 'B'


def test_explicit_same_port_waits_and_parent_leases_cross_shell_boundary(fake_campaign):
    root, control, launch = fake_campaign
    with evaluation_lock(tag_lock_path(root, 'R0'), tag='R0') as tag_fd:
        with reserve_port(root, tag='R0') as (port, port_fd):
            with launch('A', 'R0', port, (tag_fd, port_fd)) as (one, log1):
                wait_for(lambda: (control / 'A.started').exists(), [one])
                with launch('B', 'R1', port) as (two, log2):
                    wait_for(lambda: 'waiting for evaluation lock held by' in log2.read_text(), [two])
                    assert not (control / 'B.started').exists()
                    # Keep the parent lease held after the fake child exits.
                    (control / 'A.release').touch()
                    assert one.wait(timeout=5) == 0, log1.read_text()
                    assert two.wait(timeout=5) != 0
                    assert 'evaluation lock timeout' in log2.read_text()
                    assert f'{os.getpid()}/R0' in log2.read_text()
    with launch('C', 'R1', port) as (three, log3):
        wait_for(lambda: (control / 'C.started').exists(), [three])
        (control / 'C.release').touch()
        assert three.wait(timeout=5) == 0, log3.read_text()
    a, c = [json.loads((control / (v + '.started')).read_text()) for v in ('A', 'C')]
    assert a['port'] == c['port'] and a['result'] != c['result']


def test_occupied_port_is_never_used(fake_campaign, monkeypatch):
    _, control, launch = fake_campaign
    with FakeSocket() as server:
        server.bind(('0.0.0.0', 0))
        server.listen()
        monkeypatch.setenv('TEST_BUSY_PORT', str(server.getsockname()[1]))
        with launch('A', 'R0', server.getsockname()[1]) as (one, log):
            assert one.wait(timeout=5) != 0
            assert not (control / 'A.started').exists()
            assert 'Address already in use' in log.read_text()
        # No lsof/kill path ran against the unrelated live listener.
        assert server.getsockname()[1] > 0


@pytest.mark.parametrize('setting', ['LOCAL_SERVER_PORT=9170', "export 'CUDA_VISIBLE_DEVICES'=0"])
def test_dotenv_cannot_redirect_reserved_resources(fake_campaign, setting):
    root, control, launch = fake_campaign
    (root / 'envs/bfcl/gorilla/berkeley-function-call-leaderboard/.env').write_text(setting + '\n')
    with launch('A', 'R0') as (one, log):
        assert one.wait(timeout=5) != 0
        assert not (control / 'A.started').exists()
        assert 'campaign resource overrides in BFCL .env' in log.read_text()
