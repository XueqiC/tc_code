"""D10b deadline and read-only phase table checks; no model or GPU loading."""
import json

import pytest

from bfas.rtd import cli, runtime
from bfas.rtd.persistence import ComputeJournal
from tools.rtd_v11_phase_times import main as phase_main, phase_times


@pytest.mark.parametrize('command', ['smoke', 'run', 'resume'])
@pytest.mark.parametrize('seconds', [None, 7200])
def test_cli_smoke_deadline_default_and_override(monkeypatch, command, seconds):
    received = []
    monkeypatch.setattr(cli, 'run_command', lambda args: received.append(args))
    argv = [command, '--arm', 'R0']
    if seconds is not None:
        argv += ['--smoke-deadline-seconds', str(seconds)]
    cli.main(argv)
    assert received[0].smoke_deadline_seconds == (900 if seconds is None else seconds)


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'nan'])
def test_cli_rejects_invalid_smoke_deadline(value):
    with pytest.raises(SystemExit) as exc:
        cli.main(['smoke', '--smoke-deadline-seconds', value])
    assert exc.value.code == 2


@pytest.mark.parametrize('seconds', [900, 7200])
def test_smoke_config_journal_alarm_share_deadline(tmp_path, monkeypatch, seconds):
    from bfas.rtd import preflight
    monkeypatch.setattr(preflight, 'preflight_config', lambda *a, **kw: None)
    config = cli.load_config(cli.ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    monkeypatch.setattr(cli, 'load_config', lambda path: config)
    monkeypatch.setattr(cli, 'bank_audit', lambda config: {})
    monkeypatch.setattr(cli, 'make_manifest', lambda *a, **kw:
                        dict(hardware=dict(uuid='test', gpu='test', memory=0)))
    monkeypatch.setattr(cli.time, 'monotonic', lambda: 100.)
    handlers, timers = [], []
    monkeypatch.setattr(cli.signal, 'signal', lambda sig, handler: handlers.append(handler))
    monkeypatch.setattr(cli.signal, 'setitimer', lambda kind, seconds: timers.append(seconds))

    def load_boundary(config, manifest, journal):
        assert config['smoke_override'] == dict(parents_per_fold=2, slots=2, rollouts=1, windows=1,
            baseline='action-independent zero', max_seconds=seconds)
        assert journal.deadline == 100.+seconds and journal.deadline_seconds == seconds
        handlers[0](None, None)

    monkeypatch.setattr(runtime, 'load_backend', load_boundary)
    argv = ['smoke', '--arm', 'R0', '--run-dir', str(tmp_path)]
    if seconds != 900:
        argv += ['--smoke-deadline-seconds', str(seconds)]
    with pytest.raises(TimeoutError, match=f'smoke exceeded {seconds} seconds; resume state retained'):
        cli.main(argv)
    assert timers == [seconds, 0] and handlers[-1] is None


def test_journal_deadline_reports_configured_seconds(tmp_path):
    journal = ComputeJournal(tmp_path/'compute.jsonl', deadline=0, deadline_seconds=7200)
    with pytest.raises(TimeoutError, match='smoke exceeded 7200 seconds'):
        with journal.measure('generation'):
            pytest.fail('expired operation must not start')
    assert not journal.events


def test_phase_table_sums_attempts_and_takes_peak_without_modifying_journal(tmp_path, capsys):
    rows = [dict(kind='compute_begin', operation='generation')]
    for status, wall, peak in [('complete', 2., 2**30), ('failed', 3., 2**29)]:
        rows.append(dict(kind='compute_end', operation='generation', status=status,
            wall_seconds=wall, gpu_seconds=wall, peak_allocated_bytes=peak, peak_reserved_bytes=2*peak))
    path = tmp_path/'compute.jsonl'
    original = ''.join(json.dumps(row)+'\n' for row in rows).encode()
    path.write_bytes(original)
    assert phase_times(tmp_path) == dict(generation=dict(n=2, wall_seconds=5., gpu_seconds=5.,
        peak_allocated_bytes=2**30, peak_reserved_bytes=2**31))
    assert phase_main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert output.splitlines()[1].split() == ['generation', '2', '5.000', '5.000', '1.000', '2.000']
    assert 'nested operations overlap' in output and 'failed attempts' in output
    assert path.read_bytes() == original
    path.write_text('')
    assert phase_main([str(tmp_path)]) == 0
