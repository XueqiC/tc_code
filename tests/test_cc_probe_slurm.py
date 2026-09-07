"""C19c launcher regression checks; stop before any model or GPU work."""

import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cc_probe_hpg.slurm"


def test_slurm_submit_root_and_ninja_path_contract():
    text = SCRIPT.read_text()
    assert 'if [[ -v SLURM_SUBMIT_DIR ]]; then' in text
    assert 'CC_ROOT=$(cd -- "${SLURM_SUBMIT_DIR:?}" && pwd)' in text
    assert 'CC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)' in text
    assert '! -f "$CC_ROOT/tools/cc_pairs.py" || ! -d "$CC_ROOT/data"' in text
    assert text.index("Invalid CC_ROOT") < text.index("mkdir -p logs")
    path_export = 'export PATH="$CC_VLLM_VENV/bin:$CC_ROOT/.venv/bin:$PATH"'
    assert path_export in text
    assert text.index(path_export) < text.index('"$CC_VLLM" serve')


@pytest.mark.parametrize("mode", ["slurm-canonical", "slurm-serve", "rai"])
def test_launcher_root_and_path_from_unrelated_cwd(tmp_path, mode):
    root = tmp_path / "checkout with spaces"
    (root / "tools").mkdir(parents=True)
    (root / "tools/cc_pairs.py").touch()
    (root / "data").mkdir()
    venv = root / ("envs/vllm-serve/.venv" if mode == "slurm-serve" else "envs/vllm/.venv")
    (venv / "bin").mkdir(parents=True)
    script = (root / "scripts/probe.sh" if mode == "rai"
              else tmp_path / "slurm/spool/job/slurm_script")
    script.parent.mkdir(parents=True)
    script.write_text(SCRIPT.read_text())
    # The first Python invocation captures startup state and stops the launcher.
    stub = tmp_path / "python-stub"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$PATH"\nexit 73\n')
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_", "CC_"))}
    env.update(CC_PYTHON=str(stub), PATH=os.defpath)
    if mode != "rai":
        env.update(SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID="41215087")
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 73, result.stderr
    expected_path = (os.defpath if mode == "rai"
                     else f"{venv}/bin:{root}/.venv/bin:{os.defpath}")
    assert result.stdout.splitlines() == [str(root), expected_path]
    assert (root / "logs").is_dir()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("missing", ["tools/cc_pairs.py", "data"])
def test_invalid_submit_root_fails_before_creating_logs(tmp_path, missing):
    (tmp_path / "tools").mkdir()
    if missing != "tools/cc_pairs.py":
        (tmp_path / "tools/cc_pairs.py").touch()
    if missing != "data":
        (tmp_path / "data").mkdir()
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path,
                            env={**os.environ, "SLURM_SUBMIT_DIR": str(tmp_path)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert f"Invalid CC_ROOT '{tmp_path}'" in result.stderr
    assert "expected tools/cc_pairs.py and data/" in result.stderr
    assert not (tmp_path / "logs").exists()
