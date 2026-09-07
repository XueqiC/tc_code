"""HPG shell contracts and downloader safety; all external work is stubbed."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/hpg_appworld_official_setup.sh",
    "scripts/sync_appworld_to_hpg.sh",
    "tools/aw_hpg_common.sh",
    "tools/aw1_train_eval_hpg.sh",
    "tools/aw2_runner_bc_hpg.sh",
    "tools/aw2_mine_more_hpg.sh",
    "scripts/aw2_bc_hpg.slurm",
    "scripts/aw2_mine_hpg.slurm",
    "scripts/rtd_run_hpg.slurm",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_bash_syntax_and_no_home_writes(script):
    subprocess.run(["bash", "-n", str(ROOT / script)], check=True)
    text = (ROOT / script).read_text()
    # Reject home-based paths and HOME reassignment, including fake-home workarounds.
    assert not re.search(r"\$HOME\b|\$\{HOME\b|\bHOME=|~/|/home/", text)
    assert "nvidia-smi" not in text
    assert not re.search(r"\bCUDA_VISIBLE_DEVICES=", text)
    assert "envs/appworld-venv" not in text
    assert "envs/appworld-data" not in text


def test_rtd_slurm_extra_args_follow_built_arguments():
    text = (ROOT / "scripts/rtd_run_hpg.slurm").read_text()
    built, tail = text.split('args+=("${extra_args[@]}")', 1)
    assert 'args+=(--run-dir "$RTD_RUN_DIR")' in built
    assert "read -r -d '' -a extra_args" in built
    assert '"${RTD_EXTRA_ARGS:-}"' in built
    assert 'tools/rtd_experiment.py "${args[@]}"' in tail


def test_setup_static_downloader_guard_and_install_provenance():
    text = (ROOT / SCRIPTS[0]).read_text()
    assert 'here=$(pwd -P)' in text
    assert '$here == "$ROOT/envs/"* && $here == "$REPO"' in text
    assert 'realpath -m -- "$here/$path"' in text
    assert 'cd -- "$REPO"\nassert_appworld_cwd' in text
    assert re.search(r'assert_appworld_cwd\s+"\$VENV/bin/appworld" download data --root \.', text)
    assert '[[ $ROOT == /blue/* ]]' in text
    assert '--python 3.12.13' in text
    assert 'python3.12 -m venv --copies "$VENV"' in text
    assert '-e "$REPO" -e "$REPO/experiments"' in text
    assert '"appworld", "0.2.0.dev0"' in text
    assert '"appworld-agents", "0.1.0.dev0"' in text
    assert 'if data_ready; then' in text
    assert '[[ -d $REPO/data/tasks ]] && data_ready' in text
    assert text.rstrip().endswith('echo APPWORLD_OFFICIAL_OK')


@pytest.mark.parametrize("script", ["scripts/aw2_bc_hpg.slurm", "scripts/aw2_mine_hpg.slurm"])
def test_slurm_resources(script):
    text = (ROOT / script).read_text()
    for option in ["account=fsu-compsci-dept", "qos=fsu-compsci-dept", "partition=hpg-b200",
                   "gres=gpu:b200:1", "cpus-per-task=14", "mem=240G", "time=06:00:00"]:
        assert f"#SBATCH --{option}" in text
    assert "--account=yd24f.fsu --qos=yd24f.fsu" in text
    assert "SLURM_SUBMIT_DIR" in text
    assert not re.search(r"\bsbatch\b", "\n".join(line for line in text.splitlines() if not line.startswith("#")))


@pytest.fixture
def project(tmp_path):
    for script in SCRIPTS:
        target = tmp_path / script
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / script, target)
    for script in ["aw2_expand_pool.py", "aw2_matched_pools.py", "aw2_mine_more.sh"]:
        shutil.copy2(ROOT / "tools" / script, tmp_path / "tools" / script)
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").symlink_to(sys.executable)
    directory = tmp_path / "data/appworld_events"
    directory.mkdir(parents=True)
    rows = [{"task_id": "a", "turn_index": turn, "_event_turn": turn,
             "_demo_len": 16, "_probe_pos": (turn + 1) / 16} for turn in (3, 7, 11, 15)]
    (directory / "pool_aw2_spread_matched.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def run_shell(project, script, *args, env=None, cwd=None):
    return subprocess.run(["bash", str(project / script), *args], cwd=cwd or project,
                          env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=20)


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


@pytest.mark.parametrize("layout", ["canonical", "serve", "serve-symlink"])
def test_hpg_environment_exports_venv_path_for_ninja(project, layout):
    vllm_venv = project / ("envs/vllm-serve/.venv" if layout == "serve" else "envs/vllm/.venv")
    executable(vllm_venv / "bin/ninja", "print('vllm-ninja')\n")
    executable(project / ".venv/bin/ninja", "print('project-ninja')\n")
    if layout == "serve-symlink":
        serving_venv = project / "envs/vllm-serve/.venv"
        serving_venv.parent.mkdir(parents=True)
        serving_venv.symlink_to("../vllm/.venv", target_is_directory=True)
    command = '''
ROOT=$(pwd -P)
source tools/aw_hpg_common.sh
aw_hpg_environment
.venv/bin/python - <<'PY'
import json, os, shutil, subprocess
print(json.dumps({"path": os.environ["PATH"], "ninja": shutil.which("ninja"),
                  "version": subprocess.check_output(["ninja", "--version"], text=True).strip()}))
PY
'''
    result = subprocess.run(["bash", "-euc", command], cwd=project,
                            env={**os.environ, "PATH": os.defpath},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    child = json.loads(result.stdout)
    assert child["path"] == f"{vllm_venv.resolve() / 'bin'}:{project / '.venv/bin'}:{os.defpath}"
    assert child["ninja"] == str(vllm_venv.resolve() / "bin/ninja")
    assert child["version"] == "vllm-ninja"


def fake_checkout(project):
    repo = project / "envs/appworld-repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "pyproject.toml").write_text('[project]\nversion="0.2.0.dev0"\n')
    (repo / "experiments/pyproject.toml").write_text('[project]\nversion="0.1.0.dev0"\n')
    return repo


@pytest.mark.parametrize("escape", ["repo", "envs", "data", ".tmp"])
def test_setup_rejects_resolved_path_escape_without_side_effects(project, escape):
    repo = fake_checkout(project)
    sentinel = project / "data/DO_NOT_DELETE"
    sentinel.write_text("project data must survive")
    if escape == "repo":
        shutil.copy2(repo / "pyproject.toml", project / "pyproject.toml")
        shutil.copytree(repo / "experiments", project / "experiments")
        shutil.rmtree(repo)
        repo.symlink_to(project, target_is_directory=True)
    elif escape == "envs":
        elsewhere = project / "outside-envs"
        (project / "envs").rename(elsewhere)
        (project / "envs").symlink_to(elsewhere, target_is_directory=True)
    else:
        (repo / escape).symlink_to(project / "data", target_is_directory=True)
    result = run_shell(project, SCRIPTS[0])
    assert result.returncode == 2
    assert "REFUSE AppWorld operation" in result.stderr
    assert sentinel.read_text() == "project data must survive"
    assert not (project / "envs/appworld-official").exists()
    assert not (project / "envs/.cache").exists()


def test_setup_refuses_non_blue_project_and_wrong_initial_cwd(project):
    fake_checkout(project)
    result = run_shell(project, SCRIPTS[0])
    assert result.returncode == 2 and "ON HPG" in result.stderr
    result = run_shell(project, SCRIPTS[0], cwd=project / "tools")
    assert result.returncode == 2 and "project root" in result.stderr
    assert not (project / "envs/appworld-official").exists()


def test_port_derivation_and_validation(project):
    command = 'source tools/aw_hpg_common.sh; aw_hpg_port "$PORT_OVERRIDE"'
    for job, override, expected in [("12345", "", "32345"), ("12346", "", "32346"),
                                    ("99999999", "", "29999"), ("12345", "08951", "8951")]:
        result = subprocess.run(["bash", "-c", command], cwd=project,
                                env={**os.environ, "SLURM_JOB_ID": job, "PORT_OVERRIDE": override},
                                capture_output=True, text=True, check=True)
        assert result.stdout.strip() == expected
    for job, override in [("", ""), ("oops", ""), ("12345", "0"), ("12345", "65536")]:
        result = subprocess.run(["bash", "-c", command], cwd=project,
                                env={**os.environ, "SLURM_JOB_ID": job, "PORT_OVERRIDE": override},
                                capture_output=True, text=True)
        assert result.returncode == 2


SLURM_ENV = {"SLURM_JOB_ID": "12345", "CUDA_VISIBLE_DEVICES": "GPU-slurm-assigned"}


def install_arm_stub(project):
    (project / "tools/aw1_train_eval_hpg.sh").write_text('''#!/bin/bash
python3 - "$@" <<'PY'
import json, os, sys
with open("calls.jsonl", "a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:], "env": dict(os.environ)}) + "\\n")
PY
echo "[aw1] $1 dev run exit=${FAKE_DEV_EXIT:-0}"
if [[ ${FAKE_NO_AGGREGATE:-0} == 0 ]]; then
  echo ' aggregate   |         21.1         |           10.5' > "logs/awoff_${1}_dev.log"
else
  : > "logs/awoff_${1}_dev.log"
fi
exit "${FAKE_TRAIN_EXIT:-0}"
''')


@pytest.mark.parametrize("selection,suffixes", [(None, ["e2", "x2"]), ("B", ["e2"]),
                                               ("C", ["x2"]), ("C,B,B", ["e2", "x2"]),
                                               ("aw2_pref_spread_x2_s0", ["x2"])])
def test_runner_selected_arms_keep_rai_pins_and_slurm_gpu(project, selection, suffixes):
    install_arm_stub(project)
    (project / "data/appworld_events/pool_aw2_spread_x2.jsonl").write_text("{}\n")
    env = {**SLURM_ENV, "AW_DDPO_EPOCHS": "99", "AW_DDPO_LR": "1", "AW_DDPO_WEIGHT": "pos",
           "AW_DDPO_WEIGHT_PERMUTE": "global", "AW_DDPO_REF_FREE": "1", "AW_ANCHOR_ADAPTIVE": "1",
           "AW2_ARMS": "B C" if selection is None else selection}
    result = run_shell(project, "tools/aw2_runner_bc_hpg.sh", env=env)
    assert result.returncode == 0, result.stderr + result.stdout
    calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
    assert [call["args"][0] for call in calls] == [f"aw2_pref_spread_{suffix}_s0" for suffix in suffixes]
    for call, suffix in zip(calls, suffixes):
        assert call["args"][2:] == ["ddpo", "slurm", "32345"]
        for key, value in {"AW_DDPO_EPOCHS": "2" if suffix == "e2" else "1", "AW_DDPO_LR": "5e-6",
                           "AW_GRAD_CKPT": "1", "AW_DISTILL": "ddpo", "AW_SFT_SKIP": "1",
                           "AW_DDPO_BETA": "0.1", "AW_DDPO_WEIGHT": "none", "AW_DDPO_REF_FREE": "0",
                           "AW_DDPO_SPAN_ONLY": "0", "AW_ANCHOR_ADAPTIVE": "0", **SLURM_ENV}.items():
            assert call["env"][key] == value
        assert "AW_DDPO_WEIGHT_PERMUTE" not in call["env"]
        assert call["env"]["HF_HOME"] == "/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache"
    assert "ALL DONE" in result.stdout


@pytest.mark.parametrize("failure", [{"FAKE_TRAIN_EXIT": "7"}, {"FAKE_DEV_EXIT": "1"},
                                     {"FAKE_NO_AGGREGATE": "1"}])
def test_runner_stops_on_failure_or_missing_official_report(project, failure):
    install_arm_stub(project)
    result = run_shell(project, "tools/aw2_runner_bc_hpg.sh", env={**SLURM_ENV, "AW2_ARMS": "B C", **failure})
    assert result.returncode != 0
    assert len((project / "calls.jsonl").read_text().splitlines()) == 1
    assert "ALL DONE" not in result.stdout


@pytest.mark.parametrize("env", [{"AW2_ARMS": "B typo"}, {"CUDA_VISIBLE_DEVICES": "0,1"},
                                 {"CUDA_VISIBLE_DEVICES": ""}, {"SLURM_JOB_ID": ""}])
def test_runner_rejects_bad_selection_or_allocation_before_launch(project, env):
    install_arm_stub(project)
    result = run_shell(project, "tools/aw2_runner_bc_hpg.sh", env={**SLURM_ENV, **env})
    assert result.returncode == 2
    assert not (project / "calls.jsonl").exists()
    assert not (project / "logs").exists()


def test_mining_dry_run_matches_rai_plan_and_uses_job_port(project):
    rai = run_shell(project, "tools/aw2_mine_more.sh", "32345", "--dry-run")
    hpg = run_shell(project, "tools/aw2_mine_more_hpg.sh", "--dry-run", env=SLURM_ENV)
    assert rai.returncode == hpg.returncode == 0, rai.stderr + hpg.stderr
    assert rai.stdout == hpg.stdout
    command = shlex.split(hpg.stdout.strip())
    for flag, value in {"--port": "32345", "--probe-positions": "0.125,0.375,0.625,0.875",
                        "--k": "3", "--probe-samples": "3", "--equality": "api_effects",
                        "--run-tag": "aw2_spread_more_k3", "--env-seed": "1"}.items():
        assert command[command.index(flag) + 1] == value
    assert not (project / "logs").exists()
    assert not (project / "data/appworld_events/aw2_spread_more_k3.jsonl").exists()


def test_mining_retains_official_bridge_and_output_sandbox():
    text = (ROOT / "tools/aw2_mine_more_hpg.sh").read_text()
    for fragment in ["aw_hpg_check_official", "envs/appworld-official/.venv/bin/python",
                     "APPWORLD_ROOT=$ROOT/envs/appworld-repo", "PYTHONPATH=$ROOT/src",
                     '[[ ! -e $OUT && ! -e $OUT.done ]]', 'grep -Fxq -- "$task" "$OUT.done"',
                     'trap cleanup EXIT', '--host 127.0.0.1', 'for ((i=0; i<120; i++))']:
        assert fragment in text
    bridge = (ROOT / "tools/appworld_event_mine.py").read_text()
    assert 'env["APPWORLD_ROOT"] = str(REPO)' in bridge
    assert 'cwd=str(REPO)' in bridge
    slurm = (ROOT / "scripts/aw2_mine_hpg.slurm").read_text()
    assert 'tools/aw2_expand_pool.py --mined data/appworld_events/aw2_spread_more_k3.jsonl' in slurm


@pytest.mark.parametrize("with_x2", [False, True])
def test_sync_only_requested_inputs_and_excludes_data_symlink(project, with_x2):
    repo = fake_checkout(project)
    (repo / "data").symlink_to("../appworld-official/data", target_is_directory=True)
    demos = project / "results/bfas/appworld/collect_shared/demos.json"
    demos.parent.mkdir(parents=True)
    demos.write_text("{}\n")
    if with_x2:
        (project / "data/appworld_events/pool_aw2_spread_x2.jsonl").write_text("{}\n")
    body = '''import json, sys
from pathlib import Path
with open("sync-calls.jsonl", "a") as stream:
    stream.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + "\\n")
'''
    for name in ("ssh", "rsync"):
        executable(project / "fake-bin" / name, body)
    result = run_shell(project, "scripts/sync_appworld_to_hpg.sh",
                       env={"PATH": f"{project / 'fake-bin'}:{os.environ['PATH']}"})
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (project / "sync-calls.jsonl").read_text().splitlines()]
    assert calls[0][:2] == ["ssh", "hpg"]
    transfers = calls[1:]
    assert transfers[0][-2] == "envs/appworld-repo/"
    for excluded in (".git", "/experiments/outputs", "/data", ".venv"):
        assert ("--exclude", excluded) in list(zip(transfers[0], transfers[0][1:]))
    assert len(transfers) == 3 + int(with_x2)
    assert all(call[-1].startswith("hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/") for call in transfers)
    assert "hub_merged" not in json.dumps(calls)
    assert "appworld-official/.venv" not in json.dumps(calls)
    assert "--delete" not in json.dumps(calls)


def install_pipeline_stubs(project):
    repo = fake_checkout(project)
    (repo / "data/tasks").mkdir(parents=True)
    config = repo / "experiments/configs/simplified_react_code_agent/local"
    config.mkdir(parents=True)
    (config / "qwen35-4b-base_dev.jsonnet").write_text(
        '{"name": "qwen35-4b-base", "base_url": "http://localhost:8950/v1", "dataset": "dev"}\n')
    # The official interpreter only receives metadata preflight here; no real package is loaded.
    executable(project / "envs/appworld-official/.venv/bin/python", "import sys\nsys.stdin.read()\n")
    recorder = '''import json, os, sys
from pathlib import Path
root = Path(os.environ["FAKE_ROOT"])
def record(kind):
    with (root / "pipeline.jsonl").open("a") as stream:
        stream.write(json.dumps({"kind": kind, "args": sys.argv[1:], "cwd": str(Path.cwd()),
                                 "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
'''
    (project / ".venv/bin/python").unlink()
    executable(project / ".venv/bin/python", recorder + '''
if sys.argv[1] == "src/appworld_train.py":
    record("train")
elif sys.argv[1] == "tools/bfcl_hub_merge_export.py":
    record("merge")
    directory = Path(sys.argv[sys.argv.index("--out") + 1])
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
else:
    sys.stdin.read()  # Port bind preflight is stubbed; no real listener is opened.
''')
    executable(project / "envs/appworld-official/.venv/bin/appworld", recorder + '''
record("official")
print("Text Evaluation Report")
if os.environ.get("FAKE_NO_AGGREGATE") != "1":
    print(" aggregate | 21.1 | 10.5")
sys.exit(int(os.environ.get("FAKE_DEV_EXIT", "0")))
''')
    executable(project / "envs/vllm-serve/.venv/bin/vllm", recorder + '''
import signal
record("vllm")
def finish(*args):
    (root / "vllm-stopped").touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, finish)
(root / "vllm-ready").touch()
while True:
    signal.pause()
''')
    executable(project / "fake-bin/curl", '''import os, sys
from pathlib import Path
if not (Path(os.environ["FAKE_ROOT"]) / "vllm-ready").exists():
    sys.exit(1)
print('{"data": [{"id": "test_arm"}]}')
''')
    executable(project / "fake-bin/sleep", "import time\ntime.sleep(0.01)\n")


@pytest.mark.parametrize("failure,expected", [({}, 0), ({"FAKE_DEV_EXIT": "7"}, 7),
                                             ({"FAKE_NO_AGGREGATE": "1"}, 1)])
def test_train_merge_serve_official_contract_and_cleanup(project, failure, expected):
    install_pipeline_stubs(project)
    result = run_shell(project, "tools/aw1_train_eval_hpg.sh", "test_arm",
                       "data/appworld_events/pool_aw2_spread_matched.jsonl", "ddpo", "99",
                       env={**SLURM_ENV, "FAKE_ROOT": str(project), **failure,
                            "PATH": f"{project / 'fake-bin'}:{os.environ['PATH']}"})
    assert result.returncode == expected, result.stdout + result.stderr
    calls = [json.loads(line) for line in (project / "pipeline.jsonl").read_text().splitlines()]
    assert [call["kind"] for call in calls] == ["train", "merge", "vllm", "official"]
    assert all(call["gpu"] == SLURM_ENV["CUDA_VISIBLE_DEVICES"] for call in calls)
    assert calls[-1]["cwd"] == str(project / "envs/appworld-repo")
    assert calls[-1]["args"] == ["run", "simplified_react_code_agent/local/test_arm_dev",
                                  "--root", ".", "--without-setup", "--num-processes", "4"]
    serve = calls[2]["args"]
    assert serve[serve.index("--port") + 1] == "32345"
    assert serve[serve.index("--served-model-name") + 1] == "test_arm"
    assert serve[serve.index("--default-chat-template-kwargs") + 1] == '{"enable_thinking": false}'
    config = project / "envs/appworld-repo/experiments/configs/simplified_react_code_agent/local/test_arm_dev.jsonnet"
    assert json.loads(config.read_text()) == {"name": "test_arm", "base_url": "http://localhost:32345/v1", "dataset": "dev"}
    assert (project / "vllm-stopped").exists()
    assert (project / "logs/aw1_train_test_arm.log").exists()
    assert (project / "logs/vllm_aw1_test_arm.log").exists()
    assert ("[aw1] test_arm DONE" in result.stdout) == (expected == 0)
