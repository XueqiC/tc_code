"""C22 coverage/retry integration using real BFCL data and a fake generator."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
from bfcl_fake_socket import fake_network_pythonpath


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"

# Only the bfcl CLI is stubbed: the completeness check reads the vendored data
# through the real BFCL loader. The stub appends missing IDs just as generate
# does, and evaluation asserts that existing rows survived the retries.
FAKE_BFCL = r'''
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from bfcl_eval.utils import (extract_test_category_from_id,
    get_directory_structure_by_category, get_file_name_by_category,
    is_format_sensitivity, load_dataset_entry, parse_test_category_argument)

args = sys.argv[1:]
root = Path(os.environ["TEST_ROOT"])
calls = root / "calls.jsonl"
previous = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
with calls.open("a") as stream:
    stream.write(json.dumps(args) + "\n")
def option(name):
    return args[args.index(name) + 1]

model_dir = Path(option("--result-dir")) / "Qwen_Qwen3.5-4B-FC"
scenario = os.environ["TEST_SCENARIO"]
if args[0] == "generate":
    category_arg = option("--test-category")
    initial = category_arg == "all"
    assert ("--allow-overwrite" in args) == initial
    for category in parse_test_category_argument([category_arg]):
        if is_format_sensitivity(category):
            continue
        grouped = {}
        for entry in load_dataset_entry(category):
            grouped.setdefault(extract_test_category_from_id(entry["id"]), []).append(entry["id"])
        for file_category, ids in grouped.items():
            path = (model_dir / get_directory_structure_by_category(category)
                    / get_file_name_by_category(file_category, is_result_file=True))
            if initial:
                if category == "multi_turn_long_context":
                    if scenario in {"recover", "exhausted", "retry_error"}:
                        ids = ids[:26]
                    elif scenario == "absent":
                        continue
                    elif scenario == "duplicate":
                        ids = ids[:-1] + [ids[0]]
                if scenario == "memory" and category == "memory_kv":
                    ids = ids[:1]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps({"id": tid, "original": True}) + "\n" for tid in ids))
            else:
                if scenario == "retry_error":
                    sys.exit(9)
                if scenario == "exhausted":
                    continue
                if scenario == "recover" and len(previous) == 1:
                    ids = ids[:100]
                existing = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as stream:
                    for tid in ids:
                        if tid not in existing:
                            stream.write(json.dumps({"id": tid}) + "\n")
else:
    assert args[0] == "evaluate"
    # Complete categories must never have been touched by resume generation.
    path = model_dir / "non_live/BFCL_v4_simple_python_result.json"
    assert all(json.loads(line)["original"] for line in path.read_text().splitlines())
    if scenario in {"recover", "complete"}:
        path = model_dir / "multi_turn/BFCL_v4_multi_turn_long_context_result.json"
        assert sum(bool(json.loads(line).get("original")) for line in path.read_text().splitlines()) == (26 if scenario == "recover" else 200)
    score = Path(option("--score-dir"))
    (score / "Qwen_Qwen3.5-4B-FC").mkdir(parents=True)
    (score / "data_overall.csv").write_text("Overall Acc\n46.06\n")
'''


@pytest.mark.parametrize("scenario,retries,success", [
    ("complete", [], True),
    ("recover", ["multi_turn_long_context"] * 2, True),
    ("absent", ["multi_turn_long_context"], True),
    ("memory", ["memory_kv"], True),
    ("exhausted", ["multi_turn_long_context"] * 2, False),
    ("retry_error", ["multi_turn_long_context"] * 2, False),
    ("duplicate", ["multi_turn_long_context"] * 2, False),
])
def test_campaign_completeness_and_resume(tmp_path, scenario, retries, success):
    if not (LEADERBOARD / "bfcl_eval/utils.py").is_file():
        pytest.skip("Vendored BFCL data required")
    root = tmp_path / "checkout with spaces"
    (root / "tools").mkdir(parents=True)
    for name in ("bfcl_std_campaign.sh", "bfcl_generation_check.py", "bfcl_campaign_lock.py"):
        shutil.copy(ROOT / "tools" / name, root / "tools")
    package = root / 'src/bfas/rtd'
    package.mkdir(parents=True)
    shutil.copy(ROOT / 'src/bfas/rtd/evaluation_lock.py', package)
    (package / '__init__.py').touch()
    (package.parent / '__init__.py').touch()
    leaderboard = root / LEADERBOARD.relative_to(ROOT)
    leaderboard.mkdir(parents=True)
    (leaderboard / "bfcl_eval").symlink_to(LEADERBOARD / "bfcl_eval", target_is_directory=True)
    # Cover both executable locations supported by the campaign.
    venv = root / ("envs/bfcl-venv/bin" if scenario == "memory" else "envs/bfcl/.venv/bin")
    venv.mkdir(parents=True)
    # Invoke the original venv path: relocating its Python symlink would lose
    # pyvenv.cfg and the BFCL loader's filelock dependency.
    (venv / "python").write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    (venv / "python").chmod(0o755)
    bfcl = venv / "bfcl"
    bfcl.write_text(f"#!{sys.executable}\n" + FAKE_BFCL)
    bfcl.chmod(0o755)
    # No real process inspection/killing, even when testing retry cleanup.
    (venv / "lsof").write_text("#!/bin/bash\nexit 1\n")
    (venv / "lsof").chmod(0o755)
    env = {**os.environ, "TEST_ROOT": str(root), "TEST_SCENARIO": scenario,
           "PYTHONPATH": fake_network_pythonpath(root),
           "BFCL_PROJECT_ROOT": str(leaderboard), "PATH": f"{venv}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", str(root / "tools/bfcl_std_campaign.sh"), "0", "43505", "base"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    log = (root / "logs/bfclstd_gen_base.log").read_text()
    assert (result.returncode == 0) == success, result.stdout + result.stderr + log
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    generations = [call for call in calls if call[0] == "generate"]
    assert [call[call.index("--test-category") + 1] for call in generations] == ["all", *retries]
    assert all("--allow-overwrite" not in call for call in generations[1:])
    assert (any(call[0] == "evaluate" for call in calls)) == success
    assert (root / "results/bfcl_std/base/data_overall.csv").exists() == success
    assert "simple_python=400/400 missing=0" in log
    assert "web_search_base=100/100 missing=0" in log
    assert "web_search_no_snippet=100/100 missing=0" in log
    assert "format_sensitivity=" not in log
    if scenario in {"recover", "exhausted", "retry_error"}:
        assert "multi_turn_long_context=26/200 missing=174" in log
    if scenario == "recover":
        assert "multi_turn_long_context=100/200 missing=100" in log
        assert "multi_turn_long_context=200/200 missing=0" in log
    if scenario == "absent":
        assert "multi_turn_long_context=0/200 missing=200" in log
    if scenario == "memory":
        assert "memory_kv=1/155 missing=154" in log
        assert "memory_kv_prereq=1/37 missing=36" in log
        assert "memory_kv_prereq=37/37 missing=0" in log
    if not success:
        assert "GENERATION INCOMPLETE after 2 retries" in result.stdout
        assert "multi_turn_long_context=" in result.stdout
        assert "CAMPAIGN FAILED" in result.stdout
        assert "CAMPAIGN COMPLETE" not in result.stdout
    if scenario == "retry_error":
        assert "GEN RETRY multi_turn_long_context FAILED" in log
