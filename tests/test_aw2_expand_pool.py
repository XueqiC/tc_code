"""AW-2 expansion and orchestration with tiny, offline fixtures; no real launches."""

from collections import Counter
import copy
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import aw2_expand_pool as aw2
from appworld_event_mine import select_probe_steps


def event(task, turn, length=16, **fields):
    return {
        "task_id": task, "turn_index": turn, "_event_turn": turn,
        "_demo_len": length, "_probe_pos": (turn + 1) / length,
        "prompt": "student context", "response": "a b", "_rejected": "c",
        "_event_dU": 0, "_probe_select": "explicit", "_event_overflow": False,
        **fields,
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


@pytest.fixture
def tokenizer_dir(tmp_path):
    path = tmp_path / "tokenizer"
    path.mkdir()
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3},
                                          unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]",
    }))
    return path


def test_per_task_sampling_filters_preserves_preferences_and_reports_shortfall():
    matched = [event("a", 3), event("b", 3), event("a", 7), event("b", 7)]
    negative = event("a", 1, _event_dU=-.4, _event_errors=["kept like AW-2"],
                     response="teacher", _rejected="student")
    agreement = event("a", 5, _student_agree_rate=1, _event_k=0, response="same", _rejected="same")
    positive = event("b", 1, _event_dU=.4)
    # An overflow copy must not consume the key of a later usable observation.
    mined = [dict(negative, _event_overflow=True), negative, agreement, positive,
             dict(negative, response="duplicate"), event("a", 3), event("outside", 1)]
    before = copy.deepcopy((matched, mined))
    expanded, counts, inputs = aw2.expand_pool(matched, mined)
    assert expanded[:4] == matched
    assert {aw2.event_key(r) for r in expanded[4:]} == {("a", 1), ("a", 5), ("b", 1)}
    assert all(r in [negative, agreement, positive] for r in expanded[4:])
    assert len({aw2.event_key(r) for r in expanded}) == len(expanded)
    assert counts["a"] == {"existing": 2, "target_new": 2, "eligible_new": 2,
                           "selected_new": 2, "expanded": 4, "shortfall": 0,
                           "residual_new_minus_target": 0}
    assert counts["b"]["shortfall"] == 1
    assert inputs == {"total_rows": 7, "excluded": {"overflow": 1, "other_task": 1,
                     "existing_key": 1, "duplicate_new_key": 1}, "eligible_rows": 3,
                     "selected_rows": 3, "unused_eligible_rows": 0}
    assert (matched, mined) == before


def test_seed_and_no_cross_task_topup():
    matched = [event("a", 30, 40), event("a", 31, 40), event("b", 30, 40)]
    mined = [event("a", i, 40) for i in range(20)]
    first = aw2.expand_pool(matched, mined, seed=0)
    assert first == aw2.expand_pool(matched, mined, seed=0)
    assert first[0][3:] != aw2.expand_pool(matched, mined, seed=7)[0][3:]
    assert len(first[0]) == 5
    assert first[1]["b"]["selected_new"] == 0
    assert first[1]["b"]["shortfall"] == 1
    assert first[2]["unused_eligible_rows"] == 18


@pytest.mark.parametrize("trailing_newline", [True, False])
def test_cli_verbatim_prefix_and_complete_report(tmp_path, tokenizer_dir, trailing_newline):
    directory = tmp_path / "data/appworld_events"
    directory.mkdir(parents=True)
    matched = [event("a", 3, response="a b c", _probe_pos=None),
               event("b", 7, response="a", _event_dU=-.2)]
    original = ("\r\n  " + json.dumps(matched[0], ensure_ascii=False, separators=(",", ":"))
                + "\r\n\r\n" + json.dumps(matched[1]) + ("\r\n" if trailing_newline else "")).encode()
    matched_path = directory / "pool_aw2_spread_matched.jsonl"
    matched_path.write_bytes(original)
    mined_path = tmp_path / "more.jsonl"
    write_rows(mined_path, [event("a", 1, response="a b", _rejected="b c"),
                            event("b", 5, response="c", _event_dU=.4)])
    command = [sys.executable, str(ROOT / "tools/aw2_expand_pool.py"), "--mined", str(mined_path),
               "--tokenizer", str(tokenizer_dir)]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=True)
    output = directory / "pool_aw2_spread_x2.jsonl"
    report_path = directory / "aw2_expand_report.json"
    report = json.loads(report_path.read_text())
    assert matched_path.read_bytes() == original
    assert output.read_bytes().startswith(original)
    rows = aw2.read_events(output)
    assert rows[:2] == matched
    assert Counter(r["task_id"] for r in rows) == {"a": 2, "b": 2}
    assert report["tasks"] == ["a", "b"]
    assert report["residual_imbalance"]["shortfall_rows"] == 0
    assert report["arms"]["A"]["stage_position_distribution"]["mean"] == .375
    assert report["arms"]["A"]["probe_pos_sources"]["derived"] == 1
    assert report["arms"]["C"]["response_token_length_distribution"]["response"]["total_tokens"] == 7
    for arm, n, epochs, steps, exposure_tokens in [("A", 2, 1, 1, 4), ("B", 2, 2, 2, 8), ("C", 4, 1, 1, 7)]:
        stats = report["arms"][arm]
        assert (stats["total_rows"], stats["epochs"], stats["steps"]) == (n, epochs, steps)
        assert stats["row_exposures"] == n * epochs
        assert stats["response_token_length_distribution"]["response"]["exposure_tokens"] == exposure_tokens
    assert report["loss_normalization"]["response_log_probability"].startswith("sum")
    assert "residual shortfall=0" in result.stdout
    first = output.read_bytes(), report_path.read_bytes()
    subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=True)
    assert first == (output.read_bytes(), report_path.read_bytes())


@pytest.mark.parametrize("n,epochs,steps", [(155, 1, 20), (155, 2, 40), (310, 1, 39), (309, 1, 39)])
def test_exposure_steps_round_each_epoch(n, epochs, steps):
    stats = aw2.arm_summary([event("a", 0)] * n, TinyTokenizer(), epochs)
    assert stats["steps"] == stats["estimated_ddpo_steps"] == steps
    assert stats["row_exposures"] == epochs * n


def test_distributions_missing_positions_and_nonpreference_rows():
    rows = [event("a", 0, _probe_pos=0, response="", _rejected=""),
            event("a", 1, _probe_pos=.25, response="a b c"),
            event("a", 2, _probe_pos=1, response="a b"),
            event("a", 3, _probe_pos=None, _demo_len=None)]
    stats = aw2.arm_summary(rows, TinyTokenizer(), 2)
    dist = stats["stage_position_distribution"]
    assert dist["count"] == 3 and dist["missing"] == 1
    assert [b["count"] for b in dist["histogram"]] == [2, 0, 0, 1]
    assert dist["quantiles"]["p50"] == .25
    tokens = stats["response_token_length_distribution"]["response"]
    assert tokens["min"] == 0 and tokens["max"] == 3
    assert tokens["quantiles"]["p50"] == 2
    assert tokens["mean"] == 1.75
    assert stats["preference_rows"] == 3
    assert aw2.arm_summary([], TinyTokenizer(), 1)["steps"] == 0


def test_turn_fallback_and_alias_conflicts():
    fallback = event("a", 1)
    del fallback["turn_index"]
    assert aw2.event_key(fallback) == ("a", 1)
    expanded, _, inputs = aw2.expand_pool([fallback], [event("a", 1), event("a", 2)])
    assert len(expanded) == 2 and inputs["excluded"]["existing_key"] == 1
    with pytest.raises(ValueError, match="Conflicting turn aliases"):
        aw2.event_key(event("a", 1, _event_turn=2))


@pytest.mark.parametrize("rows,message", [
    ([], "empty"), ([event("a", 1)] * 2, "duplicate"),
    ([event("a", 1, _event_overflow=True)], "overflow"),
    ([event("a", 1, turn_index=None, _event_turn=None)], "nonnegative"),
])
def test_invalid_matched_pool(rows, message):
    with pytest.raises(ValueError, match=message):
        aw2.expand_pool(rows, [])


def test_no_new_events_fails_without_artifacts(tmp_path):
    matched, mined = tmp_path / "matched.jsonl", tmp_path / "mined.jsonl"
    write_rows(matched, [event("a", 1)])
    write_rows(mined, [event("a", 1), event("other", 2), event("a", 2, _event_overflow=True)])
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        aw2.main(["--matched", str(matched), "--mined", str(mined), "--out-dir", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_reject_output_alias_before_writing(tmp_path):
    matched = tmp_path / "pool_aw2_spread_x2.jsonl"
    mined = tmp_path / "mined.jsonl"
    write_rows(matched, [event("a", 1)])
    write_rows(mined, [event("a", 2)])
    before = matched.read_bytes()
    with pytest.raises(SystemExit):
        aw2.main(["--matched", str(matched), "--mined", str(mined), "--out-dir", str(tmp_path)])
    assert matched.read_bytes() == before


def test_midpoints_and_short_demo_rounding():
    rows = ([event("a", t) for t in (3, 7, 11, 15)] +
            [event("b", t, 7) for t in (1, 3, 5, 6)])
    plan = aw2.midpoint_plan(rows)
    assert plan["a"]["probe_positions"] == [.125, .375, .625, .875]
    assert plan["a"]["probe_turns"] == [1, 5, 9, 13]
    assert plan["b"]["probe_turns"] == [0, 2, 4]
    assert plan["b"]["shortfall"] == 1
    for task, item in plan.items():
        steps = select_probe_steps(item["demo_length"], item["planned_new"], "spread",
                                   tuple(item["probe_positions"]))
        assert steps == item["probe_turns"]
        assert not set(steps) & set(item["existing_turns"])


def test_midpoints_fill_uneven_gaps_and_handle_exhausted_demos():
    # Exhaust small layouts to cover adjacent states, boundaries and gap bisection.
    for length in range(1, 9):
        for count in range(1, min(length, 4) + 1):
            for turns in itertools.combinations(range(length), count):
                item = aw2.midpoint_plan([event("a", t, length) for t in turns])["a"]
                assert item["planned_new"] == min(count, length - count)
                selected = select_probe_steps(length, item["planned_new"], "spread",
                                              tuple(item["probe_positions"]))
                assert selected == item["probe_turns"]
                assert not set(selected) & set(turns)


@pytest.mark.parametrize("fields", [{"_demo_len": None}, {"_probe_pos": .99}, {"_demo_len": 1}])
def test_midpoint_plan_rejects_inconsistent_metadata(fields):
    with pytest.raises(ValueError):
        aw2.midpoint_plan([event("a", 3, **fields)])


@pytest.fixture
def shell_project(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").symlink_to(sys.executable)
    for name in ["aw2_expand_pool.py", "aw2_matched_pools.py", "aw2_mine_more.sh", "aw2_runner_bc.sh"]:
        shutil.copy2(ROOT / "tools" / name, tmp_path / "tools" / name)
    write_rows(tmp_path / aw2.MATCHED, [event("a", t) for t in (3, 7, 11, 15)])
    return tmp_path


def test_mining_script_dry_run_uses_real_midpoint_plan(shell_project):
    result = subprocess.run(["bash", "tools/aw2_mine_more.sh", "8989", "--dry-run"],
                            cwd=shell_project, capture_output=True, text=True, check=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 1
    command = commands[0]
    assert command[:2] == [".venv/bin/python", "tools/appworld_event_mine.py"]
    for flag, value in {"--tasks": "a", "--probe-select": "spread", "--k": "3", "--port": "8989",
                        "--probe-positions": "0.125,0.375,0.625,0.875", "--max-probes": "4",
                        "--equality": "api_effects", "--thinking": "off",
                        "--out": "data/appworld_events/aw2_spread_more_k3.jsonl"}.items():
        assert command[command.index(flag) + 1] == value
    assert not (shell_project / "logs").exists()
    assert not (shell_project / "data/appworld_events/aw2_spread_more_k3.jsonl").exists()


@pytest.mark.parametrize("script", ["aw2_mine_more.sh", "aw2_runner_bc.sh"])
def test_scripts_reject_wrong_working_directory(shell_project, script):
    result = subprocess.run(["bash", str(shell_project / "tools" / script), "8989"],
                            cwd=shell_project / "tools", capture_output=True, text=True)
    assert result.returncode == 2
    assert "project root" in result.stderr
    assert not (shell_project / "logs").exists()


def install_train_stub(project):
    # Entire training/evaluation launcher is replaced with an environment recorder.
    (project / "tools/aw1_train_eval.sh").write_text('''#!/bin/bash
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' "$1" "$2" "$3" "$4" "$5" "$AW_DDPO_EPOCHS" "$AW_DDPO_LR" "$AW_GRAD_CKPT" "$AW_DDPO_WEIGHT" >> calls
status=${FAKE_DEV_EXIT:-0}
echo "[aw1] $1 dev run exit=$status"
echo ' aggregate   |         21.1         |           10.5' | tee "logs/awoff_${1}_dev.log"
exit "${FAKE_TRAIN_EXIT:-0}"
''')


@pytest.mark.parametrize("selection,expected", [(None, ["e2", "x2"]), ("B", ["e2"]),
                                               ("C", ["x2"]), ("C,B,B", ["e2", "x2"]),
                                               ("aw2_pref_spread_x2_s0", ["x2"])])
def test_runner_subset_epochs_settings_and_log(shell_project, selection, expected):
    install_train_stub(shell_project)
    if "x2" in expected:
        write_rows(shell_project / "data/appworld_events/pool_aw2_spread_x2.jsonl", [event("a", 1)])
    env = {**os.environ, "AW_DDPO_EPOCHS": "99", "AW_DDPO_LR": "1", "AW_DDPO_WEIGHT": "pos"}
    env.pop("AW2_ARMS", None)
    if selection is not None:
        env["AW2_ARMS"] = selection
    subprocess.run(["bash", "tools/aw2_runner_bc.sh", "3", "8959"], cwd=shell_project,
                   env=env, text=True, capture_output=True, check=True)
    calls = [line.split("|") for line in (shell_project / "calls").read_text().splitlines()]
    assert [c[0] for c in calls] == [f"aw2_pref_spread_{suffix}_s0" for suffix in expected]
    for call, suffix in zip(calls, expected):
        assert call[2:] == ["ddpo", "3", "8959", "2" if suffix == "e2" else "1", "5e-6", "1", "none"]
    log = (shell_project / "logs/aw2_runner_bc.log").read_text()
    assert log.count("aggregate") == len(expected)
    assert log.count("[aw2-runner] === aw2_pref_spread_") == 2 * len(expected)
    assert "ALL DONE" in log


@pytest.mark.parametrize("setting", [{"FAKE_TRAIN_EXIT": "7"}, {"FAKE_DEV_EXIT": "1"}])
def test_runner_stops_on_train_or_swallowed_dev_failure(shell_project, setting):
    install_train_stub(shell_project)
    env = {**os.environ, "AW2_ARMS": "B C", **setting}
    result = subprocess.run(["bash", "tools/aw2_runner_bc.sh"], cwd=shell_project,
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert len((shell_project / "calls").read_text().splitlines()) == 1
    assert "ALL DONE" not in result.stdout


def test_runner_rejects_unknown_arm_without_launch(shell_project):
    install_train_stub(shell_project)
    result = subprocess.run(["bash", "tools/aw2_runner_bc.sh"], cwd=shell_project,
                            env={**os.environ, "AW2_ARMS": "B typo"}, text=True, capture_output=True)
    assert result.returncode == 2
    assert not (shell_project / "calls").exists()
