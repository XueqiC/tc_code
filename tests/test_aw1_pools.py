"""CLI coverage for AppWorld pools using synthetic events only."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "aw1_pools.py"


@pytest.mark.parametrize("options,prefix,keep_overflow", [
    ([], "aw1", False),
    (["--prefix", "aw1s"], "aw1s", False),
    (["--prefix", "aw1s", "--exclude-overflow"], "aw1s", False),
    (["--prefix", "aw1s", "--no-exclude-overflow"], "aw1s", True),
])
def test_pools(tmp_path, options, prefix, keep_overflow):
    rows = [
        {"task_id": "t1", "_event_dU": 0.4, "_event_agree": True},
        {"task_id": "t1", "_event_dU": -0.2},
        {"task_id": "t2", "_event_dU": 0, "_student_agree_rate": 1 / 3,
         "_event_good": "teacher action", "_event_bad": "student action"},
        # API-effect agreement can have different teacher and student code.
        {"task_id": "t2", "_event_dU": 0, "_student_agree_rate": 1,
         "_event_good": "teacher action", "_event_bad": "equivalent action"},
        {"task_id": "t3", "_event_dU": 0, "_event_agree": True},
        {"task_id": "t3", "_event_dU": 0,
         "_event_good": "same action", "_event_bad": "same action"},
        {"task_id": "t3", "_event_dU": 0,
         "response": "same reply é", "_rejected": "same reply é"},
        # Missing action fields must not compare equal and imply agreement.
        {"task_id": "t4", "_event_dU": 0},
        {"task_id": "overflow", "_event_dU": 0, "_event_overflow": True,
         "_event_k": 0, "_student_agree_rate": None,
         "response": "teacher reply", "_rejected": ""},
    ]
    out_dir = tmp_path / "data" / "appworld_events"
    out_dir.mkdir(parents=True)
    src = out_dir / "aw1_base_k3.jsonl"
    src.write_text("\n" + "\n\n".join(json.dumps(r) for r in rows) + "\n")
    # The first case also checks the original default input path.
    command = [sys.executable, str(SCRIPT)]
    if options:
        command.append(str(src))
    result = subprocess.run(command + options, cwd=tmp_path, text=True,
                            capture_output=True, check=True)

    expected = {"all": rows if keep_overflow else rows[:-1],
                "conseq": rows[:2], "pos": rows[:1]}
    assert {p.name for p in out_dir.glob("pool_*.jsonl")} == {
        f"pool_{prefix}_{name}.jsonl" for name in expected
    }
    for name, pool in expected.items():
        path = out_dir / f"pool_{prefix}_{name}.jsonl"
        assert [json.loads(line) for line in path.read_text().splitlines()] == pool
    assert "same reply é" in (out_dir / f"pool_{prefix}_all.jsonl").read_text()
    assert result.stdout.splitlines() == [
        f"all {9 if keep_overflow else 8} tasks {5 if keep_overflow else 4} -> data/appworld_events/pool_{prefix}_all.jsonl",
        f"conseq 2 tasks 1 -> data/appworld_events/pool_{prefix}_conseq.jsonl",
        f"pos 1 tasks 1 -> data/appworld_events/pool_{prefix}_pos.jsonl",
        f"dU hist [(-0.2, 1), (0, {7 if keep_overflow else 6}), (0.4, 1)]",
        f"agreements 4 divergences 4 overflows {int(keep_overflow)}",
    ]
