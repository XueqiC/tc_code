from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.adapter import BenchmarkAdapter, Demo, Rollout, TaskRef
from bfas.audit import AuditError, audit_pool
from bfas.protocol import AdaptiveSampler, make_support_split, support_size
from bfas.adapters.bfcl import extract_verdicts


class SyntheticAdapter(BenchmarkAdapter):
    name = "synthetic"

    def task_pool(self) -> list[TaskRef]:
        return [TaskRef("call", "calls"), TaskRef("prose", "prose")]

    def official_eval_split_disjoint(self) -> bool:
        return True

    def rollout(self, policy, task_ids, temperature, guided_demos=None):
        return []

    def teacher_demo(self, task_ids, attempts):
        return {}

    def evaluate(self, policy_ref, out_dir):
        return {}

    def serving_probe(self) -> None:
        return None

    def generation_suffix(self) -> str:
        return "<GEN>"

    def target_policy(self, category: str) -> str:
        return "call_required" if category == "calls" else "prose_ok"

    def rerender(self, row) -> str:
        return row["_render_context"]


def valid_row() -> dict[str, Any]:
    prompt = "task<GEN>"
    return {
        "task_id": "call",
        "category": "calls",
        "prompt": prompt,
        "response": '<tool_call>\n{"name":"f"}\n</tool_call>',
        "_render_context": prompt,
        "_task_phat": 0.5,
    }


def test_split_determinism_sizing_and_calibration() -> None:
    tasks = [TaskRef(f"task-{i:04d}", f"cat-{i % 4}") for i in range(2000)]
    first = make_support_split(tasks)
    second = make_support_split(reversed(tasks))
    assert first == second
    assert len(first.support) == 100
    assert len(first.demand) == 80
    assert len(first.calibration) == 20
    assert set(first.demand).isdisjoint(first.calibration)
    assert set(first.demand) | set(first.calibration) == set(first.support)
    assert support_size(20) == 20
    assert support_size(100) == 50
    assert support_size(10_000) == 250


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(messages=[]), "forbidden messages"),
        (lambda row: row.update(prompt="task<WRONG>"), "generation suffix"),
        (lambda row: row.update(response="plain prose"), "requires a call"),
    ],
)
def test_pool_audit_rejects_contract_violations(mutation, message: str) -> None:
    row = valid_row()
    mutation(row)
    with pytest.raises(AuditError, match=message):
        audit_pool([row], SyntheticAdapter())


def _write_score(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_bfcl_verdicts_require_present_positive_summary(tmp_path: Path) -> None:
    _write_score(
        tmp_path / "model/non_live/BFCL_v4_alpha_score.json",
        [
            {"accuracy": 0.5, "correct_count": 1, "total_count": 2},
            {"id": "alpha-fail", "valid": False},
        ],
    )
    _write_score(
        tmp_path / "model/live/BFCL_v4_beta_score.json",
        [{"accuracy": 1.0, "correct_count": 2, "total_count": 2}],
    )
    verdicts = extract_verdicts(tmp_path, {
        "alpha-pass": "alpha",
        "alpha-fail": "alpha",
        "beta-1": "beta",
        "beta-2": "beta",
        "missing-1": "missing",
    })
    assert verdicts == {
        "alpha-pass": True,
        "alpha-fail": False,
        "beta-1": True,
        "beta-2": True,
        "missing-1": False,
    }


def test_beta_posterior_adaptive_stopping() -> None:
    sampler = AdaptiveSampler(["solved", "failed"])
    for _ in range(3):
        sampler.observe("solved", True)
        sampler.observe("failed", False)
    assert "solved" not in sampler.active()
    assert "failed" in sampler.active()
    while "failed" in sampler.active():
        sampler.observe("failed", False)
    assert sampler.rounds() == {"solved": 3, "failed": 8}
    assert sampler.p_hats()["solved"] == pytest.approx(4 / 5)
    assert sampler.p_hats()["failed"] == pytest.approx(1 / 10)
