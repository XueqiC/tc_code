from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfas.adapters.tau2 import LUNA_MODEL, NativeRun, Tau2Adapter
from bfas.tau2_budget import BudgetStopped, RequestBudget, cost_usd, write_json
from bfas.tau2_eval import evaluate, parser


def budget(tmp_path, max_usd=3, task_id="retail:1"):
    state_path = tmp_path / "budget.json"
    write_json(state_path, {"max_usd": max_usd, "estimated_usd": 0.0,
                           "charged_upper_bound_usd": 0.0, "events": [], "stop_reason": None})
    config = tmp_path / "config.json"
    write_json(config, {"state_path": str(state_path), "task_id": task_id,
                        "ledger_path": str(tmp_path / "ledger/tau2.jsonl"),
                        "max_completion_tokens": 100})
    return RequestBudget(config), state_path


def paid_args(purpose="user_sim"):
    return {"model": LUNA_MODEL, "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7, "metadata": {"bfas_purpose": purpose}}


def reply():
    return {"usage": {"prompt_tokens": 100, "completion_tokens": 50,
                      "prompt_tokens_details": {"cached_tokens": 40}}}


def test_cache_cost_and_request_reservation(tmp_path):
    guard, state_path = budget(tmp_path)
    def completion(**kwargs):
        assert "temperature" not in kwargs
        assert kwargs["service_tier"] == "flex"
        assert kwargs["max_completion_tokens"] == 100
        assert kwargs["num_retries"] == kwargs["max_retries"] == 0
        state = json.loads(state_path.read_text())
        assert state["events"][0]["status"] == "reserved"
        assert state["charged_upper_bound_usd"] > 0
        return reply()
    guard.call(completion, **paid_args())
    state = json.loads(state_path.read_text())
    assert state["estimated_usd"] == pytest.approx(0.0000364)
    assert state["charged_upper_bound_usd"] == pytest.approx(state["estimated_usd"])
    assert cost_usd({"prompt_tokens": 100, "cached_tokens": 40, "completion_tokens": 50}) == pytest.approx(0.0000364)


def test_cap_blocks_before_request_and_within_episode(tmp_path):
    guard, state_path = budget(tmp_path, max_usd=0.0006)
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        return reply()
    guard.call(completion, **paid_args())
    with pytest.raises(BudgetStopped, match="max_usd"):
        guard.call(completion, **{**paid_args(), "messages": [{"role": "user", "content": "long" * 2000}]})
    assert len(calls) == 1
    assert json.loads(state_path.read_text())["charged_upper_bound_usd"] <= 0.0006


@pytest.mark.parametrize("failure", ["timeout", "missing_usage"])
def test_unknown_charge_is_reserved_and_never_retried(tmp_path, failure):
    guard, state_path = budget(tmp_path)
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        if failure == "timeout":
            raise TimeoutError("simulated")
        return {}
    with pytest.raises(BudgetStopped, match="unknown_charge"):
        guard.call(completion, **paid_args())
    with pytest.raises(BudgetStopped):
        guard.call(completion, **paid_args())
    state = json.loads(state_path.read_text())
    assert len(calls) == 1
    assert state["charged_upper_bound_usd"] > 0
    assert state["estimated_usd"] == 0


def test_judge_is_metered_with_its_own_ledger_purpose(tmp_path):
    guard, state_path = budget(tmp_path)
    guard.call(lambda **kwargs: reply(), **paid_args("teacher_judge"))
    state = json.loads(state_path.read_text())
    assert state["estimated_usd"] == pytest.approx(0.0000364)
    row = json.loads((tmp_path / "ledger/tau2.jsonl").read_text())
    assert row["purpose"] == "teacher_judge" and row["verified"] is False


def test_student_request_does_not_receive_official_key(tmp_path, monkeypatch):
    guard, _ = budget(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "official-secret")
    monkeypatch.delenv("BFAS_STUDENT_API_KEY", raising=False)
    calls = []
    guard.call(lambda **kwargs: calls.append(kwargs), model="openai/gemma4-12b-base",
               metadata={"bfas_purpose": "student"})
    assert calls[0]["api_key"] == "EMPTY"


@pytest.fixture
def stub_cli(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "stub-key")
    monkeypatch.setattr(Tau2Adapter, "_splits", lambda self, domain: {"test": ["1", "2", "3"]})
    def run(self, **kwargs):
        calls.append(kwargs)
        assert kwargs["split"] == "test"
        assert kwargs["num_trials"] == 1
        assert kwargs["max_steps"] == 30
        assert kwargs["agent_args"]["metadata"]["bfas_purpose"] in {"student", "teacher_probe"}
        guard = RequestBudget(kwargs["budget_config"])
        purpose = kwargs["agent_args"]["metadata"]["bfas_purpose"]
        try:
            guard.call(lambda **kw: reply(), **paid_args())
            if purpose == "teacher_probe":
                guard.call(lambda **kw: reply(), **paid_args(purpose))
        except BudgetStopped:
            return NativeRun({"simulations": [{"task_id": kwargs["task_ids"][0],
                                              "termination_reason": "infrastructure_error"}]}, tmp_path)
        return NativeRun({"simulations": [{"task_id": kwargs["task_ids"][0],
                                          "termination_reason": "user_stop",
                                          "reward_info": {"reward": int(kwargs["domain"] == "retail")}}]}, tmp_path)
    monkeypatch.setattr(Tau2Adapter, "_run_cli", run)
    return calls


def test_student_eval_weighted_metrics_and_task_cap(tmp_path, stub_cli):
    args = parser().parse_args(["--out-dir", str(tmp_path / "eval"), "--max-tasks", "4"])
    metrics = evaluate(args)
    assert len(stub_cli) == 4
    assert [c["domain"] for c in stub_cli] == ["retail", "airline", "telecom", "retail"]
    assert all(c["agent_model"] == "openai/gemma4-12b-base" for c in stub_cli)
    assert all(c["agent_args"]["temperature"] == 0 for c in stub_cli)
    assert metrics["pass^1"] == 0.5
    assert metrics["per_domain"]["retail"]["pass^1"] == 1
    assert metrics["per_domain"]["airline"]["pass^1"] == 0
    assert metrics["simulator_usage"] == {"prompt_tokens": 400, "cached_tokens": 160, "completion_tokens": 200}
    assert metrics["estimated_usd"] == pytest.approx(4 * 0.0000364)
    assert metrics["stop_reason"] == "max_tasks" and not metrics["complete"]
    assert len((args.out_dir / "tasks.jsonl").read_text().splitlines()) == 4
    assert len((args.out_dir / "ledger/tau2.jsonl").read_text().splitlines()) == 4


def test_teacher_probe_counts_both_sides(tmp_path, stub_cli):
    args = parser(True).parse_args(["--out-dir", str(tmp_path / "probe"), "--n", "2"])
    assert parser(True).parse_args(["--out-dir", "unused"]).tasks_per_domain == 5
    metrics = evaluate(args, teacher=True)
    assert len(stub_cli) == 6
    assert all(c["agent_model"] == LUNA_MODEL for c in stub_cli)
    assert all("temperature" not in c["agent_args"] for c in stub_cli)
    assert metrics["pass^1"] == pytest.approx(1 / 3)
    assert metrics["estimated_usd"] == pytest.approx(12 * 0.0000364)
    assert metrics["teacher_usage"] == metrics["simulator_usage"]
    records = [json.loads(s) for s in (args.out_dir / "ledger/tau2.jsonl").read_text().splitlines()]
    assert {r["purpose"] for r in records} == {"user_sim", "teacher_probe"}


@pytest.mark.parametrize("limit", [["--max-tasks", "0"], ["--max-usd", "0"]])
def test_zero_caps_make_no_cli_calls(tmp_path, stub_cli, limit):
    metrics = evaluate(parser().parse_args(["--out-dir", str(tmp_path / "zero"), *limit]))
    assert not stub_cli and metrics["tasks"] == 0
    assert metrics["pass^1"] is None and metrics["estimated_usd"] == 0


def test_budget_stop_counts_interrupted_task_as_failure(tmp_path, stub_cli):
    metrics = evaluate(parser().parse_args(["--out-dir", str(tmp_path / "limited"), "--max-usd", "0.000001"]))
    assert len(stub_cli) == 1
    assert metrics["stop_reason"] == "max_usd"
    assert metrics["pass^1"] == 0 and metrics["infrastructure_errors"] == 1
    assert metrics["estimated_usd"] == 0


def test_refuse_missing_test_split(tmp_path, monkeypatch):
    monkeypatch.setattr(Tau2Adapter, "_splits", lambda *args: {"base": ["1"]})
    with pytest.raises(ValueError, match="no official test split"):
        evaluate(parser().parse_args(["--out-dir", str(tmp_path / "bad")]))


def test_cli_failure_retains_usage_and_counts_failure(tmp_path, stub_cli, monkeypatch):
    def run(self, **kwargs):
        RequestBudget(kwargs["budget_config"]).call(lambda **kw: reply(), **paid_args())
        return NativeRun({"simulations": []}, tmp_path)
    monkeypatch.setattr(Tau2Adapter, "_run_cli", run)
    args = parser().parse_args(["--out-dir", str(tmp_path / "error"), "--max-tasks", "1"])
    metrics = evaluate(args)
    assert metrics["pass^1"] == 0 and metrics["tasks"] == 1
    assert metrics["estimated_usd"] == pytest.approx(0.0000364)
    assert metrics["simulator_usage"]["prompt_tokens"] == 100


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_invalid_cost_caps(value):
    with pytest.raises(SystemExit):
        parser().parse_args(["--out-dir", "unused", "--max-usd", value])
