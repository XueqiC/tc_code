"""Source-fidelity oracles; all teachers, models and campaigns are local stubs."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from bfas.rtd.baselines.paper_acquisition import (KangAcquisition, KangAcquisitionConfig,
    build_prefix_memory, first_paragraph_prefix, prefix_dry_run, read_budget)
from bfas.rtd.baselines.paper_fidelity import KANG_PREFIX, KANG_SUMMARY
from bfas.rtd.ledger import Ledger, BudgetError, LedgerError
from bfas.rtd.persistence import ComputeJournal
from tools import baseline_run, kang_prefix_acquire
from test_baseline_run_data import make_bank
from test_baseline_run_training import Backend, trainer


def test_cot_prompt_matches_official_repository():
    import yaml
    from bfas.rtd.baselines.paper_acquisition import OFFICIAL_COT_SYSTEM_PROMPT
    path = ROOT/"envs/baseline_repos/agent-distillation/src/smolagents/prompts/teacher_model.yaml"
    assert OFFICIAL_COT_SYSTEM_PROMPT == yaml.safe_load(path.read_text())["system_prompt"]


class Teacher:
    supports_assistant_prefix = True

    def __init__(self, ledger, outputs):
        self.ledger, self.outputs, self.calls = ledger, iter(outputs), []

    def __call__(self, **kwargs):
        # The real reservation must already exist when ANY model call starts.
        assert self.ledger.events[-1]["kind"] == "reserve"
        assert self.ledger.events[-1]["cap"] == kwargs["max_output_tokens"]
        self.calls.append(kwargs)
        output, cost = next(self.outputs)
        return dict(output=output, cost=cost, confidence="exact",
                    usage=dict(completion_tokens=cost, reasoning_tokens=2))


@pytest.mark.parametrize("completion,expected", [
    ("  First line\nsecond line\n\nDiscard me", "Thought:   First line\nsecond line\n\n"),
    ("No blank line", "Thought: No blank line\n\n"),
    ("\n\nAnother paragraph", "Thought: \n\n"),
    ("", "Thought: \n\n"),
])
def test_literal_official_first_paragraph(completion, expected):
    assert first_paragraph_prefix(completion) == expected


def test_build_prefix_memory_plain_cot_question_keys_and_durable_charges(tmp_path):
    ledger = Ledger(100, tmp_path/"ledger.jsonl")
    config = KangAcquisitionConfig("new_ftp", 20, 10)
    teacher = Teacher(ledger, [("First\n\nDiscard", 7), ("No blank line", 5)])
    memory = build_prefix_memory(["Question A?", "Question B?"], teacher,
        ledger=ledger, config=config, output_path=tmp_path/"prefix_memory.json")
    assert memory == {"Question A?": "Thought: First\n\n", "Question B?": "Thought: No blank line\n\n"}
    assert json.loads((tmp_path/"prefix_memory.json").read_text()) == memory
    assert [c["messages"][-1] for c in teacher.calls] == [
        dict(role="user", content="Question A?"), dict(role="user", content="Question B?")]
    assert all(set(c) == {"messages", "max_output_tokens"} for c in teacher.calls)
    assert ledger.spent == 12 and ledger.remaining == 88  # no double-count of reasoning
    assert [e["kind"] for e in ledger.events] == ["reserve", "reveal"]*2
    assert Ledger.resume(100, tmp_path/"ledger.jsonl", read_only=True).spent == 12


def test_duplicate_question_occurrences_are_charged_and_last_memory_wins():
    ledger = Ledger(40)
    teacher = Teacher(ledger, [("First", 3), ("Second", 4)])
    memory = build_prefix_memory(["Q", "Q"], teacher, ledger=ledger,
                                 config=KangAcquisitionConfig("duplicates", 10, 10))
    assert memory == {"Q": "Thought: Second\n\n"}
    assert len(teacher.calls) == 2 and ledger.spent == 7


def test_acquisition_prefix_is_first_assistant_continuation_only():
    ledger = Ledger(100)
    config = KangAcquisitionConfig("new_ftp", 20, 10)
    acquisition = KangAcquisition(ledger, config)
    teacher = Teacher(ledger, [("Plan\n\nrest", 5), ("Action: search[x]", 7), ("Finish[x]", 3)])
    memory = acquisition.build_prefix_memory(["Q"], teacher)
    def purchase_trajectory(*, question, model):
        messages = [dict(role="user", content=question)]
        first = model(messages=messages)
        second = model(messages=messages + [dict(role="assistant", content=first),
                                           dict(role="user", content="Observation: x")])
        return [first, second]
    trajectory = acquisition.acquire_with_prefix("Q", teacher, prefix_memory=memory,
        attempt_id="attempt0", purchase_trajectory=purchase_trajectory)
    assert trajectory == ["Thought: Plan\n\nAction: search[x]", "Finish[x]"]
    assert teacher.calls[1]["prefix"] == "Thought: Plan\n\n"
    assert "prefix" not in teacher.calls[0] and "prefix" not in teacher.calls[2]
    assert ledger.spent == 15 and len(ledger.charges) == 3


def test_budget_refusal_precedes_teacher_and_duplicate_request_cannot_recharge():
    ledger = Ledger(9)
    teacher = Teacher(ledger, [("Plan", 3)])
    acquisition = KangAcquisition(ledger, KangAcquisitionConfig("new", 10, 10))
    with pytest.raises(BudgetError):
        acquisition.build_prefix_memory(["Q"], teacher)
    assert not teacher.calls and not ledger.events
    ledger.authorize(30)
    acquisition.build_prefix_memory(["Q"], teacher)
    with pytest.raises(LedgerError):
        acquisition.build_prefix_memory(["Q"], teacher)
    assert len(teacher.calls) == 1


def test_cot_and_trajectory_compete_for_one_cell_budget():
    ledger = Ledger(30)
    acquisition = KangAcquisition(ledger, KangAcquisitionConfig("shared_budget", 10, 15))
    teacher = Teacher(ledger, [("Plan", 8), ("Action: go", 12), ("Next plan", 8)])
    def purchase_trajectory(*, question, model):
        return model(messages=[dict(role="user", content=question)])
    for attempt, question in enumerate(("Q1", "Q2")):
        memory = acquisition.build_prefix_memory([question], teacher)
        if attempt == 0:
            acquisition.acquire_with_prefix(question, teacher, prefix_memory=memory,
                attempt_id=str(attempt), purchase_trajectory=purchase_trajectory)
        else:
            with pytest.raises(BudgetError):
                acquisition.acquire_with_prefix(question, teacher, prefix_memory=memory,
                    attempt_id=str(attempt), purchase_trajectory=purchase_trajectory)
    # The second CoT stays paid even though its trajectory cannot be purchased.
    assert ledger.budget == 30 and ledger.spent == 28 and ledger.remaining == 2
    assert len(teacher.calls) == 3 and not ledger.reservations


def test_invalid_billed_output_stays_charged_and_uncertain_usage_stays_reserved():
    ledger = Ledger(100)
    config = KangAcquisitionConfig("new", 10, 10)
    with pytest.raises(ValueError, match="output must be text"):
        build_prefix_memory(["Q"], Teacher(ledger, [(None, 3)]), ledger=ledger, config=config)
    assert ledger.spent == 3
    def uncertain(**kwargs):
        raise RuntimeError("usage unknown")
    with pytest.raises(RuntimeError):
        build_prefix_memory(["Other"], uncertain, ledger=ledger, config=config)
    assert sum(ledger.reservations.values()) == 10


def test_unsupported_prefix_provider_is_rejected_before_agent_call():
    ledger = Ledger(100)
    teacher = Teacher(ledger, [])
    teacher.supports_assistant_prefix = False
    acquisition = KangAcquisition(ledger, KangAcquisitionConfig("new", 10, 10))
    with pytest.raises(ValueError, match="true assistant-prefix"):
        acquisition.acquire_with_prefix("Q", teacher, prefix_memory={"Q": "Thought: Plan\n\n"},
            attempt_id="a", purchase_trajectory=lambda question, model: model(messages=[]))
    assert not teacher.calls and not ledger.events


def test_dry_run_reads_outstanding_reservations_without_writes(tmp_path):
    path = tmp_path/"ledger.jsonl"
    ledger = Ledger(100, path)
    ledger.reserve("paid", 30)
    ledger.settle("paid", 12, confidence="exact", usage={"completion_tokens": 12})
    ledger.reserve("uncertain", 50)
    before = path.read_bytes()
    report = prefix_dry_run(["A", "B", "A"], KangAcquisitionConfig("new", 20, 10),
                            ledger_path=path, budget=100)
    assert report["cot_requests"] == 3 and report["unique_questions"] == 2
    assert report["reserved_worst_case_output_tokens"] == 60
    assert report["remaining"] == 38 and report["additional_tokens_required"] == 22
    assert report["new_teacher_calls"] == report["new_teacher_tokens"] == 0
    assert not report["reservation_performed"] and not report["fits_remaining_budget"]
    assert path.read_bytes() == before
    path.write_bytes(before + b'{"torn":')
    with pytest.raises(LedgerError, match="read-only"):
        read_budget(path, budget=100)
    assert path.read_bytes() == before + b'{"torn":'
    assert not path.with_suffix(".torn").exists()


@pytest.mark.parametrize("benchmark,count,remaining", [("hotpotqa", 17, 337), ("alfworld", 13, 771)])
def test_real_archive_dry_run_is_read_only(benchmark, count, remaining, capsys):
    path = ROOT/f"archive/table1/table1_{benchmark}_kang/manifest.json"
    before = path.read_bytes()
    assert kang_prefix_acquire.main(["--attempt-manifest", str(path), "--ledger", str(path),
        "--cot-max-output-tokens", "4096", "--run-name", f"{benchmark}_ftp_v2", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["cot_requests"] == count and report["remaining"] == remaining
    assert report["reserved_worst_case_output_tokens"] == count*4096
    assert path.read_bytes() == before


@pytest.mark.parametrize("method", ["sad", "smartad"])
def test_paper_method_names_survive_manifest_and_metrics(tmp_path, monkeypatch, method):
    from bfas.rtd import hardware
    from bfas.rtd.baselines import paper_evaluation
    from bfas.rtd.persistence import tree_hash
    bank = make_bank(tmp_path)
    directory = tmp_path/f"{method}_v2"
    assert baseline_run.main(["--method", method, "--benchmark", "bfcl", "--bank", str(bank),
        "--budget-tokens", "33", "--run-dir", str(directory), "--prepare-only"]) == 0
    manifest = json.loads((directory/"manifest.json").read_text())
    assert manifest["method"] == manifest["config"]["method"] == manifest["hyperparameters"]["method"] == method
    assert manifest["fidelity"] == {"implementation_basis": "paper_reimplementation"}
    assert "sad" in manifest["hyperparameters"]
    # Exercise the real metrics writer with external work replaced by stubs.
    (directory/"checkpoint").mkdir()
    (directory/"model").mkdir()
    (directory/"checkpoint/adapter.json").write_text("{}")
    (directory/"model/config.json").write_text("{}")
    manifest.update(benchmark="alfworld", checkpoint_sha256=tree_hash(directory/"checkpoint"),
        model_path=str(directory/"model"), base_checkpoint_hash=tree_hash(directory/"model"), hardware={"hard": "cpu"})
    monkeypatch.setattr(hardware, "hardware_identity", lambda: {"hard": "cpu"})
    from contextlib import nullcontext
    monkeypatch.setattr(paper_evaluation, "evaluation_policy", lambda *a: nullcontext(directory/"model"))
    monkeypatch.setattr(paper_evaluation, "run_alfworld", lambda *a, **kw: dict(complete=True, overall_accuracy_percent=50.))
    paper_evaluation.evaluate_run(ROOT, directory, manifest)
    for name in ("manifest.json", "metrics.json", "official_metrics.json"):
        value = json.loads((directory/name).read_text())
        assert value["method"] == method
        if name != "manifest.json":
            assert not {"fidelity", "caveat", "limitation", "disclaimer"} & value.keys()


def test_kang_default_refuses_unprefixed_bank_and_legacy_flag_is_explicit(tmp_path):
    bank = make_bank(tmp_path)
    directory = tmp_path/"kang_ftp_v2"
    args = ["--method", "kang", "--benchmark", "bfcl", "--bank", str(bank),
            "--budget-tokens", "33", "--run-dir", str(directory), "--prepare-only"]
    with pytest.raises(ValueError, match="newly acquired"):
        baseline_run.main(args)
    assert not directory.exists()
    baseline_run.main(args + ["--kang-mode", KANG_SUMMARY])
    manifest = json.loads((directory/"manifest.json").read_text())
    assert manifest["method"] == KANG_SUMMARY
    assert "NOT the published mechanism" in manifest["fidelity"]["limitation"]


def test_ftp_training_preserves_acquired_first_target(tmp_path):
    from bfas.rtd.baselines.paper_train import PaperTrainer
    t = trainer(tmp_path, KANG_SUMMARY)
    target = "Thought: Plan\n\nAction: go"
    rows = [replace(t.rows[0], target=target, acquisition_method=KANG_PREFIX)]
    t = PaperTrainer(Backend(), rows, {**t.config, "smoke": True}, KANG_PREFIX,
                     tmp_path, ComputeJournal(tmp_path/"ftp_compute.jsonl", cuda=False))
    t.train()
    assert json.loads((tmp_path/"training_rows.json").read_text())[0]["target"] == target


def test_new_run_and_internal_worker_cannot_write_archive(tmp_path):
    args = ["--method", "smartad", "--benchmark", "bfcl", "--bank", str(tmp_path/"bank"),
            "--budget-tokens", "1", "--run-dir", str(tmp_path/"archive/new_cell")]
    for phase in (["--prepare-only"], ["--_phase", "train"], ["--_phase", "evaluate"]):
        with pytest.raises(ValueError, match="archive is sealed"):
            baseline_run.main(args + phase)
    assert not (tmp_path/"archive").exists()


def test_cached_purchase_cannot_be_mislabelled_as_ftp():
    ledger = Ledger(100)
    acquisition = KangAcquisition(ledger, KangAcquisitionConfig("new", 10, 10))
    with pytest.raises(ValueError, match="no model calls"):
        acquisition.acquire_with_prefix("Q", Teacher(ledger, []),
            prefix_memory={"Q": "Thought: Plan\n\n"}, attempt_id="a",
            purchase_trajectory=lambda **kw: {"behaviors": []})


def test_fresh_purchase_payload_carries_ftp_provenance():
    ledger = Ledger(100)
    acquisition = KangAcquisition(ledger, KangAcquisitionConfig("new", 10, 10))
    def purchase_trajectory(*, question, model):
        return {"behaviors": [{"text": model(messages=[{"role": "user", "content": question}])}],
                "provenance": {"task_id": "task"}}
    result = acquisition.acquire_with_prefix("Q", Teacher(ledger, [("Action: go", 3)]),
        prefix_memory={"Q": "Thought: Plan\n\n"}, attempt_id="a", purchase_trajectory=purchase_trajectory)
    assert result["provenance"]["acquisition_method"] == KANG_PREFIX
    assert result["provenance"]["task_id"] == "task"
    assert result["behaviors"][0]["text"] == "Thought: Plan\n\nAction: go"


def test_table_reader_preserves_paper_method_names():
    from tools import table1_aggregate
    summary = table1_aggregate.aggregate(ROOT/"archive/table1")
    assert "sad" in summary["method_order"] and "sad" in summary["methods"]
    for benchmark in summary["benchmarks"].values():
        for method in ("sad", "smartad"):
            for run in benchmark[method]["runs"]:
                assert run["method"] == method
    latex = table1_aggregate.latex_body(summary)
    assert "\nSAD & " in latex and "\nSmartAD & " in latex


def test_archived_run_name_cannot_be_reused_for_new_behavior(tmp_path):
    from bfas.rtd.baselines.paper_fidelity import require_unsealed_output
    with pytest.raises(ValueError, match="new fidelity run name"):
        require_unsealed_output(tmp_path/"table1_hotpotqa_smartad")
