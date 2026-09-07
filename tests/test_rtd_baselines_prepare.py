import json
from pathlib import Path

import pytest

from bfas.rtd.baselines.evidence import prepare, read_chain
from bfas.rtd.persistence import atomic_json, file_hash
from test_rtd_baselines_helpers import FIXTURE, TinyTokenizer, h, make_fixture, prepared_fixture


def test_prepare_reproduces_hand_built_real_format_fixture_without_writes_to_source(tmp_path, monkeypatch):
    source, bank, support, manifest = make_fixture(tmp_path)
    before = {str(p): file_hash(p) for d in (source, bank) for p in d.rglob("*") if p.is_file()}
    # Loading the unavailable sealed payload would violate the evidence boundary.
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.name != h("unavailable")+".json"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    out = tmp_path/"evidence.json"
    result = prepare(source, bank, out, root=tmp_path, expected_hardware_hash=manifest["hardware_hash"],
                     tokenizer=TinyTokenizer(), support=support)
    expected = json.loads(FIXTURE.read_text())["expected"]
    assert result["owned"] == sorted([h("paid_a"), h("paid_b"), h("paid_a_independent")])
    assert result["budgets"]["teacher_tokens"] == expected["teacher_tokens"]
    assert len(result["states"]) == expected["unique_states"]
    assert sum(len(s["teachers"]) for s in result["states"].values()) == expected["teacher_empirical_items"]
    assert len([g for g in result["feedback"] if not g["reused"]]) == expected["physical_feedback_groups"]
    assert sum(t["rollout_count"] for g in result["feedback"] if not g["reused"] for t in g["tasks"]) == expected["feedback_rollouts"]
    assert result["budgets"]["update_tokens"] == [672]*3
    assert result["budgets"]["pg_reserved_tokens"] == [144]*3
    assert result["charges"][2]["dependencies"] == [h("paid_a")]
    assert result["charges"][1]["usage"]["input_tokens"] is None
    assert before == {str(p): file_hash(p) for d in (source, bank) for p in d.rglob("*") if p.is_file()}
    assert json.loads(out.read_text()) == result


@pytest.mark.parametrize("corrupt", ["partial", "torn", "hash", "pending", "audit", "owned", "feedback", "costs", "hardware"])
def test_prepare_refuses_incomplete_or_inconsistent_source_without_repair(tmp_path, corrupt):
    source, bank, support, manifest = make_fixture(tmp_path)
    expected_hardware = manifest["hardware_hash"]
    trajectory = json.loads((source/"trajectory.json").read_text())
    if corrupt == "partial":
        trajectory["steps"].pop()
    elif corrupt == "torn":
        with (source/"teacher.jsonl").open("ab") as stream:
            stream.write(b'{"incomplete":')
    elif corrupt == "hash":
        text = (source/"teacher.jsonl").read_text()
        (source/"teacher.jsonl").write_text(text.replace('"cost": 7', '"cost": 8'))
    elif corrupt == "pending":
        trajectory["pending"] = ["unmerged"]
    elif corrupt == "audit":
        atomic_json(source/"audit.json", dict(passed=True, steps=12))
    elif corrupt == "owned":
        trajectory["checkpoints"][-1]["owned"].pop()
    elif corrupt == "feedback":
        trajectory["steps"][0]["truncation"]["actual_feedback_reused"] = False
    elif corrupt == "costs":
        del trajectory["steps"][0]["old_exposure"]["prompt_tokens"]
    elif corrupt == "hardware":
        expected_hardware = "another-hardware"
    atomic_json(source/"trajectory.json", trajectory)
    before = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
    out = tmp_path/"evidence.json"
    with pytest.raises((ValueError, KeyError)):
        prepare(source, bank, out, root=tmp_path, expected_hardware_hash=expected_hardware,
                tokenizer=TinyTokenizer(), support=support)
    assert not out.exists()
    assert before == {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}


def test_prepare_refuses_overwriting_exports(tmp_path):
    evidence, out, support = prepared_fixture(tmp_path)
    with pytest.raises(FileExistsError):
        prepare(tmp_path/"source", tmp_path/"bank", out, root=tmp_path,
                expected_hardware_hash=evidence["hardware_class_hash"], tokenizer=TinyTokenizer(), support=support)


def test_prepare_output_cannot_mutate_source_or_bank(tmp_path):
    source, bank, support, manifest = make_fixture(tmp_path)
    for directory in (source, bank):
        out = directory/"new-but-forbidden.json"
        with pytest.raises(ValueError, match="read-only"):
            prepare(source, bank, out, root=tmp_path, expected_hardware_hash=manifest["hardware_hash"],
                    tokenizer=TinyTokenizer(), support=support)
        assert not out.exists()
