"""Exercise official-list mining without the BFCL harness or a student server."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import bfcl_event_mine_single as miner
import bfas.adapters.bfcl as bfcl


@pytest.fixture
def fake_miner(tmp_path, monkeypatch, capsys):
    entries = {
        f"simple_{i}": {
            "id": f"simple_{i}",
            "question": [[{"role": "user", "content": f"simple_{i}"}]],
            "ground_truth": [{"wrong_entry_answer": {}}],
        }
        for i in range(4)
    }
    functions = [{"name": "lookup", "parameters": {"type": "object"}}]
    truth = lambda task_id: [{"lookup": {"value": [task_id]}}]
    populated_ids, rendered_ids, checked = [], [], []

    class FakeAdapter:
        def _load_entries(self):
            return entries, {}

        def _render(self, messages, schemas):
            assert schemas == functions
            prompt = messages[0]["content"]
            assert prompt.startswith("populated:")
            rendered_ids.append(prompt.removeprefix("populated:"))
            return prompt

    def populate(rows):
        for row in rows:
            populated_ids.append(row["id"])
            row["question"][0][0]["content"] = f"populated:{row['id']}"
            row["function"] = functions
        return rows

    def checker(schemas, calls, answer, language, category, model):
        checked.append((schemas, answer, language, category, model))
        return {"valid": calls == [{"lookup": {"value": answer[0]["lookup"]["value"][0]}}]}

    harness = tmp_path / "harness"
    answers_dir = harness / "bfcl_eval/data/possible_answer"
    answers_dir.mkdir(parents=True)
    (answers_dir / "BFCL_v4_simple.json").write_text("\n".join(
        json.dumps({"id": task_id, "ground_truth": truth(task_id)})
        for task_id in entries
    ))
    # Supply only the modules main imports, redirecting possible_answer loading
    # to the tiny fixture above instead of an installed BFCL checkout.
    for name, attributes in {
        "bfcl_eval.eval_checker.ast_eval.ast_checker": {"ast_checker": checker},
        "bfcl_eval.utils": {"populate_test_cases_with_predefined_functions": populate},
        "bfcl_generate": {"BFCL": harness, "CHECKER_MODEL": "fake-checker"},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(bfcl, "BFCLAdapter", FakeAdapter)
    monkeypatch.setattr(miner, "ROOT", tmp_path)
    monkeypatch.setattr(miner, "language_for", lambda category: "python")

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "bfcl_support_split.json").write_text(json.dumps({
        "demand": ["simple_0"], "calibration": ["simple_3"],
    }))
    selected_ids = ["simple_2", "simple_1"]
    (tmp_path / "official_ids.json").write_text(json.dumps(selected_ids))
    pool = tmp_path / "data/bfcl_sft/gen_pool_v3.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text(json.dumps({
        "id": "generated_0", "seed_category": "simple", "verified": True,
        "question": [[{"role": "user", "content": "populated:generated_0"}]],
        "function": functions, "ground_truth": truth("generated_0"),
    }) + "\n")

    sample_counts = {}
    correct_replies = {}

    def student_reply(prompt, port, temperature):
        task_id = prompt.removeprefix("populated:")
        sample_counts[task_id] = sample_counts.get(task_id, 0) + 1
        correct = sample_counts[task_id] == 1
        reply = '<tool_call>' + json.dumps({
            "name": "lookup", "arguments": {"value": task_id if correct else "wrong"},
        }) + '</tool_call>'
        if correct:
            correct_replies[task_id] = reply
        return reply

    monkeypatch.setattr(miner, "student_reply", student_reply)

    def run(*args):
        anchors = tmp_path / "anchors.jsonl"
        events = tmp_path / "events.jsonl"
        rates = tmp_path / "rates.json"
        monkeypatch.setattr(sys, "argv", [
            "bfcl_event_mine_single.py", "--k", "2",
            "--anchors-out", str(anchors), "--out", str(events),
            "--rates-out", str(rates), *args,
        ])
        assert miner.main() == 0
        final = next(line for line in capsys.readouterr().out.splitlines()
                     if line.startswith("[events1] FINAL "))
        return SimpleNamespace(
            anchors=[json.loads(line) for line in anchors.read_text().splitlines()],
            events=[json.loads(line) for line in events.read_text().splitlines()],
            rates=json.loads(rates.read_text()),
            stats=ast.literal_eval(final.split("FINAL ", 1)[1].split(" -> ", 1)[0]),
        )

    return SimpleNamespace(
        run=run, selected_ids=selected_ids, populated_ids=populated_ids,
        rendered_ids=rendered_ids, checked=checked, entries=entries,
        correct_replies=correct_replies, functions=functions, truth=truth,
    )


@pytest.mark.parametrize("teacher_demos", ["", "missing_demos", "verified_demos"])
def test_official_ids_anchors_and_teacher_events(fake_miner, tmp_path, teacher_demos):
    fake = fake_miner
    if teacher_demos == "verified_demos":
        demos = tmp_path / teacher_demos
        demos.mkdir()
        (demos / "BFCL_v4_simple_result.json").write_text("\n".join(
            json.dumps({"id": task_id, "result": [{"lookup": json.dumps({"value": "teacher"})}]})
            for task_id in fake.selected_ids
        ))

    result = fake.run(
        "--official-ids", "official_ids.json", "--no-support", "--gen",
        "--teacher-demos", teacher_demos,
    )

    assert fake.populated_ids == fake.rendered_ids == fake.selected_ids
    assert result.rates == dict.fromkeys(fake.selected_ids, 0.5)
    assert fake.checked == [
        (fake.functions, fake.truth(task_id), "python", "simple", "fake-checker")
        for task_id in fake.selected_ids for _ in range(2)
    ]
    # Population must operate on copies, preserving the adapter's cached entries.
    assert all(entry["question"][0][0]["content"] == task_id
               for task_id, entry in fake.entries.items())
    assert [row["task_id"] for row in result.anchors] == fake.selected_ids
    for row in result.anchors:
        assert row["_source"] == "official_list"
        assert row["teacher"] == "self_anchor"
        assert row["_anchor"] is True
        assert row["response"] == fake.correct_replies[row["task_id"]]
        assert row["_event_pass"] == 1
    assert result.stats["tasks"] == result.stats["anchors"] == 2
    assert result.stats["serve_err"] == result.stats["skipped_no_truth"] == 0
    if teacher_demos == "verified_demos":
        assert result.stats["events"] == 2
        assert [row["task_id"] for row in result.events] == fake.selected_ids
        assert all(row["_source"] == "official_list" and row["teacher"] == "deepseek-v4-pro-FC"
                   and miner.parsed_calls(row["response"]) == [{"lookup": {"value": "teacher"}}]
                   for row in result.events)
    else:
        assert result.events == []
        assert result.stats["events"] == 0
        assert result.stats["skipped_no_teacher"] == 2


@pytest.mark.parametrize("include_official_list", [False, True])
def test_support_and_generated_defaults_unchanged(fake_miner, include_official_list):
    fake = fake_miner
    args = ["--teacher-demos", ""]
    if include_official_list:
        args += ["--official-ids", "official_ids.json"]
    result = fake.run(*args)

    expected = ["simple_0", *fake.selected_ids, "generated_0"] if include_official_list else ["simple_0", "generated_0"]
    assert fake.rendered_ids == expected
    assert list(result.rates) == expected
    assert [row["task_id"] for row in result.anchors] == expected
    assert [(row["task_id"], row["_source"], row["teacher"]) for row in result.events] == [
        ("simple_0", "support", "oracle_gt"),
        ("generated_0", "gen_pool_v3.jsonl", "teacher_authored_gt"),
    ]
