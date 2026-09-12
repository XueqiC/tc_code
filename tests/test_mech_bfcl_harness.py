"""Offline harness batches: dependencies, capture provenance, and student cost."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.adapters.bfcl import BFCLAdapter
from bfas.mech_bfcl import harness, pipeline
from bfas.mech_bfcl.common import append_row, digest, read_json, write_json


@pytest.fixture
def batch(tmp_path, monkeypatch):
    adapter = BFCLAdapter()
    entries, categories, chains = {}, {}, {}
    for backend in ("kv", "vector", "rec_sum"):
        category = f"memory_{backend}"
        # Only immediate dependencies are listed to exercise transitive closure.
        chain = [f"{category}_prereq_{32+n}-notetaker-{n}" for n in range(3)]
        chain += [f"{category}_132-notetaker-2", f"{category}_141-notetaker-11"]
        chains[backend] = chain
        for n, tid in enumerate(chain):
            entries[tid] = dict(id=tid, depends_on=chain[n-1:n] if n else [], scenario="notetaker")
            categories[tid] = category
        # Later questions and other personas must not enter the episode set.
        for tid in (f"{category}_142-notetaker-12", f"{category}_35-healthcare-5"):
            entries[tid] = dict(id=tid, depends_on=[], scenario=tid.split("-")[1])
            categories[tid] = category
    plain = "simple_python_0"
    entries[plain] = dict(id=plain)
    categories[plain] = "simple_python"
    adapter._entries, adapter._categories = entries, categories
    adapter._prereq_ids = {tid for tid in entries if "_prereq_" in tid}
    adapter._memory_prereqs = {tid: e.get("depends_on", []) for tid, e in entries.items()
                               if tid not in adapter._prereq_ids}
    requested = [chains["kv"][-1], plain, chains["kv"][-2], chains["vector"][-1]]
    info = {tid: dict(category=categories[tid], parent_id=e.get("scenario", tid),
                      source_hash=digest(e)) for tid, e in entries.items()
            if tid not in adapter._prereq_ids}
    splits = dict(seed=0, support=requested, calibration=[plain], evaluation=[chains["rec_sum"][-1]],
                  items=info)
    args = SimpleNamespace(run_dir=tmp_path, base_url="http://stub.invalid/v1",
                           tokenizer="stub", served_model="stub", bfcl_python="stub-python",
                           seed=0, arm="base", repeat="main")
    calls, generated = [], []
    missing_results, missing_frames = set(), set()

    def run(command, *, env, cwd, stdout, stderr, check=False):
        calls.append(command)
        root = Path(env["BFCL_PROJECT_ROOT"])
        selection = read_json(root / "test_case_ids_to_generate.json")
        selected = [tid for group in selection.values() for tid in group]
        assert len(selected) == len(set(selected))
        if "generate" in command:
            assert "--run-ids" in command and "--skip-server-setup" in command
            assert command[command.index("--num-threads") + 1] == "1"
            assert env["MECH_CAPTURE_DIR"] == str(root / "trajectories")
            completed = set()
            for tid in selected:
                assert set(entries[tid].get("depends_on", [])) <= completed
                completed.add(tid)
                generated.append(tid)
                if tid not in missing_results:
                    append_row(root / "result" / (categories[tid] + "_result.json"),
                               dict(id=tid, result="stub", input_token_count=[[3, 3]],
                                    output_token_count=[[7, 7]]))
                # Two model steps per trajectory, so cost must count queries,
                # not tasks. query_started alone is not a completed generation.
                for step in range(2):
                    append_row(root / "trajectories" / (tid + ".jsonl"), dict(
                        event="query_started" if tid in missing_frames else "query",
                        frame_id=f"{tid}:{step}", task_id=tid, response="stub", functions=[],
                        messages=[dict(role="user", content="stub")], snapshot={},
                        snapshot_error=None, involved_classes=[], finish_reason="stop",
                        usage=dict(prompt_tokens=3, completion_tokens=7)))
        else:
            assert "evaluate" in command and "--partial-eval" in command
            for category, group in selection.items():
                scored = [tid for tid in group if tid not in adapter._prereq_ids]
                path = root / "score" / f"BFCL_v4_{category}_score.json"
                append_row(path, dict(correct_count=0, total_count=len(scored)))
                for tid in scored:
                    append_row(path, dict(id=tid, valid=False))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness.subprocess, "run", run)
    monkeypatch.setattr(harness, "trajectory_metrics", lambda states, correct: {})
    return SimpleNamespace(**locals())


def test_official_batch_expands_chains_and_keeps_only_support_seeds(batch):
    b = batch
    original = copy.deepcopy(b.entries)
    destination = b.tmp_path / "support"
    items = harness.official_run(b.args, b.requested, destination, b.adapter, b.splits)
    expected = b.chains["kv"] + [b.plain] + b.chains["vector"]
    assert b.generated == expected
    assert [r["id"] for r in items] == b.requested
    assert len(b.calls) == 2  # One generate and one evaluate for the entire set.
    assert b.entries == original
    prerequisites = read_json(destination / "prerequisites.json")
    assert [r["id"] for r in prerequisites] == [t for t in expected if t not in b.requested]
    assert all(r["role"] == "prerequisite" and len(r["frame_ids"]) == 2 for r in prerequisites)
    assert all((destination / r["trajectory_path"]).exists() for r in prerequisites)
    cost = read_json(destination / "cost.json")
    assert cost["generations"] == len(expected) * 2
    assert cost["output_tokens"] == len(expected) * 14
    assert cost["input_tokens"] == len(expected) * 6
    assert cost["prerequisites"]["output_tokens"] == len(prerequisites) * 14
    assert cost["requested"]["output_tokens"] == len(b.requested) * 14
    captured = harness.frames(destination / "trajectories")
    seeds = pipeline.select_seeds(items + prerequisites, captured, b.splits)
    assert {s["context"]["task_id"] for s in seeds} <= set(b.requested)
    assert b.chains["kv"][-2] in {s["context"]["task_id"] for s in seeds}
    assert harness.official_run(b.args, b.requested, destination, b.adapter, b.splits) == items
    assert len(b.calls) == 2  # Resume does not rerun or double-charge a chain.


def test_missing_ids_reported_together_after_batch_and_cost_recorded(batch):
    b = batch
    b.missing_frames.update([b.chains["kv"][0], b.chains["kv"][-1], b.chains["vector"][-1]])
    b.missing_results.update([b.chains["vector"][1], b.plain])
    destination = b.tmp_path / "failed"
    with pytest.raises(RuntimeError, match="batch finished after memory prerequisite expansion") as exc:
        harness.official_run(b.args, b.requested, destination, b.adapter, b.splits)
    assert len(b.calls) == 2
    assert b.generated == b.chains["kv"] + [b.plain] + b.chains["vector"]
    assert f"missing result ids: {sorted(b.missing_results)}" in str(exc.value)
    assert f"missing completed student trajectory ids: {sorted(b.missing_frames)}" in str(exc.value)
    assert "generate.log" in str(exc.value)
    assert not (destination / "items.json").exists()
    cost = read_json(destination / "cost.json")
    assert cost["output_tokens"] == (len(b.generated) - len(b.missing_frames)) * 14
    with pytest.raises(ValueError, match="Incomplete harness run"):
        harness.official_run(b.args, b.requested, destination, b.adapter, b.splits)
    assert len(b.calls) == 2


def test_evaluate_expands_memory_and_charges_prerequisites(batch, monkeypatch):
    b = batch
    # Keep the full 256-question evaluation size; only this requested memory
    # question's chain may add episodes, and those episodes are not scored items.
    for n in range(1, 256):
        tid = f"simple_python_{n}"
        b.entries[tid] = dict(id=tid)
        b.categories[tid] = "simple_python"
        b.splits["items"][tid] = dict(category="simple_python", parent_id=tid,
                                       source_hash=digest(b.entries[tid]))
        b.splits["evaluation"].append(tid)
    monkeypatch.setattr(pipeline, "inventory", lambda: (b.adapter, b.entries))
    monkeypatch.setattr(pipeline, "check_server", lambda *args: {"id": "stub"})
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())))
    monkeypatch.setattr(pipeline, "OfficialValidator", lambda: SimpleNamespace(
        score=lambda *args: {"correct": True}))
    monkeypatch.setattr(pipeline, "student_reply", lambda *args: ("stub", {}))
    exercises = [dict(id=f"local_{layer}", task_id=b.plain, parent_id=b.plain,
                      category="simple_python", layer=layer, generation_group="heldout")
                 for layer in (1, 2)]
    write_json(b.tmp_path / "heldout.json", exercises)
    pipeline.evaluate(b.args, b.splits)
    destination = b.tmp_path / "evaluation/base/main"
    items = read_json(destination / "items.json")
    assert [r["id"] for r in items if r["layer"] == 3] == b.splits["evaluation"]
    assert len(items) == 258
    assert b.generated == b.chains["rec_sum"] + b.splits["evaluation"][1:]
    assert len(b.calls) == 2
    cost = read_json(destination / "cost.json")["layer_3"]
    assert cost == read_json(destination / "full/cost.json")
    assert cost["requested"]["generations"] == 512
    assert cost["prerequisites"]["generations"] == 8
    assert cost["generations"] == 520
    assert cost["output_tokens"] == 520 * 7


def test_non_memory_batch_adds_no_episodes(batch):
    b = batch
    destination = b.tmp_path / "plain"
    items = harness.official_run(b.args, [b.plain], destination, b.adapter, b.splits)
    assert b.generated == [b.plain] and len(b.calls) == 2
    assert [r["id"] for r in items] == [b.plain]
    assert read_json(destination / "prerequisites.json") == []
    assert read_json(destination / "cost.json")["prerequisites"]["generations"] == 0
