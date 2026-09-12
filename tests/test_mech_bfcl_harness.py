"""Offline harness batches: dependencies, capture provenance, and student cost."""
import copy
from datetime import datetime, timezone
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
    missing_results, missing_frames, missing_categories = set(), set(), set()

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
                if category in missing_categories:
                    continue
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
    assert "excluded_categories" not in read_json(destination / "run.json")
    assert all("evaluation_scope" not in r for r in items)


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
    # A retry keeps the failed attempt byte-for-byte and starts a clean batch.
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    b.missing_frames.clear()
    b.missing_results.clear()
    items = harness.official_run(b.args, b.requested, destination, b.adapter, b.splits)
    assert len(b.calls) == 4
    assert [r["id"] for r in items] == b.requested
    archives = list(b.tmp_path.glob("failed.failed-*"))
    assert len(archives) == 1
    datetime.strptime(archives[0].name.removeprefix("failed.failed-"), "%Y%m%dT%H%M%S.%fZ")
    assert {p.relative_to(archives[0]): p.read_bytes() for p in archives[0].rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("arm,repeat", [("base", "main"), ("base", "repeat"), ("C", "main"), ("D", "main")])
def test_evaluate_excludes_web_search_and_charges_prerequisites(batch, monkeypatch, arm, repeat):
    b = batch
    b.args.arm, b.args.repeat = arm, repeat
    # Frozen subset: 246 usable questions and ten excluded web-search questions.
    # The requested memory question still expands through unscored prerequisites.
    for n in range(1, 256):
        category = "simple_python" if n < 246 else "web_search"
        tid = f"{category}_{n}"
        b.entries[tid] = dict(id=tid)
        b.categories[tid] = category
        b.splits["items"][tid] = dict(category=category, parent_id=tid,
                                       source_hash=digest(b.entries[tid]))
        b.splits["evaluation"].append(tid)
    original = copy.deepcopy(b.splits)
    split_hash = digest(b.splits)
    b.missing_categories.add("web_search")  # Reproduce the official checker's omission.
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
    destination = b.tmp_path / "evaluation" / arm / repeat
    # Simulate the old 256-ID run, with different metadata and no items.json.
    write_json(destination / "full/run.json", dict(ids=b.splits["evaluation"], split_hash=split_hash))
    (destination / "full/evaluate.log").write_text("old web_search omission\n")
    pipeline.evaluate(b.args, b.splits)
    items = read_json(destination / "items.json")
    full = [r for r in items if r["layer"] == 3]
    expected = b.splits["evaluation"][:246]
    excluded = b.splits["evaluation"][246:]
    assert [r["id"] for r in full] == expected
    assert len(items) == 248
    assert all("evaluation_scope" not in r for r in items if r["layer"] in (1, 2))
    assert b.generated == b.chains["rec_sum"] + expected[1:]
    assert not set(excluded) & set(b.generated)
    assert len(b.calls) == 2
    run = read_json(destination / "full/run.json")
    assert run["ids"] == expected
    assert run["split_hash"] == split_hash == digest(b.splits)
    assert b.splits == original
    assert run["excluded_categories"] == ["web_search"]
    assert run["excluded_ids"] == excluded
    assert (run["evaluated_questions"], run["subset_questions"]) == (246, 256)
    for r in full:
        assert all(run[k] == v for k, v in r["evaluation_scope"].items())
        assert r["evaluation_scope"]["excluded_ids"] == excluded
    assert read_json(destination / "full/items.json") == full
    archive, = destination.glob("full.failed-*")
    assert read_json(archive / "run.json")["ids"] == b.splits["evaluation"]
    assert (archive / "evaluate.log").read_text() == "old web_search omission\n"
    cost = read_json(destination / "cost.json")["layer_3"]
    assert cost == read_json(destination / "full/cost.json")
    assert cost["requested"]["generations"] == 492
    assert cost["prerequisites"]["generations"] == 8
    assert cost["generations"] == 500
    assert cost["output_tokens"] == 500 * 7
    monkeypatch.setattr(pipeline, "student_reply", lambda *args: pytest.fail("local result should be reused"))
    pipeline.evaluate(b.args, b.splits)
    assert len(b.calls) == 2
    assert len(list(destination.glob("full.failed-*"))) == 1


def test_non_memory_batch_adds_no_episodes(batch):
    b = batch
    destination = b.tmp_path / "plain"
    items = harness.official_run(b.args, [b.plain], destination, b.adapter, b.splits)
    assert b.generated == [b.plain] and len(b.calls) == 2
    assert [r["id"] for r in items] == [b.plain]
    assert read_json(destination / "prerequisites.json") == []
    assert read_json(destination / "cost.json")["prerequisites"]["generations"] == 0


def test_missing_categories_error_names_expected_and_scored_sets(batch):
    b = batch
    b.missing_categories.update(["memory_kv", "simple_python"])
    destination = b.tmp_path / "omitted"
    with pytest.raises(RuntimeError) as exc:
        harness.official_run(b.args, b.requested, destination, b.adapter, b.splits)
    message = str(exc.value)
    assert "omitted categories: ['memory_kv', 'simple_python']" in message
    assert "expected categories: ['memory_kv', 'memory_vector', 'simple_python']" in message
    assert "scored categories: ['memory_vector']" in message
    assert str(destination / "evaluate.log") in message
    assert not (destination / "items.json").exists()


def test_move_aside_timestamp_collision_preserves_both_attempts(batch, monkeypatch):
    b = batch
    now = datetime(2026, 9, 11, 23, 0, 1, 123456, tzinfo=timezone.utc)
    def frozen_now(tz):
        assert tz == timezone.utc
        return now
    monkeypatch.setattr(harness, "datetime", SimpleNamespace(now=frozen_now))
    destination = b.tmp_path / "full"
    archive = b.tmp_path / "full.failed-20260911T230001.123456Z"
    write_json(archive / "run.json", {"attempt": "older"})
    write_json(destination / "run.json", {"attempt": "failed"})
    harness.official_run(b.args, [b.plain], destination, b.adapter, b.splits)
    assert read_json(archive / "run.json") == {"attempt": "older"}
    assert read_json(b.tmp_path / (archive.name + "-1") / "run.json") == {"attempt": "failed"}
    assert (destination / "items.json").exists()


def test_unknown_directory_and_changed_completed_run_are_preserved(batch):
    b = batch
    unknown = b.tmp_path / "unknown"
    write_json(unknown / "unrelated.json", {"preserve": True})
    with pytest.raises(ValueError, match="Incomplete harness run"):
        harness.official_run(b.args, [b.plain], unknown, b.adapter, b.splits)
    assert read_json(unknown / "unrelated.json") == {"preserve": True}
    assert not b.calls and not list(b.tmp_path.glob("unknown.failed-*"))
    destination = b.tmp_path / "completed"
    items = harness.official_run(b.args, [b.plain], destination, b.adapter, b.splits)
    with pytest.raises(ValueError, match="changed inputs"):
        harness.official_run(b.args, b.requested, destination, b.adapter, b.splits)
    assert read_json(destination / "items.json") == items
    assert len(b.calls) == 2 and not list(b.tmp_path.glob("completed.failed-*"))
