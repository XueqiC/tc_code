"""Offline CPU tests using real, tiny parquet shards and mocked HTTP bodies."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import socket
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import setup_hotpotqa as setup


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("HotpotQA setup CPU tests must never contact the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(setup.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(setup.time, "sleep", lambda _: None)


def hf_row(task_id):
    return {
        "id": task_id, "question": " Where is Café?\n", "answer": "Paris",
        "type": "bridge", "level": "hard",
        "supporting_facts": {"title": ["Café", "Other", "Café"], "sent_id": [1, 0, 0]},
        "context": {"title": ["Other", "Café", "Empty"],
                    "sentences": [["Other sentence."], [" First. ", "Second!"], []]},
    }


def expected_row(task_id):
    return {
        "_id": task_id, "question": " Where is Café?\n", "answer": "Paris",
        "type": "bridge", "level": "hard",
        "supporting_facts": [["Café", 1], ["Other", 0], ["Café", 0]],
        "context": [["Other", ["Other sentence."]], ["Café", [" First. ", "Second!"]], ["Empty", []]],
    }


def write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    data = root / "envs/hotpotqa/data"
    data.mkdir(parents=True)
    (root / "configs").mkdir()
    monkeypatch.setattr(setup, "ROOT", root)
    monkeypatch.setattr(setup, "DATA", data)
    monkeypatch.setattr(setup, "SOURCES", {
        "train": ("hotpot_train_v1.1.json", 3), "dev": ("hotpot_dev_distractor_v1.json", 1),
    })
    ids = {"train": ["train-a", "train-b", "train-c"], "dev": ["dev-a"]}
    manifests = {}
    for split, name in (("train", "support"), ("dev", "eval")):
        manifest = {
            "source_count": len(ids[split]),
            "source_ids_sha256": hashlib.sha256(
                json.dumps(ids[split], separators=(",", ":")).encode()).hexdigest(),
            "ids": list(reversed(ids[split])),
        }
        if split == "train":
            manifest.update(demand=["train-c", "train-a"], calibration=["train-b"])
        manifests[split] = manifest
        (root / f"configs/hotpotqa_{name}_split.json").write_text(json.dumps(manifest))
    for remote, task_ids in zip(
        (*setup.PARQUETS["train"], *setup.PARQUETS["dev"]),
        (ids["train"][:2], ids["train"][2:], ids["dev"]), strict=True,
    ):
        write_parquet(data / Path(remote).name, [hf_row(task_id) for task_id in task_ids])
    return data, ids, manifests


def test_local_parquets_exact_json_and_persistent_provenance(dataset):
    data, ids, _ = dataset
    setup.main()
    source = json.loads((data / "SOURCE.json").read_text())
    assert source == json.loads((data / "manifest.json").read_text())
    for split, (filename, count) in setup.SOURCES.items():
        output = data / filename
        assert json.loads(output.read_text()) == [expected_row(task_id) for task_id in ids[split]]
        entry = source[split]
        assert entry["source"] == "huggingface"
        assert entry["dataset"] == "hotpotqa/hotpot_qa" and entry["config"] == "distractor"
        assert entry["split"] == ("train" if split == "train" else "validation")
        assert entry["count"] == count and entry["sha256"] == setup.sha256(output)
        assert [item["path"] for item in entry["inputs"]] == list(setup.PARQUETS[split])
        for item in entry["inputs"]:
            assert item["transport"] == "local"
            assert item["url"] == setup.HF_BASE + item["path"]
            assert item["sha256"] == setup.sha256(data / item["file"])
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.iterdir()}
    setup.main()
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.iterdir()}
    assert not list(data.glob("*.part"))


@pytest.mark.parametrize("keep_first_train_shard", [False, True])
def test_timeout_fallback_downloads_only_missing_shards(dataset, monkeypatch, keep_first_train_shard):
    data, ids, _ = dataset
    payloads, local_names = {}, set()
    for split, remotes in setup.PARQUETS.items():
        for index, remote in enumerate(remotes):
            path = data / Path(remote).name
            payloads[setup.HF_BASE + remote] = path.read_bytes()
            if keep_first_train_shard and split == "train" and index == 0:
                local_names.add(path.name)
            else:
                path.unlink()
    calls = []

    def urlopen(request, timeout):
        assert timeout == 60
        calls.append(request.full_url)
        if request.full_url.startswith(setup.BASE):
            raise TimeoutError("synthetic CMU timeout")
        return io.BytesIO(payloads[request.full_url])

    monkeypatch.setattr(setup.urllib.request, "urlopen", urlopen)
    setup.main()
    source = json.loads((data / "SOURCE.json").read_text())
    for split, (filename, _) in setup.SOURCES.items():
        expected_attempts = 0 if split == "train" and keep_first_train_shard else 3
        assert calls.count(setup.BASE + filename) == expected_attempts
        assert json.loads((data / filename).read_text()) == [expected_row(task_id) for task_id in ids[split]]
        for item in source[split]["inputs"]:
            is_local = item["file"] in local_names
            assert item["transport"] == ("local" if is_local else "download")
            assert calls.count(item["url"]) == (0 if is_local else 1)
    assert not list(data.glob("*.part"))


@pytest.mark.parametrize("failure", ["count", "duplicate", "order", "ids", "demand", "calibration", "nested"])
def test_conversion_rejects_invalid_inventory_or_structure_atomically(dataset, failure):
    data, _, manifests = dataset
    paths = [data / Path(remote).name for remote in setup.PARQUETS["train"]]
    count, manifest = 3, deepcopy(manifests["train"])
    message = failure
    if failure == "count":
        count, message = 4, "expected 4 questions"
    elif failure == "duplicate":
        write_parquet(paths[1], [hf_row("train-a")])
    elif failure == "order":
        paths.reverse()
    elif failure in ("ids", "demand", "calibration"):
        manifest[failure].append("missing")
        message = f"unresolved frozen {failure}"
    else:
        row = hf_row("train-c")
        row["context"]["title"].append("Unpaired")
        write_parquet(paths[1], [row])
        message = "Malformed HuggingFace parquet record"
    destination = data / "converted.json"
    with pytest.raises(ValueError, match=message):
        setup.convert_parquets(paths, destination, count, manifest)
    assert not destination.exists()
    assert not list(data.glob("*.part"))


def test_existing_json_without_provenance_is_not_misattributed(dataset):
    data, ids, _ = dataset
    for split, (filename, _) in setup.SOURCES.items():
        (data / filename).write_text(json.dumps([expected_row(task_id) for task_id in ids[split]]))
    setup.main()
    source = json.loads((data / "SOURCE.json").read_text())
    assert all(entry["source"] == "existing_json" for entry in source.values())


def test_corrupt_local_parquet_is_preserved_and_not_published(dataset):
    data, _, _ = dataset
    corrupt = data / Path(setup.PARQUETS["train"][1]).name
    corrupt.write_bytes(b"unfinished download")
    with pytest.raises((OSError, ValueError)):
        setup.main()
    assert corrupt.read_bytes() == b"unfinished download"
    assert not (data / setup.SOURCES["train"][0]).exists()
    assert not list(data.glob("*.part"))
