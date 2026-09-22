"""CPU integration oracles for frozen subset bytes, pi1 recipes, and exposure."""
from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from bfas.rtd.baselines.pi1 import encode_teacher_turn, load_bank, load_config
from bfas.rtd.benchmarks.alfworld_support import FrozenRenderer, audit_verified_bank
from bfas.rtd.persistence import file_hash
from tools import alf_bank_subset as subset
from tools.bfcl_hub_merge_export import _snapshot_for_model

NAMES = ("d0_lowsweep", "d0_highsweep", "smartad_all87")
EXPECTED = {"d0_lowsweep": (15, 136, 2490), "d0_highsweep": (15, 288, 7159),
            "smartad_all87": (87, 1343, 33436)}


def config(name):
    return load_config(ROOT / f"configs/rtd/pi1_alfworld_k32_{name}.yaml")


def bank_path(name):
    return ROOT / config(name)["bank"]


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(key, "1")

    def forbidden(*args, **kwargs):
        pytest.fail("subset verification attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(scope="module")
def renderer():
    return FrozenRenderer(_snapshot_for_model(config("d0_lowsweep")["student"]))


@pytest.mark.parametrize("name", NAMES)
def test_config_matches_bank_and_unchanged_recipe(name, renderer):
    cfg = config(name)
    kang = load_config(ROOT / "configs/rtd/pi1_alfworld_k32_kang.yaml")
    assert {k: v for k, v in cfg.items() if k not in subset.BANK_FIELDS} == {
        k: v for k, v in kang.items() if k not in subset.BANK_FIELDS}
    path = bank_path(name)
    assert file_hash(path / "sealed/manifest.json") == cfg["sealed_manifest_sha256"]
    rows, identity = load_bank(path, cfg, renderer)
    assert cfg["support_size"] == 32
    assert len({r.package_id for r in rows}) == identity["demonstrations"] == EXPECTED[name][0]
    assert len(rows) == cfg["supervised_turns"] == EXPECTED[name][1]
    tokenizer = renderer.adapter._tokenizer
    encoded = [encode_teacher_turn(tokenizer, row, cfg["max_context_tokens"]) for row in rows]
    assert all(r["target_ids"][-1] == 106 for r in encoded)
    tokens = sum(len(r["target_ids"]) for r in encoded)
    # Independent denominator: authored replies without automatic special tokens,
    # plus exactly one native boundary per turn (no prompt/observation tokens).
    assert tokens == sum(len(tokenizer.encode(row.target, add_special_tokens=False)) + 1
                         for row in rows) == cfg["bank_supervised_tokens"]
    assert tokens == EXPECTED[name][2]
    assert identity["support_manifest_hash"] == subset.read_json(path / "public/support.json")["manifest_hash"]


@pytest.mark.parametrize("name", NAMES)
def test_real_cli_preflight(name):
    completed = subprocess.run([sys.executable, str(ROOT / "tools/alf_pi1_train.py"),
        "--config", str(ROOT / f"configs/rtd/pi1_alfworld_k32_{name}.yaml"), "--preflight"],
        cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1"), check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    cfg = config(name)
    assert report["assertions_passed"]
    assert report["exposure"]["turns"] == cfg["supervised_turns"]
    assert report["exposure"]["bank_supervised_tokens"] == cfg["bank_supervised_tokens"]
    assert report["exposure"]["target_tokens"] == [p * cfg["bank_supervised_tokens"] for p in (3, 10)]


@pytest.mark.parametrize("option", ["--package-ids", "--package-id"])
def test_explicit_subset_roundtrip(tmp_path, renderer, option, capsys):
    source = bank_path("d0_lowsweep")
    source_manifest = subset.read_json(source / "sealed/manifest.json")
    packages = subset.usable_packages(source)
    ids = sorted(packages)[:2]
    output, cfg_path = tmp_path / "subset", tmp_path / "subset.yaml"
    if option == "--package-ids":
        id_path = tmp_path / "ids.json"
        id_path.write_text(json.dumps(ids[::-1]))
        selection = [option, str(id_path)]
    else:
        selection = [arg for q in reversed(ids) for arg in (option, q)]
    assert subset.main(["--source", str(source), "--output", str(output), "--config", str(cfg_path),
                        *selection]) == 0
    assert json.loads(capsys.readouterr().out)["audit"]["passed"]
    cfg = load_config(cfg_path)
    rows, identity = load_bank(output, cfg, renderer)
    assert len({r.package_id for r in rows}) == 2
    assert len(rows) == sum(len(packages[q]["teacher_react_turns"]) for q in ids)
    assert identity["sealed_manifest_sha256"] == cfg["sealed_manifest_sha256"]
    assert cfg["bank_supervised_tokens"] == sum(len(encode_teacher_turn(
        renderer.adapter._tokenizer, r, cfg["max_context_tokens"])["target_ids"]) for r in rows)
    for q in ids:
        assert (output / "sealed" / f"{q}.json").read_bytes() == (source / "sealed" / f"{q}.json").read_bytes()
    assert subset.read_json(output / "public/support.json") == subset.read_json(source / "public/support.json")
    assert subset.read_json(output / "sealed/audit.json")["historical_inventory"] == subset.read_json(
        source / "sealed/audit.json")["historical_inventory"]
    manifest = subset.read_json(output / "sealed/manifest.json")
    assert manifest["source_files"] == source_manifest["source_files"]
    assert manifest["derivation"]["source_bank"] == str(source)
    assert manifest["derivation"]["source_sealed_manifest_sha256"] == file_hash(source / "sealed/manifest.json")
    assert manifest["derivation"]["included_package_ids"] == ids
    assert manifest["derivation"]["new_teacher_calls"] == 0
    # Identical input IDs in a different order must produce identical bank bytes.
    again = tmp_path / "again"
    subset.materialize_subset(source, again, ids)
    assert file_hash(again / "sealed/manifest.json") == cfg["sealed_manifest_sha256"]
    with pytest.raises(FileExistsError):
        subset.materialize_subset(source, output, ids)
    # A changed package must fail audit before the reader can consume it.
    with (again / "sealed" / f"{ids[0]}.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="artifact integrity"):
        audit_verified_bank(again)


@pytest.mark.parametrize("case", ["empty", "duplicate", "unknown", "bad-source-hash", "inside-source"])
def test_invalid_subset_is_not_published(tmp_path, case):
    source = bank_path("d0_lowsweep")
    qid = subset.read_json(source / "sealed/manifest.json")["derivation"]["included_package_ids"][0]
    ids = {"empty": [], "duplicate": [qid, qid], "unknown": ["0" * 64]}.get(case, [qid])
    output = source / "must-not-exist" if case == "inside-source" else tmp_path / "bad"
    kwargs = {"expected_source_sha256": "0" * 64} if case == "bad-source-hash" else {}
    with pytest.raises(ValueError):
        subset.materialize_subset(source, output, ids, **kwargs)
    assert not output.exists()


def test_sweep_split_is_exact_deterministic_disjoint_and_complete():
    low = subset.usable_packages(bank_path("d0_lowsweep"))
    high = subset.usable_packages(bank_path("d0_highsweep"))
    assert len(low) == len(high) == 15
    assert not low.keys() & high.keys()
    packages = dict(low, **high)
    split = subset.sweep_split(packages)
    assert split == subset.sweep_split(dict(reversed(list(packages.items()))))
    oracle = sorted(packages, key=lambda q: (Fraction(sum(
        re.fullmatch(r"(?:go to|open|close) (?:cabinet|drawer) \d+", command) is not None
        for command in packages[q]["commands"]), len(packages[q]["commands"])), q))
    assert split["low_ids"] == oracle[:15]
    assert split["high_ids"] == oracle[15:]
    assert set(split["low_ids"]) == set(low)
    assert set(split["high_ids"]) == set(high)
    # The split cuts an exact 1/4 tie; package ID must decide it.
    assert split["packages"][14]["sweep_share"] == split["packages"][15]["sweep_share"] == 0.25
    assert split["low_ids"][-1] < split["high_ids"][0]
    assert sum(len(p["commands"]) for p in packages.values()) == 424
    assert config("d0_lowsweep")["bank_supervised_tokens"] + config("d0_highsweep")["bank_supervised_tokens"] == 9649


@pytest.mark.parametrize("name", NAMES)
def test_derivation_preserves_original_packages_and_historical_costs(name):
    path = bank_path(name)
    manifest = subset.read_json(path / "sealed/manifest.json")
    derivation = manifest["derivation"]
    source = Path(derivation["source_bank"])
    if not source.is_dir():
        pytest.skip("original frozen source bank is not installed")
    assert file_hash(source / "sealed/manifest.json") == derivation["source_sealed_manifest_sha256"]
    original = subset.read_json(source / "sealed/audit.json")["historical_inventory"]
    assert subset.read_json(path / "sealed/audit.json")["historical_inventory"] == original
    assert manifest["source_files"] == original["source_files"]
    for entry in original["source_files"]:
        assert file_hash(entry["path"]) == entry["sha256"]
    selected = subset.usable_packages(path)
    assert sorted(selected) == derivation["included_package_ids"]
    for q in selected:
        assert (path / "sealed" / f"{q}.json").read_bytes() == (source / "sealed" / f"{q}.json").read_bytes()
    if name == "smartad_all87":
        candidates = subset.read_json(Path(str(source) + ".collection") / "candidate_sets.json")
        assert candidates["bank_manifest_sha256"] == derivation["source_sealed_manifest_sha256"]
        assert sorted(selected) == sorted(a["query_id"] for group in candidates["tasks"].values()
                                         for a in group["attempts"] if a["status"] == "usable")
        per_task = Counter(p["provenance"]["task_id"] for p in selected.values())
        assert Counter(per_task.values()) == {3: 28, 2: 1, 1: 1}
        assert all(p["collection"]["method"] == "smartad" for p in selected.values())
        assert config(name)["method"] == "pi1_ce"
