"""Offline P0 audits: synthetic accounting/isolation regressions plus real splits."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.behavior_atom import cost_ledger as cost
from tools.behavior_atom import split_check as split


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def pool_row(index: int, parent: str = "source_task_1") -> dict:
    return {"_traj": f"gen_{parent}_{index}#ev1", "prompt": f"prompt {index}",
            "_rejected": f"bad {index}"}


def manifest_row(index: int) -> dict:
    row = pool_row(index)
    return {"pool_row": index, "event_key": split.event_key(row),
            "prompt_sha1": sha1(row["prompt"]), "prompt_hash": sha1(row["prompt"])[:16],
            "yT_sha1": sha1(f"good {index}"), "yS_sha1": sha1(row["_rejected"]),
            "task_id": row["_traj"].split("#")[0], "_traj": row["_traj"]}


@pytest.fixture
def manifests():
    sources = {"version": "test", "created": "2026-09-04", "pool_path": "pool.jsonl", "units": [
        {"source_id": sid, "unit_type": kind, "role": role, "parent_task_id": "source_task_1",
         "group": "source_task_1", "factor": "boundary", "family": "source",
         "rows": [manifest_row(i) for i in indices]}
        for sid, kind, role, indices in [("s1", "pack", "formal", [1, 2]),
                                         ("s2", "contrastive", "formal", [3, 4]),
                                         ("s3", "single", "pilot", [5])]] + [
        {"source_id": f"noop{i}", "unit_type": "single", "role": "noop", "rows": [],
         "parent_task_id": None, "group": "control", "factor": "control", "family": None}
        for i in (1, 2)]}
    probes = {"version": "test", "created": "2026-09-04", "probes": [
        {"probe_id": f"p{i}", "set": part, "source": origin, "task_id": task,
         "parent_task_id": parent, "group": parent, "prompt": f"probe {i}",
         "function": [], "truth": [], "category": "test", "factor": "boundary", "family": "probe",
         "expected": expected, "base_rate": None, "base_margin": None,
         "prompt_sha1": sha1(f"probe {i}"), "prompt_hash": sha1(f"probe {i}")[:16]}
        for i, part, origin, task, parent, expected in [
            (1, "P_disc", "generated", "gen_disc_task_1_0", "disc_task_1", "call"),
            (2, "P_confirm", "official", "confirm_task_1", "confirm_task_1", "abstain")]]}
    folds = {"n_folds": 2, "assignment": {"s1": 0, "s2": 0, "s3": 0},
             "groups": {f"s{i}": "source_task_1" for i in (1, 2, 3)},
             "roles": {u["source_id"]: u["role"] for u in sources["units"]},
             "controls": ["noop1", "noop2"]}
    return sources, probes, folds


@pytest.fixture
def anchor_unit(manifests):
    sources, _, folds = manifests
    unit = {**deepcopy(sources["units"][2]), "source_id": "anchor1", "role": "anchor",
            "rows": [manifest_row(6)], "pool_path": "anchors.jsonl"}
    sources["units"].append(unit)
    folds["roles"]["anchor1"] = "anchor"
    return unit


@pytest.fixture
def pairs():
    anchor = {"anchor_file": "anchors.jsonl", "row": 0, "task_id": "anchor_task_1",
              "parent_task_id": "anchor_task_1", "prompt_sha1": sha1("anchor")}
    return {"n_pairs": 2, "pairs": [
        {"pair_id": f"cp{i}", "family": "source",
         "correction": {**manifest_row(9 + i), "parent_task_id": "source_task_1"},
         "anchor_match": deepcopy(anchor) if i == 1 else None, "anchor_random": deepcopy(anchor)}
        for i in (1, 2)]}


def test_split_passes_with_advisory_counts_and_null_noops(manifests, pairs):
    original = deepcopy((manifests, pairs))
    report = split.check_split(*manifests, set(), pairs)
    assert report["passed"], json.dumps(report, indent=2)
    assert report["checks"]["counts"]["report_only"]
    assert not report["checks"]["counts"]["passed"]
    assert report["counts"] == {"formal": 2, "pilot": 1, "noop": 2, "anchor": 0, "P_disc": 1,
                                "P_confirm": 1, "pairs": 2, "anchor_match_null": 1}
    assert report["checks"]["unique_event_key"]["passed"]
    assert (manifests, pairs) == original
    assert split.check_split(*manifests, set())["passed"]


@pytest.mark.parametrize("unit_index", [0, 1, 2])
@pytest.mark.parametrize("parent", [None, "control"])
@pytest.mark.parametrize("unit_type", ["single", "noop"])
def test_noop_with_duplicated_rows_passes(manifests, unit_index, parent, unit_type):
    sources, _, _ = manifests
    for unit in sources["units"]:
        if unit["role"] == "noop":
            unit.update(unit_type=unit_type, parent_task_id=parent,
                        rows=deepcopy(sources["units"][unit_index]["rows"]))
    original = deepcopy(manifests)
    report = split.check_split(*manifests, set())
    assert report["passed"], json.dumps(report, indent=2)
    for check in ("noop_rows_reference_existing", "unique_event_key", "unique_prompt_sha1"):
        assert report["checks"][check]["passed"]
    assert manifests == original


@pytest.mark.parametrize("parent", [None, "control"])
def test_noop_type_with_empty_rows_passes(manifests, parent):
    manifests[0]["units"][3].update(unit_type="noop", parent_task_id=parent)
    assert split.check_split(*manifests, set())["passed"]


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("reference", ["unknown", "anchor"])
def test_noop_rows_require_formal_or_pilot_reference(manifests, anchor_unit, strict, reference):
    sources, _, _ = manifests
    row = manifest_row(99) if reference == "unknown" else anchor_unit["rows"][0]
    # A later unmatched row must fail, even if another noop has the same row.
    sources["units"][3].update(unit_type="noop", parent_task_id="control",
                               rows=[deepcopy(sources["units"][0]["rows"][0]), deepcopy(row)])
    sources["units"][4]["rows"] = [deepcopy(row)]
    report = split.check_split(*manifests, set(), strict=strict)
    assert not report["passed"]
    check = report["checks"]["noop_rows_reference_existing"]
    assert check["offending_ids"] == ["noop1", "noop2"]
    assert check["details"] == [
        {"ids": ["noop1"], "unit_row": 1, "event_key": row["event_key"]},
        {"ids": ["noop2"], "unit_row": 0, "event_key": row["event_key"]}]
    assert all(c["passed"] for name, c in report["checks"].items()
               if name not in {"noop_rows_reference_existing", "counts"})


def test_noop_rows_excluded_from_probe_isolation_and_calibration(manifests):
    sources, probes, _ = manifests
    row = deepcopy(sources["units"][0]["rows"][0])
    # Only event_key establishes the noop reference; other row fields are ignored
    # by isolation and calibration checks.
    row.update(prompt_sha1=probes["probes"][0]["prompt_sha1"],
               _traj=probes["probes"][0]["task_id"] + "#ev1",
               task_id="gen_held_out_31_7#ev2", parent_task_id="held_out_31",
               gen_ids=["genmt_held_out_31_8"])
    sources["units"][3].update(unit_type="noop", parent_task_id="control", rows=[row])
    report = split.check_split(*manifests, {"held_out_31"})
    assert report["passed"], json.dumps(report, indent=2)


def test_noop_requires_null_or_control_parent(manifests):
    manifests[0]["units"][3].update(unit_type="noop", parent_task_id="source_task_1")
    report = split.check_split(*manifests, set())
    assert not report["passed"]
    assert report["checks"]["manifest_fields"]["offending_ids"] == ["noop1"]


@pytest.mark.parametrize("unit_index", [0, 2, 5])
def test_noop_unit_type_requires_noop_role(manifests, anchor_unit, unit_index):
    unit = manifests[0]["units"][unit_index]
    unit["unit_type"] = "noop"
    report = split.check_split(*manifests, set())
    assert not report["passed"]
    assert report["checks"]["manifest_fields"]["offending_ids"] == [unit["source_id"]]


def test_anchor_unit_passes_without_fold_membership(manifests, anchor_unit):
    original = deepcopy(manifests)
    report = split.check_split(*manifests, set())
    assert report["passed"], json.dumps(report, indent=2)
    assert report["counts"]["anchor"] == 1
    assert report["counts"]["formal"] == 2
    assert report["counts"]["pilot"] == 1
    assert report["checks"]["anchor_not_assigned"]["passed"]
    assert manifests == original


@pytest.mark.parametrize("field,value", [("rows", []), ("parent_task_id", None)])
def test_anchor_unit_requires_rows_and_parent(manifests, anchor_unit, field, value):
    anchor_unit[field] = value
    report = split.check_split(*manifests, set())
    assert not report["passed"]
    assert report["checks"]["manifest_fields"]["offending_ids"] == ["anchor1"]


@pytest.mark.parametrize("field,value", [("assignment", 0), ("groups", "source_task_1")])
def test_anchor_unit_cannot_appear_in_folds(manifests, anchor_unit, field, value):
    manifests[2][field]["anchor1"] = value
    report = split.check_split(*manifests, set())
    assert not report["passed"]
    assert report["checks"]["anchor_not_assigned"]["offending_ids"] == ["anchor1"]
    if field == "assignment":
        assert report["checks"]["fold_membership"]["offending_ids"] == ["anchor1"]


def test_anchor_prompt_equal_to_probe_prompt_fails(manifests, anchor_unit):
    anchor_unit["rows"][0]["prompt_sha1"] = manifests[1]["probes"][0]["prompt_sha1"]
    report = split.check_split(*manifests, set())
    assert not report["passed"]
    for check in ("probe_source_prompt_hashes", "unique_prompt_sha1"):
        assert report["checks"][check]["offending_ids"] == ["anchor1", "p1"]


@pytest.mark.parametrize("target", ["unit", "row"])
@pytest.mark.parametrize("field", ["parent_task_id", "task_id", "_traj"])
def test_anchor_calibration_exclusion(manifests, anchor_unit, target, field):
    row = anchor_unit if target == "unit" else anchor_unit["rows"][0]
    row[field] = "gen_held_out_31_7#ev2"
    report = split.check_split(*manifests, {"held_out_31"})
    assert not report["passed"]
    assert "anchor1" in report["checks"]["calibration_exclusion"]["offending_ids"]


@pytest.mark.parametrize("unit_index", [0, 2, 5])
def test_deliberate_probe_parent_overlap_lists_offending_ids(manifests, anchor_unit, unit_index):
    sources, probes, folds = manifests
    source = sources["units"][unit_index]
    probes["probes"][0]["parent_task_id"] = source["parent_task_id"]
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert {source["source_id"], "p1"} <= set(report["checks"]["probe_source_parent_tasks"]["offending_ids"])


def test_deliberate_probe_prompt_overlap_lists_offending_ids(manifests):
    sources, probes, folds = manifests
    probes["probes"][1]["prompt_sha1"] = sources["units"][1]["rows"][1]["prompt_sha1"]
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert {"s2", "p2"} <= set(report["checks"]["probe_source_prompt_hashes"]["offending_ids"])
    assert {"s2", "p2"} <= set(report["checks"]["unique_prompt_sha1"]["offending_ids"])


def test_split_duplicate_source_id(manifests):
    sources, probes, folds = manifests
    sources["units"][1]["source_id"] = "s1"
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert report["checks"]["unique_source_id"]["offending_ids"] == ["s1"]


@pytest.mark.parametrize("field", ["prompt_sha1", "event_key"])
@pytest.mark.parametrize("unit_index", [0, 1, 2, 5])
def test_split_duplicate_source_row_identity(manifests, anchor_unit, field, unit_index):
    sources, probes, folds = manifests
    sources["units"][unit_index]["rows"][-1][field] = sources["units"][0]["rows"][0][field]
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert "s1" in report["checks"][f"unique_{field}"]["offending_ids"]


@pytest.mark.parametrize("field", ["probe_id", "prompt_sha1"])
def test_split_duplicate_probe_identity(manifests, field):
    sources, probes, folds = manifests
    probes["probes"][1][field] = probes["probes"][0][field]
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert "p1" in report["checks"][f"unique_{field}"]["offending_ids"]


def test_split_full_hashes_and_event_keys_are_authoritative(manifests):
    sources, probes, folds = manifests
    first, second = sources["units"][0]["rows"]
    # A short-hash collision or repeated trajectory is not a duplicate identity.
    second["prompt_sha1"] = first["prompt_sha1"][:16] + "0" * 24
    second["prompt_hash"] = first["prompt_hash"]
    second["_traj"] = first["_traj"]
    second["task_id"] = first["task_id"]
    sources["units"][0]["content_hash"] = sources["units"][1]["content_hash"] = "same"
    assert split.check_split(sources, probes, folds, set())["passed"]


def calibration_record(manifests, pairs, target):
    sources, probes, _ = manifests
    if target == "unit":
        return sources["units"][0], "s1"
    if target == "unit_row":
        return sources["units"][0]["rows"][1], "s1"
    if target == "probe":
        return probes["probes"][0], "p1"
    return pairs["pairs"][0][target], "cp1"


@pytest.mark.parametrize("target", ["unit", "unit_row", "probe", "correction", "anchor_match", "anchor_random"])
@pytest.mark.parametrize("field", ["parent_task_id", "task_id"])
def test_split_calibration_parent_or_task(manifests, pairs, target, field):
    row, rid = calibration_record(manifests, pairs, target)
    row[field] = "held_out_31"
    report = split.check_split(*manifests, {"held_out_31"}, pairs)
    assert not report["passed"]
    assert rid in report["checks"]["calibration_exclusion"]["offending_ids"]


@pytest.mark.parametrize("prefix", ["gen_", "oos_", "genmt_"])
@pytest.mark.parametrize("field", ["parent_task_id", "task_id", "_traj", "gen_ids"])
def test_split_generated_calibration_ids_and_official_suffix(manifests, pairs, prefix, field):
    for target in ("unit", "unit_row", "probe", "correction", "anchor_match", "anchor_random"):
        row, _ = calibration_record(manifests, pairs, target)
        value = f"{prefix}held_out_31_7#ev2"
        row[field] = [value] if field == "gen_ids" else value
    report = split.check_split(*manifests, {"held_out_31"}, pairs)
    assert not report["passed"]
    assert {"s1", "p1", "cp1"} <= set(report["checks"]["calibration_exclusion"]["offending_ids"])
    assert split.parent_id("simple_python_31#ev2") == "simple_python_31"


@pytest.mark.parametrize("source_field", ["_traj", "task_id"])
@pytest.mark.parametrize("probe_field", ["task_id", "parent_task_id", "gen_ids"])
@pytest.mark.parametrize("unit_index", [0, 5])
def test_split_trajectory_base_overlap(manifests, anchor_unit, source_field, probe_field, unit_index):
    sources, probes, folds = manifests
    unit = sources["units"][unit_index]
    unit["rows"][-1][source_field] = "shared_task_1#ev1"
    value = "shared_task_1#ev9"
    probes["probes"][0][probe_field] = [value] if probe_field == "gen_ids" else value
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert {unit["source_id"], "p1"} <= set(report["checks"]["probe_source_traj_bases"]["offending_ids"])


def test_confirm_disc_parents_disjoint(manifests):
    sources, probes, folds = manifests
    probes["probes"][1]["parent_task_id"] = probes["probes"][0]["parent_task_id"]
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert report["checks"]["confirm_disc_parent_tasks"]["offending_ids"] == ["p1", "p2"]


@pytest.mark.parametrize("sid", ["s2", "s3"])
def test_fold_splitting_a_parent_fails(manifests, sid):
    sources, probes, folds = manifests
    folds["assignment"][sid] = 1
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    check = report["checks"]["parent_task_single_fold"]
    assert {"s1", sid} <= set(check["offending_ids"])
    assert check["details"][0]["folds"] == [0, 1]
    assert report["checks"]["formal_source_fold_once"]["passed"]
    assert report["checks"]["pilot_source_fold_once"]["passed"]


@pytest.mark.parametrize("sid,role", [("s2", "formal"), ("s3", "pilot")])
def test_formal_and_pilot_fold_assignment_required(manifests, sid, role):
    sources, probes, folds = manifests
    del folds["assignment"][sid]
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert report["checks"][f"{role}_source_fold_once"]["offending_ids"] == [sid]


@pytest.mark.parametrize("group", [None, "wrong_parent"])
def test_fold_parent_coverage(manifests, group):
    sources, probes, folds = manifests
    if group is None:
        del folds["groups"]["s2"]
    else:
        folds["groups"]["s2"] = group
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert report["checks"]["fold_parent_coverage"]["offending_ids"] == ["s2"]


@pytest.mark.parametrize("fold", [-1, 2, "0", True, [0, 1]])
def test_invalid_fold_assignment(manifests, fold):
    sources, probes, folds = manifests
    folds["assignment"]["s2"] = fold
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert report["checks"]["fold_membership"]["offending_ids"] == ["s2"]


@pytest.mark.parametrize("sid", ["noop1", "unknown"])
def test_only_formal_and_pilot_units_assigned(manifests, sid):
    sources, probes, folds = manifests
    folds["assignment"][sid] = 0
    report = split.check_split(sources, probes, folds, set())
    assert not report["passed"]
    assert report["checks"]["fold_membership"]["offending_ids"] == [sid]
    if sid == "noop1":
        assert report["checks"]["noop_not_assigned"]["offending_ids"] == [sid]


@pytest.mark.parametrize("component", ["correction", "anchor_match", "anchor_random"])
@pytest.mark.parametrize("probe_index", [0, 1])
def test_pair_parent_overlaps_probe_parent(manifests, pairs, component, probe_index):
    _, probes, _ = manifests
    pairs["pairs"][0][component]["parent_task_id"] = probes["probes"][probe_index]["parent_task_id"]
    report = split.check_split(*manifests, set(), pairs)
    assert not report["passed"]
    check = report["checks"]["pair_probe_parent_tasks"]
    assert {"cp1", f"p{probe_index + 1}"} <= set(check["offending_ids"])
    assert check["details"][0]["component"] == component


@pytest.mark.parametrize("unit_index", [0, 2, 5])
def test_pair_correction_is_separate_from_source_rows(manifests, pairs, anchor_unit, unit_index):
    sources, _, _ = manifests
    pairs["pairs"][0]["correction"]["event_key"] = sources["units"][unit_index]["rows"][-1]["event_key"]
    report = split.check_split(*manifests, set(), pairs)
    assert not report["passed"]
    assert {"cp1", sources["units"][unit_index]["source_id"]} <= set(
        report["checks"]["pair_source_event_keys"]["offending_ids"])


@pytest.mark.parametrize("field,value", [("prompt_sha1", "a" * 16), ("prompt_hash", "z" * 16)])
def test_split_strict_hash_fields(manifests, field, value):
    sources, probes, folds = manifests
    sources["units"][0]["rows"][0][field] = value
    assert not split.check_split(sources, probes, folds, set())["checks"]["manifest_fields"]["passed"]
    assert split.check_split(sources, probes, folds, set(), strict=False)["passed"]
    probes["probes"][0]["parent_task_id"] = "source_task_1"
    assert not split.check_split(sources, probes, folds, set(), strict=False)["passed"]


def test_real_split_manifests(capsys):
    paths = {name: ROOT / "data/behavior_atom_v1" / f"{name}.json"
             for name in ("sources", "probes", "folds")}
    paths.update({"pairs": ROOT / "data/behavior_atom_v1/complementarity_pairs.json",
                  "support-split": ROOT / "configs/bfcl_support_split.json",
                  "calibration-v2": ROOT / "data/bfcl_sft/calibration_ids_official_v2.json",
                  "pool": ROOT / "data/bfcl_sft/pool_events_pref_v3t.jsonl",
                  "anchor-pool": ROOT / "data/behavior_atom_v1/pool_anchors_legal.jsonl"})
    if not all(p.exists() for p in paths.values()):
        pytest.skip("Real split manifests/pairs/calibration/pool inputs are unavailable")
    status = split.main([arg for name, path in paths.items() for arg in (f"--{name}", str(path))])
    report = json.loads(capsys.readouterr().out)
    assert status == 0, json.dumps(report, indent=2)
    assert report["passed"], json.dumps(report, indent=2)
    for check in ("counts", "noop_rows_reference_existing", "pool_event_keys", "anchor_pool_event_keys"):
        assert report["checks"][check]["passed"]
    pairs = json.loads(paths["pairs"].read_text())
    assert report["counts"]["anchor_match_null"] == sum(p["anchor_match"] is None for p in pairs["pairs"])


def test_split_cli_pool_occurs_exactly_once(tmp_path, manifests, pairs):
    sources, probes, folds = manifests
    pool_rows = [pool_row(r["pool_row"]) for unit in sources["units"] for r in unit["rows"]]
    pool_rows += [pool_row(p["correction"]["pool_row"]) for p in pairs["pairs"]]
    args = []
    for flag, value in (("sources", sources), ("probes", probes), ("folds", folds), ("pairs", pairs),
                        ("support-split", {"calibration": []}), ("calibration-v2", {"ids": []})):
        args += [f"--{flag}", str(write_json(tmp_path / f"{flag}.json", value))]
    pool = write_jsonl(tmp_path / "pool.jsonl", pool_rows)
    out = tmp_path / "reports/split.json"
    cmd = [sys.executable, str(ROOT / "tools/behavior_atom/split_check.py"), *args,
           "--pool", str(pool), "--out", str(out), "--expect-formal", "2", "--expect-disc", "1"]
    passing = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert passing.returncode == 0, passing.stderr + passing.stdout
    report = json.loads(passing.stdout)
    assert report == json.loads(out.read_text())
    assert report["expected_counts"]["formal"] == 2
    assert report["counts"]["pairs"] == 2
    assert report["inputs"]["pairs"]["status"] == "ok"
    # Check a later pack row, the pilot row, and a separate pair correction.
    for index, rid in [(1, "s1"), (4, "s3"), (5, "cp1")]:
        for occurrences in (0, 2):
            changed = pool_rows[:index] + pool_rows[index + 1:] + [pool_rows[index]] * occurrences
            write_jsonl(pool, changed)
            failed = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
            assert failed.returncode == 1, failed.stderr + failed.stdout
            report = json.loads(failed.stdout)
            assert report == json.loads(out.read_text())
            check = report["checks"]["pool_event_keys"]
            assert check["offending_ids"] == [rid]
            assert check["details"][0]["pool_occurrences"] == occurrences
    write_jsonl(pool, pool_rows)
    for filename, field in [("support-split", "calibration"), ("calibration-v2", "ids")]:
        path = tmp_path / f"{filename}.json"
        write_json(path, {field: ["anchor_task_1"]})
        failed = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
        assert failed.returncode == 1, failed.stderr + failed.stdout
        assert json.loads(failed.stdout)["checks"]["calibration_exclusion"]["offending_ids"] == ["cp1", "cp2"]
        write_json(path, {field: []})


@pytest.mark.parametrize("role", ["formal", "pilot", "anchor"])
@pytest.mark.parametrize("unit_pool", [None, "pool.jsonl", "anchors.jsonl"])
def test_pool_checks_only_units_from_top_level_pool(tmp_path, manifests, role, unit_pool):
    unit = {**manifests[0]["units"][2], "role": role}
    if unit_pool is not None:
        unit["pool_path"] = unit_pool
    sources = {**manifests[0], "units": [unit]}
    pool = write_jsonl(tmp_path / "pool.jsonl", [])
    report = split.check_pool(sources, pool)
    assert report["passed"] == (unit_pool == "anchors.jsonl")
    if not report["passed"]:
        assert report["offending_ids"] == ["s3"]
        assert report["details"][0]["pool_occurrences"] == 0
    write_jsonl(pool, [pool_row(5)])
    assert split.check_pool(sources, pool)["passed"]


def test_split_cli_separate_anchor_pool(tmp_path, manifests, pairs, anchor_unit, capsys):
    sources, probes, folds = manifests
    args = []
    for flag, value in (("sources", sources), ("probes", probes), ("folds", folds), ("pairs", pairs),
                        ("support-split", {"calibration": []}), ("calibration-v2", {"ids": []})):
        args += [f"--{flag}", str(write_json(tmp_path / f"{flag}.json", value))]
    pool_rows = [pool_row(r["pool_row"]) for unit in sources["units"]
                 if unit["role"] != "anchor" for r in unit["rows"]]
    pool_rows += [pool_row(p["correction"]["pool_row"]) for p in pairs["pairs"]]
    pool_args = ["--pool", str(write_jsonl(tmp_path / "pool.jsonl", pool_rows))]
    anchor_pool = write_jsonl(tmp_path / "anchors.jsonl", [pool_row(6)])
    anchor_args = ["--anchor-pool", str(anchor_pool)]
    # Either pool audit can be requested independently, or both together.
    for options in (pool_args, anchor_args, pool_args + anchor_args):
        assert split.main(args + options) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["passed"]
        assert ("anchor_pool_event_keys" in report["checks"]) == ("--anchor-pool" in options)
    for occurrences in (0, 2):
        write_jsonl(anchor_pool, [pool_row(6)] * occurrences)
        assert split.main(args + pool_args + anchor_args) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["checks"]["pool_event_keys"]["passed"]
        check = report["checks"]["anchor_pool_event_keys"]
        assert check["offending_ids"] == ["anchor1"]
        assert check["details"][0]["pool_occurrences"] == occurrences


@pytest.mark.parametrize("value,expected", [([1, [2, [], [3, "4"]], None], 10), ([], 0), ("19", 19)])
def test_flatten_nested_tokens(value, expected):
    assert cost.flatten_tokens(value) == expected


@pytest.mark.parametrize("task,expected", [("official_1", "demand"), ("memory_kv_prereq_1", "memory_prereq"), ("other_2", "other_official")])
def test_group_demo_ids(task, expected):
    assert cost.group_demo_id(task, {"official_1", "memory_kv_prereq_1"}) == expected


@pytest.fixture
def demo_campaign(tmp_path):
    dirs = [tmp_path / f"a{i}" for i in (1, 2, 3)]
    write_jsonl(dirs[0] / "nested/one_result.json", [{"id": "official_1", "input_token_count": [[10, 20]],
        "output_token_count": [[2, [3]], 5], "verified": False, "reasoning_tokens": 999,
        "usage": {"completion_tokens_details": {"reasoning_tokens": 999}}}])
    write_jsonl(dirs[1] / "two_result.json", [{"id": "official_1", "input_token_count": 30,
        "output_token_count": [4, 6], "verified": False}])
    write_jsonl(dirs[2] / "three_result.json", [
        {"id": "memory_kv_prereq_1", "input_token_count": 5, "output_token_count": 7},
        {"id": "other_2", "input_token_count": 3, "output_token_count": 9}])
    verified = tmp_path / "verified"
    write_jsonl(verified / "merged_result.json", [{"id": "other_2", "input_token_count": 1000000, "output_token_count": 1000000}])
    return dirs, verified


def test_demo_attempts_failed_billed_and_reasoning_not_added(demo_campaign):
    dirs, verified = demo_campaign
    report = cost.audit_demos(dirs, verified, {"official_1"})
    demand = report["groups"]["demand"]
    assert (demand["n_ids"], demand["n_rows"]) == (1, 2)
    assert demand["output_tokens"]["value"] == 20
    assert demand["input_tokens"]["value"] == 60
    assert report["totals"]["output_tokens"]["value"] == 36
    assert report["totals"]["input_tokens"]["value"] == 68
    assert report["totals"]["n_rows"] == 4
    assert report["per_id"]["official_1"]["verified"] is False
    assert report["per_id"]["other_2"]["verified"] is True
    assert [t["value"] for t in report["per_id"]["official_1"]["output_tokens_per_attempt"]] == [10, 10]
    assert "must NOT be added again" in report["note"]


def test_dedup_ledger_retries_and_failed_attempts(tmp_path):
    rows = [
        {"task_id": "task", "attempt_index": 0, "tokens_spent": "7", "verified": "False", "purpose": "teacher"},
        {"task_id": "task", "attempt_index": "0", "tokens_spent": "13", "verified": "True", "purpose": "user_sim"},
        {"task_id": "task", "attempt_index": "0", "tokens_spent": "999", "verified": "True", "purpose": "teacher"},
        {"task_id": "task", "attempt_index": 1, "tokens_spent": "11", "verified": "True", "purpose": "teacher"},
        {"task_id": "other", "attempt_index": 0, "tokens_spent": "3", "verified": "False", "purpose": "user_sim"},
        {"task_id": "zero", "attempt_index": 0, "tokens_spent": "0", "verified": "False", "purpose": "teacher"},
    ]
    for row in rows:
        row["timestamp"] = "2026-09-03T12:00:00Z"
    distinct = cost.dedup_ledger_rows(rows[:2])
    assert distinct["rows"] == rows[:2]
    assert distinct["duplicate_rows"] == distinct["duplicate_pairs"] == 0
    assert distinct["warnings"] == []
    dedup = cost.dedup_ledger_rows(rows)
    assert dedup["rows"] == [*rows[:2], *rows[3:]]
    assert dedup["duplicate_rows"] == dedup["duplicate_pairs"] == 1
    assert dedup["warnings"] == [
        "Excluded 1 duplicate rows in 1 groups keyed by (task_id, attempt_index, purpose, timestamp), "
        "or (task_id, attempt_index, purpose) when timestamp is missing; kept first occurrence."
    ]
    group = dedup["duplicates"][0]
    assert (group["task_id"], group["attempt_index"], group["purpose"], group["first_row"]) == ("task", "0", "teacher", 1)
    assert group["timestamp"] == rows[0]["timestamp"]
    assert group["duplicates"][0]["row"] == 3
    assert group["duplicates"][0]["purpose"] == "teacher"
    assert group["duplicates"][0]["tokens_spent"]["value"] == 999
    path = write_jsonl(tmp_path / "ledger.jsonl", rows)
    report = cost.audit_ledger(path, "alfworld", date(2026, 9, 4))
    assert (report["rows"], report["billed_rows"]) == (6, 5)
    assert report["tokens_spent"]["value"] == 34
    assert report["tokens_by_purpose"]["teacher"]["value"] == 18
    assert report["tokens_by_purpose"]["user_sim"]["value"] == 16
    assert report["attempts_per_task_histogram"] == {1: 2, 3: 1}
    assert report["zero_token_rows"] == 1
    assert report["verified_rows"] == 3 and report["billed_verified_rows"] == 2
    assert "0 new calls today" in report["note"]


def test_dedup_ledger_distinct_user_sim_turns(tmp_path):
    rows = [
        {"task_id": "task", "attempt_index": 0, "purpose": "user_sim", "tokens_spent": 7,
         "timestamp": "2026-09-03T12:00:00Z"},
        {"task_id": "task", "attempt_index": 0, "purpose": "user_sim", "tokens_spent": 13,
         "timestamp": "2026-09-03T12:00:01Z"},
    ]
    dedup = cost.dedup_ledger_rows(rows)
    assert dedup["rows"] == rows
    assert dedup["duplicate_rows"] == dedup["duplicate_pairs"] == 0
    assert dedup["warnings"] == []
    path = write_jsonl(tmp_path / "ledger.jsonl", rows)
    report = cost.audit_ledger(path, "tau2", date(2026, 9, 4))
    assert report["billed_rows"] == 2
    assert report["tokens_spent"]["value"] == 20
    assert report["tokens_by_purpose"]["user_sim"]["value"] == 20


@pytest.mark.parametrize("timestamp_fields", [{}, {"timestamp": None}, {"timestamp": ""}])
def test_dedup_ledger_missing_timestamp_falls_back_with_warning(tmp_path, timestamp_fields):
    row = {"task_id": "task", "attempt_index": 0, "purpose": "user_sim", "tokens_spent": 7,
           **timestamp_fields}
    rows = [row, {**row, "tokens_spent": 999}, {**row, "timestamp": "2026-09-03T12:00:00Z"}]
    dedup = cost.dedup_ledger_rows(rows)
    assert dedup["rows"] == [rows[0], rows[2]]
    assert dedup["duplicate_rows"] == dedup["duplicate_pairs"] == 1
    assert dedup["warnings"] == [
        "Row 1 lacks timestamp; falling back to (task_id, attempt_index, purpose) for deduplication.",
        "Row 2 lacks timestamp; falling back to (task_id, attempt_index, purpose) for deduplication.",
        "Excluded 1 duplicate rows in 1 groups keyed by (task_id, attempt_index, purpose, timestamp), "
        "or (task_id, attempt_index, purpose) when timestamp is missing; kept first occurrence.",
    ]
    assert dedup["duplicates"][0]["timestamp"] is None
    assert dedup["duplicates"][0]["duplicates"][0]["row"] == 2
    path = write_jsonl(tmp_path / "ledger.jsonl", rows)
    report = cost.audit_ledger(path, "tau2", date(2026, 9, 4))
    assert report["billed_rows"] == 2
    assert report["tokens_spent"]["value"] == 14


def test_pool_shared_demo_query_generated_unresolved_and_separate_costs(demo_campaign):
    demos = cost.audit_demos(*demo_campaign, {"official_1"})
    rows = [
        {"task_id": "official_1", "teacher": "deepseek-v4-pro-FC", "prompt": "first", "token_hint": 2, "_traj": "official_1#ev1", "_rejected": "bad"},
        {"task_id": "official_1", "teacher": "deepseek-v4-pro-FC", "prompt": "second", "token_hint": 3, "_traj": "official_1#ev2", "_rejected": "worse"},
        {"task_id": "gen_a_0", "teacher": "teacher_authored_gt", "prompt": "new question", "token_hint": 5},
        {"task_id": "gen_b_0", "teacher": "teacher_authored_gt", "prompt": "new question", "token_hint": 7},
    ]
    report = cost.pool_provenance(rows, demos["per_id"])
    assert report["rows"] == 4 and report["unique_prompts"] == 3
    assert report["unique_task_ids"] == 3 and report["unique_teacher_queries"] == 2
    assert report["exact_tokens"]["value"] == 20
    assert report["exact_input_tokens"]["value"] == 60
    assert report["unresolved_estimate_rows"] == 2
    assert report["unresolved_estimate_queries"] == 1
    generated = next(q for q in report["queries"] if q["kind"] == "generated")
    assert generated["status"] == "unresolved_estimate"
    assert generated["estimated_tokens"]["value"] is None
    assert report["selected_supervision_text_tokens"]["value"] == 17
    prompt_hash = hashlib.sha1(b"new question").hexdigest()[:16]
    resolved = cost.pool_provenance(rows, demos["per_id"], {prompt_hash: {"value": 13.5, "rule": "fixture estimate"}})
    assert resolved["unresolved_estimate_rows"] == 0
    assert resolved["exact_tokens"]["value"] == 20
    assert resolved["estimated_tokens"]["value"] == 13.5
    assert resolved["exact_tokens"]["basis"] == "exact"
    assert resolved["estimated_tokens"]["basis"] == "estimate"
    assert "total_tokens" not in resolved  # No combined exact + estimate total.
    demo = next(q for q in resolved["queries"] if q["kind"] == "demo")
    assert len(demo["trajectories"]) == 2 and len(demo["events"]) == 2
    assert demo["events"][0]["event_key"] == split.event_key(rows[0])


def test_pool_unresolved_demo_and_generated_reused_task_id():
    rows = [{"task_id": "reused", "teacher": "teacher_authored", "prompt": p, "token_hint": 1} for p in ("a", "b")]
    rows.append({"task_id": "missing", "teacher": "demo-model", "prompt": "c", "token_hint": 1})
    report = cost.pool_provenance(rows, {})
    assert report["unique_teacher_queries"] == 3
    assert report["unresolved_estimate_queries"] == 2
    assert report["unresolved_exact_rows"] == 1


def test_generation_chars_estimate_smoke_and_exact_usage(tmp_path):
    row = {"question": [[{"role": "user", "content": "abcd"}]], "ground_truth": ["ef"], "scenario": {"setting": "gh"}}
    generated = write_jsonl(tmp_path / "gen_pool.jsonl", [row])
    smoke = write_jsonl(tmp_path / "gen_smoke.jsonl", [row])
    usage = write_jsonl(tmp_path / "usage.jsonl", [
        {"usage": {"prompt_tokens": [5, [8]], "completion_tokens": [[7, 9]], "total_tokens": 29,
                   "completion_tokens_details": {"reasoning_tokens": 999}}, "output_tokens": 16},
        {"output_tokens": "20", "reasoning_tokens": 888},
        {"unrecognized": True},
    ])
    report = cost.audit_generation([generated, smoke], usage)
    expected_text = 'abcd\n["ef"]\n{"setting":"gh"}'
    assert cost.generation_text(row) == expected_text
    assert report["rows"] == 1
    assert report["excluded_smoke_files"] == [str(smoke)]
    assert report["totals"]["estimated_output_tokens"]["value"] == pytest.approx(len(expected_text) / 4 * 1.3)
    assert report["method"]["label"] == "estimate:chars/4x1.3"
    assert report["totals"]["exact_output_tokens"]["value"] == 36
    assert report["totals"]["exact_input_tokens"]["value"] == 13
    assert report["exact_usage"]["unrecognized_rows"] == [3]
    assert report["exact_usage"]["fields_found"]["usage.completion_tokens"] == 1
    assert cost.audit_generation([generated, smoke], usage, include_smoke=True)["rows"] == 2


def test_tokenizer_offline_fallback_and_optional_proxy(monkeypatch):
    def unavailable(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        raise OSError("no cached tokenizer")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=unavailable)))
    counter, method = cost.load_token_counter("Qwen/Qwen3.5-4B")
    assert counter("abcd") == 1
    assert method["basis"] == "estimate" and method["rule"] == "chars/4x1.3"
    assert "no cached tokenizer" in method["fallback_reason"]
    def cached(*args, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return SimpleNamespace(encode=lambda text, add_special_tokens: [1, 2, 3])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=cached)))
    counter, method = cost.load_token_counter("local-tokenizer")
    assert counter("abcd") == 3
    assert method["basis"] == "estimate" and method["rule"] == "tokenizer:local-tokenizerx1.3"


def test_training_and_student_event_costs(tmp_path):
    index = write_jsonl(tmp_path / "index.jsonl", [
        {"row": 0, "prompt_tokens": 10, "n_tok_T": 3, "n_tok_S": 4},
        {"row": 1, "prompt_tokens": 20, "n_tok_T": 5, "n_tok_S": 6},
    ])
    report = cost.audit_training(index, 3)
    assert report["per_epoch"]["loss_tokens"]["value"] == 18
    assert report["per_epoch"]["context_tokens"]["value"] == 60
    assert report["all_epochs"]["loss_tokens"]["value"] == 54
    assert report["all_epochs"]["context_tokens"]["value"] == 180
    events = write_jsonl(tmp_path / "events.jsonl", [
        {"_cost": {"student_calls": 2, "student_completion_tokens": 12, "episodes": 3, "wall_s": 1.5}, "_teacher_tokens": 4},
        {"_cost": {"student_calls": 4, "student_completion_tokens": 20, "episodes": 5, "wall_s": 2.5}, "_teacher_tokens": 6},
    ])
    student = cost.audit_appworld_events(events)
    assert student["student_calls"] == 6 and student["episodes"] == 8 and student["wall_s"] == 4
    assert student["student_completion_tokens"]["value"] == 32
    assert student["teacher_tokens"]["value"] == 10
    assert student["teacher_tokens"]["basis"] == "exact"


def test_cost_cli_missing_inputs_and_six_buckets(tmp_path):
    out = tmp_path / "reports/cost.json"
    result = subprocess.run([sys.executable, str(ROOT / "tools/behavior_atom/cost_ledger.py"), "--out", str(out)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(out.read_text())
    assert report["bfcl_demos"]["status"] == "missing"
    assert report["bfcl_generation"]["status"] == "missing"
    assert report["bfcl_generation"]["exact_usage"]["status"] == "missing"
    assert report["ledgers"]["alfworld"]["status"] == "missing"
    assert report["training_tokens"]["status"] == "missing"
    assert report["student_environment_cost"]["alfworld"]["status"] == "missing"
    assert report["student_environment_cost"]["bfcl_mining"]["status"] == "not_logged"
    assert set(report["summary"]) == {"provided_expert_demos", "fixed_bootstrap_overhead",
        "new_paid_teacher_calls_today", "selected_supervision_text_tokens", "training_tokens", "student_inference_env_cost"}
    assert report["summary"]["new_paid_teacher_calls_today"]["calls"] == 0


def test_cost_cli_overrides_combined_pools_and_future_call(tmp_path, demo_campaign, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    dirs, verified = demo_campaign
    pool_rows = [{"task_id": "official_1", "teacher": "deepseek-v4-pro-FC", "prompt": "shared", "token_hint": 3}]
    pool_a = write_jsonl(tmp_path / "pool_a.jsonl", pool_rows)
    pool_b = write_jsonl(tmp_path / "pool_b.jsonl", pool_rows)
    support = write_json(tmp_path / "support.json", {"demand": ["official_1"]})
    ledger = write_jsonl(tmp_path / "ledger.jsonl", [
        {"task_id": "task", "attempt_index": 0, "tokens_spent": 5, "verified": "False", "timestamp": "2026-09-04T23:59:59Z"},
        {"task_id": "task", "attempt_index": 0, "tokens_spent": 999, "verified": "True", "purpose": "teacher", "timestamp": "2026-09-05T00:00:00Z"},
        {"task_id": "task", "attempt_index": 0, "tokens_spent": 13, "verified": "False", "purpose": "user_sim", "timestamp": "2026-09-04T23:59:59Z"},
    ])
    out = tmp_path / "out.json"
    args = ["--out", str(out), "--support-split", str(support), "--demo-dirs", *map(str, dirs),
            "--verified-dir", str(verified), "--pool", str(pool_a), "--pool", str(pool_b),
            "--bfcl-ledger", str(ledger), "--today", "2026-09-04"]
    assert cost.main(args) == 1
    report = json.loads(out.read_text())
    assert report["pool_provenance"]["combined"]["exact_tokens"]["value"] == 20
    assert report["pool_provenance"]["combined"]["unique_teacher_queries"] == 1
    assert report["pool_provenance"]["combined"]["selected_supervision_text_tokens"]["value"] == 6
    check = report["checks"]["no_ledger_rows_after_today"]
    assert not check["passed"] and check["offending_rows"][0]["row"] == 2
    billed = report["ledgers"]["bfcl"]
    assert billed["billed_rows"] == 3
    assert billed["tokens_spent"]["value"] == 1017
    assert billed["tokens_by_purpose"]["teacher"]["value"] == 1004
    assert billed["tokens_by_purpose"]["user_sim"]["value"] == 13
    assert billed["duplicate_rows"] == billed["duplicate_pairs"] == 0
    assert billed["warnings"] == []
    assert report["summary"]["provided_expert_demos"]["exact_tokens"]["value"] == 29
    assert report["summary"]["fixed_bootstrap_overhead"]["output_tokens"]["value"] == 7
    assert cost.main([*args[:-1], "2026-09-05"]) == 0
    capsys.readouterr()
