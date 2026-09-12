import copy
import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl.common import ROOT
from bfas.mech_bfcl.splits import families, make_splits, parent_id


def stub_inventory():
    entries, source = {}, []
    categories = ["memory_kv", "multi_turn_base", "irrelevance", "live_simple", "simple_python"]
    for cat in categories:
        for n in range(30):
            tid = f"{cat}_{n}"
            entry = dict(id=tid, function=[dict(name=f"{cat}.api_{n}")])
            if cat.startswith("memory"):
                entry.update(scenario=f"persona_{n}", involved_classes=[f"MemoryFamily_{n}"])
            entries[tid] = entry
            source.append(tid)
    for cat in categories + ["web_search"]:
        for n in range(100, 200):
            tid = f"{cat}_{n}"
            entries[tid] = dict(id=tid, function=[dict(name=f"eval_{cat}_{n}")])
            if cat.startswith("memory"):
                entries[tid].update(scenario=f"persona_{n}", involved_classes=[f"MemoryFamily_{n}"])
    return entries, dict(demand=source[:100], calibration=source[100:])


def test_split_determinism_disjoint_parents_families():
    entries, source = stub_inventory()
    original = copy.deepcopy(entries)
    a = make_splits(entries, source)
    assert a == make_splits(entries, source)
    assert entries == original
    assert [len(a[k]) for k in ("support", "calibration", "evaluation")] == [24, 16, 256]
    assert set(a["support"] + a["calibration"]) <= set(source["demand"]+source["calibration"])
    assert not set(a["evaluation"]) & set(a["source_pool_ids"])
    assert not {parent_id(entries[t]) for t in a["support"]} & {parent_id(entries[t]) for t in a["calibration"]}
    used = set().union(*(families(entries[t]) for t in a["support"]+a["calibration"]))
    assert all(not families(entries[t]) & used for t in a["evaluation"])


def test_memory_parent_cross_backend_and_namespace_family():
    assert parent_id(dict(id="memory_kv_3-student-2", scenario="student")) == parent_id(
        dict(id="memory_vector_97-student-12", scenario="student"))
    assert families(dict(function=[dict(name="Hotels_4_ReserveHotel")])) == families(
        dict(function=[dict(name="Hotels_2_ReserveHotel")]))
    assert families(dict(function=[dict(name="api.read"), dict(name="api.write")])) == {"api:api"}


def test_missing_source_rejected():
    entries, source = stub_inventory()
    del entries[source["demand"][0]]
    with pytest.raises(ValueError, match="missing"):
        make_splits(entries, source)


def test_committed_split_has_exact_counts_and_no_group_overlap():
    s = json.loads((ROOT/"configs/mech_bfcl_splits.json").read_text())
    assert [len(s[k]) for k in ("support", "calibration", "evaluation")] == [24,16,256]
    groups = {k: {s["items"][t]["parent_id"] for t in s[k]} for k in ("support","calibration","evaluation")}
    assert not groups["support"] & groups["calibration"]
    assert not groups["evaluation"] & (groups["support"] | groups["calibration"])
    used_families = {f for t in s["support"]+s["calibration"] for f in s["items"][t]["families"]}
    assert all(not used_families & set(s["items"][t]["families"]) for t in s["evaluation"])
