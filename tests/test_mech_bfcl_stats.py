import copy
from types import SimpleNamespace

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl.common import ROOT, digest, read_json, write_json
from bfas.mech_bfcl.pipeline import LAYER3_EXCLUDED_CATEGORIES, layer3_selection
from bfas.mech_bfcl.stats import pair, paired_interval, refresh_pairs, report


def row(i, parent, correct):
    return dict(id=i, parent_id=parent, correct=correct, layer=1, category="memory")


def test_cluster_weighting_not_number_of_generated_variants():
    a = [row(str(i),"large",False) for i in range(100)] + [row("one","small",True)]
    b = [dict(r, correct=not r['correct']) for r in a]
    paired = pair(a,b)
    result = paired_interval(paired, samples=200)
    assert result['delta'] == 0
    assert result['item_weighted_delta'] > .98
    assert result['independent_parents'] == 2
    assert result['outcomes'] == {'01':100, '10':1}
    assert result == paired_interval(paired, samples=200)


def test_missing_duplicates_parent_mismatch_fail_closed():
    a = [row('a','p',True)]
    with pytest.raises(ValueError, match="Incomplete"):
        pair(a, [])
    with pytest.raises(ValueError, match="Duplicate"):
        pair(a+a,a)
    with pytest.raises(ValueError, match="parent"):
        pair(a,[row('a','q',False)])
    assert paired_interval(pair(a,a))['ci95'] is None


def test_all_four_paired_outcomes():
    a = [row(str(i), str(i), bool(i//2)) for i in range(4)]
    b = [row(str(i), str(i), bool(i%2)) for i in range(4)]
    assert paired_interval(pair(a,b), samples=100)['outcomes'] == {'00':1,'01':1,'10':1,'11':1}


def test_layer3_selection_preserves_committed_split_and_hash():
    splits = read_json(ROOT / "configs/mech_bfcl_splits.json")
    original = copy.deepcopy(splits)
    ids, scope = layer3_selection(splits)
    assert LAYER3_EXCLUDED_CATEGORIES == ("web_search",)
    assert scope["excluded_categories"] == ["web_search"]
    assert len(scope["excluded_ids"]) == 10
    assert scope["excluded_ids"] == [tid for tid in splits["evaluation"]
                                      if splits["items"][tid]["category"] == "web_search"]
    assert len(ids) == scope["evaluated_questions"] == 246
    assert scope["subset_questions"] == 256
    assert ids == [tid for tid in splits["evaluation"] if tid not in scope["excluded_ids"]]
    assert splits == original
    assert digest(splits) == "20fefdb2feb7f6479f5f88af1066b2f51d34af4d378a664c194538687e044792"


@pytest.fixture
def evaluations(tmp_path):
    splits = read_json(ROOT / "configs/mech_bfcl_splits.json")
    ids, scope = layer3_selection(splits)
    for arm in ("base", "C", "D"):
        local = [dict(row(f"local_{layer}", "calibration", True), layer=layer) for layer in (1, 2)]
        full = [dict(id=tid, **splits["items"][tid], layer=3, correct=True,
                     evaluation_scope=scope) for tid in ids]
        write_json(tmp_path / "evaluation" / arm / "main/items.json", local + full)
        if arm != "base":
            write_json(tmp_path / arm / "training/metrics.json", {})
    return splits, scope


def test_report_and_paired_statistics_record_full_task_scope(tmp_path, evaluations):
    splits, scope = evaluations
    report(SimpleNamespace(run_dir=tmp_path), splits)
    result = read_json(tmp_path / "report.json")
    assert result["layer_3_scope"] == scope
    intervals = read_json(tmp_path / "paired/intervals.json")
    assert intervals == result["intervals"]
    assert set(intervals) == {"base_vs_C", "base_vs_D", "C_vs_D"}
    for name, layers in intervals.items():
        assert layers["3"]["evaluation_scope"] == scope
        assert layers["3"]["items"] == 246
        assert layers["1"]["items"] == layers["2"]["items"] == 1
        assert not any(r["category"] == "web_search" for r in read_json(tmp_path / "paired" / (name + ".json")))
    text = (tmp_path / "report.md").read_text()
    assert "Layer 3 covers 246 of the 256 subset questions" in text
    assert "Excluded categories: web_search" in text
    assert all(tid in text for tid in scope["excluded_ids"])
    assert "| C_vs_D | 3 | 246 |" in text


@pytest.mark.parametrize("changed", ["ids", "metadata"])
def test_pairs_reject_wrong_layer3_selection_or_scope(tmp_path, evaluations, changed):
    splits, _ = evaluations
    path = tmp_path / "evaluation/base/main/items.json"
    items = read_json(path)
    if changed == "ids":
        items.pop()
    else:
        items[-1]["evaluation_scope"]["excluded_ids"] = []
    write_json(path, items)
    with pytest.raises(ValueError, match="layer-3"):
        refresh_pairs(tmp_path, splits)
