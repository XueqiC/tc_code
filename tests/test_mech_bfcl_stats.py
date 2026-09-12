import copy
from types import SimpleNamespace

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl.common import ROOT, digest, read_json, write_json
from bfas.mech_bfcl.pipeline import LAYER3_EXCLUDED_CATEGORIES, layer3_selection
from bfas.mech_bfcl.stats import pair, paired_interval, pool_seed_pairs, refresh_pairs, report


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


def test_report_records_training_dose_and_total_exposure(tmp_path, evaluations):
    splits, _ = evaluations
    metrics = dict(passes=20, learning_rate=1e-4, supervised_tokens_per_pass=611,
                   supervised_tokens=12220, optimizer_steps=40, wall_seconds=12.5)
    for arm in ("C", "D"):
        write_json(tmp_path / arm / "training/metrics.json", metrics)
    report(SimpleNamespace(run_dir=tmp_path), splits)
    result = read_json(tmp_path / "report.json")
    text = (tmp_path / "report.md").read_text()
    assert "| Passes | Learning rate | Total supervised tokens | Optimizer steps |" in text
    for arm in ("C", "D"):
        entry = next(r for r in result["mechanism_table"] if r["arm"] == arm)
        assert all(entry[key] == value for key, value in metrics.items())
        assert f"| {arm} | 100.00 | 100.00 | 100.00 | 0 | 20 | 0.0001 | 12220 | 40 | 12.5 |" in text


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


@pytest.fixture(params=[0, 19])
def seed_evaluations(tmp_path, request):
    split_seed = request.param
    splits = dict(seed=split_seed, evaluation=[f"task_{i}" for i in range(3)],
                  items={f"task_{i}": dict(parent_id="large" if i < 2 else "small", category="memory")
                         for i in range(3)})
    _, scope = layer3_selection(splits)
    variants = [("base", None, [False, False, False]),
                ("C", split_seed, [False, False, True]), ("D", split_seed, [True, True, False]),
                ("C", 7, [False, True, False]), ("D", 7, [True, True, True]),
                ("C", 11, [True, True, True])]
    for arm, seed, scores in variants:
        name = arm if seed in (None, split_seed) else f"{arm}-s{seed}"
        items = []
        for layer in (1, 2, 3):
            for i, score in enumerate(scores):
                item = dict(row(f"task_{i}" if layer == 3 else f"local_{layer}_{i}",
                                "large" if i < 2 else "small", score), layer=layer)
                if layer == 3:
                    item["evaluation_scope"] = scope
                items.append(item)
        write_json(tmp_path / "evaluation" / name / "main/items.json", list(reversed(items)))
        if arm != "base":
            folder = "training" if seed == split_seed else f"training_s{seed}"
            write_json(tmp_path / arm / folder / "metrics.json", dict(train_seed=seed, passes=3,
                       learning_rate=1e-3, supervised_tokens=30, optimizer_steps=9, wall_seconds=seed))
    # Decode repeats and partial evaluations must not become extra training seeds.
    write_json(tmp_path / "evaluation/C-s99/repeat/items.json", [])
    write_json(tmp_path / "evaluation/D-s99/main/protocol.json", {})
    return splits, scope


def test_report_discovers_all_seed_variants_and_pools_questions_before_parents(tmp_path, seed_evaluations):
    splits, scope = seed_evaluations
    baseline = splits["seed"]
    report(SimpleNamespace(run_dir=tmp_path), splits)
    result = read_json(tmp_path / "report.json")
    entries = {(r["arm"], r["train_seed"]): r for r in result["mechanism_table"]}
    assert set(entries) == {("base", None), ("C", baseline), ("D", baseline), ("C", 7), ("D", 7), ("C", 11)}
    assert entries[("C", 7)]["variant"] == "C-s7"
    assert entries[("C", 7)]["wall_seconds"] == 7
    assert entries[("C", baseline)]["wall_seconds"] == baseline
    assert result["pooled_training_seeds"] == sorted([baseline, 7])
    assert result["unpooled_training_seeds"] == {"C": [11], "D": []}
    assert set(result["category_breakdown"]) == {r["variant"] for r in entries.values()}
    intervals = result["intervals"]
    assert intervals == read_json(tmp_path / "paired/intervals.json")
    assert {"base_vs_C", "base_vs_D", "C_vs_D", "base_vs_C_s7", "base_vs_D_s7",
            f"C_s{baseline}_vs_C_s7", f"D_s{baseline}_vs_D_s7", "C_s7_vs_D_s7", "C_vs_D_pooled"} <= intervals.keys()
    for layer in ("1", "2", "3"):
        c_noise = intervals[f"C_s{baseline}_vs_C_s7"][layer]
        assert c_noise["delta"] == -0.25
        assert c_noise["mean_absolute_item_delta"] == pytest.approx(2/3)
        assert c_noise["outcomes"] == {"00": 1, "01": 1, "10": 1}
        assert intervals[f"D_s{baseline}_vs_D_s7"][layer]["mean_absolute_item_delta"] == pytest.approx(1/3)
        pooled = intervals["C_vs_D_pooled"][layer]
        assert pooled["items"] == 3  # Training seeds never multiply the item or parent count.
        assert pooled["independent_parents"] == 2
        assert pooled["delta"] == 0.375
        assert pooled["item_weighted_delta"] == 0.5
        assert pooled["ci95"] == [0.0, 0.75]
        assert pooled["outcomes"] == {}  # Fractional means have no binary catches/regressions.
        assert pooled["train_seeds"] == sorted([baseline, 7])
    assert intervals["C_vs_D_pooled"]["3"]["evaluation_scope"] == scope
    pooled_rows = read_json(tmp_path / "paired/C_vs_D_pooled.json")
    assert len(pooled_rows) == 9
    assert [r["left_mean_correct"] for r in pooled_rows[:3]] == [0.0, 0.5, 0.5]
    assert [r["right_mean_correct"] for r in pooled_rows[:3]] == [1.0, 1.0, 0.5]
    assert [r["delta"] for r in pooled_rows[:3]] == [1.0, 0.5, 0.0]
    text = (tmp_path / "report.md").read_text()
    assert "| C-s7 |" in text and "| D-s7 |" in text and "| C-s11 |" in text
    assert "| C_vs_D_pooled | 3 | 3 | 37.50 |" in text
    assert "conditional on these training seeds" in text


@pytest.mark.parametrize("change", ["seed_set", "items", "parent", "category", "unscored", "duplicate"])
def test_seed_pooling_rejects_mismatched_data(change):
    left = {0: [row("a", "p", True)], 7: [row("a", "p", False)]}
    right = copy.deepcopy(left)
    if change == "seed_set":
        del right[7]
    elif change == "items":
        left[7] = right[7] = [row("b", "p", False)]
    elif change == "parent":
        left[7][0]["parent_id"] = right[7][0]["parent_id"] = "q"
    elif change == "category":
        right[7][0]["category"] = "other"
    elif change == "unscored":
        right[7][0]["correct"] = 0.5
    else:
        right[7] *= 2
    with pytest.raises(ValueError):
        pool_seed_pairs(left, right)


@pytest.mark.parametrize("change", ["heldout_hash", "seed", "train_seed"])
def test_report_rejects_mixed_seed_protocols(tmp_path, seed_evaluations, change):
    splits, _ = seed_evaluations
    protocol = dict(split_hash=digest(splits), heldout_hash="frozen", temperature=0.001,
                    top_k=1, seed=splits["seed"], train_seed=splits["seed"])
    write_json(tmp_path / "evaluation/C/main/protocol.json", protocol)
    changed = dict(protocol, train_seed=7)
    changed[change] = "changed"
    write_json(tmp_path / "evaluation/C-s7/main/protocol.json", changed)
    with pytest.raises(ValueError, match="same split|training seed mismatch"):
        refresh_pairs(tmp_path, splits)
