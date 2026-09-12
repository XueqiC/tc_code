import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl.stats import pair, paired_interval


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
