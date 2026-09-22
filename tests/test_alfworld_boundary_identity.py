"""Frozen v1 fixtures remain readable; v2 changes exactly two scoped symbols."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType

import pytest

from bfas.rtd.benchmarks import alfworld_identity as identity
from bfas.rtd.persistence import digest, file_hash
from rtd_alfworld_evaluation_fixtures import ROOT, campaign

FIXTURES = Path(__file__).parent/"fixtures/alfworld_scoring_v1"
V1_HASH = "a475f62fd4014f1ec43d1e1ea0b885283a43a2038b396ab9aec7964557f010e9"


def legacy_module(name, monkeypatch):
    module = ModuleType("bfas.rtd.benchmarks._legacy_"+name)
    module.__package__ = "bfas.rtd.benchmarks"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile((FIXTURES/(name+".py.txt")).read_text(), str(FIXTURES), "exec"), module.__dict__)
    return module


def frozen_v1_files():
    files = {}
    for name in identity.SCOPES:
        archived = FIXTURES/(Path(name).name+".txt")
        source = archived if archived.exists() else ROOT/name
        files[name] = digest(identity.scoring_projection(name, source.read_text()))
    files["src/bfas/rtd/benchmarks/alfworld_identity.py"] = file_hash(FIXTURES/"alfworld_identity.py.txt")
    return files


def test_v1_fixture_projection_remains_frozen():
    assert digest(frozen_v1_files()) == V1_HASH


def test_only_two_scoped_symbols_changed(monkeypatch):
    legacy = legacy_module("alfworld_identity", monkeypatch)
    assert identity.SCOPES == legacy.SCOPES
    name = "src/bfas/rtd/benchmarks/alfworld_evaluation.py"
    def symbols(source):
        tree = ast.parse(source)
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old = symbols((FIXTURES/"alfworld_evaluation.py.txt").read_text())
    new = symbols((ROOT/name).read_text())
    assert {symbol for symbol in identity.SCOPES[name] if "." not in symbol
            and old[symbol] != new[symbol]} == {"HFBackend", "VLLMBackend"}
    # Replacing the two classes must restore the entire old projection,
    # including imported bindings, coordinator calls, and all other symbols.
    tree = ast.parse((ROOT/name).read_text())
    old_tree = ast.parse((FIXTURES/"alfworld_evaluation.py.txt").read_text())
    replacements = {n.name: n for n in old_tree.body if isinstance(n, ast.ClassDef)}
    tree.body = [replacements[n.name] if isinstance(n, ast.ClassDef) and n.name in
                 {"HFBackend", "VLLMBackend"} else n for n in tree.body]
    assert identity.scoring_projection(name, ast.unparse(tree)) == identity.scoring_projection(
        name, (FIXTURES/"alfworld_evaluation.py.txt").read_text())


def test_v2_campaign_is_distinct_from_v1_fixture(campaign):
    current = campaign.manifest
    old = deepcopy(current)
    old["version"] = old["evaluation_harness"]["version"] = "alfworld-evaluation-scoring-c26d-v1"
    old["evaluation_harness"]["scoring_files"] = frozen_v1_files()
    old["harness_hash"] = digest(old["evaluation_harness"])
    assert current["version"] == "alfworld-evaluation-scoring-c26d-v2"
    assert identity.campaign_identity(old) != identity.campaign_identity(current)


@pytest.mark.parametrize("ids", [[20, 21, 2], [20]+[22]*255, [20]+[22]*254+[2], [2]])
def test_existing_v1_backend_contract_against_v1_fixture(campaign, monkeypatch, ids):
    import test_alf_vllm as original
    legacy = legacy_module("alfworld_evaluation", monkeypatch)
    monkeypatch.setattr(original, "evaluation", legacy)
    original.test_server_matches_hf_generation_contract(campaign, monkeypatch, ids)
