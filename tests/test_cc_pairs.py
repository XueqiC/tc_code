"""C19/C21 CPU tests: historical evidence, leakage, probing and pair evaluation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from bfas import cc_pairs as cc


def row(tid, value, seed="seed_1", *, function=None, question=None, abstain=False):
    return dict(id=tid, seed_task=seed, seed_category="simple_python", out_of_scope=abstain,
                function=function or [dict(name="f", parameters={"type": "dict"})],
                question=[[dict(role="user", content=question or f"Set x to {value} ({tid})")]],
                ground_truth=[] if abstain else [{"f": {"x": [value]}}])


def fake_render(question, functions):
    return json.dumps([question, functions]) + "<|im_start|>assistant\n"


def candidates(rows, pool=None, exclusions=None, renderer=fake_render):
    return cc.build_candidates([("gen.jsonl", rows)], ("pool.jsonl", pool or []),
                               exclusions or cc.Exclusions(), renderer=renderer)


def test_content_identity_reused_ids_and_exact_pool_join():
    a, b, c = row("gen_seed_1_0", 1), row("gen_seed_1_0", 2), row("gen_seed_1_1", 3)
    text = '<tool_call>\n{"name": "f", "arguments": {"x": 2}}\n</tool_call>'
    pool = [dict(task_id=b["id"], prompt=fake_render(b["question"], b["function"]),
                 response=text, teacher="teacher_authored_gt", _seed_category="simple_python")]
    pairs, _, audit = candidates([a, b, c, a], pool)
    assert audit["unique_states"] == 3
    assert len(pairs) == 3
    sides = {s["state_id"]: s for p in pairs for s in (p["s1"], p["s2"])}
    assert len(sides) == 3
    target = next(s for s in sides.values() if s["ground_truth"] == b["ground_truth"])
    assert target["teacher_text"] == text
    assert target["rendered_from_gt"] is False
    assert target["provenance"]["teacher_text_source"] == "pool.jsonl:1"
    others = [s for s in sides.values() if s is not target]
    assert all(s["rendered_from_gt"] for s in others)
    assert all(s["provenance"]["ledger"] == cc.LEDGER for s in sides.values())
    assert pairs == candidates([a, b, c, a], pool)[0]


def test_render_truth_first_nonempty_alternatives_and_empty_eos_target():
    truth = [{"f": {"list_arg": [[1, 2], [3]], "default": ["", 0], "city": ["東京"]}}]
    text = cc.render_truth(truth)
    assert cc.parse_calls(text) == [{"f": {"list_arg": [1, 2], "default": 0, "city": "東京"}}]
    from tools.bfcl_event_mine_single import gt_call_strings, target_block
    assert cc.legacy_render_truth(truth) == target_block(gt_call_strings(truth))
    assert cc.render_truth([]) == ""


def test_render_truth_uses_schema_without_defaults_or_new_values():
    values = dict(text=[12], count=["", "7", 8], ratio=[2], enabled=[None, "", False, True],
                  items=[[], [1]], mapping=[{}], absent=["", None])
    props = dict(text={"type": "string"}, count={"type": "integer"}, ratio={"type": "float"},
                 enabled={"type": "boolean"}, items={"type": "array", "items": {"type": "integer"}},
                 mapping={"type": "dict"}, absent={"type": "float", "default": 4.5},
                 never_supplied={"type": "integer", "default": 10})
    functions = [dict(name="f", parameters=dict(properties=props, required=[]))]
    original = json.dumps(values)
    args = cc.parse_calls(cc.render_truth([{"f": values}], functions))[0]["f"]
    assert args == dict(text="12", count=7, ratio=2.0, enabled=False, items=[], mapping={})
    assert type(args["count"]) is int and type(args["ratio"]) is float
    assert json.dumps(values) == original
    assert cc.typed_value(["1", "2"], {"type": "array", "items": {"type": "integer"}}) == [1, 2]
    assert cc.typed_value({"ratio": 2}, {"type": "dict", "properties": {"ratio": {"type": "float"}}}) == {"ratio": 2.0}


@pytest.mark.parametrize("value,schema", [(1.5, {"type": "integer"}), (True, {"type": "integer"}),
                                         ("maybe", {"type": "boolean"}), ("invented", {"type": "float"}),
                                         (float("inf"), {"type": "float"}), (3, {"type": "array"})])
def test_typed_rendering_rejects_lossy_or_invented_values(value, schema):
    with pytest.raises(ValueError):
        cc.typed_value(value, schema)


def test_typed_rendering_rejects_missing_schema_and_empty_accepted_values():
    with pytest.raises(ValueError, match="no accepted values"):
        cc.render_truth([{"f": {"x": []}}])
    with pytest.raises(ValueError, match="function absent"):
        cc.render_truth([{"unknown": {}}], [])
    with pytest.raises(ValueError, match="argument absent"):
        cc.render_truth([{"f": {"unknown": [1]}}], [{"name": "f"}])


@pytest.mark.parametrize("category,props,values", [
    ("simple_javascript", dict(t={"type": "float"}, n={"type": "integer"}, text={"type": "String"},
                               flag={"type": "Boolean"}, items={"type": "array", "items": {"type": "integer"}},
                               data={"type": "dict"}, big={"type": "Bigint"}),
     dict(t=[7.0], n=[2], text=["Tokyo"], flag=[False], items=[[1, 2]], data=[{"a": [1]}], big=[12])),
    ("simple_java", dict(t={"type": "float"}, n={"type": "long"}, text={"type": "String"},
                         flag={"type": "boolean"}, items={"type": "Array", "items": {"type": "integer"}},
                         data={"type": "HashMap"}, ratio={"type": "double"}),
     dict(t=[7.0], n=[2], text=["Tokyo"], flag=[False], items=[[1, 2]], data=[{"a": [1]}], ratio=[0.5])),
])
def test_javascript_java_schema_literals_pass_real_checker(category, props, values):
    from tools.behavior_atom.checker_bridge import CheckerBridge
    functions = [dict(name="f", parameters=dict(properties=props, required=list(props)))]
    truth = [{"f": values}]
    side = dict(function=functions, category=category, ground_truth=truth, out_of_scope=False)
    text = cc.render_truth(truth, functions, category)
    assert all(isinstance(v, str) for v in cc.parse_calls(text)[0]["f"].values())
    with CheckerBridge() as checker:
        result = cc.checked_output(text, side, checker)
    assert result["outcome"] == 1, result


def test_parallel_optional_arguments_use_each_function_schema():
    from tools.behavior_atom.checker_bridge import CheckerBridge
    functions = [dict(name="first", parameters=dict(required=["query"], properties=dict(
        query={"type": "string"}, page={"type": "integer"}, heartbeat={"type": "boolean"}))),
        dict(name="second", parameters=dict(required=["query"], properties=dict(
            query={"type": "integer"}, urgent={"type": "boolean"})))]
    truth = [{"first": dict(query=["wifi"], page=["", None], heartbeat=["", False])},
             {"second": dict(query=[3], urgent=["", None])}]
    side = dict(function=functions, category="live_parallel_multiple", ground_truth=truth, out_of_scope=False)
    text = cc.render_truth(truth, functions, side["category"])
    assert cc.parse_calls(text) == [{"first": dict(query="wifi", heartbeat=False)}, {"second": dict(query=3)}]
    with CheckerBridge() as checker:
        assert cc.checked_output(cc.legacy_render_truth(truth), side, checker)["outcome"] == 0
        assert cc.checked_output(text, side, checker)["outcome"] == 1


def test_dictionary_truth_selects_nested_accepted_values_and_preserves_arrays():
    from tools.behavior_atom.checker_bridge import CheckerBridge
    props = dict(data={"type": "dict", "properties": {"count": {"type": "integer"}}},
                 records={"type": "array", "items": {"type": "dict"}})
    functions = [dict(name="f", parameters=dict(properties=props, required=list(props)))]
    truth = [{"f": dict(data=[dict(count=["", 3, 4], missing=["", None], array=[[1, 2], [3]])],
                        records=[[dict(city=["Tokyo", "Kyoto"])]])}]
    text = cc.render_truth(truth, functions)
    assert cc.parse_calls(text) == [{"f": dict(data=dict(count=3, array=[1, 2]), records=[dict(city="Tokyo")])}]
    side = dict(function=functions, ground_truth=truth, category="simple_python", out_of_scope=False)
    with CheckerBridge() as checker:
        assert cc.checked_output(text, side, checker)["outcome"] == 1


def test_teacher_audit_preserves_passing_bytes_repairs_others_and_updates_all_copies():
    from tools.behavior_atom.checker_bridge import CheckerBridge
    functions = [dict(name="f", parameters=dict(required=["x"], properties=dict(
        x={"type": "integer"}, rating={"type": "float"})))]
    rows = [row(tid, value, function=functions) for tid, value in (("a", 1), ("b", 2), ("c", 3))]
    rows[0]["ground_truth"][0]["f"]["x"] = [1, 10]
    rows[1]["ground_truth"][0]["f"]["rating"] = ["", None]
    passing = '  <tool_call>{"name":"f","arguments":{"x":10}}</tool_call>\n'
    broken = cc.legacy_render_truth(rows[1]["ground_truth"])
    malformed = '<tool_call>{broken}</tool_call>'
    pool = [dict(task_id=r["id"], prompt=fake_render(r["question"], r["function"]),
                 response=text, teacher="teacher_authored_gt")
            for r, text in zip(rows, (passing, broken, malformed))]
    pairs = json.loads(json.dumps(candidates(rows, pool)[0]))  # independent repeated side copies
    truths = {s["state_id"]: json.dumps(s["ground_truth"]) for p in pairs for s in (p["s1"], p["s2"])}
    with CheckerBridge() as checker:
        audit = cc.validate_teachers(pairs, checker)
        assert cc.validate_teachers(pairs, checker) == audit  # preserve original before counts on rerun
    assert audit["valid_states_before"] == 1 and audit["valid_states"] == 3
    assert audit["retyped_states"] == 2
    for p in pairs:
        for s in (p["s1"], p["s2"]):
            assert json.dumps(s["ground_truth"]) == truths[s["state_id"]]
            assert s["y_plus"] == s["teacher_text"]
            if s["task_id"] == "a":
                assert s["teacher_text"] == passing and "y_plus_original" not in s
                assert s["provenance"]["kind"] == "teacher_authored_gt"
            else:
                assert s["y_plus_original"] == {"b": broken, "c": malformed}[s["task_id"]]
                assert s["provenance"]["kind"] == "teacher_authored_gt/rendered_typed"
                assert s["rendered_from_gt"] is True


def test_javascript_any_nonstring_gt_remains_invalid_without_fabricating_truth():
    from tools.behavior_atom.checker_bridge import CheckerBridge
    functions = [dict(name="f", parameters=dict(required=["x"], properties=dict(x={"type": "any"})))]
    rows = [row("a", ["taskA", "taskB"], function=functions), row("b", 42, function=functions)]
    for r in rows:
        r["seed_category"] = "simple_javascript"
    pairs = candidates(rows)[0]
    with CheckerBridge() as checker:
        audit = cc.validate_teachers(pairs, checker)
    assert audit["valid_states"] == 0 and audit["retyped_states"] == 2
    assert audit["errors_before"] == {"type_error:js": 2}
    assert audit["errors_after"] == {"value_error:others": 2}
    assert {json.dumps(pairs[0][s]["ground_truth"]) for s in ("s1", "s2")} == {json.dumps(r["ground_truth"]) for r in rows}


def test_pair_rules_cap_and_no_fabricated_type2():
    rows = [row(f"gen_seed_1_{i}", i) for i in range(4)]
    rows += [row(f"oos_seed_1_{i}", i, abstain=True) for i in range(8)]
    pool = [dict(task_id="genmt_memory_kv_1_0", _seed_category="memory_kv", prompt="historic state",
                 response="historic action", teacher="teacher_authored_gt", turn_index=2)]
    pairs, todo, audit = candidates(rows, pool)
    assert audit["before_type3_cap"]["by_type"]["binding"] == 6
    assert audit["before_type3_cap"]["by_type"]["call_vs_abstain"] == 32
    assert sum(p["type"] == "call_vs_abstain" for p in pairs) == 3
    assert len(todo) == 1 and todo[0]["missing_side"] == "s2"
    assert "s2" not in todo[0] and todo[0]["state"] == pool[0]
    assert candidates([row("a", 1), row("b", 2, abstain=True)])[0] == []  # boundary-only capped to zero


def test_same_query_same_action_different_function_and_allowed_alternatives_are_not_pairs():
    a, b = row("a", 1), row("b", 1)
    assert candidates([a, b])[0] == []
    b["ground_truth"] = [{"f": {"x": [1, 2]}}]
    assert candidates([a, b])[0] == []
    b["ground_truth"] = [{"g": {"x": [2]}}]
    assert candidates([a, b])[0] == []
    b = row("b", 2, question=a["question"][0][0]["content"])
    assert candidates([a, b])[0] == []
    assert candidates([a, row("b", 2, seed="other", function=[{"name": "g"}])])[0] == []
    assert len(candidates([a, row("b", 2, seed="other")])[0]) == 1  # identical schema


def test_exclusions_reused_ids_hashes_parents_support_and_probe_metadata(tmp_path):
    a, b, c = row("gen_seed_1_0", 1), row("gen_seed_1_0", 2), row("gen_seed_2_0", 3, seed="seed_2")
    path = tmp_path / "probes.json"
    cc.write_json(path, dict(probes=[dict(task_id=a["id"], parent_task_id="seed_1",
                                         prompt_sha1=hashlib.sha1(fake_render(a["question"], a["function"]).encode()).hexdigest())], spare_unused=[]))
    ex = cc.Exclusions()
    ex.load(path, probe=True)
    pairs, _, audit = candidates([a, b, row("sibling", 8), c], exclusions=ex)
    assert {r["task_id"] for r in audit["excluded"]} == {a["id"]}
    assert len(audit["excluded"]) == 1
    assert len(pairs) == 3  # reused-ID variant and parent's other siblings survive
    assert [r["task_id"] for r in audit["recovered_rows"]] == [a["id"]]
    hashed = tmp_path / "calibration_ids.json"
    prompt = fake_render(c["question"], c["function"])
    cc.write_json(hashed, dict(hashes=[hashlib.sha1(prompt.encode()).hexdigest()[:16]]))
    ex.load(hashed)
    assert ex.reasons("different_id", "other", prompt) == [str(hashed)]
    support = tmp_path / "support_split.json"
    cc.write_json(support, dict(support=["seed_1"], demand=["seed_2"], held_out={"ids": ["seed_3"]},
                               calibration=["seed_4"], certification=["seed_5"]))
    ex.load(support, heldout_only=True)
    assert not ex.reasons("sibling", "seed_1", "new prompt")
    assert not ex.reasons("sibling", "seed_2", "new prompt")
    for parent in ("seed_3", "seed_4", "seed_5"):
        assert ex.reasons("gen_" + parent + "_0", parent, "new prompt")


def test_probe_spares_are_unprotected_and_parent_flag_is_opt_in(tmp_path):
    path = tmp_path / "probes.json"
    spare_prompt, protected_prompt = "unused spare state", "active protected state"
    cc.write_json(path, dict(probes=[dict(task_id="gen_seed_1_0", parent_task_id="seed_1",
                                         gen_ids=["gen_seed_1_1"],
                                         prompt_sha256=hashlib.sha256(protected_prompt.encode()).hexdigest())],
                              spare_unused=[dict(task_id="gen_spare_0", parent_task_id="spare",
                                                 prompt_hash=hashlib.sha1(spare_prompt.encode()).hexdigest()[:16])]))
    for enabled in (False, True):
        ex = cc.Exclusions(exclude_probe_parents=enabled)
        ex.load(path, probe=True)
        ex.index_prompts([("gen_seed_1_0", protected_prompt), ("gen_seed_1_0", "another reused-id prompt"),
                          ("gen_seed_1_1", "unique generated prompt")])
        for tid in ("gen_seed_1_0", "gen_seed_1_1#turn2"):
            assert bool(ex.reasons(tid, "seed_1", "another reused-id prompt")) is (enabled or tid.endswith("#turn2"))
        assert ex.reasons("renamed", "unrelated", protected_prompt)
        assert not ex.reasons("gen_spare_0", "spare", spare_prompt)
        assert bool(ex.reasons("gen_seed_1_9", "seed_1", "sibling state")) is enabled
        matches = ex.matches("gen_seed_1_9", "seed_1", "sibling state")
        assert all(m["mechanism"] == "parent" and m["probe_parent"] for m in matches)


def test_calibration_children_protected_without_banning_siblings(tmp_path):
    path = tmp_path / "calibration_ids.json"
    cc.write_json(path, dict(ids=["gen_seed_1_0", "official_2"]))
    ex = cc.Exclusions()
    ex.load(path)
    ex.index_prompts([("gen_seed_1_0", "new")])
    assert ex.reasons("gen_seed_1_0#turn1", "seed_1", "new")
    assert ex.reasons("gen_gen_seed_1_0_3", "gen_seed_1_0", "new")
    assert not ex.reasons("gen_seed_1_9", "seed_1", "new")
    assert ex.reasons("official_2", "official_2", "new")
    assert ex.reasons("oos_official_2_3", "official_2", "new")


def test_leakage_audit_mechanisms_partition_source_rows_and_compare_parent_policy(tmp_path):
    path = tmp_path / "probes.json"
    rows = [row("gen_seed_1_0", 1), row("gen_seed_1_0", 2), row("gen_seed_1_2", 3),
            row("renamed", 4, seed="seed_2"), row("free", 5, seed="seed_2")]
    cc.write_json(path, dict(probes=[dict(task_id=rows[0]["id"], parent_task_id="seed_1",
                                         prompt_hash=hashlib.sha1(fake_render(rows[3]["question"], rows[3]["function"]).encode()).hexdigest()[:16])]))
    for enabled in (False, True):
        ex = cc.Exclusions(exclude_probe_parents=enabled)
        ex.load(path, probe=True)
        audit = candidates(rows, exclusions=ex)[2]
        off, on = (audit["leakage_comparison"][k] for k in ("probe_parents_off", "probe_parents_on"))
        assert off["total"] == 1 and on["total"] == 4
        assert len(audit["excluded"]) == (4 if enabled else 1)
        assert "seed_1" not in off["by_seed_function"]
        assert off["by_seed_function"]["seed_2"]["by_reason"] == dict(unique_id=0, official_id=0, content_hash=0, prompt_hash=1, parent=0)
        assert on["by_seed_function"]["seed_1"]["by_reason"] == dict(unique_id=0, official_id=0, content_hash=0, prompt_hash=0, parent=3)
        assert len(audit["recovered_rows"]) == (0 if enabled else 2)
        assert audit["previous_rule"]["total"] == (4 if enabled else 3)
        assert all(r["matches"] and r["reasons"] == [str(path)] for r in audit["excluded"])


@pytest.mark.parametrize("field", ["prompt_sha1", "prompt_hash", "prompt_sha256", "content_hash"])
def test_hash_exclusion_recovers_reused_ids_without_leaking_renamed_states(tmp_path, field):
    protected = row("gen_seed_1_0", 1, question="Set x to 1 in 東京")
    variant = row(protected["id"], 2)
    renamed = dict(protected, id="renamed")
    prompt = fake_render(protected["question"], protected["function"])
    if field == "content_hash":
        value = cc.question_content_hash(protected["question"], protected["function"], protected["ground_truth"])[:16]
    elif field == "prompt_sha256":
        value = hashlib.sha256(prompt.encode()).hexdigest()
    else:
        value = hashlib.sha1(prompt.encode()).hexdigest()
        if field == "prompt_hash":
            value = value[:16]
    path = tmp_path / "probes.json"
    cc.write_json(path, {"probes": [dict(task_id=protected["id"], **{field: value})]})
    ex = cc.Exclusions()
    ex.load(path, probe=True)
    pairs, _, audit = candidates([protected, variant, renamed, row("free", 3)], exclusions=ex)
    assert {r["source"] for r in audit["excluded"]} == {"gen.jsonl:1", "gen.jsonl:3"}
    assert len(pairs) == 1
    assert audit["recovery_by_seed_function"]["seed_1"]["recovered"] == 1
    assert audit["recovery_by_seed_function"]["seed_1"]["newly_excluded"] == (0 if field != "content_hash" else 1)
    # Content protection survives a changed renderer and also checks GT.
    if field == "content_hash":
        assert len(candidates([protected, variant, renamed, row("free", 3)], exclusions=ex,
                              renderer=lambda q, f: "new renderer\n" + fake_render(q, f))[2]["excluded"]) == 2
        assert not ex.matches("unrelated", "other", prompt, content_hashes=[
            cc.question_content_hash(protected["question"], protected["function"], variant["ground_truth"])])


def test_unique_id_index_spans_pools_turn_suffixes_duplicates_and_skipped_rows(tmp_path):
    a, b, duplicate = row("gen_seed_1_0", 1), row("gen_seed_1_1", 2), row("gen_seed_1_2", 3)
    pool = [dict(task_id=a["id"] + "#turn1", prompt="another prompt", response="", teacher="other")]
    unverified = dict(b, question=row("unused", 9)["question"], verified=False)
    path = tmp_path / "probes.json"
    cc.write_json(path, {"probes": [dict(task_id=r["id"]) for r in (a, b, duplicate)]})
    ex = cc.Exclusions()
    ex.load(path, probe=True)
    pairs, _, audit = candidates([a, b, duplicate, duplicate, unverified], pool=pool, exclusions=ex)
    assert len(pairs) == 1
    assert audit["source_id_prompt_counts"] == {a["id"]: 2, b["id"]: 2, duplicate["id"]: 1}
    assert [r["source"] for r in audit["excluded"]] == ["gen.jsonl:3", "gen.jsonl:4"]
    assert audit["leakage_comparison"]["probe_parents_off"]["by_seed_function"]["seed_1"]["by_reason"]["unique_id"] == 2


def test_ambiguous_calibration_generator_ids_do_not_ban_variants_or_unrelated_children(tmp_path):
    path = tmp_path / "calibration_ids.json"
    cc.write_json(path, {"ids": ["gen_seed_1_0", "official_2"]})
    ex = cc.Exclusions()
    ex.load(path)
    ex.index_prompts([("gen_seed_1_0", "a"), ("gen_seed_1_0", "b"),
                      ("official_2", "c"), ("official_2", "d")])
    assert not ex.reasons("gen_seed_1_0", "seed_1", "a")
    assert not ex.reasons("gen_gen_seed_1_0_3", "gen_seed_1_0", "unrelated variant")
    assert ex.reasons("official_2", "official_2", "c")
    assert ex.matches("oos_official_2_0", "official_2", "derived child")[0]["mechanism"] == "parent"


def test_content_protection_covers_pool_rows_with_authored_gt_alternatives(tmp_path):
    a = row("gen_seed_1_0", 1, function=[dict(name="f", parameters=dict(type="dict", properties={}))])
    a["ground_truth"] = [{"f": {"x": [1, 2]}}]
    prompt = cc.render_prompt(a["question"], a["function"])
    pool = [dict(task_id=a["id"], prompt=prompt, response=cc.render_truth(a["ground_truth"]), teacher="teacher_authored_gt")]
    path = tmp_path / "probes.json"
    cc.write_json(path, {"probes": [dict(content_hash=cc.question_content_hash(a["question"], a["function"], a["ground_truth"])[:16])]})
    ex = cc.Exclusions()
    ex.load(path, probe=True)
    audit = candidates([a, row(a["id"], 3)], pool=pool, exclusions=ex, renderer=cc.render_prompt)[2]
    assert {r["source"] for r in audit["excluded"]} == {"gen.jsonl:1", "pool.jsonl:1"}
    assert all(r["matches"][0]["mechanism"] == "content_hash" for r in audit["excluded"])


def test_probe_prompt_assertion_checks_unpaired_states_and_hash_integrity(tmp_path):
    a = row("a", 1)
    prompt = fake_render(a["question"], a["function"])
    path = tmp_path / "probes.json"
    sha1 = hashlib.sha1(prompt.encode()).hexdigest()
    cc.write_json(path, {"probes": [dict(prompt=prompt, prompt_sha1=sha1, prompt_hash=sha1[:16])]})
    ex = cc.Exclusions()
    ex.load(path, probe=True)
    audit = candidates([a], exclusions=ex)[2]
    assert audit["probe_prompt_check"]["passed"]
    assert audit["probe_prompt_check"]["by_source"][str(path)]["present_in_source"] == 1
    skipped = candidates([dict(a, verified=False)], exclusions=ex)[2]
    assert skipped["excluded"] == [] and len(skipped["skipped"]) == 1
    assert skipped["probe_prompt_check"]["passed"]
    assert skipped["probe_prompt_check"]["by_source"][str(path)]["present_in_source"] == 1
    with pytest.raises(AssertionError, match="probe prompt exclusion failed"):
        ex.check_probe_prompts([dict(prompt=prompt)], [], [])
    cc.write_json(path, {"probes": [dict(prompt=prompt, prompt_sha1="0" * 40)]})
    ex = cc.Exclusions()
    ex.load(path, probe=True)
    with pytest.raises(AssertionError, match="probe prompt exclusion failed"):
        candidates([a], exclusions=ex)


def test_real_renderer_roundtrip_and_pool_only_historical_state():
    from bfas.adapters.bfcl import BFCLAdapter
    a, b = row("gen_seed_1_0", 1), row("gen_seed_1_1", 2)
    expected = BFCLAdapter()._render(b["question"][0], b["function"])
    assert cc.render_prompt(b["question"], b["function"]) == expected
    assert cc.unpack_pool_prompt(expected, cc.render_prompt) == (b["question"], b["function"])
    pool = [dict(task_id=b["id"], prompt=expected, response=cc.render_truth(b["ground_truth"]),
                 teacher="teacher_authored_gt", _seed_category="simple_python")]
    pairs, _, audit = candidates([a], pool, renderer=cc.render_prompt)
    assert len(pairs) == 1
    recovered = next(pairs[0][s] for s in ("s1", "s2") if pairs[0][s]["task_id"] == b["id"])
    assert recovered["ground_truth"] == b["ground_truth"]
    assert recovered["provenance"]["truth_source"] == "pool_teacher_response_exact_binding"
    assert not audit["skipped"]
    pool[0]["prompt"] = expected.replace("Set x", "<|im_start|>tool\nSet x")
    assert candidates([a], pool, renderer=cc.render_prompt)[0] == []


def test_render_falls_back_to_isolated_bfcl_environment(monkeypatch):
    from bfas.adapters.bfcl import BFCLAdapter
    entry = row("a", 1)
    expected = cc.render_prompt(entry["question"], entry["function"])
    monkeypatch.setattr(BFCLAdapter, "_render", lambda *args: (_ for _ in ()).throw(ImportError("isolated dependency")))
    monkeypatch.setenv("BFCL_VENV_PYTHON", sys.executable)
    assert cc.render_prompt(entry["question"], entry["function"]) == expected


@pytest.mark.parametrize("output", ["<tool_call>{bad}</tool_call>", "<tool_call>", "</tool_call>",
                                    '<tool_call>{"name":"f","arguments":[]}</tool_call>'])
def test_malformed_call_cannot_pass_abstention(output):
    checker = SimpleNamespace(checker_version="fake", check=lambda *a: pytest.fail("broken text reached checker"))
    result = cc.checked_output(output, {"out_of_scope": True}, checker)
    assert result["outcome"] == 0 and result["calls"] is None


def test_binding_distance_assignment_numeric_and_abstain():
    assert cc.binding_distance([{"f": {"x": 8}}], [{"f": {"x": [9]}}]) < cc.binding_distance(
        [{"f": {"x": 8}}], [{"f": {"x": [1]}}])
    assert cc.binding_distance([{"f": {"x": 2}}, {"f": {"x": 1}}],
                               [{"f": {"x": [1]}}, {"f": {"x": [2]}}]) == 0
    assert cc.binding_distance([], []) == 0
    assert cc.binding_distance([], [{"f": {}}]) == 1
    assert cc.binding_distance([{"f": {"x": True}}], [{"f": {"x": [1]}}]) > 0


class FakeChecker:
    checker_version = "test_checker"

    def check(self, probe, calls):
        valid = cc.binding_distance(calls, probe["truth"]) == 0
        return dict(valid=valid, checker_version=self.checker_version)


class FakeClient:
    def __init__(self, text):
        self.text, self.requests = text, []

    def generate(self, prompt, **kwargs):
        self.requests.append((prompt, kwargs))
        return dict(output=self.text, finish_reason="stop", **kwargs)


def probe_fixture():
    pairs = candidates([row("a", 1), row("b", 9), row("c", 6)])[0]
    client = FakeClient(cc.render_truth([{"f": {"x": [9]}}]))
    generations = cc.generate_states(pairs, client, FakeChecker(), seed=123)
    scores = {}
    for p in pairs:
        for label in ("s1", "s2"):
            for target in ("s1", "s2"):
                scores[cc.digest([p[label]["prompt"], p[target]["teacher_text"]])] = dict(logp=-1.0 if label == target else -2.0)
    return pairs, client, generations, scores


def test_generate_deduplicates_states_uses_greedy_four_samples_and_fixed_seeds():
    pairs, client, generations, scores = probe_fixture()
    assert len(client.requests) == 15  # 3 states, not 6 duplicated sides
    assert Counter(kwargs["temperature"] for _, kwargs in client.requests) == {0.0: 3, 0.7: 12}
    assert len({kwargs["seed"] for _, kwargs in client.requests}) == 15
    assert client.requests == probe_fixture()[1].requests
    results = cc.assemble_results(pairs, generations, scores)
    assert all(r["status"] == "complete" for r in results)
    assert any(r[s]["condition_insensitive"] for r in results for s in ("s1", "s2"))
    assert all(r[s]["kappa"] == -1 for r in results for s in ("s1", "s2"))
    bundle, _ = cc.select_pairs(pairs, results)
    assert bundle["eligible"]["total"] > 0


def test_checker_errors_propagate_and_do_not_become_student_failures():
    checker = FakeChecker()
    checker.check = lambda *args: (_ for _ in ()).throw(RuntimeError("checker offline"))
    pair = candidates([row("a", 1), row("b", 2)])[0][0]
    with pytest.raises(RuntimeError, match="offline"):
        cc.checked_output(pair["s1"]["teacher_text"], pair["s1"], checker)


def test_selection_same_side_positive_kappa_incomplete_and_stale_rejected():
    pairs, _, generations, scores = probe_fixture()
    pair = pairs[0]
    result = cc.assemble_results([pair], generations, scores)[0]
    for s in ("s1", "s2"):
        result[s].update(wrong=False, condition_insensitive=False, kappa=-1)
    result["s1"]["wrong"] = True
    result["s2"]["kappa"] = 2
    assert cc.select_pairs([pair], [result])[0]["selected"]["total"] == 0
    result["s1"]["kappa"] = 1
    assert cc.select_pairs([pair], [result])[0]["selected"]["total"] == 1
    result["s1"]["teacher_check"]["outcome"] = 0
    assert cc.select_pairs([pair], [result])[0]["invalid_teacher_pairs"] == 1
    result["status"] = "generated"
    with pytest.raises(ValueError, match="incomplete"):
        cc.select_pairs([pair], [result])
    with pytest.raises(ValueError, match="exactly once"):
        cc.select_pairs([pair], [])


def abstain_probe_fixture(seed="seed_1"):
    pairs = candidates([row(f"{seed}_{i}", i, seed=seed) for i in range(3)]
                       + [row(f"{seed}_oos", 0, seed=seed, abstain=True)])[0]
    pair = next(p for p in pairs if p["type"] == "call_vs_abstain")
    cc.validate_teachers([pair], FakeChecker())
    label = next(s for s in ("s1", "s2") if pair[s]["out_of_scope"])
    # Calls are wrong on both sides; the abstain side has verified alternatives.
    client = FakeClient(cc.render_truth([{"f": {"x": [99]}}]))
    generations = cc.generate_states([pair], client, FakeChecker())
    record = generations[pair[label]["state_id"]]
    for index, text in ((0, "  The available tools cannot do that.\n"),
                        (1, "That request is outside the available tools.")):
        record["outputs"][index].update(output=text, **cc.checked_output(text, pair[label], FakeChecker()))
    record["greedy_outcome"] = 1
    record["sample_pass_rate"] = .25
    scores = {key: dict(logp=-1) for key in cc.teacher_score_jobs([pair])}
    return pair, cc.assemble_results([pair], generations, scores)[0], label


@pytest.mark.parametrize("greedy", ["verified", "wrong", "empty", "whitespace", "unverified"])
def test_empty_teacher_uses_verified_greedy_then_sample_and_wrong_call(greedy):
    pair, result, label = abstain_probe_fixture()
    record = result[label]
    outputs = record["outputs"]
    if greedy == "wrong":
        outputs[0].update(output=outputs[2]["output"], **cc.checked_output(
            outputs[2]["output"], pair[label], FakeChecker()))
    elif greedy in ("empty", "whitespace"):
        outputs[0]["output"] = "" if greedy == "empty" else " \n"
    elif greedy == "unverified":
        outputs[0].pop("verdict")
    expected = outputs[0 if greedy == "verified" else 1]
    expected_wrong = outputs[0 if greedy == "wrong" else 2]
    # A sample stored before greedy must not change the preference.
    record["outputs"] = [outputs[1], outputs[0], *outputs[2:]]
    before = json.dumps([pair, result])
    bundle, split = cc.select_pairs([pair], [result], type3_cap=1, within_fraction=0)
    side = bundle["pairs"][0][label]
    assert side["teacher_text"] == "" and side["ground_truth"] == []
    assert side["y_plus"] == expected["output"]
    assert side["y_plus_from_student"] is True
    assert side["y_plus_provenance"] == "student_verified_" + expected["mode"]
    assert side["y_minus"] == expected_wrong["output"]
    assert side["y_minus_provenance"] == "student_wrong"
    assert side["provenance"] == pair[label]["provenance"]  # historical teacher evidence
    assert bundle["student_target_fallback"] == dict(
        filled_sides=1, filled_states=1, by_provenance={side["y_plus_provenance"]: 1},
        student_wrong_sides=1, excluded_pairs=[])
    assert json.dumps([pair, result]) == before
    assert bundle["pairs"][0]["pair_id"] == pair["pair_id"]
    assert split == cc.split_pairs([pair], within_fraction=0)


@pytest.mark.parametrize("problem", ["failed", "unverified", "empty", "temperature", "missing"])
def test_empty_teacher_without_verified_nonempty_target_is_reported_and_excluded(problem):
    pair, result, label = abstain_probe_fixture()
    for output in result[label]["outputs"]:
        if output["outcome"] != 1:
            continue
        if problem == "failed":
            output.update(outcome=0, verdict=dict(valid=False))
        elif problem == "unverified":
            output.pop("verdict")
        elif problem == "empty":
            output["output"] = " \n"
        elif problem == "temperature":
            output["temperature"] = 1.0
    if problem == "missing":
        result[label]["outputs"] = []
    patched = cc.training_targets(pair, result)
    assert patched[label]["y_plus"] == ""
    assert patched[label]["y_plus_from_student"] is False
    assert "y_plus_provenance" not in patched[label]
    bundle, _ = cc.select_pairs([pair], [result], type3_cap=1)
    assert bundle["selected"]["total"] == bundle["eligible"]["total"] == 0
    assert bundle["student_target_fallback"]["excluded_pairs"] == [dict(
        pair_id=pair["pair_id"], sides=[label], reason="empty_teacher_text_no_verified_student_output")]
    assert pair[label]["y_plus"] == "" and pair[label]["y_plus_from_student"] is False


def test_empty_teacher_requires_matching_state_and_does_not_invent_negative():
    pair, result, label = abstain_probe_fixture()
    result[label]["outputs"] = result[label]["outputs"][:2]
    side = cc.training_targets(pair, result)[label]
    assert side["y_plus_from_student"] and "y_minus" not in side and "y_minus_provenance" not in side
    result[label]["state_id"] = "another_state"
    with pytest.raises(ValueError, match="base state ID mismatch"):
        cc.select_pairs([pair], [result], type3_cap=1)


def test_empty_teacher_cli_selection_loads_train_split_and_keeps_eval_prompt_checks(tmp_path):
    from bfas.pair_unit import load_pairs
    source, out = tmp_path / "probe", tmp_path / "selected"
    write_probe_fixture(source)
    items = [abstain_probe_fixture(seed) for seed in ("train", "held")]
    pairs, results = [i[0] for i in items], [i[1] for i in items]
    cc.write_jsonl(source / "candidates.jsonl", pairs)
    cc.write_jsonl(source / "probe_results.jsonl", results)
    cc.write_json(source / "probe_metadata.json", dict(status="complete", candidates_sha256=cc.digest(pairs)))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    args = ["select", "--out", str(out), "--probe-from", str(source), "--type3-cap", "1.0",
            "--target", "64", "--within-fraction", "0", "--heldout-seed", "held"]
    assert cc.main(args) == 0
    path, split_path = out / "confused_pairs.json", out / "split.json"
    bundle = json.loads(path.read_text())
    split = json.loads(split_path.read_text())
    assert split == cc.split_pairs(pairs, heldout_seed="held", within_fraction=0)
    loaded = load_pairs(path, split_path)
    assert [p["pair_id"] for p in loaded] == split["train"]
    assert all(s["y_plus"] for p in loaded for s in p["sides"])
    assert bundle["student_target_fallback"]["filled_sides"] == 2
    assert "Filled training sides: **2**" in (out / "report.md").read_text()
    assert cc.load_eval_inputs(path, split_path, source / "probe_results.jsonl")[0] == bundle["pairs"]
    generated = {p.name: p.read_bytes() for p in out.iterdir()}
    assert cc.main(args) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == generated
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    for field in ("prompt", "y_plus", "y_minus", "y_plus_provenance"):
        changed = json.loads(path.read_text())
        side = next(p[s] for p in changed["pairs"] for s in ("s1", "s2") if p[s].get("y_plus_from_student"))
        side[field] += "changed"
        cc.write_json(path, changed)
        if field == "prompt":
            with pytest.raises(ValueError, match="stale"):
                cc.load_eval_inputs(path, split_path, source / "probe_results.jsonl")
        else:
            assert cc.load_eval_inputs(path, split_path, source / "probe_results.jsonl")[0] == changed["pairs"]
        path.write_bytes(generated[path.name])


@pytest.mark.parametrize("identity", ["state_id", "prompt_sha256"])
@pytest.mark.parametrize("problem", [None, "prompt_s1", "prompt_s2", "missing_s1", "missing_s2",
                                     "missing_outputs", "missing_outcome"])
def test_evaluate_filled_side_matches_probe_without_source_candidates(tmp_path, identity, problem):
    pair, result, filled_label = abstain_probe_fixture()
    if identity == "state_id":
        for label in ("s1", "s2"):
            result[label].pop("prompt_sha256")  # Historical C19 probe format.
    bundle, split = cc.select_pairs([pair], [result], type3_cap=1, within_fraction=0)
    filled = bundle["pairs"][0]
    assert filled[filled_label]["y_plus"] and pair[filled_label]["y_plus"] == ""
    assert filled[filled_label]["teacher_text"] == pair[filled_label]["teacher_text"] == ""
    assert cc.digest(filled) != result["candidate_sha256"] == cc.digest(pair)
    for label in ("s1", "s2"):
        assert filled[label]["state_id"] == result[label]["state_id"]
        assert filled[label]["prompt"] == pair[label]["prompt"]
    if problem and problem.startswith("prompt_"):
        label = problem.removeprefix("prompt_")
        filled[label]["prompt"] += "changed"
        # Even a refreshed candidate prompt hash cannot hide a stale base state.
        filled[label]["prompt_sha256"] = hashlib.sha256(filled[label]["prompt"].encode()).hexdigest()
    elif problem in ("missing_s1", "missing_s2"):
        result.pop(problem.removeprefix("missing_"))
    elif problem == "missing_outputs":
        result[filled_label].pop("outputs")
    elif problem == "missing_outcome":
        result[filled_label]["outputs"][0].pop("outcome")
    # A moved selection must not depend on its original probe_from directory.
    bundle["probe_from"] = str(tmp_path / "unavailable_source")
    paths = [tmp_path / name for name in ("selected/pairs.json", "selected/split.json", "probe/base.jsonl")]
    cc.write_json(paths[0], bundle)
    cc.write_json(paths[1], split)
    cc.write_jsonl(paths[2], [result])
    if problem:
        with pytest.raises(ValueError, match="stale|outcome|checked greedy"):
            cc.load_eval_inputs(*paths)
    else:
        loaded, partitions, base = cc.load_eval_inputs(*paths)
        assert loaded == [filled] and partitions[filled["pair_id"]] in cc.EVAL_PARTITIONS
        assert base[filled["pair_id"]] == result


def split_fixture():
    pairs = []
    for group in ("alpha", "beta", "gamma"):
        for i in range(6):
            pairs.append(dict(pair_id=f"{group}_{i}", seed_function=group, type="binding",
                              s1=dict(prompt_sha256=f"{group}_{2*i}"), s2=dict(prompt_sha256=f"{group}_{2*i+1}")))
    # Deliberate shared states to ensure cross edges are never trained.
    pairs.append(dict(pair_id="beta_cross", seed_function="beta", type="binding",
                      s1=pairs[6]["s1"], s2=pairs[7]["s2"]))
    return pairs


@pytest.mark.parametrize("within_fraction", [0.01, 0.2, 0.5, 0.99])
def test_split_whole_seed_and_within_seed_prompt_isolation(within_fraction):
    pairs = split_fixture()
    split = cc.split_pairs(pairs, heldout_seed="alpha", within_fraction=within_fraction)
    assert split == cc.split_pairs(pairs, heldout_seed="alpha", within_fraction=within_fraction)
    by_id = {p["pair_id"]: p for p in pairs}
    assert all(by_id[k]["seed_function"] == "alpha" for k in split["heldout_seed"])
    assert all(by_id[k]["seed_function"] != "alpha" for k in split["train"] + split["heldout_within_seed"])
    held_states = {by_id[k][s]["prompt_sha256"] for k in split["heldout_within_seed"] for s in ("s1", "s2")}
    train_states = {by_id[k][s]["prompt_sha256"] for k in split["train"] for s in ("s1", "s2")}
    assert not held_states & train_states
    assigned = [k for field in ("train", "heldout_seed", "heldout_within_seed", "excluded_cross_partition") for k in split[field]]
    assert sorted(assigned) == sorted(by_id)
    assert split["heldout_within_seed"] and split["train"]


@pytest.mark.parametrize("heldout_seed", [None, "alpha", "beta"])
def test_zero_within_fraction_trains_every_pair_outside_whole_seed(heldout_seed):
    pairs = split_fixture()
    for p in pairs:
        if p["seed_function"] == "beta":
            p["s1"] = dict(prompt_sha256="beta_shared")
    split = cc.split_pairs(pairs, heldout_seed=heldout_seed, within_fraction=0)
    assert split == cc.split_pairs(list(reversed(pairs)), heldout_seed=heldout_seed, within_fraction=0)
    heldout = split["heldout_seed_function"]
    assert sorted(split["heldout_seed"]) == sorted(p["pair_id"] for p in pairs if p["seed_function"] == heldout)
    assert sorted(split["train"]) == sorted(p["pair_id"] for p in pairs if p["seed_function"] != heldout)
    assert split["heldout_within_seed"] == split["excluded_cross_partition"] == []
    assert sum(split["counts"].values()) == len(pairs)
    if heldout_seed == "alpha":
        positive = cc.split_pairs(pairs, heldout_seed=heldout_seed)
        assert positive["excluded_cross_partition"]
        assert set(positive["excluded_cross_partition"]) <= set(split["train"])


def test_zero_within_fraction_empty_and_single_pair():
    assert cc.split_pairs([], within_fraction=0)["counts"] == dict(
        train=0, heldout_seed=0, heldout_within_seed=0, excluded_cross_partition=0)
    split = cc.split_pairs(split_fixture()[:1], within_fraction=0)
    assert split["counts"] == dict(train=0, heldout_seed=1, heldout_within_seed=0, excluded_cross_partition=0)
    with pytest.raises(ValueError, match="requested heldout seed has no selected pairs"):
        cc.split_pairs(split_fixture(), heldout_seed="missing", within_fraction=0)


@pytest.mark.parametrize("within_fraction", [-0.1, 1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_within_fraction(within_fraction):
    with pytest.raises(ValueError, match="within_fraction must be at least zero and less than one"):
        cc.split_pairs(split_fixture(), within_fraction=within_fraction)


def test_final_type3_cap_is_reapplied_after_selection():
    rows = split_fixture()
    rows += [dict(pair_id=f"boundary_{i}", seed_function="delta", type="call_vs_abstain") for i in range(100)]
    for limit in (1, 2, 3, 5, 32):
        selected = cc.cap_pairs(rows, 0, limit)
        assert len(selected) <= limit
        assert sum(p["type"] == "call_vs_abstain" for p in selected) * 3 <= len(selected)


def selection_fixture():
    pairs = split_fixture()
    pairs += [dict(pair_id=f"boundary_{i}", seed_function="delta", type="call_vs_abstain",
                   s1=dict(prompt_sha256=f"boundary_{2*i}"), s2=dict(prompt_sha256=f"boundary_{2*i+1}"))
              for i in range(100)]
    for p in pairs:
        for s in ("s1", "s2"):
            p[s].update(teacher_text="historical target", y_plus="historical target")
    results = [dict(pair_id=p["pair_id"], candidate_sha256=cc.digest(p), status="complete",
                    **{s: dict(teacher_check=dict(outcome=1), wrong=True, condition_insensitive=False, kappa=1)
                       for s in ("s1", "s2")}) for p in pairs]
    return pairs, results


@pytest.mark.parametrize("cap", [0, .1, .3333, .4, .5, .9, 1.0])
@pytest.mark.parametrize("limit", [1, 3, 32, 64, None])
def test_configurable_cap_and_target(cap, limit):
    pairs, _ = selection_fixture()
    selected = cc.cap_pairs(pairs, 0, limit, type3_cap=cap)
    assert selected == cc.cap_pairs(list(reversed(pairs)), 0, limit, type3_cap=cap)
    assert len(selected) <= (limit or len(pairs))
    assert len({p["pair_id"] for p in selected}) == len(selected)
    n3 = sum(p["type"] == "call_vs_abstain" for p in selected)
    assert n3 <= len(selected) * (1 / 3 if cap == .3333 else cap)
    if cap == 1:
        assert len(selected) == min(limit or len(pairs), len(pairs))
    if cap == .3333:
        binding = cc.balanced([p for p in pairs if p["type"] == "binding"], 0)
        boundary = cc.balanced([p for p in pairs if p["type"] == "call_vs_abstain"], 0)
        old_n3 = min(len(boundary), len(binding) // 2, limit // 3 if limit else len(boundary))
        old_selected = binding[:limit - old_n3 if limit else None] + boundary[:old_n3]
        assert selected == sorted(old_selected, key=lambda p: p["pair_id"])


def test_uncapped_boundary_only_and_multiturn_are_retained():
    pairs, _ = selection_fixture()
    boundary = [p for p in pairs if p["type"] == "call_vs_abstain"]
    assert cc.cap_pairs(boundary, 0, 64) == []
    assert len(cc.cap_pairs(boundary, 0, 64, type3_cap=1)) == 64
    multi_turn = dict(pairs[0], type="multi_turn")
    selected = cc.cap_pairs([multi_turn, *boundary], 0, 64, type3_cap=.5)
    assert cc.counts(selected)["by_type"] == dict(binding=0, multi_turn=1, call_vs_abstain=1)


def test_reporting_minimum_does_not_change_selection_or_split():
    pairs, results = selection_fixture()
    before = json.dumps(pairs)
    bundle, split = cc.select_pairs(pairs, results, limit=64, type3_cap=1)
    reported, reported_split = cc.select_pairs(pairs, results, limit=64, type3_cap=1, min_per_type=100)
    assert bundle["pairs"] == reported["pairs"]
    assert split == reported_split
    assert json.dumps(pairs) == before
    assert bundle["target_pairs"] == 64 and bundle["type3_cap"] == 1
    assert bundle["selected"]["total"] == 64
    assert bundle["selected"]["by_type"]["call_vs_abstain"] > 64 / 3
    assert reported["min_per_type"] == 100
    assert reported["shortfall_by_type"] == {t: 100 - bundle["selected"]["by_type"][t] for t in cc.TYPES}
    assigned = [pid for k in ("train", "heldout_seed", "heldout_within_seed", "excluded_cross_partition")
                for pid in split[k]]
    assert sorted(assigned) == sorted(p["pair_id"] for p in bundle["pairs"])
    assert split == cc.split_pairs(bundle["pairs"])


def write_probe_fixture(out):
    pairs, _, generations, scores = probe_fixture()
    audit = candidates([row("a", 1), row("b", 9), row("c", 6)])[2]
    cc.write_jsonl(out / "candidates.jsonl", pairs)
    cc.write_json(out / "candidate_audit.json", audit)
    cc.write_jsonl(out / "probe_results.jsonl", cc.assemble_results(pairs, generations, scores))
    cc.write_json(out / "probe_metadata.json", dict(status="complete", candidates_sha256=cc.digest(pairs)))


def test_default_selection_cli_aliases_remain_byte_identical(tmp_path):
    write_probe_fixture(tmp_path)
    assert cc.main(["select", "--out", str(tmp_path)]) == 0
    expected = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                for name in ("confused_pairs.json", "split.json", "report.md")}
    for options in ([], ["--type3-cap", "0.3333", "--target", "32"], ["--limit", "32"]):
        assert cc.main(["select", "--out", str(tmp_path), *options]) == 0
        assert {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in expected} == expected


@pytest.mark.parametrize("explicit_source", [False, True])
def test_select_reads_fallback_inputs_without_changing_source(tmp_path, monkeypatch, explicit_source):
    monkeypatch.setattr(cc, "ROOT", tmp_path)
    source = tmp_path / ("custom_probe" if explicit_source else "data/cc_pairs_v1")
    write_probe_fixture(source)
    # Existing selection/report bytes in the source must also remain untouched.
    assert cc.main(["select", "--out", str(source)]) == 0
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()}
    out = tmp_path / "selection"
    args = ["select", "--out", str(out), "--seed", "0", "--type3-cap", "1.0", "--target", "64"]
    if explicit_source:
        args += ["--probe-from", str(source)]
    assert cc.main(args) == 0
    bundle = json.loads((out / "confused_pairs.json").read_text())
    split_bytes = (out / "split.json").read_bytes()
    assert bundle["type3_cap"] == 1.0 and bundle["target_pairs"] == 64
    assert bundle["probe_from"] == str(source)
    report = (out / "report.md").read_text()
    assert "--type3-cap 1.0` (disabled)" in report and "--target 64" in report
    assert str(source) in report
    assert cc.main([*args, "--min-per-type", "4"]) == 0
    reported = json.loads((out / "confused_pairs.json").read_text())
    assert reported["min_per_type"] == 4
    assert reported["pairs"] == bundle["pairs"]
    assert (out / "split.json").read_bytes() == split_bytes
    assert "Reporting only: `--min-per-type 4`" in (out / "report.md").read_text()
    assert {p.name for p in out.iterdir()} == {"confused_pairs.json", "split.json", "report.md"}
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()} == before


def test_select_zero_within_fraction_reports_train_composition(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    write_probe_fixture(source)
    pairs, results = selection_fixture()
    cc.write_jsonl(source / "candidates.jsonl", pairs)
    cc.write_jsonl(source / "probe_results.jsonl", results)
    cc.write_json(source / "probe_metadata.json", dict(status="complete", candidates_sha256=cc.digest(pairs)))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    assert cc.main(["select", "--out", str(out), "--probe-from", str(source), "--seed", "0",
                    "--type3-cap", "1.0", "--target", "64", "--within-fraction", "0",
                    "--heldout-seed", "alpha"]) == 0
    bundle = json.loads((out / "confused_pairs.json").read_text())
    split = json.loads((out / "split.json").read_text())
    assert bundle["pairs"] == cc.select_pairs(pairs, results, limit=64, type3_cap=1)[0]["pairs"]
    assert split["counts"] == dict(train=58, heldout_seed=6, heldout_within_seed=0, excluded_cross_partition=0)
    assert split == cc.split_pairs(bundle["pairs"], heldout_seed="alpha", within_fraction=0)
    report = (out / "report.md").read_text()
    assert "train **58** / heldout_seed **6** (`alpha`)" in report
    assert "--within-fraction 0" in report
    assert "| beta | 7 | 0 | 0 | 7 |" in report
    assert "| gamma | 6 | 0 | 0 | 6 |" in report
    assert "| delta | 0 | 0 | 45 | 45 |" in report
    assert "| Total | 13 | 0 | 45 | 58 |" in report
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_local_probe_takes_precedence_and_stale_fallback_is_rejected(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    write_probe_fixture(source)
    metadata = json.loads((source / "probe_metadata.json").read_text())
    cc.write_json(source / "probe_metadata.json", dict(metadata, status="generated"))
    with pytest.raises(ValueError, match="complete probe"):
        cc.main(["select", "--out", str(out), "--probe-from", str(source)])
    assert not (out / "confused_pairs.json").exists()
    cc.write_json(source / "probe_metadata.json", metadata)
    results = cc.read_jsonl(source / "probe_results.jsonl")
    results[0]["candidate_sha256"] = "stale"
    cc.write_jsonl(source / "probe_results.jsonl", results)
    with pytest.raises(ValueError, match="stale or incomplete"):
        cc.main(["select", "--out", str(out), "--probe-from", str(source)])
    write_probe_fixture(out)
    assert cc.main(["select", "--out", str(out), "--probe-from", str(source)]) == 0
    assert "probe_from" not in json.loads((out / "confused_pairs.json").read_text())
    # A local incomplete probe must fail, without substituting fallback results.
    write_probe_fixture(source)
    cc.write_json(out / "probe_metadata.json", dict(metadata, status="generated"))
    with pytest.raises(ValueError, match="complete probe"):
        cc.main(["select", "--out", str(out), "--probe-from", str(source)])


@pytest.mark.parametrize("options", [
    ["--type3-cap", "-0.1"], ["--type3-cap", "1.1"], ["--type3-cap", "nan"],
    ["--type3-cap", "inf"], ["--target", "0"], ["--target", "-1"], ["--target", "1.5"],
    ["--min-per-type", "-1"], ["--min-per-type", "1.5"],
    ["--within-fraction", "-0.1"], ["--within-fraction", "1"], ["--within-fraction", "1.1"],
    ["--within-fraction", "nan"], ["--within-fraction", "inf"],
])
def test_invalid_selection_cli_options_fail_before_writing(tmp_path, options):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        cc.main(["select", "--out", str(out), *options])
    assert exc.value.code == 2
    assert not out.exists()


def test_native_likelihood_is_margin_full_sequence_with_eos_no_truncation():
    torch = pytest.importorskip("torch")
    from tools.behavior_atom.gpu_driver import Student
    import torch.nn.functional as F

    class Tokenizer:
        eos_token_id, pad_token_id = 0, 0

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) % 3 + 1 for c in text]}

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0., 0., 1., -1.], dtype=torch.bfloat16))

        def forward(self, input_ids, attention_mask=None):
            position = torch.arange(input_ids.shape[1], dtype=self.weight.dtype)
            logits = self.weight[None, None, :] + position[None, :, None] * torch.tensor([0., 0., .01, -.01])
            return SimpleNamespace(logits=logits.to(self.weight.dtype).expand(len(input_ids), -1, -1))

    pair = candidates([row("a", 1), row("b", 2)])[0][0]
    pair["s1"]["teacher_text"] = ""  # empty target must score EOS, not NA
    pair["s2"]["teacher_text"] = "x" * 600  # exceeds the trainer's response cap
    student = Student(Tiny(), Tokenizer(), "unused", None, None)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        scores = cc.score_teacher_texts([pair], student, device="cpu", context_limit=2048)
    finally:
        torch.set_num_threads(old_threads)
    for side in ("s1", "s2"):
        prompt = student.tokenizer(cc.thinking_off(pair[side]["prompt"]))["input_ids"]
        for target in ("s1", "s2"):
            response = student.tokenizer(pair[target]["teacher_text"])["input_ids"] + [0]
            logits = student.model(torch.tensor([prompt + response])).logits[0, len(prompt)-1:-1]
            expected = -F.cross_entropy(logits, torch.tensor(response), reduction="none").float().sum()
            actual = scores[cc.digest([pair[side]["prompt"], pair[target]["teacher_text"]])]
            assert actual["logp"] == expected.item()
            assert actual["counts"]["plus"]["effective_loss_tokens"] == len(response)
            assert actual["counts"]["plus"]["response_tokens_dropped"] == 0
    with pytest.raises(ValueError, match="refusing truncation"):
        cc.score_teacher_texts([pair], student, device="cpu", context_limit=10)


def test_vllm_request_preserves_bfcl_prompt_and_sampling_contract(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"choices":[{"text":"hi","finish_reason":"stop"}]}'

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(cc.urllib.request, "urlopen", urlopen)
    client = cc.VLLMClient("http://localhost:1/v1")
    client.generate("<|im_start|>assistant\n", seed=5, temperature=.7)
    body = requests[0]
    assert body["model"] == cc.BASE_MODEL
    assert body["prompt"] == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert body["temperature"] == .7 and body["seed"] == 5 and body["n"] == 1


def test_base_server_validation_rejects_adapter_or_mislabelled_checkpoint(monkeypatch):
    from io import StringIO
    model = dict(id=cc.BASE_MODEL, root=cc.BASE_MODEL, parent=None)
    monkeypatch.setattr(cc.urllib.request, "urlopen", lambda *a, **kw: StringIO(json.dumps(dict(data=[model]))))
    client = cc.VLLMClient("http://localhost:1/v1")
    assert client.validate_base() == model
    model["root"] = "results/adapted_model"
    with pytest.raises(ValueError, match="unadapted"):
        client.validate_base()
    model.update(root=cc.BASE_MODEL, parent="a_lora")
    with pytest.raises(ValueError, match="unadapted"):
        client.validate_base()


def test_cpu_candidate_artifacts_and_generate_select_integration(tmp_path, monkeypatch):
    # All synthetic outputs stay in pytest's temporary root, never real data.
    function = [dict(name="f", description="Set x.", parameters=dict(type="dict", required=["x"],
                properties=dict(x=dict(type="integer", description="Value"))))]
    rows = [row(f"gen_seed_1_{i}", i, function=function) for i in range(5)]
    cc.write_jsonl(tmp_path / "data/bfcl_sft/gen_pool_v3.jsonl", rows)
    cc.write_jsonl(tmp_path / "data/bfcl_sft/gen_oos.jsonl", [])
    cc.write_jsonl(tmp_path / "data/bfcl_sft/pool_events_pref_v3t.jsonl", [])
    cc.write_json(tmp_path / "data/bfcl_sft/calibration_ids.json", {"ids": []})
    for name in ("probes.json", "probes_disc.json"):
        cc.write_json(tmp_path / "data/behavior_atom_v1" / name,
                      {"probes": [dict(task_id="gen_seed_1_99", parent_task_id="seed_1")]})
    cc.write_json(tmp_path / "configs/support_split.json", {"calibration": []})
    cc.write_json(tmp_path / cc.LEDGER, {})
    out = tmp_path / "out"
    assert cc.main(["candidates", "--root", str(tmp_path), "--out", str(out)]) == 0
    required = {"candidates.jsonl", "candidates_type2_todo.jsonl", "probe_results.jsonl",
                "confused_pairs.json", "split.json", "report.md"}
    assert required <= {p.name for p in out.iterdir()}
    assert (out / "probe_results.jsonl").read_text() == ""
    assert json.loads((out / "confused_pairs.json").read_text())["status"] == "pending_probe"
    validation = json.loads((out / "teacher_validation.json").read_text())
    assert validation["invalid_states"] == 0
    assert validation["valid_states_before"] == validation["valid_states"] == 5
    assert validation["retyped_states"] == 0
    assert "5/5 pass before → 5/5 after" in (out / "report.md").read_text()
    audit = json.loads((out / "candidate_audit.json").read_text())
    assert audit["exclude_probe_parents"] is False
    assert audit["leakage_comparison"]["probe_parents_off"]["total"] == 0
    assert audit["leakage_comparison"]["probe_parents_on"]["total"] == 5
    fake = FakeClient(cc.render_truth([{"f": {"x": [3]}}]))
    fake.validate_base = lambda: dict(id=cc.BASE_MODEL, root=cc.BASE_MODEL)
    monkeypatch.setattr(cc, "VLLMClient", lambda *args: fake)
    assert cc.main(["probe", "--phase", "generate", "--out", str(out)]) == 0
    with pytest.raises(ValueError, match="complete probe"):
        cc.main(["select", "--out", str(out)])
    pairs = cc.read_jsonl(out / "candidates.jsonl")
    generations = json.loads((out / "probe_generations.json").read_text())["states"]
    scores = {cc.digest([p[s]["prompt"], p[t]["teacher_text"]]): {"logp": -1.0}
              for p in pairs for s in ("s1", "s2") for t in ("s1", "s2")}
    cc.write_jsonl(out / "probe_results.jsonl", cc.assemble_results(pairs, generations, scores))
    meta = json.loads((out / "probe_metadata.json").read_text())
    cc.write_json(out / "probe_metadata.json", dict(meta, status="complete"))
    assert cc.main(["select", "--out", str(out)]) == 0
    assert json.loads((out / "confused_pairs.json").read_text())["status"] == "complete"
    assert "shortfall" in (out / "report.md").read_text()
    previous_results = (out / "probe_results.jsonl").read_bytes()
    previous_mtime = (out / "probe_results.jsonl").stat().st_mtime_ns
    assert cc.main(["candidates", "--root", str(tmp_path), "--out", str(out)]) == 0
    assert (out / "probe_results.jsonl").read_bytes() == previous_results
    assert (out / "probe_results.jsonl").stat().st_mtime_ns == previous_mtime
    assert json.loads((out / "probe_metadata.json").read_text())["status"] == "pending_probe"
    assert "probe must be re-run for the new candidates" in (out / "report.md").read_text()
    with pytest.raises(ValueError, match="complete probe"):
        cc.main(["select", "--out", str(out)])
    assert json.loads((out / "split.json").read_text())["status"] == "pending_probe"
    assert cc.main(["candidates", "--root", str(tmp_path), "--out", str(out), "--exclude-probe-parents"]) == 0
    assert cc.read_jsonl(out / "candidates.jsonl") == []
    audit = json.loads((out / "candidate_audit.json").read_text())
    assert audit["exclude_probe_parents"] is True and len(audit["excluded"]) == 5


def test_slurm_resources_and_cpu_cli_help():
    script = ROOT / "scripts/cc_probe_hpg.slurm"
    text = script.read_text()
    assert "#SBATCH --account=fsu-compsci-dept" in text
    assert "#SBATCH --gres=gpu:b200:1" in text
    assert "#SBATCH --time=01:00:00" in text
    assert 'export HOME=' not in text
    assert text.index("--phase generate") < text.index("# Native CE") < text.index("--phase score")
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run([sys.executable, str(ROOT / "tools/cc_pairs.py"), "--help"],
                            check=True, capture_output=True, text=True)
    assert "candidates,probe,select" in result.stdout


class StateClient:
    def __init__(self, outputs):
        self.outputs, self.requests = outputs, []

    def generate(self, prompt, **kwargs):
        self.requests.append((prompt, kwargs))
        return dict(output=self.outputs[prompt], finish_reason="stop", **kwargs)


@pytest.fixture
def eval_data(tmp_path):
    sides = {}
    for sid, value in zip("abcde", (1, 2, 3, 4, 5)):
        r = row(sid, value, abstain=sid == "e")
        sides[sid] = dict(state_id=sid, task_id=sid, prompt=fake_render(r["question"], r["function"]),
                          teacher_text=cc.render_truth(r["ground_truth"]), function=r["function"],
                          ground_truth=r["ground_truth"], out_of_scope=r["out_of_scope"], category="simple_python")
    pairs = [dict(pair_id=name, type=kind, seed_function=seed, s1=sides[a], s2=sides[b])
             for name, a, b, kind, seed in [("ab", "a", "b", "binding", "train_seed"),
                                           ("ac", "a", "c", "binding", "train_seed"),
                                           ("de", "d", "e", "call_vs_abstain", "held_seed")]]
    before = {s["prompt"]: s["teacher_text"] for s in sides.values()}
    before[sides["a"]["prompt"]] = "No call"
    after = {s["prompt"]: s["teacher_text"] for s in sides.values()}
    after[sides["b"]["prompt"]] = "<tool_call>{broken}"
    after[sides["e"]["prompt"]] = sides["d"]["teacher_text"]
    base_scores, scores = {}, {}
    for p in pairs:
        for label in ("s1", "s2"):
            for target in ("s1", "s2"):
                key = cc.digest([p[label]["prompt"], p[target]["teacher_text"]])
                base_scores[key] = dict(logp=-2.0 if label == target else -1.0)
                scores[key] = dict(logp=-2.0 if label == target or p["pair_id"] == "de" else -3.0)
    generations = cc.generate_states(pairs, StateClient(before), FakeChecker())
    base = cc.assemble_results(pairs, generations, base_scores)
    split = dict(train=["ab", "ac"], heldout_seed=["de"], heldout_within_seed=[], excluded_cross_partition=[])
    cc.write_json(tmp_path / "pairs.json", dict(status="complete", pairs=pairs))
    cc.write_json(tmp_path / "split.json", split)
    cc.write_jsonl(tmp_path / "base.jsonl", base)
    return SimpleNamespace(pairs=pairs, split=split, base=base, scores=scores, before=before, after=after,
                           paths=[tmp_path / name for name in ("pairs.json", "split.json", "base.jsonl")])


def test_evaluate_greedy_repairs_damage_kappa_and_unique_state_call_rates(eval_data):
    pairs, partitions, base = cc.load_eval_inputs(*eval_data.paths)
    client = StateClient(eval_data.after)
    generations = cc.generate_states(pairs, client, FakeChecker(), samples=0)
    assert len(client.requests) == 5  # Six side occurrences, five distinct states.
    assert all(kwargs["temperature"] == 0 for _, kwargs in client.requests)
    assert all(len(g["outputs"]) == 1 and g["sample_pass_rate"] is None for g in generations.values())
    result = cc.assemble_eval_results(pairs, partitions, base, generations, eval_data.scores)
    ab, ac, de = result
    assert ab["s1"]["repaired"] and ab["s2"]["damaged"] and not ab["both_sides_correct"]
    assert ac["both_sides_correct"] and not ac["base_both_sides_correct"]
    assert ab["s2"]["emitted_call"]  # Broken call counts as an attempt and fails AST.
    assert de["s2"]["kind"] == "should_abstain" and de["s2"]["damaged"]
    assert de["s1"]["kappa"] == 0 and de["s2"]["kappa"] == 0 and de["kappa_flipped"]
    summary = cc.summarize_eval(result)
    train, held = summary["partitions"]["train"], summary["partitions"]["heldout_seed"]
    assert train["pairs"] == 2 and train["both_sides_correct_rate"] == .5
    assert train["sides"]["repaired"] == 2 and train["unique_states"]["repaired"] == 1
    assert train["sides"]["damaged"] == 1 and held["sides"]["damaged"] == 1
    assert train["by_side"]["s1"]["correct"] == 2 and train["by_side"]["s2"]["correct"] == 1
    assert summary["overall"]["call_rates"]["overall"] == dict(
        states=5, calls=5, base_calls=3, call_rate=1.0, base_call_rate=.6, delta=.4)
    assert summary["overall"]["call_rates"]["should_call"]["base_call_rate"] == .75
    assert held["call_rates"]["should_abstain"]["delta"] == 1
    assert train["kappa_flip_rate"] == 1 and held["kappa_flip_rate"] == 1
    assert held["by_type"]["binding"]["both_sides_correct_rate"] is None
    assert summary["partitions"]["heldout_within_seed"]["kappa_flip_rate"] is None


def test_evaluate_kappa_flip_requires_both_now_nonpositive_and_reports_denominators(eval_data):
    pairs, partitions, base = cc.load_eval_inputs(*eval_data.paths)
    generations = cc.generate_states(pairs, StateClient(eval_data.after), FakeChecker(), samples=0)
    # Base sample-level wrong flags are deliberately different from greedy.
    for record in base.values():
        record["s1"]["wrong"] = False
        record["s2"]["wrong"] = True
    for label in ("s1", "s2"):
        base["ac"][label]["kappa"] = 0
    pair = pairs[0]
    eval_data.scores[cc.digest([pair["s2"]["prompt"], pair["s1"]["teacher_text"]])]["logp"] = -1
    rows = cc.assemble_eval_results(pairs, partitions, base, generations, eval_data.scores)
    assert rows[0]["any_side_kappa_flipped"] and not rows[0]["kappa_flipped"]
    assert rows[0]["s1"]["repaired"] and rows[1]["s2"]["repaired"] is False
    assert not rows[1]["base_kappa_positive"] and not rows[1]["kappa_flipped"]
    m = cc.eval_metrics(rows)
    assert m["base_kappa_positive_pairs"] == 2 and m["kappa_flipped_pairs"] == 1
    assert m["kappa_flip_rate"] == .5 and m["kappa_flip_fraction_all_pairs"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("problem", ["stale", "missing", "duplicate_base", "nan", "overlap", "unknown",
                                     "unassigned", "alias_conflict", "state_mismatch", "duplicate_pair"])
def test_evaluate_rejects_bad_inputs_before_model_loading(eval_data, problem):
    data = eval_data
    if problem == "stale":
        data.pairs[0]["s1"]["prompt"] += "changed"
    elif problem == "missing":
        data.base.pop()
    elif problem == "duplicate_base":
        data.base.append(data.base[0])
    elif problem == "nan":
        data.base[0]["s1"]["kappa"] = None
    elif problem == "overlap":
        data.split["heldout_seed"].append("ab")
    elif problem == "unknown":
        data.split["train"].append("unknown")
    elif problem == "unassigned":
        data.split["train"].pop()
    elif problem == "alias_conflict":
        data.split["train_pair_ids"] = ["ab"]
    elif problem == "state_mismatch":
        data.base[0]["s1"]["state_id"] = "other"
    elif problem == "duplicate_pair":
        data.pairs.append(data.pairs[0])
    cc.write_json(data.paths[0], data.pairs)
    cc.write_json(data.paths[1], data.split)
    cc.write_jsonl(data.paths[2], data.base)
    with pytest.raises(ValueError):
        cc.load_eval_inputs(*data.paths)


def test_evaluate_within_seed_partition_and_excluded_edges(eval_data):
    eval_data.split.update(train=[], heldout_within_seed=["ac"], excluded_cross_partition=["ab"])
    cc.write_json(eval_data.paths[1], eval_data.split)
    pairs, partitions, _ = cc.load_eval_inputs(*eval_data.paths)
    assert {p["pair_id"] for p in pairs} == {"ac", "de"}
    assert partitions["ac"] == "heldout_within_seed" and partitions["ab"] == "excluded_cross_partition"


def test_evaluate_missing_config_reports_tried_directory(eval_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = Path("test_tag/adapter")
    model.mkdir(parents=True)
    (model / "model.safetensors").touch()
    monkeypatch.setattr(cc, "load_eval_student", lambda *args: pytest.fail("loaded an unmerged adapter"))
    with pytest.raises(ValueError, match="containing config.json") as error:
        cc.main(["evaluate", "--model", str(model),
                 *[arg for flag, path in zip(("--pairs", "--split", "--base-results"), eval_data.paths)
                   for arg in (flag, str(path))]])
    assert f"Tried directories: {model.resolve()}" in str(error.value)
    assert "tools/bfcl_hub_merge_export.py" in str(error.value)


@pytest.mark.parametrize("filled_side", [False, True])
def test_evaluate_cli_local_writes_complete_outputs(eval_data, tmp_path, monkeypatch, filled_side):
    from tools.behavior_atom import checker_bridge
    from contextlib import nullcontext
    state_matches = []
    if filled_side:
        original, previous = eval_data.pairs[-1], eval_data.base[-1]
        side = original["s2"]
        assert side["out_of_scope"] and side["teacher_text"] == ""
        output = previous["s2"]["outputs"][0]
        output.update(output="No call", **cc.checked_output("No call", side, FakeChecker()))
        filled = cc.training_targets(original, previous)
        assert filled["s2"]["y_plus"] == "No call"
        assert cc.digest(filled) != previous["candidate_sha256"] == cc.digest(original)
        eval_data.pairs[-1] = filled
        cc.write_json(eval_data.paths[0], dict(status="complete", pairs=eval_data.pairs))
        cc.write_jsonl(eval_data.paths[2], eval_data.base)
        state_matches = [dict(pair_id=filled["pair_id"], base_candidate_sha256=previous["candidate_sha256"],
                              candidate_sha256=cc.digest(filled))]
    monkeypatch.setattr(checker_bridge, "CheckerBridge", lambda: nullcontext(FakeChecker()))
    model = tmp_path / "test_tag" / "hub_merged"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    student = object()
    monkeypatch.setattr(cc, "load_eval_student", lambda path, seed: student)
    client = StateClient(eval_data.after)
    monkeypatch.setattr(cc, "LocalEvalClient", lambda obj, **kwargs: client if obj is student else pytest.fail("wrong model"))
    monkeypatch.setattr(cc, "score_teacher_texts", lambda pairs, obj, **kwargs: eval_data.scores)
    out = tmp_path / "output"
    assert cc.main(["evaluate", "--model", str(model), "--out", str(out),
                    *[arg for flag, path in zip(("--pairs", "--split", "--base-results"), eval_data.paths)
                      for arg in (flag, str(path))]]) == 0
    assert {p.name for p in out.iterdir()} == {"pair_eval.jsonl", "summary.json", "report.md"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["tag"] == "test_tag"
    assert summary["model"] == str(model) and "native CE" in summary["likelihood_backend"]
    assert summary["base_probe_state_identity_matches"] == state_matches
    assert len(cc.read_jsonl(out / "pair_eval.jsonl")) == 3
    report = (out / "report.md").read_text()
    assert all(s in report for s in ("## train", "## heldout_seed", "binding", "call_vs_abstain", "should_abstain", "N/A"))
    # An interrupted rerun invalidates previous successful outputs.
    def fail(*args, **kwargs):
        raise RuntimeError("scorer unavailable")
    monkeypatch.setattr(cc, "score_teacher_texts", fail)
    with pytest.raises(RuntimeError, match="scorer unavailable"):
        cc.main(["evaluate", "--model", str(model), "--out", str(out),
                 *[arg for flag, path in zip(("--pairs", "--split", "--base-results"), eval_data.paths)
                   for arg in (flag, str(path))]])
    interrupted = json.loads((out / "summary.json").read_text())
    assert interrupted["status"] == "running"
    assert interrupted["base_probe_state_identity_matches"] == state_matches
    assert cc.read_jsonl(out / "pair_eval.jsonl") == []
    assert "incomplete" in (out / "report.md").read_text()


def test_local_eval_generation_preserves_tokens_and_disables_thinking():
    torch = pytest.importorskip("torch")
    class Tokenizer:
        eos_token_id = 0
        def __call__(self, text, **kwargs):
            assert text == cc.thinking_off("prompt") and kwargs == dict(add_special_tokens=False)
            return dict(input_ids=[1, 2, 3])
        def decode(self, ids, **kwargs):
            assert ids == [7, 0] and kwargs == dict(skip_special_tokens=False)
            return "decoded response"
    def generate(ids, mask, settings):
        assert ids.tolist() == [[1, 2, 3]] and mask.tolist() == [[1, 1, 1]]
        assert settings == dict(max_new_tokens=10)
        return torch.tensor([[1, 2, 3, 7, 0]])
    student = SimpleNamespace(tokenizer=Tokenizer(), model=SimpleNamespace(eval=lambda: None), generate=generate)
    client = cc.LocalEvalClient(student, max_tokens=10, context_limit=20, device="cpu")
    result = client.generate("prompt", seed=3, temperature=0)
    assert result["output"] == "decoded response" and result["finish_reason"] == "stop"
    client.context_limit = 12
    with pytest.raises(ValueError, match="refusing truncation"):
        client.generate("prompt", seed=3, temperature=0)


@pytest.mark.parametrize("broken", [None, "missing", "short", "nonfinite"])
def test_served_teacher_forcing_scores_only_response_and_eos(eval_data, monkeypatch, broken):
    from io import StringIO
    requests = []
    class Tokenizer:
        eos_token_id = 99
        def __call__(self, text, **kwargs):
            assert kwargs == dict(add_special_tokens=False)
            if text.endswith("</think>\n\n"):
                return dict(input_ids=[1, 2, 3])
            return dict(input_ids=[] if text == "" else [7, 8])
    def urlopen(request, timeout):
        body = json.loads(request.data)
        requests.append(body)
        assert body["model"] == "arm_b" and body["prompt"][:3] == [1, 2, 3] and body["prompt"][-1] == 99
        assert body["echo"] and body["max_tokens"] == 0 and body["logprobs"] == 1
        assert body["add_special_tokens"] is False
        values = [None, -1000, -1000] + [-.5] * (len(body["prompt"]) - 3)
        if broken == "short":
            values.pop()
        if broken == "nonfinite":
            values[-1] = None
        return StringIO(json.dumps(dict(choices=[dict(logprobs=None if broken == "missing" else dict(token_logprobs=values))])))
    monkeypatch.setattr(cc.urllib.request, "urlopen", urlopen)
    client = cc.VLLMClient("http://localhost:1/v1", model="arm_b")
    if broken:
        with pytest.raises(ValueError, match="logprobs"):
            cc.score_served_teacher_texts(eval_data.pairs, client, Tokenizer())
        return
    scores = cc.score_served_teacher_texts(eval_data.pairs, client, Tokenizer())
    assert len(requests) == len(cc.teacher_score_jobs(eval_data.pairs))
    for score in scores.values():
        assert score["logp"] == -.5 * score["counts"]["plus"]["effective_loss_tokens"]
    assert any(s["logp"] == -.5 for s in scores.values())  # Empty abstention target still scores EOS.
    with pytest.raises(ValueError, match="refusing truncation"):
        cc.score_served_teacher_texts(eval_data.pairs, client, Tokenizer(), context_limit=4)


def test_eval_server_selection_allows_adapted_model_but_keeps_probe_guard(monkeypatch):
    from io import StringIO
    models = [dict(id="adapted", root="results/arm_b/hub_merged", parent=None)]
    monkeypatch.setattr(cc.urllib.request, "urlopen", lambda *args, **kwargs: StringIO(json.dumps(dict(data=models))))
    client = cc.VLLMClient("http://localhost:1/v1")
    assert client.select_model() == models[0] and client.model == "adapted"
    with pytest.raises(ValueError, match="unadapted"):
        client.validate_base()
    models.append(dict(id="another", root="elsewhere"))
    with pytest.raises(ValueError, match="one served model"):
        client.select_model()
    assert client.select_model("adapted") == models[0]


def test_evaluate_cli_served_model_never_loads_local_weights(eval_data, tmp_path, monkeypatch):
    from contextlib import nullcontext
    from tools.behavior_atom import checker_bridge
    from transformers import AutoTokenizer
    monkeypatch.setattr(checker_bridge, "CheckerBridge", lambda: nullcontext(FakeChecker()))
    client = StateClient(eval_data.after)
    client.model = "served_b"
    client.select_model = lambda name: dict(id="served_b", root="/remote/hub_merged")
    monkeypatch.setattr(cc, "VLLMClient", lambda *args: client)
    monkeypatch.setattr(cc, "load_eval_student", lambda *args: pytest.fail("served evaluation loaded local weights"))
    tokenizer = object()
    def load_tokenizer(path, **kwargs):
        assert path == "local-tokenizer" and kwargs == dict(trust_remote_code=False)
        return tokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    def score(pairs, remote, tok, **kwargs):
        assert remote is client and tok is tokenizer
        return eval_data.scores
    monkeypatch.setattr(cc, "score_served_teacher_texts", score)
    out = tmp_path / "remote_output"
    assert cc.main(["evaluate", "--base-url", "http://localhost:1/v1", "--tag", "remote_b",
                    "--tokenizer", "local-tokenizer", "--out", str(out),
                    *[arg for flag, path in zip(("--pairs", "--split", "--base-results"), eval_data.paths)
                      for arg in (flag, str(path))]]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["model"] == "served_b" and summary["thinking"] is False
    assert summary["base_url"] == "http://localhost:1/v1" and "echo" in summary["likelihood_backend"]


@pytest.mark.parametrize("mode", ["slurm-canonical", "slurm-serve", "local"])
def test_eval_launcher_sequential_tags_paths_and_submit_root(tmp_path, mode):
    import os
    script_source = ROOT / "scripts/cc_eval_hpg.slurm"
    text = script_source.read_text()
    assert "#SBATCH --account=fsu-compsci-dept" in text and "#SBATCH --gres=gpu:b200:1" in text
    assert "#SBATCH --time=01:30:00" in text and "--kill-after=15s 30m" in text
    subprocess.run(["bash", "-n", str(script_source)], check=True)
    root = tmp_path / "checkout with spaces"
    (root / "tools").mkdir(parents=True)
    (root / "tools/cc_pairs.py").touch()
    (root / "data").mkdir()
    venv = root / ("envs/vllm-serve/.venv" if mode == "slurm-serve" else "envs/vllm/.venv")
    (venv / "bin").mkdir(parents=True)
    for path in ("results/appworld_students/A/adapter", "results/bfcl_students/B/hub_merged"):
        (root / path).mkdir(parents=True)
        (root / path / "config.json").write_text("{}")
    script = root / "scripts/eval.sh" if mode == "local" else tmp_path / "slurm/spool/job/slurm_script"
    script.parent.mkdir(parents=True)
    script.write_text(text)
    stub = tmp_path / "python-stub"
    stub.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    "with open('invocations.jsonl', 'a') as stream:\n"
                    "    stream.write(json.dumps(dict(cwd=os.getcwd(), path=os.environ['PATH'], "
                    "gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), args=sys.argv[1:])) + '\\n')\n")
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "CC_"))}
    env.update(CC_PYTHON=str(stub), PATH=os.defpath, CUDA_VISIBLE_DEVICES="7")
    if mode != "local":
        env.update(SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID="123")
    env["CC_EVAL_TAGS"] = "A,B"
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    invocations = cc.read_jsonl(root / "invocations.jsonl")
    assert len(invocations) == 2
    for call, tag, model in zip(invocations, ("A", "B"),
                                ("results/appworld_students/A/adapter", "results/bfcl_students/B/hub_merged")):
        assert call["cwd"] == str(root) and call["gpu"] == "7"
        expected_path = os.defpath if mode == "local" else f"{venv}/bin:{root}/.venv/bin:{os.defpath}"
        assert call["path"] == expected_path
        args = call["args"]
        assert args[:2] == ["tools/cc_pairs.py", "evaluate"]
        assert args[args.index("--model") + 1] == model and args[args.index("--tag") + 1] == tag
        assert args[args.index("--pairs") + 1] == "data/cc_pairs_v1_stage1/confused_pairs.json"
        assert args[args.index("--base-results") + 1] == "data/cc_pairs_v1/probe_results.jsonl"
    assert not (tmp_path / "logs").exists()


def test_eval_launcher_merge_fallback_contract():
    script = ROOT / "scripts/cc_eval_hpg.slurm"
    for pattern in (
        '"$CC_PYTHON" tools/bfcl_hub_merge_export.py',
        '--adapter "$adapter" --out "$tmp_model" --model Qwen/Qwen3.5-4B',
        'tmp_model="$CC_EVAL_OUT/$tag/hub_merged_tmp"',
        '[[ -f "$tmp_model/config.json" ]]',
        'model=$tmp_model',
        'tools/cc_pairs.py evaluate --model "$model"',
        '"$CC_ROOT/_trash/cc_eval_${tag}_XXXXXX"',
        'mv -- "$tmp_model" "$trash/"',
        'trap trash_tmp_model EXIT',
    ):
        subprocess.run(["grep", "-Fq", "--", pattern, str(script)], check=True)
    assert "rm -rf" not in script.read_text()


@pytest.mark.parametrize("tree,failure,exit_code", [
    ("appworld_students", "", 0), ("bfcl_students", "", 0),
    ("appworld_students", "merge", 73), ("appworld_students", "evaluate", 74),
    ("appworld_students", "config", 2),
])
def test_eval_launcher_merges_adapter_and_archives_temp(tmp_path, tree, failure, exit_code):
    import os
    root = tmp_path / "checkout with spaces"
    (root / "tools").mkdir(parents=True)
    (root / "tools/cc_pairs.py").touch()
    (root / "data").mkdir()
    tag = "cc_s1_A_s0"
    adapter = root / f"results/{tree}/{tag}/adapter"
    adapter.mkdir(parents=True)
    (adapter / "model.safetensors").write_text("training weights")
    merged = root / f"results/cc_eval/{tag}/hub_merged_tmp"
    merged.mkdir(parents=True)
    (merged / "stale").write_text("previous interrupted export")
    ready = root / "results/bfcl_students/ready/hub_merged"
    ready.mkdir(parents=True)
    (ready / "config.json").write_text("{}")
    # Prefer an existing hub_merged even when another tree has a native adapter/.
    other = root / "results/appworld_students/ready/adapter"
    other.mkdir(parents=True)
    (other / "config.json").write_text("{}")
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        f"failure = {failure!r}\n"
        "args = sys.argv[1:]\n"
        "with open('invocations.jsonl', 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if args[0] == 'tools/bfcl_hub_merge_export.py':\n"
        "    out = Path(args[args.index('--out') + 1])\n"
        "    assert not out.exists(), 'stale export was not archived'\n"
        "    out.mkdir(parents=True)\n"
        "    (out / 'weights').write_text('temporary weights')\n"
        "    if failure == 'merge': sys.exit(73)\n"
        "    if failure != 'config': (out / 'config.json').write_text('{}')\n"
        "else:\n"
        "    assert args[:2] == ['tools/cc_pairs.py', 'evaluate']\n"
        "    model = Path(args[args.index('--model') + 1])\n"
        "    assert (model / 'config.json').is_file()\n"
        "    if failure == 'evaluate': sys.exit(74)\n"
        "    if args[args.index('--tag') + 1] == 'ready':\n"
        f"        assert not Path({str(merged)!r}).exists(), 'previous tag was not cleaned up'\n")
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "CC_"))}
    env.update(CC_PYTHON=str(stub), SLURM_SUBMIT_DIR=str(root), PATH=os.defpath)
    result = subprocess.run(["bash", str(ROOT / "scripts/cc_eval_hpg.slurm"), tag, "ready"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    calls = cc.read_jsonl(root / "invocations.jsonl")
    assert calls[0] == ["tools/bfcl_hub_merge_export.py", "--adapter", str(adapter.relative_to(root)),
                        "--out", str(merged.relative_to(root)), "--model", "Qwen/Qwen3.5-4B", "--verify"]
    if failure in ("merge", "config"):
        assert len(calls) == 1  # Never evaluate failed/incomplete exports.
    else:
        assert calls[1][calls[1].index("--model") + 1] == str(merged.relative_to(root))
        assert len(calls) == (2 if failure else 3)
        if not failure:
            assert calls[2][calls[2].index("--model") + 1] == str(ready.relative_to(root))
    assert not merged.exists()
    archives = list((root / "_trash").glob("*/hub_merged_tmp"))
    assert len(archives) == 2
    assert sorted(p.name for archive in archives for p in archive.iterdir()) == (
        ["stale", "weights"] if failure in ("merge", "config") else ["config.json", "stale", "weights"])
    assert (adapter / "model.safetensors").read_text() == "training weights"
    assert (ready / "config.json").read_text() == (other / "config.json").read_text() == "{}"
