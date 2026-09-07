"""C26-B parent isolation and sealed-bank tests use synthetic CPU archives."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import random

import pytest

from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_caps import public_cap
from bfas.rtd.benchmarks.alfworld_state import canonical_hash, parent_fold, parent_hash
from bfas.rtd.benchmarks.alfworld_support import (
    ALFWorldSupport, _signed, audit_verified_bank, freeze_support, prompt_messages,
    seal_verified_bank, validate_support, verify_package,
)
from bfas.rtd.broker import PurchasedEvidencePackage, SealedReplayBroker, UnavailableError
from bfas.rtd.ledger import Ledger
from bfas.rtd.selector import StudentSnapshot, select_public
from bfas.rtd.transport import FullState


def write_json(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v) + "\n" for v in value) if lines else json.dumps(value))


@pytest.fixture
def support_archive(tmp_path):
    root = tmp_path / "archive"
    # Enough parents in each fold for a full 4-parent feedback block.
    tids = [f"pick_and_place_simple-Apple-None-Table-{i}/trial-1" for i in range(16)]
    tids.append(tids[0].replace("trial-1", "trial-2"))
    probe, calibration = "probe/trial-1", "calibration/trial-1"
    tids += [probe, "probe/trial-2", "calibration/trial-2"]
    write_json(root / bank.SUPPORT, dict(demand=tids, calibration=[calibration], seed=50))
    write_json(root / bank.PROBE, [dict(task_id=probe)], lines=True)
    # Unique full worlds, including the two trials under one game name.
    for tid in tids:
        world = root / "envs/alfworld/data/json_2.1.1/train" / tid
        write_json(world / "game.tw-pddl", dict(grammar={"task": [{"rhs": "Your task is to: put apple on table."}]}))
        write_json(world / "traj_data.json", dict(task_id=tid))
        write_json(world / "initial_state.pddl", dict(room=tid))
    write_json(root / bank.LEDGER, [dict(task_id=t) for t in tids], lines=True)
    identity = dict(fixture="fake CPU environment", max_episode_steps=40)
    environment = {**identity, "environment_hash": canonical_hash(identity)}
    return root, freeze_support(root, environment)


def test_freeze_groups_trials_excludes_entire_probe_and_calibration_parents(support_archive):
    root, m = support_archive
    assert m["m"] == 16 and len(m["training_task_ids"]) == 17
    assert m["historical_parent_groups"] == 18
    assert freeze_support(root, m["environment"]) == m
    two = next(p for p in m["parents"].values() if len(p["task_ids"]) == 2)
    assert two["selected_task_id"] == min(two["task_ids"])
    assert len({m["tasks"][t]["fold"] for t in two["task_ids"]}) == 1
    assert all(t not in m["training_task_ids"] for t in ("probe/trial-2", "calibration/trial-2"))
    assert len(m["manifest_hash"]) == 64
    assert all(m["tasks"][t]["request_hash"] == canonical_hash(m["tasks"][t]["request"]) for t in m["tasks"])


def test_two_trials_in_different_folds_rejected_even_after_resigning(support_archive):
    _, m = support_archive
    two = next(p for p in m["parents"].values() if len(p["task_ids"]) == 2)
    m["tasks"][two["task_ids"][1]]["fold"] ^= 1
    with pytest.raises(ValueError, match="parent fold"):
        validate_support(_signed(m))


@pytest.mark.parametrize("use", ["inner", "P", "normalizer", "projection", "source", "candidate", "pending"])
@pytest.mark.parametrize("round_number", [1, 2, 3])
def test_probe_calibration_feedback_never_enters_inner_fitting(support_archive, use, round_number):
    _, m = support_archive
    support = ALFWorldSupport(m)
    inner = support.parent_hashes(round_number, use=use)
    feedback = support.parent_hashes(round_number, use="feedback")
    assert inner.isdisjoint(feedback) and inner | feedback == set(m["parents"])
    feedback_tid = m["parents"][next(iter(feedback))]["selected_task_id"]
    for tid in (feedback_tid, "probe/trial-1", "probe/trial-2", "calibration/trial-1", "calibration/trial-2"):
        with pytest.raises(ValueError, match="rejected"):
            support.guard_tasks([tid], round_number, use=use)
    legal = m["parents"][next(iter(inner))]["selected_task_id"]
    assert support.guard_tasks([legal], round_number, use=use) == (legal,)


def test_feedback_uses_frozen_trial_and_includes_parents_without_teacher_payload(support_archive):
    _, m = support_archive
    support = ALFWorldSupport(m)
    for round_number in (1, 2, 3):
        tasks = support.feedback_tasks(round_number, random.Random(9))
        assert tasks == support.feedback_tasks(round_number, random.Random(9))
        assert len(tasks) == len({parent_hash(t) for t in tasks}) == 4
        assert all(t == m["parents"][parent_hash(t)]["selected_task_id"] for t in tasks)
        assert support.guard_tasks(tasks, round_number, use="feedback") == tasks
    assert support.parent_hashes(1) == support.parent_hashes(3)
    assert support.parent_hashes(1) == support.parent_hashes(2, use="feedback")


def test_support_is_defensively_copied_and_manifest_tampering_rejected(support_archive):
    _, m = support_archive
    support = ALFWorldSupport(m)
    m["m"] = 999
    with pytest.raises(ValueError, match="integrity"):
        validate_support(m)
    assert support.manifest["m"] == 16
    copy = support.manifest
    copy["parents"].clear()
    assert support.manifest["parents"]


def test_state_guards_bind_world_and_reject_teacher_prefix_without_ownership(support_archive):
    _, m = support_archive
    support = ALFWorldSupport(m)
    h = next(iter(support.parent_hashes(1)))
    tid = m["parents"][h]["selected_task_id"]
    request = m["tasks"][tid]["request"]
    history = [dict(role="user", index=0, content="full reset", admissible=["look"], done=False)]
    reset = FullState.create(request, history, "same prompt", h)
    prefix_history = history + [dict(role="assistant", index=1, content="look"),
                                dict(role="tool", index=1, content="full observation", admissible=["look"], done=False)]
    prefix = FullState.create(request, prefix_history, "same prompt", h)
    assert support.guard_states([reset], 1, use="P") == (reset,)
    with pytest.raises(ValueError, match="not owned"):
        support.guard_states([prefix], 1, use="normalizer")
    from bfas.rtd.transport import Behavior
    package = PurchasedEvidencePackage("owned", (), (Behavior(prefix, "look"),), 10, "estimated", {}, {}, {})
    assert support.guard_states([prefix], 1, owned_packages={"owned": package}) == (prefix,)
    with pytest.raises(ValueError, match="rejected"):
        support.guard_states([prefix], 2, owned_packages={"owned": package})
    request = {**request, "environment_hash": "f" * 64}
    changed = FullState.create(request, history, "same prompt", h)
    with pytest.raises(ValueError, match="frozen support"):
        support.guard_states([changed], 1)
    dependency = replace(package, query_id="dependency", behaviors=(Behavior(reset, "look"),))
    package = replace(package, dependencies=("dependency",))
    with pytest.raises(ValueError, match="not owned"):
        support.guard_states([prefix], 1, owned_packages={"owned": package})
    other_h = next(iter(support.parent_hashes(1, use="feedback")))
    other_tid = m["parents"][other_h]["selected_task_id"]
    other_state = FullState.create(m["tasks"][other_tid]["request"], history, "same prompt", other_h)
    dependency = replace(dependency, behaviors=(Behavior(other_state, "look"),))
    # Ownership persists across folds. Unrelated owned packages must not block
    # legal states, but a dependency on their evidence must be rejected.
    independent = replace(package, dependencies=())
    assert support.guard_states([prefix], 1, owned_packages={"owned": independent, "dependency": dependency}) == (prefix,)
    with pytest.raises(ValueError, match="rejected"):
        support.guard_states([prefix], 1, owned_packages={"owned": package, "dependency": dependency})
    middle = replace(independent, query_id="middle", dependencies=("dependency",))
    package = replace(package, dependencies=("middle",))
    with pytest.raises(ValueError, match="rejected"):
        support.guard_states([prefix], 1, owned_packages={"owned": package, "middle": middle, "dependency": dependency})


def test_identical_world_bytes_across_different_parent_names_fail_closed(support_archive):
    root, m = support_archive
    a, b = m["training_task_ids"][0], m["training_task_ids"][-1]
    base = root / "envs/alfworld/data/json_2.1.1/train"
    for name in ("game.tw-pddl", "traj_data.json", "initial_state.pddl"):
        (base / b / name).write_bytes((base / a / name).read_bytes())
    with pytest.raises(ValueError, match="identical world bytes"):
        freeze_support(root, m["environment"])


def test_sealed_verified_bank_roundtrip_cap_denominator_and_broker(support_archive, tmp_path):
    from test_rtd_alfworld_state import FakeStepper, make_payload, render
    root, m = support_archive
    tid = m["training_task_ids"][0]
    request = m["tasks"][tid]["request"]
    payload = make_payload(request)
    q = payload["query_id"]
    verified = verify_package(payload, request, FakeStepper(request), render)
    assert verified["status"] == "usable"
    failed = deepcopy(payload)
    row = failed["historical_response"]
    row.update(verified=False, attempt_index=1, tokens_spent=90)
    row.pop("demo")
    failed["provenance"].update(attempt_index=1, line=2)
    q2 = bank.query_id(failed["provenance"]["ledger_sha256"], row, 2)
    failed.update(query_id=q2, success=False, status="unavailable", unavailable_reason="failed attempt: response/trajectory missing",
                  cost=90, commands=[], payload_kind=None, raw_ledger_line=json.dumps(row), behaviors=[])
    failed["usage"]["estimated_output_tokens"] = 90
    archive = bank.Archive([bank.public_record(q, request), bank.public_record(q2, request)],
                           {q: payload, q2: failed}, {q: request, q2: request}, [], {"source_files": []})
    out = tmp_path / "sealed"
    reset = verified["verification"]["states"][0]
    summary = seal_verified_bank(out, archive, m, {q: verified, q2: failed}, {tid: reset})
    assert audit_verified_bank(out)["passed"]
    cap = public_cap("alf_demo_episode")[0]
    assert summary["cap_audit"]["B_bank"] == cap
    assert summary["unavailable_packages"] == 1
    assert [r["budget"] for r in summary["cap_audit"]["budget_affordability"]] == [cap * p // 100 for p in (10, 25, 50)]
    ledger = Ledger(cap)
    broker = SealedReplayBroker(out, ledger, inner_parent_hashes={parent_hash(tid)})
    candidates = broker.list_candidates(StudentSnapshot("s", frozenset({parent_hash(tid)})), (), cap)
    assert [c.query_id for c in candidates] == [q]
    assert candidates[0].state_hash == FullState(**reset).state_hash
    package = broker.acquire(q)
    assert len(package.behaviors) == 2
    assert broker.acquire(q) == package and ledger.spent == 40
    broker.set_inner_parents(set())
    with pytest.raises(UnavailableError):
        broker.acquire(q)
    with pytest.raises(PermissionError):
        select_public(candidates, lambda _: audit_verified_bank(out))
    with pytest.raises(FileExistsError):
        seal_verified_bank(out, archive, m, {q: verified, q2: failed}, {tid: reset})
    # Even re-signing transport integrity/artifact hashes cannot launder reordered states.
    pfile = out / f"sealed/{q}.json"
    tampered = json.loads(pfile.read_text())
    tampered["verification"]["states"][1:] = reversed(tampered["verification"]["states"][1:])
    write_json(pfile, tampered)
    from bfas.cc_pairs import digest
    integrity = json.loads((out / "sealed/integrity.json").read_text())
    integrity[q] = digest(tampered)
    write_json(out / "sealed/integrity.json", integrity)
    manifest = json.loads((out / "sealed/manifest.json").read_text())
    for name in (f"sealed/{q}.json", "sealed/integrity.json"):
        manifest["artifacts"][name] = bank.file_hash(out / name)
    write_json(out / "sealed/manifest.json", manifest)
    with pytest.raises(ValueError, match="history|state"):
        audit_verified_bank(out)
