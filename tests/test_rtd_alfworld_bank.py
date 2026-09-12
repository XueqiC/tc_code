"""C26-A synthetic CPU contracts; never reads a real bank or starts an env."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_caps import CapConfiguration, cap_audit, public_cap
from bfas.rtd.benchmarks.alfworld_state import (
    Observation, canonical_hash, parent_fold, parent_hash, reconstruct_package,
    replay_commands, validate_full_state, validate_parent_folds,
)
from bfas.rtd.broker import RequestRecord, SealedReplayBroker, UnavailableError, seal_bank
from bfas.rtd.ledger import Ledger
from bfas.rtd.selector import PublicFeatures, StudentSnapshot, select_public
from bfas.rtd.transport import FullState
from tools.rtd_alfworld_bank import main, write_report

FIXTURE = Path(__file__).parent / "fixtures/alfworld_archive.json"


def write(path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in value) if jsonl else json.dumps(value))


@pytest.fixture
def archive(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    root = tmp_path / "archive"
    rows = fixture["rows"]
    rows[0]["demo"] = dict(turns=fixture["turns"], worked_example="hidden teacher explanation")
    write(root / bank.LEDGER, rows, jsonl=True)
    tids = sorted({r["task_id"] for r in rows})
    write(root / bank.SUPPORT, dict(demand=tids, calibration=["protected/trial-0"], support=tids, seed=50))
    write(root / bank.PROBE, [], jsonl=True)
    write(root / bank.CACHE, dict(demos={fixture["task_id"]: dict(turns=fixture["turns"])},
                                  completed_task_ids=tids, complete=True))
    for tid in tids:
        world = root / "envs/alfworld/data/json_2.1.1/train" / tid
        write(world / "game.tw-pddl", dict(grammar={"task": [{"rhs": "Your task is to: " + fixture["goal"]}]},
                                          walkthrough=["HIDDEN SOLUTION"], solvable=True))
        write(world / "traj_data.json", dict(plan="HIDDEN TRAJECTORY"))
        write(world / "initial_state.pddl", dict(room="synthetic"))
    event = dict(task_id=fixture["task_id"], prompt="same event prompt", response="THOUGHT: derived\nACTION: put apple 1 on table 1",
                 _event_good="put apple 1 on table 1", _event_turn=1, _prefix_len=1,
                 _traj="legacy-name", _event_dU=999, _rejected="not current source sampling")
    write(root / "data/alf_sft/events_v1.jsonl", [event, event], jsonl=True)
    write(root / "data/alf_sft/pool_A_all.jsonl", [event, event], jsonl=True)
    write(root / "data/alf_sft/pool_C_conseq.jsonl", [event], jsonl=True)
    # A local synthetic tokenizer, no hub or network access.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    return root


def collect(root, **kwargs):
    return bank.collect_archive(root, tokenizer_path=root / "tokenizer.json", **kwargs)


@dataclass(frozen=True)
class FakeStepper:
    world_hash: str
    wrong_index: bool = False
    terminal_early: bool = False
    won: bool = True

    def reset(self, request):
        return 0, Observation(0, "Full reset observation " + "x" * 2100,
                              ("go to table 1",), self.world_hash)

    def step(self, cursor, command):
        index = cursor + 1
        return index, Observation(index + int(self.wrong_index),
                                  "Full observation " + str(index) + "x" * 400,
                                  ("put apple 1 on table 1",) if index == 1 else (),
                                  self.world_hash, done=index == 2 or self.terminal_early,
                                  won=self.won if index == 2 else False)


def renderer(request, history):
    return "fixed prompt"  # deliberately identical; history must determine state equality


def successful(archive):
    return next(p for p in archive.payloads.values() if p["status"] == "candidate")


def replay(archive):
    payload = successful(archive)
    request = archive.reset_requests[payload["query_id"]]
    return replay_commands(request, payload["commands"], FakeStepper(request["world_hash"]), renderer)


def test_inventory_boundaries_costs_gaps_and_protected_parents(archive):
    a = collect(archive)
    s = a.summary
    assert (s["attempts"], s["task_ids"], s["successful_attempts"], s["failed_attempts"]) == (3, 2, 1, 2)
    assert (s["candidate_packages"], s["unavailable_packages"], s["usable_packages"]) == (1, 2, 0)
    assert s["legacy_token_estimate"] == 210
    assert s["successful_legacy_token_estimate"] == 40
    assert s["retained_command_tokens"] != 40
    assert s["cap_audit"]["B_bank"] == 0
    assert s["cap_audit"]["inventory_by_class"]["alf_teacher_turn"] == 0
    assert s["historical_attempt_gaps"][0]["missing_attempt_indices"] == [0]
    assert s["historical_attempt_gaps"][0]["cost"] is None
    assert s["proposed_support"]["m"] == 1
    assert len({p["fold"] for p in s["proposed_support"]["parents"]}) == 1  # two trials, one parent
    assert s["discrepancies"]  # synthetic numbers must never be forced to historical counts
    assert s["cache_audit"]["entries"][0]["status"] == "copy"
    p = successful(a)
    assert len(p["provenance"]["event_aliases"]) == 5
    assert p["payload_kind"] == "extracted_teacher_commands" and p["cost_basis"] == "estimated"
    assert all(p["usage"][k] is None for k in ("output_tokens", "input_tokens", "reasoning_tokens", "monetary_cost"))


def test_failed_payload_and_all_candidates_cannot_be_reserved_or_revealed(archive, tmp_path):
    out = tmp_path / "bank"
    bank.build_alfworld_bank(archive, out, tokenizer_path=archive / "tokenizer.json")
    a = collect(archive)
    ledger = Ledger(10_000_000)
    parents = frozenset(r.parent_hash for r in a.records)
    broker = SealedReplayBroker(out, ledger, inner_parent_hashes=parents)
    assert broker.list_candidates(StudentSnapshot("s", parents), (), ledger.remaining) == []
    for q, p in a.payloads.items():
        if not p["success"]:
            assert p["commands"] == [] and p["payload_kind"] is None and p["cost"] > 0
        with pytest.raises(UnavailableError):
            broker.acquire(q)
    assert not ledger.events and not ledger.reservations


def test_aliases_charge_once_with_existing_broker_and_ledger(archive, tmp_path):
    a = collect(archive)
    p = deepcopy(successful(a))
    q = p["query_id"]
    result = replay(a)
    p.update(status="usable", behaviors=[dict(state=asdict(b.state), text=b.text) for b in result.behaviors])
    r = next(r for r in a.records if r.spec.query_id == q)
    r = replace(r, unavailable_reason=None, spec=replace(r.spec, state_hash=result.states[0].state_hash))
    out = tmp_path / "synthetic-validated-bank"
    seal_bank(out, [r], {q: p})
    ledger = Ledger(r.spec.cost_upper_bound)
    broker = SealedReplayBroker(out, ledger, inner_parent_hashes={r.parent_hash})
    specs = broker.list_candidates(StudentSnapshot("s", frozenset({r.parent_hash})), (), ledger.remaining)
    choice, trace = select_public(specs, lambda candidates: candidates[0].query_id)
    for alias in p["provenance"]["event_aliases"]:
        package = broker.acquire(alias["package_id"])
        assert package.query_id == choice and package.cost == 40
        assert len(package.behaviors) == 2
    assert ledger.spent == 40 and ledger.remaining == r.spec.cost_upper_bound - 40
    assert len([e for e in ledger.events if e["kind"] == "reveal"]) == 1
    assert broker.assert_no_hidden_access(trace).passed
    restored = SealedReplayBroker(out, ledger, inner_parent_hashes={r.parent_hash})
    assert restored.acquire(q).cost == 40 and ledger.spent == 40
    broker.set_inner_parents(set())
    with pytest.raises(UnavailableError):
        broker.acquire(q)  # ownership does not bypass fold rotation


def test_same_name_different_requests_never_merge_and_ambiguous_alias_is_unavailable(archive):
    rows = [json.loads(l) for l in (archive / bank.LEDGER).read_text().splitlines()]
    duplicate_name = deepcopy(rows[0])
    duplicate_name["timestamp"] = "2026-02-02T00:00:00Z"
    rows.append(duplicate_name)
    write(archive / bank.LEDGER, rows, jsonl=True)
    a = collect(archive)
    assert len(a.records) == 4 and len(a.payloads) == 4
    assert len({r.spec.query_id for r in a.records}) == 4
    assert len({r.spec.state_hash for r in a.records if a.reset_requests[r.spec.query_id]["task_id"] == rows[0]["task_id"]}) == 1
    assert all(alias["package_id"] is None for alias in a.aliases)
    assert any(len(alias["candidate_package_ids"]) == 2 for alias in a.aliases)


def test_query_identity_uses_file_hash_task_attempt_timestamp_and_physical_line(archive):
    row = json.loads((archive / bank.LEDGER).read_text().splitlines()[0])
    h = bank.file_hash(archive / bank.LEDGER)
    q = bank.query_id(h, row, 1)
    assert len(q) == 64 and row["task_id"] not in q
    changes = [bank.query_id("a" * 64, row, 1), bank.query_id(h, row, 2)]
    for field, value in (("task_id", "different/trial"), ("attempt_index", 3), ("timestamp", "later")):
        changes.append(bank.query_id(h, {**row, field: value}, 1))
    assert q not in changes and len(set(changes)) == len(changes)
    # A blank line changes physical provenance; no filtered enumerate shortcut.
    path = archive / bank.LEDGER
    path.write_text("\n" + path.read_text())
    assert min(p["provenance"]["line"] for p in collect(archive).payloads.values()) == 2


def test_identical_ledger_copy_deduplicates_but_new_archive_request_does_not(archive):
    copy = archive / "data/teacher_ledger/alfworld_copy.jsonl"
    copy.write_bytes((archive / bank.LEDGER).read_bytes())
    a = collect(archive, ledger_paths=(bank.LEDGER, str(copy.relative_to(archive))))
    assert len(a.records) == 3 and a.summary["legacy_token_estimate"] == 210
    rows = [json.loads(l) for l in copy.read_text().splitlines()]
    rows[0]["timestamp"] = "different request archive"
    write(copy, rows, jsonl=True)
    assert len(collect(archive, ledger_paths=(bank.LEDGER, str(copy.relative_to(archive)))).records) == 6


def test_public_view_is_independent_of_hidden_text_usage_and_outcomes(archive):
    a = collect(archive)
    public_before = [asdict(r) for r in a.records]
    q = successful(a)["query_id"]
    p = a.payloads[q]
    p.update(success=False, won=False, commands=["HIDDEN ALTERED COMMAND"], cost=999, usage={"output_tokens": 12345})
    p["historical_response"]["demo"]["worked_example"] = "HIDDEN ALTERED EXPLANATION"
    rebuilt = [asdict(bank.public_record(r.spec.query_id, a.reset_requests[r.spec.query_id])) for r in a.records]
    assert rebuilt == public_before
    serialized = json.dumps([rebuilt, a.reset_requests])
    for text in ("success", "won", "usage", "HIDDEN", "derived", "tokens_spent", "integrity", "event_aliases"):
        assert text not in serialized
    assert all(r.spec.L == 40 and r.spec.features == PublicFeatures() for r in a.records)
    assert len({r.unavailable_reason for r in a.records}) == 1  # no success-dependent public status


def test_selector_leak_denied_even_for_cached_alfworld_module(archive, tmp_path):
    a = collect(archive)
    out = tmp_path / "bank"
    bank.build_alfworld_bank(archive, out)
    # Module already loaded, ordinary data reads and cached privileged calls denied.
    candidates = [a.records[0].spec]
    with pytest.raises(PermissionError):
        select_public(candidates, lambda _: (out / "sealed/audit.json").read_text())
    with pytest.raises(PermissionError):
        select_public(candidates, lambda _: bank.collect_archive(archive))
    with pytest.raises(PermissionError):
        select_public(candidates, lambda _: bank.audit_bank(out))
    with pytest.raises(PermissionError):
        select_public(candidates, lambda _: importlib.import_module("bfas.rtd.broker"))


def test_caps_come_only_from_configuration_and_budget_example(archive):
    a = collect(archive)
    cap, provenance = public_cap("alf_demo_episode")
    assert cap == 1_310_720 and "2048 × 40 × 2 × 8" in provenance
    assert public_cap("alf_teacher_turn")[0] == 32768
    assert "project-lead" not in provenance
    configuration = CapConfiguration(max_completion_tokens=100)
    assert public_cap("alf_demo_episode", configuration=configuration)[0] == 100 * 40 * 2 * 8
    audit = cap_audit(a.records, a.payloads)
    example = audit["conditional_107_candidate_example"]
    assert example["B_bank"] == 140_247_040
    assert example["ceilings"] == {"10": 14_024_704, "25": 35_061_760, "50": 70_123_520}
    assert audit["B_bank"] == 0
    # Once independently validated, only that package enters the denominator.
    q = successful(a)["query_id"]
    usable_records = [replace(r, unavailable_reason=None) if r.spec.query_id == q else r for r in a.records]
    usable_payloads = deepcopy(a.payloads)
    usable_payloads[q]["status"] = "usable"
    usable = cap_audit(usable_records, usable_payloads)
    assert usable["B_bank"] == cap
    assert [p["budget"] for p in usable["budget_affordability"]] == [cap * v // 100 for v in (10, 25, 50)]
    assert all(sum(p["by_inner_fold"].values()) == 0 for p in usable["budget_affordability"])
    changed = replace(a.records[0], spec=replace(a.records[0].spec, cost_upper_bound=41))
    with pytest.raises(ValueError, match="public configuration"):
        cap_audit([changed, *a.records[1:]], a.payloads)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            CapConfiguration(max_completion_tokens=value)
    with pytest.raises(ValueError, match="unknown"):
        public_cap("generator_item")


@pytest.mark.parametrize("cost", [1_310_721, -1, True])
def test_cost_overflow_or_invalid_cost_fails_before_writing(archive, tmp_path, cost):
    rows = [json.loads(l) for l in (archive / bank.LEDGER).read_text().splitlines()]
    rows[1]["tokens_spent"] = cost  # failed attempt must be audited too
    write(archive / bank.LEDGER, rows, jsonl=True)
    out = tmp_path / "bank"
    with pytest.raises(ValueError, match="cost"):
        bank.build_alfworld_bank(archive, out)
    assert not out.exists()


def test_missing_cost_is_unknown_unavailable_not_zero(archive):
    rows = [json.loads(l) for l in (archive / bank.LEDGER).read_text().splitlines()]
    del rows[0]["tokens_spent"]
    write(archive / bank.LEDGER, rows, jsonl=True)
    a = collect(archive)
    missing = next(p for p in a.payloads.values() if p["cost"] is None)
    assert missing["status"] == "unavailable" and "cost missing" in missing["unavailable_reason"]
    assert a.summary["cap_audit"]["missing_cost_query_ids"] == [missing["query_id"]]


def test_replay_deterministic_full_history_and_prompt_alone_never_aliases(archive):
    a = collect(archive)
    r = replay(a)
    assert len(r.behaviors) == 2 and len(r.states) == 3 and r.done and r.won
    assert r.transcript_hash == replay(a).transcript_hash
    assert len({s.state_hash for s in r.states}) == 3
    history = json.loads(r.states[-1].history_json)
    assert [h["role"] for h in history] == ["user", "assistant", "tool", "assistant", "tool"]
    assert len(history[0]["content"]) > 2000 and len(history[2]["content"]) > 300
    assert history[2]["admissible"] == ["put apple 1 on table 1"]
    with pytest.raises(ValueError, match="full state mismatch"):
        r.states[0].assert_matches(r.states[1])
    p = successful(a)
    request = a.reset_requests[p["query_id"]]
    different_won = replay_commands(request, p["commands"], FakeStepper(request["world_hash"], won=False), renderer)
    assert [s.state_hash for s in r.states] == [s.state_hash for s in different_won.states]


@pytest.mark.parametrize("fault", ["goal", "world", "step_world", "order", "terminal"])
def test_replay_rejects_invalid_request_or_history(archive, fault):
    a = collect(archive)
    p = successful(a)
    request = deepcopy(a.reset_requests[p["query_id"]])
    stepper = FakeStepper(request["world_hash"])
    if fault == "goal":
        request["goal"] = ""
    elif fault == "world":
        request["world_hash"] = "0" * 64
    elif fault == "step_world":
        stepper = replace(stepper, world_hash="0" * 64)
    elif fault == "order":
        stepper = replace(stepper, wrong_index=True)
    else:
        stepper = replace(stepper, terminal_early=True)
    with pytest.raises(ValueError):
        replay_commands(request, p["commands"], stepper, renderer)


def test_shuffled_fullstate_history_rejected_even_if_generic_fullstate_accepts(archive):
    state = replay(collect(archive)).states[-1]
    history = json.loads(state.history_json)
    history[1], history[3] = history[3], history[1]
    broken = FullState.create(json.loads(state.task_json), history, state.prompt, state.parent_hash)
    with pytest.raises(ValueError, match="out-of-order"):
        validate_full_state(broken, expected_world_hash=json.loads(state.task_json)["world_hash"])


def test_owned_prefix_and_fold_checks_precede_stepper_calls(archive):
    a = collect(archive)
    p = successful(a)
    q = p["query_id"]
    request = a.reset_requests[q]
    parents = {parent_hash(request["task_id"])}
    with pytest.raises(ValueError, match="owned"):
        reconstruct_package(p, request, None, renderer, owned_ids=(), inner_parent_hashes=parents)
    with pytest.raises(ValueError, match="inner fold"):
        reconstruct_package(p, request, None, renderer, owned_ids={q}, inner_parent_hashes=())
    p["dependencies"] = ["unowned"]
    with pytest.raises(ValueError, match="owned"):
        reconstruct_package(p, request, None, renderer, owned_ids={q}, inner_parent_hashes=parents)
    p["dependencies"] = []
    result = reconstruct_package(p, request, FakeStepper(request["world_hash"]), renderer,
                                 owned_ids={q}, inner_parent_hashes=parents)
    assert result.transcript_hash == replay(a).transcript_hash


def test_protected_probe_and_calibration_group_excluded(archive):
    write(archive / bank.PROBE, [dict(task_id="room-a/trial-2")], jsonl=True)
    a = collect(archive)
    assert a.summary["proposed_support"]["m"] == 0
    assert all(p["exclusion_reasons"] for p in a.payloads.values())
    p = successful(a)
    with pytest.raises(ValueError, match="protected"):
        reconstruct_package(p, a.reset_requests[p["query_id"]], None, renderer,
                             owned_ids={p["query_id"]}, inner_parent_hashes={parent_hash("room-a/trial-1")})


def test_two_trials_cannot_be_assigned_different_folds():
    fold = parent_fold(parent_hash("room-a/trial-1"))
    assert validate_parent_folds({"room-a/trial-1": fold, "room-a/trial-2": fold})
    with pytest.raises(ValueError, match="parent fold"):
        validate_parent_folds({"room-a/trial-1": fold, "room-a/trial-2": 1 - fold})


def test_broker_missing_dependency_and_cross_fold_rejected_before_reservation(archive, tmp_path):
    a = collect(archive)
    p = successful(a)
    base = next(r for r in a.records if r.spec.query_id == p["query_id"])
    base = replace(base, unavailable_reason=None)
    dep_q = "f" * 64
    child = replace(base, dependencies=(dep_q,))
    with pytest.raises(ValueError, match="missing/self"):
        seal_bank(tmp_path / "missing", [child], {p["query_id"]: p})
    other_parent = next(parent_hash(f"room-{i}/trial") for i in range(100)
                        if parent_fold(parent_hash(f"room-{i}/trial")) != parent_fold(base.parent_hash))
    dep = RequestRecord(replace(base.spec, query_id=dep_q), other_parent)
    out = tmp_path / "dependencies"
    seal_bank(out, [child, dep], {p["query_id"]: p, dep_q: p})
    ledger = Ledger(10_000_000)
    parents = frozenset({base.parent_hash})
    broker = SealedReplayBroker(out, ledger, inner_parent_hashes=parents)
    assert not broker.list_candidates(StudentSnapshot("s", parents), (), ledger.remaining)
    with pytest.raises(UnavailableError):
        broker.acquire(p["query_id"])
    broker.set_inner_parents({base.parent_hash, other_parent})
    with pytest.raises(UnavailableError, match="dependencies"):
        broker.acquire(p["query_id"])
    assert not ledger.events


def test_build_audit_no_overwrite_and_source_integrity(archive, tmp_path):
    out = tmp_path / "bank"
    bank.build_alfworld_bank(archive, out, tokenizer_path=archive / "tokenizer.json")
    h = bank.file_hash(out / "sealed/manifest.json")
    assert bank.audit_bank(out, root=archive, expected_manifest_sha256=h)["passed"]
    with pytest.raises(FileExistsError):
        bank.build_alfworld_bank(archive, out)
    with pytest.raises(ValueError, match="manifest"):
        bank.audit_bank(out, expected_manifest_sha256="0" * 64)
    path = archive / bank.LEDGER
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source integrity"):
        bank.audit_bank(out, root=archive)


@pytest.mark.parametrize("artifact", ["public/requests.json", "public/reset_requests.json", "sealed/audit.json",
                                      "sealed/event_aliases.json", "sealed/integrity.json", "payload"])
def test_tampering_with_any_bank_component_rejected(archive, tmp_path, artifact):
    out = tmp_path / "bank"
    bank.build_alfworld_bank(archive, out)
    if artifact == "payload":
        q = json.loads((out / "public/requests.json").read_text())[0]["spec"]["query_id"]
        artifact = f"sealed/{q}.json"
    path = out / artifact
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        bank.audit_bank(out)


def test_cli_inventory_build_audit_and_reports(archive, tmp_path, capsys):
    args = ["--root", str(archive), "--tokenizer", str(archive / "tokenizer.json")]
    out = tmp_path / "report"
    assert main(["inventory", *args, "--out", str(out)]) == 0
    report = json.loads((out / "inventory.json").read_text())
    assert "| attempts | 3 | 219 |" in (out / "inventory.md").read_text()
    with pytest.raises(FileExistsError):
        write_report(report, out)
    output = tmp_path / "bank"
    assert main(["build", *args, "--out", str(output)]) == 0
    assert main(["audit", "--bank", str(output), "--root", str(archive)]) == 0
    assert '"passed": true' in capsys.readouterr().out


def test_benchmark_package_init_has_no_import_side_effects():
    # Parent-process sys.path edits do not propagate to a fresh interpreter.
    script = """
import sys
sys.path.insert(0, sys.argv[1])
import bfas.rtd.benchmarks
assert 'torch' not in sys.modules
assert 'bfas.rtd.broker' not in sys.modules
"""
    source = Path(__file__).resolve().parents[1] / "src"
    subprocess.run([sys.executable, "-B", "-c", script, str(source)], check=True, timeout=15)
