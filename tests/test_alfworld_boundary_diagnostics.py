"""Additive parser diagnostics: observe the frozen parser, including valid look."""
import pytest

from bfas.adapters.alfworld import ALFWorldAdapter
from bfas.rtd.benchmarks.alfworld_diagnostics import EpisodeParserDiagnostics, parse_with_diagnostics
from bfas.rtd.benchmarks.alfworld_evaluation import official_episode
from bfas.rtd.benchmarks.alfworld_identity import campaign_identity
from rtd_alfworld_evaluation_fixtures import FakeBackend, FakeEnv, campaign, run


@pytest.mark.parametrize("reply,commands,reason", [
    ("", ["look"], "empty_reply"),
    ("   ", ["look"], "empty_reply"),
    ("THOUGHT: uncertain", ["look"], "no_action_marker"),
    ("ACTION: fly away", ["look"], "candidate_not_admissible"),
    ("ACTION:", ["look"], "empty_candidate"),
    ("ACTION: look", [], "candidate_not_admissible"),
    ("ACTION: look", ["look"], None),
    ("look", ["look"], None),
    ("ACTION: LOOK", ["look"], None),
    ('ACTION: `look`', ["look"], None),
    ("ACTION: please look", ["look"], None),
])
def test_each_parser_path_reports_reason_without_changing_command(reply, commands, reason):
    command, fields = parse_with_diagnostics(reply, commands)
    assert command == ALFWorldAdapter._teacher_command(reply, commands)
    assert fields["parser_fallback"] is (reason is not None)
    assert fields.get("parser_fallback_reason") == reason
    if reason is None:
        assert "parser_fallback_reason" not in fields
    if "fly away" in reply:
        assert fields["parser_candidate"] == "fly away"


def test_actual_official_loop_records_fallback_and_keeps_existing_fields(campaign):
    from bfas.rtd.benchmarks.alfworld_evaluation import Generation
    c = campaign
    tid = c.manifest["evaluation_harness"]["expected"]["task_ids"][0]
    identity = campaign_identity(c.manifest)
    for reply in ("", "ACTION: no such command", "ACTION: look", "look"):
        backend = FakeBackend()
        backend.generate = lambda *a, **k: Generation(reply, 1, False)
        factory = lambda tid: FakeEnv(tid, won=True)
        original = official_episode(tid, backend, env_factory=factory, identity=identity)
        with EpisodeParserDiagnostics(factory) as (diagnostics, factory):
            recorded = official_episode(tid, backend, env_factory=factory, identity=identity)
        diagnostics.annotate(recorded)
        for before, after in zip(original["turns"], recorded["turns"]):
            assert {k: after[k] for k in before} == before
            assert after["parser_fallback"] == (reply in ("", "ACTION: no such command"))
            assert ("parser_fallback_reason" in after) == after["parser_fallback"]


def test_serial_campaign_persists_parser_fields(campaign):
    import json
    run(campaign)
    records = (campaign.directory/"artifacts/tasks").glob("*.json")
    for path in records:
        for turn in json.loads(path.read_text())["record"]["turns"]:
            assert "parser_candidate" in turn and "parser_fallback" in turn
