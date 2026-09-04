from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.events_schema import (  # noqa: E402
    REQUIRED_FIELDS, SPEC_FIELDS, UnifiedEvent, from_alfworld_row, from_bfcl_row,
    read_events, state_hash_of, write_events,
)


class StubTok:
    """Whitespace tokenizer standing in for Qwen's (tests must not load a model)."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


TOK = StubTok()

BFCL_ROW = {
    "task_id": "live_multiple_923-191-11", "teacher": "oracle_gt", "turn_index": 0,
    "prompt": "<|im_start|>system\nTools...<|im_end|>\n<|im_start|>user\nhousekeeper<|im_end|>\n<|im_start|>assistant\n",
    "response": "<tool_call>\n{\"name\": \"get_service_providers\", \"arguments\": {\"a\": 1}}\n</tool_call>",
    "token_hint": 135, "_task_phat": 0.1667, "_traj": "live_multiple_923-191-11#ev1",
    "_seed_category": "live_multiple", "_event_dU": 0.8333, "_event_u_plus": 1.0,
    "_event_u_minus": 0.1667, "_event_k": 6, "_event_minus_ok": 1, "_event_plus_ok": 6,
    "_rejected": "<tool_call>\n{\"name\": \"get_service_providers\", \"arguments\": {}}\n</tool_call>",
}

ALF_ROW = {
    "task_id": "look_at_obj_in_light-Book-None-DeskLamp-303/trial_T2019", "teacher": "demo_replay",
    "turn_index": 1, "prompt": "<|im_start|>system\nALFWorld agent<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "response": "ACTION: go to desk 1", "_rejected": "I should look around first.\nACTION: look",
    "token_hint": 215, "_task_phat": 0.2, "_traj": "look_at_obj_in_light-Book-None-DeskLamp-303/trial_T2019#p2",
    "_seed_category": "look_at_obj_in_light", "_event_turn": 1, "_event_good": "go to desk 1",
    "_event_bad": "look", "_event_u_plus": 0.5, "_event_u_minus": 0.125, "_event_dU": 0.375,
    "_event_k": 8, "_prefix_len": 1, "_event_wins": [4, 1],
}


def test_spec_fields_match_section_13():
    assert SPEC_FIELDS == (
        "benchmark", "split", "task_id", "state_id", "state_hash", "student_checkpoint",
        "student_continuation", "teacher_model", "teacher_continuation",
        "student_branch_outcomes", "teacher_branch_outcomes", "student_output_tokens",
        "teacher_output_tokens", "fingerprint_version", "fingerprint_path",
        "atom_dictionary_version", "atom_loading", "provenance")
    assert set(REQUIRED_FIELDS) == set(SPEC_FIELDS) | {"state_text"}
    assert not any(("y+" in f or "y-" in f or "chosen" in f or "rejected" in f) for f in REQUIRED_FIELDS)


def test_bfcl_conversion_ownership_and_outcomes():
    ev = from_bfcl_row(BFCL_ROW, TOK, source="x.jsonl")
    assert ev.benchmark == "bfcl" and ev.split == "train"
    assert ev.teacher_continuation == BFCL_ROW["response"]      # y^T = GT call block
    assert ev.student_continuation == BFCL_ROW["_rejected"]      # y^S = student's own call
    assert ev.state_id == BFCL_ROW["_traj"]
    assert ev.state_hash == hashlib.sha1(BFCL_ROW["prompt"].encode()).hexdigest()
    assert ev.state_hash == state_hash_of(ev.state_text)
    assert ev.teacher_branch_outcomes == [{"kind": "aggregate", "metric": "success_rate",
                                           "value": 1.0, "k": 6, "n_success": 6}]
    assert ev.student_branch_outcomes[0]["value"] == pytest.approx(0.1667)
    assert ev.student_branch_outcomes[0]["n_success"] == 1
    assert ev.teacher_output_tokens == len(BFCL_ROW["response"].split())
    assert ev.student_output_tokens == len(BFCL_ROW["_rejected"].split())
    assert ev.provenance["seed_category"] == "live_multiple"
    assert ev.provenance["event_dU"] == pytest.approx(0.8333)
    assert ev.fingerprint_version == "" and ev.atom_loading == []


def test_alfworld_conversion_wins_and_commands():
    ev = from_alfworld_row(ALF_ROW, TOK, source="alf.jsonl", split="dev")
    assert ev.benchmark == "alfworld" and ev.split == "dev"
    assert ev.teacher_model == "demo_replay"
    assert ev.teacher_branch_outcomes[0] == {"kind": "aggregate", "metric": "success_rate",
                                             "value": 0.5, "k": 8, "n_success": 4}
    assert ev.student_branch_outcomes[0]["n_success"] == 1
    assert ev.provenance["teacher_command"] == "go to desk 1"
    assert ev.provenance["student_command"] == "look"
    assert ev.provenance["prefix_len"] == 1


def test_round_trip_json_and_file(tmp_path):
    evs = [from_bfcl_row(BFCL_ROW, TOK, source="x"), from_alfworld_row(ALF_ROW, TOK, source="y")]
    for ev in evs:
        d = json.loads(ev.to_json())
        assert list(d) == list(REQUIRED_FIELDS)
        assert UnifiedEvent.from_dict(d) == ev
        assert UnifiedEvent.from_json(ev.to_json()) == ev
    path = tmp_path / "events.jsonl"
    assert write_events(path, evs) == 2
    back = read_events(path)
    assert back == evs


def test_required_fields_enforced():
    d = from_bfcl_row(BFCL_ROW, TOK, source="x").to_dict()
    for name in ("state_hash", "provenance", "student_continuation", "state_text"):
        bad = dict(d)
        del bad[name]
        with pytest.raises(ValueError, match="missing required"):
            UnifiedEvent.from_dict(bad)
    with pytest.raises(ValueError, match="unknown fields"):
        UnifiedEvent.from_dict({**d, "y_plus": "no"})


def test_validate_rejects_bad_hash_and_enum():
    ev = from_bfcl_row(BFCL_ROW, TOK, source="x")
    ev.state_hash = "0" * 40
    with pytest.raises(ValueError, match="state_hash"):
        ev.validate()
    ev = from_bfcl_row(BFCL_ROW, TOK, source="x")
    ev.benchmark = "webarena"
    with pytest.raises(ValueError, match="benchmark"):
        ev.validate()


@pytest.mark.parametrize("path,conv", [
    ("data/bfcl_sft/pool_events_pref_v2.jsonl", from_bfcl_row),
    ("data/alf_sft/events_v1.jsonl", from_alfworld_row),
])
def test_real_rows_convert_if_present(path, conv):
    p = ROOT / path
    if not p.exists():
        pytest.skip(f"{path} not present")
    with open(p) as f:
        rows = [json.loads(next(f)) for _ in range(3)]
    for r in rows:
        ev = conv(r, TOK, source=path)
        assert ev.state_hash == hashlib.sha1(r["prompt"].encode()).hexdigest()
        assert ev.teacher_continuation == r["response"]
        assert ev.student_continuation == r["_rejected"]
