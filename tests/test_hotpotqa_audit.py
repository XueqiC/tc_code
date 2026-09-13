"""CPU-only shared lexical audit; annotations never affect the episode stream."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from bfas import hotpotqa as hp, hotpotqa_audit as audit, multihop as mh

FIRST = "The Lumen Dial was created by Mara Voss."
SECOND = "Mara Voss was born in Bellhaven."
Q = dict(_id="synthetic", question="Where was the creator born?", answer="GOLD_SECRET", type="bridge")
ANNOTATIONS = dict(annotation_level="sentence", supporting_facts=[["Lumen Dial", 0], ["Mara Voss", 0]],
                   context=[["Lumen Dial", [FIRST]], ["Mara Voss", [SECOND]]])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("audit test attempted a network or real model call")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(hp, "make_client", forbidden)
    monkeypatch.setattr(hp.Wikipedia, "_fetch", forbidden)


class Wiki:
    def reset(self):
        self.queries = []

    def search(self, query):
        return FIRST if query == "Lumen Dial" else SECOND

    lookup = search


def run(annotations=ANNOTATIONS, actions=None):
    replies = iter(actions or ["search[Lumen Dial]", "search[Mara Voss]", "finish[wrong]"])
    prompts = []
    def generate(messages, *args):
        prompts.append(deepcopy(messages))
        return next(replies)
    record = hp.run_episode(Q, Wiki(), generate)
    original = deepcopy(record)
    audit.annotate_episode(record, annotations, hp.parse_action)
    assert original == {k: v for k, v in record.items() if k in original}
    return record, prompts


@pytest.mark.parametrize("text,needle,expected", [
    ("(Result 1 / 1) MARA—Voss\nwas born.", "Mara Voss was born.", ["MARA—Voss\nwas born"]),
    ("🌙 Straße: A! STRASSE a.", "straße a", ["Straße: A", "STRASSE a"]),
    ("a a a", "A a", ["a a", "a a"]), ("Mara Vossen", "Mara Voss", []),
    ("Mara talented Voss", "Mara Voss", []), ("M. Voss", "Mara Voss", []),
    ("Anything", "...", []),
])
def test_original_unicode_matching_and_offsets(text, needle, expected):
    spans = audit.matching_spans(text, needle)
    assert [text[s["start"]:s["end"]] for s in spans] == expected


def test_hotpotqa_and_2wiki_sentence_coverage_and_title_reuse():
    record, _ = run()
    first, second, end = record["step_audit"]
    assert record["supporting_facts_audit"]["status"] == "available"
    assert first["supporting_fact_matches"][0] == dict(fact_index=0, title="Lumen Dial", sentence_index=0,
        status="matched", spans=[dict(start=0, end=len(FIRST)-1)])
    assert [s["all_supporting_facts_seen_before_step"] for s in record["step_audit"]] == [False, False, True]
    assert all(s["supporting_paragraph_matches"] is None for s in record["step_audit"])
    overlap = second["query_previous_observation_overlap"]
    assert overlap["shared_strings"] == ["Mara", "Voss"] and overlap["count"] == 2
    assert next(t for t in overlap["supporting_title_reuse"] if t["title"] == "Mara Voss")["reused"] is True
    assert end["step_end_reason"] == record["episode_end_reason"] == "finish_called"


def test_musique_coarse_paragraph_coverage_never_fabricates_sentence_indices():
    annotations = dict(annotation_level="paragraph", supporting_paragraphs=[
        dict(title="Lumen Dial", paragraph_index=7, text=FIRST),
        dict(title="Mara Voss", paragraph_index=3, text=SECOND)])
    record, _ = run(annotations)
    catalog = record["supporting_facts_audit"]
    assert catalog["status"] == "coarser_than_sentence_level" and catalog["annotation_complete"]
    for step in record["step_audit"]:
        assert step["annotation_status"] == "coarser_than_sentence_level"
        assert step["supporting_fact_matches"] is step["all_supporting_facts_seen_before_step"] is None
        assert all("sentence_index" not in m for m in step["supporting_paragraph_matches"])
    assert [s["all_supporting_paragraphs_seen_before_step"] for s in record["step_audit"]] == [False, False, True]
    # A single sentence from a longer paragraph cannot certify whole-paragraph exposure.
    annotations["supporting_paragraphs"][0]["text"] += " An extra sentence."
    record, _ = run(annotations)
    assert record["step_audit"][0]["supporting_paragraph_matches"][0]["status"] == "no_match"
    assert record["step_audit"][-1]["all_supporting_paragraphs_seen_before_step"] is False


@pytest.mark.parametrize("annotations", [{"annotation_level": "none"}, {},
    dict(annotation_level="sentence", supporting_facts=[]),
    dict(annotation_level="paragraph", supporting_paragraphs=[]),
    dict(annotation_level="sentence", supporting_facts=[["missing", 0]], context=[]),
    dict(annotation_level="paragraph", supporting_paragraphs=[dict(title="empty", paragraph_index=0, text="")])])
def test_absent_annotation_is_unknown_never_zero_or_vacuously_complete(annotations):
    record, _ = run(annotations)
    assert record["supporting_facts_audit"]["status"] == "unavailable"
    for step in record["step_audit"]:
        assert step["annotation_status"] == "unavailable"
        assert step["all_supporting_facts_seen_before_step"] is None
        assert step["all_supporting_paragraphs_seen_before_step"] is None
    overlap = record["step_audit"][1]["query_previous_observation_overlap"]
    assert overlap["shared_strings"] == ["Mara", "Voss"] and overlap["count"] == 2
    if not record["supporting_facts_audit"]["facts"]:
        assert overlap["supporting_title_reuse"] is None


def test_partial_annotation_keeps_completeness_unknown():
    annotations = deepcopy(ANNOTATIONS)
    annotations["supporting_facts"].append(["missing", 10])
    record, _ = run(annotations)
    assert not record["supporting_facts_audit"]["annotation_complete"]
    assert record["step_audit"][0]["supporting_fact_matches"][-1]["status"] == "unavailable"
    assert all(s["all_supporting_facts_seen_before_step"] is None for s in record["step_audit"])


def test_post_episode_audit_cannot_change_prompts_actions_or_scores():
    annotated = deepcopy(ANNOTATIONS)
    annotated["context"].append(["AUDIT_TITLE_SECRET", ["AUDIT_SENTENCE_SECRET"]])
    annotated["supporting_facts"].append(["AUDIT_TITLE_SECRET", 0])
    baseline, baseline_prompts = run({})
    record, prompts = run(annotated)
    assert prompts == baseline_prompts
    assert record["history"] == baseline["history"] and record["em"] == baseline["em"]
    assert "AUDIT_SENTENCE_SECRET" in json.dumps(record["supporting_facts_audit"])
    for secret in ("AUDIT_TITLE_SECRET", "AUDIT_SENTENCE_SECRET", Q["answer"]):
        assert secret not in json.dumps(prompts)


@pytest.mark.parametrize("actions,reason", [
    (["search[Lumen Dial]"] * 7, "step_limit_reached"),
    (["invalid"] * 14, "invalid_action"),
    (["finish[wrong]"], "finish_called"),
])
def test_annotation_independent_termination_reasons(actions, reason):
    record, _ = run(dict(annotation_level="none"), actions)
    assert record["step_audit"][-1]["step_end_reason"] == reason


def test_hotpotqa_loader_reads_annotations_separately_and_reports_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(hp, "load_manifest", lambda split: {"source": "dev.json"})
    source = tmp_path / "dev.json"
    missing = mh.hotpotqa_annotations([Q], data_dir=tmp_path)
    assert "audit_error" in missing[Q["_id"]]
    source.write_text(json.dumps([Q | ANNOTATIONS]))
    assert mh.hotpotqa_annotations([Q], data_dir=tmp_path) == {Q["_id"]: ANNOTATIONS}
    assert mh.hotpotqa_annotations([Q | dict(answer="different")], data_dir=tmp_path) == {}
