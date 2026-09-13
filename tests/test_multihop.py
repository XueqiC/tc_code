"""CPU-only local construction/evaluation; network, teacher, and training forbidden."""
from copy import deepcopy
import json
from pathlib import Path
import random
import socket
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from bfas import hotpotqa as hp, multihop as mh
from tools import hotpotqa_eval, multihop_prepare

GOLD = "GOLD_ANSWER_SENTINEL"
ALIAS = "ALIAS_SENTINEL"
FACT = "AUDIT_ONLY_SENTENCE_SENTINEL"
TITLE = "AUDIT_ONLY_TITLE_SENTINEL"
DECOMPOSITION = "DECOMPOSITION_ANSWER_SENTINEL"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU test attempted network, model, teacher, purchase, or training")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(hp, "make_client", forbidden)
    monkeypatch.setattr(hp.Wikipedia, "_fetch", forbidden)
    import appworld_teacher
    from bfas import ledger
    monkeypatch.setattr(appworld_teacher, "generate_reply", forbidden)
    monkeypatch.setattr(appworld_teacher, "load_teacher_config", forbidden)
    monkeypatch.setattr(ledger, "acquire_demos", forbidden)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("BFAS_HOTPOTQA_OFFLINE", raising=False)


def source_rows(dataset):
    if dataset == "2wiki":
        return [dict(_id=f"wiki-{i}", question=f"Public synthetic question {i}?", answer=GOLD,
                     type="compositional", supporting_facts=[[TITLE, 0]],
                     context=[[TITLE, [FACT]], ["Distractor", ["DISTRACTOR_SENTINEL"]]],
                     evidences=[[DECOMPOSITION]]) for i in range(4)]
    if dataset == "musique":
        return [dict(id=f"musique-{i}", question=f"Public synthetic question {i}?", answer=GOLD,
                     answer_aliases=[ALIAS, "Another alias"], answerable=i != 1,
                     paragraphs=[dict(idx=7, title=TITLE, paragraph_text=FACT, is_supporting=True),
                                 dict(idx=2, title="Distractor", paragraph_text="DISTRACTOR_SENTINEL", is_supporting=False)],
                     question_decomposition=[dict(question="Hidden question", answer=DECOMPOSITION)])
                for i in range(4)]
    return [dict(Question=f"Public synthetic question {i}?", Answer=GOLD) for i in range(4)]


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    data, manifests = tmp_path / "data", tmp_path / "configs"
    data.mkdir()
    for dataset, spec in mh.DATASETS.items():
        monkeypatch.setitem(mh.DATASETS, dataset, spec | dict(count=4))
        rows = source_rows(dataset)
        text = "\n".join(json.dumps(r) for r in rows) if dataset == "musique" else json.dumps(rows)
        (data / spec["source"]).write_text(text)
    multihop_prepare.prepare(data, manifests)
    return dict(data_dir=data, manifest_dir=manifests)


@pytest.mark.parametrize("dataset", mh.DATASETS)
def test_load_and_normalize_both_source_formats(inventory, dataset):
    questions, annotations, manifest = mh.load_dataset(dataset, **inventory)
    expected, expected_annotations = mh.normalize_rows(dataset, source_rows(dataset))
    expected_ids = random.Random(0).sample([q["_id"] for q in expected], 4)
    assert [q["_id"] for q in questions] == manifest["ids"] == expected_ids
    assert annotations == expected_annotations and manifest["excluded_rows"] == 0
    for q in questions:
        assert q["answer"] == GOLD
        assert not {"supporting_facts", "context", "paragraphs", "question_decomposition", "evidences"} & q.keys()
    if dataset == "musique":
        assert manifest["answerable_counts"] == {"true": 3, "false": 1}
        assert all(q["answer_aliases"] == [ALIAS, "Another alias"] for q in questions)
        assert all(a["supporting_paragraphs"] == [dict(title=TITLE, paragraph_index=7, text=FACT)]
                   for a in annotations.values())
    elif dataset == "2wiki":
        assert all(a["context"] == [[TITLE, [FACT]]] for a in annotations.values())
    else:
        assert set(annotations) == {f"bamboogle-{i:06d}" for i in range(4)}
        assert all(a == {"annotation_level": "none"} for a in annotations.values())


@pytest.mark.parametrize("dataset", mh.DATASETS)
def test_selection_is_frozen_prefix_without_filtering_or_replacement(inventory, dataset):
    first, _, small = mh.select_questions(dataset, 1, **inventory)
    larger, _, large = mh.select_questions(dataset, 3, **inventory)
    all_rows, _, full = mh.select_questions(dataset, **inventory)
    assert first == larger[:1] == all_rows[:1]
    assert larger == all_rows[:3] and len(all_rows) == 4
    assert small["task_ids"] == large["task_ids"][:1] == full["task_ids"][:1]
    assert small["seed"] == 0 and small["sample_size"] == 4
    assert "No filtering, replacement, or selection by student outcomes." in small["rule"]
    assert full["selected_answerable_counts"].get("false", 0) == (1 if dataset == "musique" else 0)
    for n in (0, -1, 5, True):
        with pytest.raises(ValueError, match="--n"):
            mh.select_questions(dataset, n, **inventory)


@pytest.mark.parametrize("mutation", ["source_answer", "source_order", "stored_order", "stored_seed"])
def test_inventory_drift_fails_before_inference(inventory, mutation):
    dataset = "bamboogle"
    source = inventory["data_dir"] / mh.DATASETS[dataset]["source"]
    manifest = inventory["manifest_dir"] / mh.manifest_name(dataset)
    if mutation.startswith("source"):
        rows = mh.read_rows(source)
        if mutation == "source_order":
            rows.reverse()
        else:
            rows[0]["Answer"] = "changed"
        source.write_text(json.dumps(rows))
    else:
        value = json.loads(manifest.read_text())
        if mutation == "stored_order":
            value["ids"].reverse()
        else:
            value["seed"] = 1
        manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frozen"):
        mh.select_questions(dataset, **inventory)


def test_construction_read_only_idempotent_and_rejects_replacement(inventory):
    paths = list(inventory["data_dir"].glob("*")) + list(inventory["manifest_dir"].glob("*"))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    multihop_prepare.prepare(**inventory)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    with pytest.raises(ValueError, match="outside source data"):
        multihop_prepare.prepare(inventory["data_dir"], inventory["data_dir"] / "nested")
    manifest = inventory["manifest_dir"] / mh.manifest_name("musique")
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="refusing to replace"):
        multihop_prepare.prepare(**inventory)


@pytest.mark.parametrize("field,value", [("answerable", None), ("answerable", "false"),
                                         ("answer_aliases", None), ("answer_aliases", [1])])
def test_invalid_musique_rows_fail_instead_of_being_dropped(field, value):
    rows = source_rows("musique")
    rows[1][field] = value
    with pytest.raises(ValueError):
        mh.normalize_rows("musique", rows)


def test_duplicate_ids_fail_instead_of_replacement():
    rows = source_rows("2wiki")
    rows[1]["_id"] = rows[0]["_id"]
    with pytest.raises(ValueError, match="duplicate"):
        mh.normalize_rows("2wiki", rows)


@pytest.mark.parametrize("prediction,em,f1", [(ALIAS, 1, 1), ("Another", 0, 2/3),
    (GOLD, 1, 1), ("nonsense", 0, 0)])
def test_musique_uses_maximum_over_canonical_and_aliases(prediction, em, f1):
    q, _ = mh.normalize_row("musique", source_rows("musique")[0], 0)
    assert hp.score_answer(prediction, q) == pytest.approx(dict(em=em, f1=f1))


@pytest.mark.parametrize("prediction", hp.ABSTENTION_ANSWERS)
def test_unanswerable_requires_explicit_whole_answer_abstention(prediction):
    q, _ = mh.normalize_row("musique", source_rows("musique")[1], 1)
    assert q["answerable"] is False and q["answer"] == GOLD
    assert hp.score_answer(prediction.upper() + ".", q) == dict(em=1, f1=1)
    for wrong in (GOLD, ALIAS, "", "answer", prediction + ", but the answer is Paris"):
        assert hp.score_answer(wrong, q) == dict(em=0, f1=0)


def test_scoring_seam_keeps_unit_mismatch_and_hotpotqa_special_answers():
    assert hp.score_answer("2240 feet", dict(answer="2240")) == pytest.approx(dict(em=0, f1=2/3))
    assert hp.score_answer("yes indeed", dict(answer="yes")) == dict(em=0, f1=0)


class Wiki:
    def preflight(self):
        pass

    def reset(self):
        self.queries = []

    def search(self, query):
        self.queries.append(dict(query=query, sha256="stub"))
        return "Only independently retrieved public text."

    lookup = search


class Client:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        reply = next(self.replies)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.mark.parametrize("dataset,status", [("2wiki", "available"),
    ("musique", "coarser_than_sentence_level"), ("bamboogle", "unavailable")])
def test_shared_evaluation_prompt_decoding_audit_and_resume(inventory, monkeypatch, tmp_path, dataset, status):
    questions, _, _ = mh.select_questions(dataset, **inventory)
    replies = []
    for q in questions:
        answer = ("cannot be answered" if q.get("answerable") is False
                  else ALIAS if dataset == "musique" else "not the gold")
        # Includes the existing thought/action fallback and multiple steps.
        replies.extend(["Plan a query", "search[public query]", "lookup[public]", f"finish[{answer}]"])
    client = Client(replies)
    monkeypatch.setattr(hp, "make_client", lambda _: client)
    monkeypatch.setattr(hp, "Wikipedia", lambda **kw: Wiki())
    kwargs = dict(base_url="http://localhost:8900/v1", model="student", dataset=dataset,
                  out=tmp_path / "eval", **inventory)
    metrics = hotpotqa_eval.evaluate(**kwargs)
    assert metrics["complete"] and metrics["n"] == 4
    assert metrics["annotation_status_counts"] == {status: 4}
    cfg = metrics["config"]
    assert cfg["max_steps"] == hp.MAX_STEPS == 7 and cfg["max_tokens"] == hp.MAX_TOKENS == 100
    assert cfg["temperature"] == 0 and cfg["prompt_version"] == hp.PROMPT_VERSION
    assert cfg["offline"] is False and cfg["selection"]["seed"] == 0
    assert metrics["em"] == metrics["f1"] == (1 if dataset == "musique" else 0)
    saved = [json.loads(line) for line in (kwargs["out"] / "records.jsonl").read_text().splitlines()]
    assert [r["task_id"] for r in saved] == [q["_id"] for q in questions]
    assert all(r["steps"] == 3 and r["model_calls"] == 4 for r in saved)
    assert all(s["annotation_status"] == status for r in saved for s in r["step_audit"])
    for call in client.calls:
        assert call["temperature"] == 0 and call["max_tokens"] == 100
        assert call["messages"][0]["content"].startswith(hp.PROMPT_FILE.read_text())
        assert call["stop"] in (["\nObservation 1:"], ["\n"], ["\nObservation 2:"], ["\nObservation 3:"])
    prompts = json.dumps(client.calls)
    for secret in (GOLD, ALIAS, FACT, TITLE, DECOMPOSITION, "DISTRACTOR_SENTINEL", "supporting_facts", "answerable"):
        assert secret not in prompts
    before = len(client.calls)
    assert hotpotqa_eval.evaluate(**kwargs) == metrics and len(client.calls) == before
    with pytest.raises(ValueError, match="first n stored"):
        hotpotqa_eval.evaluate(**(kwargs | dict(start=1, n=1)))
    with pytest.raises(ValueError, match="identity"):
        hotpotqa_eval.evaluate(**(kwargs | dict(n=2)))


@pytest.mark.parametrize("dataset", mh.DATASETS)
def test_same_shared_seven_step_limit_for_every_dataset(dataset):
    q, _ = mh.normalize_row(dataset, source_rows(dataset)[0], 0)
    record = hp.run_episode(q, Wiki(), lambda *a: "search[public]")
    assert record["steps"] == record["model_calls"] == 7
    assert record["termination_reason"] == "step_limit"
    assert not record["finished"] and record["em"] == record["f1"] == 0


def test_cli_dispatches_new_dataset_without_a_second_harness(inventory, monkeypatch, tmp_path, capsys):
    client = Client(["finish[wrong]"])
    monkeypatch.setattr(hp, "make_client", lambda _: client)
    monkeypatch.setattr(hp, "Wikipedia", lambda **kw: Wiki())
    hotpotqa_eval.main(["--dataset", "bamboogle", "--base-url", "http://localhost:8900/v1",
        "--model", "student", "--n", "1", "--out", str(tmp_path / "eval"),
        "--data-dir", str(inventory["data_dir"]), "--manifest-dir", str(inventory["manifest_dir"])])
    assert json.loads(capsys.readouterr().out)["complete"] is True
