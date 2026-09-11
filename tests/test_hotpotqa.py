"""CPU protocol, data, serving, cache, accounting and pool regression tests."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import appworld_teacher
from bfas import hotpotqa as hp, ledger
from bfas.adapter import Demo
from bfas.adapters.hotpotqa import HotpotQAAdapter
from bfas.hotpotqa_budget import Budget, BudgetStopped, Limits, TEACHER_MAX_TOKENS, prompt_bound
from tools import hotpotqa_eval, hotpotqa_teacher_pool as pool
from scripts import setup_hotpotqa as setup

REAL_GENERATE_REPLY = appworld_teacher.generate_reply
REAL_LOAD_CONFIG = appworld_teacher.load_teacher_config
Q = {"_id": "train-a", "question": "Where is the fictional clock?", "answer": "Paris", "type": "bridge"}


class WikiStub:
    def reset(self):
        self.queries = []

    def search(self, entity):
        self.queries.append({"query": entity, "sha256": "stub"})
        return "The clock is in Paris."

    def lookup(self, keyword):
        return "(Result 1 / 1) The clock is in Paris."


class TokenizerStub:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        return "".join(f"<{m['role']}>{m['content']}<END>" for m in messages) + ("<GEN>" if add_generation_prompt else "")


class ClientStub:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("HotpotQA CPU tests must never contact a network or real model")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(appworld_teacher, "generate_reply", forbidden)
    monkeypatch.setattr(appworld_teacher, "load_teacher_config", forbidden)
    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "ledger")
    for name in ("BFAS_TEACHER", "BFAS_HOTPOTQA_TEACHER_POOL", "BFAS_HOTPOTQA_OFFLINE", "BFAS_OPENAI_SERVICE_TIER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BFAS_TEACHER_MIN_INTERVAL_S", "0")


@pytest.fixture
def adapter(monkeypatch):
    obj = HotpotQAAdapter()
    obj._tokenizer, obj._loaded_policy = TokenizerStub(), "stub"
    obj._questions["train"] = {Q["_id"]: Q}
    monkeypatch.setattr(obj, "_wiki", WikiStub)
    return obj


@pytest.fixture
def teacher(monkeypatch):
    state = SimpleNamespace(calls=[], replies=None, missing=False, lock=threading.Lock())
    monkeypatch.setattr(appworld_teacher, "load_teacher_config", lambda name:
                        SimpleNamespace(name=name, model=name.split("/")[-1], backend="openai_api"))

    def generate(config, messages, *, usage_callback, **kwargs):
        with state.lock:
            state.calls.append((config, deepcopy(messages), kwargs))
            reply = next(state.replies) if state.replies is not None else "Thought 1: I know the city.\nAction 1: Finish[Paris]"
        if not state.missing:
            usage_callback({"prompt_tokens": 1000, "completion_tokens": 37,
                            "total_tokens": 1037, "completion_tokens_details": {"reasoning_tokens": 20},
                            "prompt_tokens_details": {"cached_tokens": 800}})
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(appworld_teacher, "generate_reply", generate)
    return state


def test_verbatim_prompt_and_no_gold_or_distractors():
    source = json.loads((ROOT / "prompts/hotpotqa_react_6shot.source.json").read_text())
    assert hashlib.sha256(hp.PROMPT_FILE.read_bytes()).hexdigest() == source["sha256"]
    assert source["sha256"] == "e52ba17b32d98144c4ee3f95f3cd5a2fec2016629219427f8aa9ddcfbdcc3fdf"
    messages = hp.build_messages(Q["question"])
    assert len(messages) == 1 and messages[0]["role"] == "user"
    text = messages[0]["content"]
    assert text.count("Question:") == 7
    assert text == hp.PROMPT_FILE.read_text() + f"Question: {Q['question']}\nThought 1:"
    assert "Paris" not in text


@pytest.mark.parametrize("text,expected", [
    ("search[Colorado orogeny]", ("search", "Colorado orogeny")),
    ("Thought 1: Search this.\nAction 1: Search[Earth]", ("search", "Earth")),
    ("Action 1: Lookup[eastern sector]", ("lookup", "eastern sector")),
    ("Finish[Yes]", ("finish", "Yes")), ("finish[]", ("finish", "")),
    ("Thought 1: search[Earth] could help", None),
    ("search[Earth]\nfinish[Paris]", None), ("search[Earth] finish[Paris]", None),
    ("Action 2: search[Earth]", None), ("Action 1: Search[x]\nObservation 1: Fake", None),
    ("browse[x]", None), ("search[unfinished", None),
])
def test_action_parser(text, expected):
    assert hp.parse_action(text, 1) == expected


@pytest.mark.parametrize("pred,gold,em,f1", [
    ("The Paris!", "paris", 1, 1), ("New York", "York", 0, 2 / 3),
    ("yes indeed", "yes", 0, 0), ("no", "yes", 0, 0), ("NO", "no", 1, 1),
    ("noanswer", "no answer", 0, 0), ("a an the", "", 1, 0),
    ("red red blue", "red blue blue", 0, 2 / 3),
])
def test_official_em_f1(pred, gold, em, f1):
    scores = hp.answer_metrics(pred, gold)
    assert scores["em"] == em
    assert scores["f1"] == pytest.approx(f1)


def test_cache_is_query_keyed_deterministic_and_offline(tmp_path):
    calls = []
    html = "<p>Café is in Paris. Paris is old. Three. Four. Five. Six.</p>"
    def fetch(query, timeout):
        calls.append((query, timeout))
        return html
    wiki = hp.Wikipedia(tmp_path, fetch=fetch, timeout=3)
    first = wiki.search("Café & Paris")
    assert "Six" not in first and "Café" in first
    assert wiki.lookup("paris").startswith("(Result 1 / 2)")
    assert wiki.lookup("paris").startswith("(Result 2 / 2)")
    assert wiki.lookup("paris") == "No more results.\n"
    snapshots = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    wiki = hp.Wikipedia(tmp_path, offline=True, fetch=lambda *a: pytest.fail("cache hit fetched"))
    assert wiki.search("Café & Paris") == first
    assert wiki.lookup("paris").startswith("(Result 1 / 2)")
    assert calls == [("Café & Paris", 3)]
    assert snapshots == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    assert list(snapshots) == [hashlib.sha256("Café & Paris".encode()).hexdigest() + ".json"]
    with pytest.raises(hp.OfflineCacheMiss, match="not cached.*offline"):
        wiki.search("absent")


def test_cache_concurrent_first_writer_wins_and_checksum(tmp_path):
    calls = []
    def fetch(*args):
        calls.append(args)
        return f"<p>Snapshot version {len(calls)} is here.</p>"
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda _: hp.Wikipedia(tmp_path, fetch=fetch).search("same"), range(8)))
    assert len(set(values)) == 1 and len(calls) == 1
    path = next(tmp_path.glob("*.json"))
    record = json.loads(path.read_text())
    record["result"]["page"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(hp.WikiError, match="checksum"):
        hp.Wikipedia(tmp_path, offline=True).search("same")


def test_wiki_search_miss_retains_lookup_page_and_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(hp.time, "sleep", lambda _: None)
    calls = []
    def fetch(query, timeout):
        calls.append(query)
        if len(calls) == 1:
            raise TimeoutError("stub timeout")
        if query == "missing":
            return '<div class="mw-search-result-heading">Other page</div>'
        return "<p>The clock is in Paris. The clock is famous.</p>"
    wiki = hp.Wikipedia(tmp_path, fetch=fetch)
    wiki.search("clock")
    assert wiki.lookup("clock").startswith("(Result 1 / 2)")
    assert wiki.search("missing") == "Could not find missing. Similar: ['Other page']."
    assert wiki.lookup("clock").startswith("(Result 2 / 2)")
    assert calls == ["clock", "clock", "missing"]
    broken = hp.Wikipedia(tmp_path / "broken", retries=1, fetch=lambda *a: "<html>rate limit</html>")
    with pytest.raises(hp.WikiError, match="after 2 requests"):
        broken.search("bad")
    assert not list((tmp_path / "broken").glob("*.json"))


def test_fixed_split_ids_and_registration():
    from bfas.run import make_adapter, parse_args
    train, dev = hp.load_manifest("train"), hp.load_manifest("dev")
    assert len(train["ids"]) == len(set(train["ids"])) == 200
    assert len(dev["ids"]) == len(set(dev["ids"])) == 500
    assert not set(train["ids"]) & set(dev["ids"])
    assert len(train["demand"]) == 160 and len(train["calibration"]) == 40
    assert set(train["ids"]) == set(train["demand"]) | set(train["calibration"])
    assert not set(train["demand"]) & set(train["calibration"])
    assert parse_args(["--benchmark", "hotpotqa", "--arm", "base"]).benchmark == "hotpotqa"
    obj = make_adapter("hotpotqa", 2, 9123)
    assert obj.server_backed and obj.port == 9123
    assert obj.support_split().support == tuple(train["ids"])
    assert obj.support_split().as_dict()["seed"] == 0


def test_student_adapter_matches_evaluator_and_guidance(adapter, monkeypatch):
    replies = ["Thought 1: Find the clock.\nAction 1: Search[clock]", "Thought 2: It is Paris.\nAction 2: Finish[Paris]"]
    client = ClientStub(replies)
    monkeypatch.setattr(adapter, "_client", lambda: client)
    rollout = adapter.rollout("stub", [Q["_id"]], 0.0)[0]
    direct = hp.run_episode(Q, WikiStub(), hp.student_generator(ClientStub(replies), "bfas-policy"))
    assert rollout.raw == direct and rollout.verified
    for turn, call in zip(rollout.turns, client.calls):
        assert turn.context == call["messages"]
        assert turn.prompt == adapter.rerender({"_render_context": turn.context})
        assert call["max_tokens"] == 100 and call["temperature"] == 0
        assert turn.prompt.endswith(adapter.generation_suffix())
    demo = Demo(Q["_id"], [], "PRIVATE GUIDANCE", {"prompt_version": hp.PROMPT_VERSION})
    monkeypatch.setattr(adapter, "_client", lambda: ClientStub(["finish[Paris]"]))
    guided = adapter.rollout("stub", [Q["_id"]], 0.7, {Q["_id"]: demo})[0]
    assert "PRIVATE GUIDANCE" in guided.turns[0].prompt
    assert "PRIVATE GUIDANCE" not in guided.raw["deployment_turns"][0].prompt


def test_seven_steps_fallback_and_no_implicit_answer():
    client = ClientStub(["I think the answer is Paris", "nonsense"] * 7)
    record = hp.run_episode(Q, WikiStub(), hp.student_generator(client, "stub"))
    assert record["steps"] == 7 and record["model_calls"] == 14
    assert record["format_failures"] == 7 and record["em"] == 0 and not record["finished"]
    assert client.calls[1]["stop"] == ["\n"]
    assert client.calls[1]["messages"][0]["content"].endswith("\nAction 1:")


def test_teacher_ledger_exact_usage_failed_then_verified(adapter, teacher):
    teacher.replies = iter(["finish[Paris France]", "finish[Paris]"])
    demos = adapter.teacher_demo([Q["_id"]], 3)
    assert Q["_id"] in demos
    rows = ledger.read_records("hotpotqa")
    assert [r["verified"] for r in rows] == [False, True]
    assert [r["temperature"] for r in rows] == [0.0, 0.7]
    for row in rows:
        assert row["teacher"] == "openai/gpt-5.6-luna"
        assert row["tokens_spent"] == 37  # hidden reasoning already included
        assert row["usage"] == {"prompt_tokens": 1000, "completion_tokens": 37, "cached_tokens": 800}
        assert row["usage_status"] == "reported"
    assert all(c[2]["retries"] == 0 and c[2]["max_completion_tokens"] == TEACHER_MAX_TOKENS for c in teacher.calls)
    assert all(c[2]["stop"] is None for c in teacher.calls)  # Luna: local observation stop
    adapter.teacher_demo([Q["_id"]], 3)
    assert len(teacher.calls) == len(ledger.read_records("hotpotqa")) == 2


def test_teacher_preserves_usage_after_error_and_restricts_support(adapter, teacher):
    teacher.replies = iter(["Action 1: search[clock]", RuntimeError("after provider usage")])
    result = adapter.teacher_episode(Q["_id"], 0, 0.0)
    assert not result.verified and result.demo is None
    assert result.tokens_spent == 74 and result.usage["prompt_tokens"] == 2000
    assert len(result.response_texts) == 1
    with pytest.raises(ValueError, match="support"):
        adapter.teacher_episode("dev-id", 0, 0.0)
    with pytest.raises(ValueError, match="three"):
        adapter.teacher_episode(Q["_id"], 3, 0.0)


def test_missing_usage_is_unverified_and_explicitly_estimated(adapter, teacher):
    teacher.missing = True
    result = adapter.teacher_episode(Q["_id"], 0, 0.0)
    assert not result.verified and result.usage_status == "estimated"
    assert result.tokens_spent == TEACHER_MAX_TOKENS


def test_eval_records_metrics_resume_and_identity(monkeypatch, tmp_path):
    questions = [Q, dict(Q, _id="dev-b", answer="Paris France")]
    monkeypatch.setattr(hp, "load_questions", lambda split: questions)
    monkeypatch.setattr(hp, "Wikipedia", lambda **kwargs: WikiStub())
    client = ClientStub(["finish[Paris]", "finish[Paris]"])
    monkeypatch.setattr(hp, "make_client", lambda _: client)
    kwargs = dict(base_url="http://stub/v1", model="stub", start=0, n=2, out=tmp_path)
    metrics = hotpotqa_eval.evaluate(**kwargs)
    assert metrics["em"] == 0.5 and metrics["f1"] == pytest.approx(5 / 6)
    assert metrics["mean_steps"] == 1 and metrics["complete"]
    assert [json.loads(line)["task_id"] for line in (tmp_path / "records.jsonl").read_text().splitlines()] == ["train-a", "dev-b"]
    assert hotpotqa_eval.evaluate(**kwargs)["em"] == 0.5 and len(client.calls) == 2
    with pytest.raises(ValueError, match="identity"):
        hotpotqa_eval.evaluate(**dict(kwargs, model="other"))
    with pytest.raises(ValueError, match="first 500"):
        hotpotqa_eval.evaluate(**dict(kwargs, start=499))


def test_eval_offline_fails_clearly_and_keeps_question_pending(monkeypatch, tmp_path):
    monkeypatch.setattr(hp, "load_questions", lambda split: [Q])
    original = hp.Wikipedia
    monkeypatch.setattr(hp, "Wikipedia", lambda **kw: original(tmp_path / "cache", **kw))
    monkeypatch.setattr(hp, "make_client", lambda _: ClientStub(["search[missing]"]))
    with pytest.raises(hp.OfflineCacheMiss, match="not cached"):
        hotpotqa_eval.evaluate(base_url="stub", model="stub", out=tmp_path, n=1, offline=True)
    assert not json.loads((tmp_path / "metrics.json").read_text())["complete"]


@pytest.fixture
def small_pool(monkeypatch):
    questions = [dict(Q, _id=f"task-{i}") for i in range(4)]
    manifest = {"ids": [q["_id"] for q in questions], "seed": 0}
    monkeypatch.setattr(hp, "load_questions", lambda split: questions)
    monkeypatch.setattr(hp, "load_manifest", lambda split: manifest)
    monkeypatch.setattr(HotpotQAAdapter, "_wiki", lambda self: WikiStub())
    return questions


def test_pool_parallel_exact_caps_resume_import(small_pool, teacher, tmp_path, adapter):
    summary = pool.collect(tmp_path, workers=3)
    assert summary["verified"] == 4 and summary["attempts"] == 4 and summary["complete"]
    assert summary["tokens"] == 4 * 1037
    rows = ledger.read_records(tmp_path / "teacher_ledger.jsonl")
    assert len({(r["task_id"], r["attempt_index"]) for r in rows}) == 4
    assert all(r["usage_status"] == "reported" for r in rows)
    pool.collect(tmp_path, workers=2)
    assert len(teacher.calls) == 4
    adapter._import_pool(tmp_path)
    adapter._import_pool(tmp_path)
    assert len(ledger.read_records("hotpotqa")) == 4
    assert all(r["demo"]["turns"][0]["prompt"].endswith("<GEN>") for r in ledger.read_records("hotpotqa"))
    identity = json.loads((tmp_path / "identity.json").read_text())
    identity["teacher"] = "other"
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="resume"):
        pool.collect(tmp_path)


@pytest.mark.parametrize("limits", [Limits(max_tokens=0), Limits(max_usd=0)])
def test_pool_zero_cap_makes_no_calls_or_attempts(small_pool, teacher, tmp_path, limits):
    summary = pool.collect(tmp_path, workers=4, limits=limits)
    assert summary["tokens"] == summary["attempts"] == 0
    assert not summary["complete"] and not teacher.calls


def test_pool_unknown_usage_keeps_reservation_stops_resume(small_pool, teacher, tmp_path):
    teacher.missing = True
    summary = pool.collect(tmp_path, workers=1)
    assert summary["uncertain_calls"] == 1 and summary["verified"] == 0
    assert summary["tokens"] > TEACHER_MAX_TOKENS
    row = ledger.read_records(tmp_path / "teacher_ledger.jsonl")[0]
    assert row["usage_status"] == "estimated"
    pool.collect(tmp_path, workers=2)
    assert len(teacher.calls) == 1


@pytest.mark.parametrize("bound", ["tokens", "money"])
def test_pool_concurrent_positive_hard_cap(small_pool, teacher, tmp_path, bound):
    prompt = prompt_bound(hp.build_messages(Q["question"]))
    reservation = {"prompt_tokens": prompt, "completion_tokens": TEACHER_MAX_TOKENS, "cached_tokens": 0}
    reported = {"prompt_tokens": 1000, "completion_tokens": 37, "cached_tokens": 800}
    limits = (Limits(max_tokens=prompt + TEACHER_MAX_TOKENS + 1037) if bound == "tokens" else
              Limits(max_usd=Limits().cost(reservation) + Limits().cost(reported)))
    summary = pool.collect(tmp_path, workers=4, limits=limits)
    assert summary["verified"] == summary["attempts"] == len(teacher.calls) == 2
    assert not summary["complete"]
    # Check every durable intermediate reservation, not only final settled spend.
    current = {}
    for line in (tmp_path / "usage.jsonl").read_text().splitlines():
        row = json.loads(line)
        current[row["id"]] = row
        usage = {k: sum(r["usage"][k] for r in current.values()) for k in reported}
        assert usage["prompt_tokens"] + usage["completion_tokens"] <= limits.max_tokens
        assert limits.cost(usage) <= limits.max_usd


def test_teacher_pool_offline_miss_charged_and_stops(small_pool, teacher, tmp_path, monkeypatch):
    monkeypatch.setattr(HotpotQAAdapter, "_wiki", lambda self: hp.Wikipedia(tmp_path / "cache", offline=True))
    teacher.replies = iter(["search[uncached]"])
    summary = pool.collect(tmp_path / "pool", offline=True)
    assert summary["attempts"] == len(teacher.calls) == 1
    assert summary["tokens"] == 1037 and "not cached" in summary["stop_reason"]
    row = ledger.read_records(tmp_path / "pool/teacher_ledger.jsonl")[0]
    assert not row["verified"] and row["usage_status"] == "reported"


def test_budget_concurrent_reservations_and_crash_recovery(tmp_path, monkeypatch, small_pool, teacher):
    messages = hp.build_messages(Q["question"])
    envelope = prompt_bound(messages) + TEACHER_MAX_TOKENS
    budget = Budget(tmp_path / "journal.jsonl", Limits(max_tokens=envelope))
    call = budget.reserve("task-0", 0, messages)
    budget.finish(call)
    with pytest.raises(BudgetStopped):
        budget.reserve("task-1", 0, messages)
    reopened = Budget(tmp_path / "journal.jsonl", Limits(max_tokens=100000))
    assert reopened.stopped and reopened.usage()["completion_tokens"] == TEACHER_MAX_TOKENS
    out = tmp_path / "pool"
    out.mkdir()
    hp.write_json(out / "identity.json", pool.collection_identity(HotpotQAAdapter.teacher_name(), Limits()))
    (out / "usage.jsonl").write_bytes((tmp_path / "journal.jsonl").read_bytes())
    summary = pool.collect(out)
    assert summary["attempts"] == 1 and not teacher.calls


def test_setup_download_atomic_idempotent_and_order_validation(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    data = root / "envs/hotpotqa/data"
    (root / "configs").mkdir(parents=True)
    rows = [dict(Q, level="easy", context=[], supporting_facts=[])]
    ids = [Q["_id"]]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    (root / "configs/hotpotqa_support_split.json").write_text(json.dumps(
        {"source_ids_sha256": digest, "source_count": 1, "ids": ids}))
    monkeypatch.setattr(setup, "ROOT", root)
    monkeypatch.setattr(setup, "DATA", data)
    monkeypatch.setattr(setup, "SOURCES", {"train": ("hotpot_train_v1.1.json", 1)})
    calls = []
    def download(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(json.dumps(rows).encode())
    monkeypatch.setattr(setup.urllib.request, "urlopen", download)
    setup.main()
    before = {p.name: p.read_bytes() for p in data.iterdir()}
    setup.main()
    assert len(calls) == 1
    assert before == {p.name: p.read_bytes() for p in data.iterdir()}
    assert not list(data.glob("*.part"))
    assert all(p.parent == data for p in data.iterdir())
    rows[0]["_id"] = "wrong"
    (data / "hotpot_train_v1.1.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="order"):
        setup.main()


def test_shared_client_request_cap_and_retries_zero(monkeypatch):
    # Exercise the actual shared transport with a fake HTTP opener.
    monkeypatch.setattr(appworld_teacher, "generate_reply", REAL_GENERATE_REPLY)
    monkeypatch.setattr(appworld_teacher, "load_teacher_config", REAL_LOAD_CONFIG)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("network"))
    calls = []
    def open_request(request, timeout):
        calls.append(json.loads(request.data))
        raise TimeoutError("stub timeout")
    monkeypatch.setattr(appworld_teacher.urllib.request, "build_opener", lambda: SimpleNamespace(open=open_request))
    config = appworld_teacher.load_teacher_config("openai/gpt-5.6-luna")
    with pytest.raises(appworld_teacher.TeacherAPIError, match="after 1 attempts"):
        appworld_teacher.generate_reply(config, hp.build_messages("question"), retries=0, max_completion_tokens=91)
    assert len(calls) == 1 and calls[0]["max_completion_tokens"] == 91
