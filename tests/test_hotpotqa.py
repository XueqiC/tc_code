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
from bfas.hotpotqa_budget import Budget, BudgetStopped, Limits, REQUEST_ATTEMPTS, TEACHER_MAX_TOKENS, prompt_bound
from tools import hotpotqa_eval, hotpotqa_teacher_pool as pool
from scripts import setup_hotpotqa as setup

REAL_GENERATE_REPLY = appworld_teacher.generate_reply
REAL_LOAD_CONFIG = appworld_teacher.load_teacher_config
Q = {"_id": "train-a", "question": "Where is the fictional clock?", "answer": "Paris", "type": "bridge"}


class WikiStub:
    def preflight(self):
        pass

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
    for name in ("BFAS_TEACHER", "BFAS_HOTPOTQA_TEACHER_POOL", "BFAS_HOTPOTQA_OFFLINE",
                 "BFAS_OPENAI_SERVICE_TIER", "OPENAI_API_KEY", "BFAS_STUDENT_API_KEY"):
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


def test_cache_is_query_keyed_deterministic_and_offline(tmp_path, monkeypatch):
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
    monkeypatch.setattr(hp, 'file_lock', lambda *a: pytest.fail('offline replay must not write cache locks'))
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


@pytest.mark.parametrize("base_url,openai_key,student_key,expected", [
    ("https://api.openai.com/v1", "openai-test", None, "openai-test"),
    ("https://api.openai.com/v1", "openai-test", "student-test", "openai-test"),
    ("http://127.0.0.1:8900/v1", "openai-test", None, "openai-test"),
    ("http://localhost:8900/v1", None, "student-test", "student-test"),
    ("http://localhost:8900/v1", None, None, "EMPTY"),
    ("http://127.0.0.1:8900/v1", "", "", "EMPTY"),
    ("http://[::1]:8900/v1", None, None, "EMPTY"),
    ("http://192.168.1.2:8900/v1", None, None, "EMPTY"),
])
def test_client_credentials(monkeypatch, base_url, openai_key, student_key, expected):
    for name, value in (("OPENAI_API_KEY", openai_key), ("BFAS_STUDENT_API_KEY", student_key)):
        if value is not None:
            monkeypatch.setenv(name, value)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: kwargs))
    assert hp.make_client(base_url) == {"base_url": base_url, "api_key": expected,
                                       "timeout": 120, "max_retries": 0}


@pytest.mark.parametrize("base_url", ["https://api.openai.com/v1", "https://inference.example.com/v1"])
def test_remote_client_requires_a_key(monkeypatch, base_url):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw: pytest.fail("missing key")))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        hp.make_client(base_url)


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-luna-2026-09-01"])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("tier", [None, "", " flex ", "priority"])
def test_luna_requests_and_local_stop(monkeypatch, model, temperature, tier):
    if tier is not None:
        monkeypatch.setenv("BFAS_OPENAI_SERVICE_TIER", tier)
    client = ClientStub(["Thought 1: City.\nAction 1: Finish[Paris]\nObservation 1: fabricated"])
    messages = hp.build_messages(Q["question"])
    reply = hp.student_generator(client, model)(messages, ["\nObservation 1:"], temperature)
    assert reply == "Thought 1: City.\nAction 1: Finish[Paris]"
    expected = {"model": model, "messages": messages, "max_completion_tokens": hp.MAX_TOKENS}
    if tier and tier.strip():
        expected["service_tier"] = tier.strip()
    assert client.calls == [expected]  # No temperature, max_tokens, or unsupported stop.


def test_luna_rejects_invalid_service_tier_and_preserves_local_decoding(monkeypatch):
    monkeypatch.setenv("BFAS_OPENAI_SERVICE_TIER", "invalid")
    client = ClientStub(["finish[Paris]"])
    with pytest.raises(ValueError, match="BFAS_OPENAI_SERVICE_TIER"):
        hp.student_generator(client, "gpt-5.6-luna")
    assert not client.calls
    messages = hp.build_messages(Q["question"])
    hp.student_generator(client, "bfas-policy")(messages, ["\n"], 0.7)
    assert client.calls == [{"model": "bfas-policy", "messages": messages, "temperature": 0.7,
                             "max_tokens": hp.MAX_TOKENS, "stop": ["\n"]}]


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
    assert result.raw["termination_reason"] == "error"
    assert result.raw["history"][0]["action"] == "search[clock]"
    assert "after provider usage" in result.raw["history"][1]["observation"]
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
    monkeypatch.setattr(hp, "make_client", lambda _: pytest.fail("preflight must precede the model client"))
    with pytest.raises(hp.OfflineCacheMiss, match="preflight.*no cached"):
        hotpotqa_eval.evaluate(base_url="stub", model="stub", out=tmp_path, n=1, offline=True)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["status"] == "failed" and not metrics["complete"]
    assert metrics["n"] == 0 and metrics["em"] is None


@pytest.mark.parametrize("offline", [None, False, True])
def test_eval_live_default_and_explicit_offline_replay(monkeypatch, tmp_path, offline):
    original = hp.Wikipedia
    cache = tmp_path / "cache"
    html = "<p>The fictional clock is in Paris.</p>"
    if offline:
        original(cache, fetch=lambda *a: html).search("clock")
    calls = []
    def fetch(query, timeout):
        assert not offline, "offline replay must not fetch"
        calls.append(query)
        return html
    monkeypatch.setattr(hp, "Wikipedia", lambda **kw: original(cache, fetch=fetch, **kw))
    monkeypatch.setattr(hp, "load_questions", lambda split: [Q])
    client = ClientStub(["search[clock]", "finish[Paris]"])
    monkeypatch.setattr(hp, "make_client", lambda _: client)
    kwargs = {} if offline is None else {"offline": offline}
    metrics = hotpotqa_eval.evaluate(base_url="stub", model="stub", n=1, out=tmp_path / "eval", **kwargs)
    assert metrics["status"] == "complete" and metrics["em"] == 1
    assert metrics["offline"] is bool(offline)
    assert calls == ([] if offline else ["Albert Einstein", "clock"])
    # A completed resume does not require connectivity or inference.
    assert hotpotqa_eval.evaluate(base_url="stub", model="stub", n=1, out=tmp_path / "eval", **kwargs) == metrics
    assert len(client.calls) == 2


@pytest.mark.parametrize("env", [None, "0", "1"])
def test_import_paths_preserve_caller_retrieval_mode(monkeypatch, env):
    import importlib
    if env is not None:
        monkeypatch.setenv("BFAS_HOTPOTQA_OFFLINE", env)
    modules = ["bfas.rtd.benchmarks.adapter_evaluation", "bfas.rtd.benchmarks.hotpotqa_evaluation",
               "bfas.rtd.benchmarks.registry", "bfas.rtd.baselines.paper_evaluation",
               "bfas.rtd.evaluation", "tools.hotpotqa_eval", "bfas.adapters.hotpotqa"]
    for name in modules:
        importlib.import_module(name)
        assert HotpotQAAdapter().offline is (env == "1")
    from bfas.rtd.benchmarks.adapter_evaluation import frozen_environment
    from bfas.rtd.baselines.paper_evaluation import protocol
    from bfas.rtd.benchmarks.registry import get_benchmark
    assert get_benchmark({"benchmark": "hotpotqa"}).official_evaluation.__module__ == "bfas.rtd.benchmarks.adapter_evaluation"
    with frozen_environment("hotpotqa"), frozen_environment("hotpotqa"):
        assert HotpotQAAdapter().offline is (env == "1")
        assert HotpotQAAdapter(offline=False).offline is False
        assert protocol("hotpotqa")["offline"] is (env == "1")
    import os
    assert os.environ.get("BFAS_HOTPOTQA_OFFLINE") == env


@pytest.mark.parametrize("failure", ["network", "html", "cache_write"])
def test_live_preflight_bypasses_warm_cache_before_client(monkeypatch, tmp_path, failure):
    original = hp.Wikipedia
    cache = tmp_path / "cache"
    original(cache, fetch=lambda *a: "<p>This is a cached article.</p>").search("Albert Einstein")
    def fetch(*args):
        if failure == "network":
            raise TimeoutError("no outbound connection")
        return "<html>blocked</html>"
    if failure == "cache_write":
        def unwritable(*args, **kwargs):
            raise PermissionError("cache is read-only")
        monkeypatch.setattr(hp.tempfile, "TemporaryFile", unwritable)
    monkeypatch.setattr(hp, "Wikipedia", lambda **kw: original(cache, fetch=fetch, **kw))
    monkeypatch.setattr(hp, "load_questions", lambda split: [Q])
    monkeypatch.setattr(hp, "make_client", lambda _: pytest.fail("model client opened before preflight"))
    with pytest.raises(hp.WikiError, match="preflight failed.*live retrieval"):
        hotpotqa_eval.evaluate(base_url="stub", model="stub", n=1, out=tmp_path / "eval")
    metrics = json.loads((tmp_path / "eval/metrics.json").read_text())
    assert metrics["status"] == "failed" and metrics["n"] == 0
    assert not metrics["complete"] and metrics["headline"] is None


def test_offline_preflight_validates_snapshots_without_writing(monkeypatch, tmp_path):
    wiki = hp.Wikipedia(tmp_path, fetch=lambda *a: "<p>This is an article.</p>")
    wiki.search("clock")
    wiki.offline = True
    def forbidden(*args, **kwargs):
        pytest.fail("offline preflight must not write or fetch")
    monkeypatch.setattr(wiki, "fetch", forbidden)
    monkeypatch.setattr(hp, "file_lock", forbidden)
    wiki.preflight()
    assert wiki.queries == []
    path = next(tmp_path.glob("*.json"))
    record = json.loads(path.read_text())
    record["sha256"] = "corrupt"
    path.write_text(json.dumps(record))
    with pytest.raises(hp.WikiError, match="preflight.*checksum"):
        wiki.preflight()


@pytest.mark.parametrize("failure", ["offline_miss", "episode_error", "interrupt", "short_inventory"])
def test_incomplete_eval_has_failure_artifact_and_no_result_scores(monkeypatch, tmp_path, failure):
    questions = [dict(Q, _id=f"dev-{i}") for i in range(500)]
    monkeypatch.setattr(hp, "load_questions", lambda split: questions[:1] if failure == "short_inventory" else questions)
    original = hp.Wikipedia
    cache = tmp_path / "cache"
    original(cache, fetch=lambda *a: "<p>This cached article exists.</p>").search("known")
    monkeypatch.setattr(hp, "Wikipedia", lambda **kw: original(cache, **kw))
    reply = {"offline_miss": "search[uncached student query]", "episode_error": RuntimeError("backend failed"),
             "interrupt": KeyboardInterrupt(), "short_inventory": "finish[Paris]"}[failure]
    client = ClientStub(["finish[Paris]", reply])
    if failure == "interrupt":
        original_create = client.create
        def create(**kwargs):
            if client.calls:
                raise KeyboardInterrupt()
            return original_create(**kwargs)
        monkeypatch.setattr(client, "create", create)
    monkeypatch.setattr(hp, "make_client", lambda _: client)
    error = {"offline_miss": hp.OfflineCacheMiss, "episode_error": hotpotqa_eval.EvaluationFailed,
             "interrupt": KeyboardInterrupt, "short_inventory": hotpotqa_eval.EvaluationFailed}[failure]
    with pytest.raises(error):
        hotpotqa_eval.evaluate(base_url="stub", model="stub", n=500, out=tmp_path / "eval", offline=True)
    metrics = json.loads((tmp_path / "eval/metrics.json").read_text())
    assert metrics["requested_n"] == 500 and metrics["status"] == "failed" and not metrics["complete"]
    assert metrics["n"] == (0 if failure == "short_inventory" else 2 if failure == "episode_error" else 1)
    assert metrics["error"]
    assert all(metrics[k] is None for k in ("em", "f1", "headline", "mean_score", "per_category"))
    if failure in ("offline_miss", "interrupt"):
        assert metrics["partial_metrics"]["em"] == 1


def test_adapter_preflight_precedes_renderer(monkeypatch, tmp_path):
    adapter = HotpotQAAdapter(offline=True)
    original = hp.Wikipedia
    monkeypatch.setattr(hp, "Wikipedia", lambda **kw: original(tmp_path / "missing", **kw))
    monkeypatch.setattr(adapter, "prepare_renderer", lambda *a: pytest.fail("renderer cost paid"))
    with pytest.raises(hp.OfflineCacheMiss, match="preflight"):
        adapter.evaluate("policy", tmp_path / "eval")


def test_eval_openai_luna_uses_exported_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "exported-test-key")
    monkeypatch.setenv("BFAS_OPENAI_SERVICE_TIER", "flex")
    monkeypatch.setattr(hp, "load_questions", lambda split: [Q])
    monkeypatch.setattr(hp, "Wikipedia", lambda **kw: WikiStub())
    client = ClientStub(["finish[Paris]"])
    def create_client(**kwargs):
        assert kwargs["api_key"] == "exported-test-key"
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        return client
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=create_client))
    metrics = hotpotqa_eval.evaluate(base_url="https://api.openai.com/v1", model="gpt-5.6-luna",
                                    out=tmp_path, n=1)
    assert metrics["complete"] and metrics["em"] == 1
    assert metrics["config"]["temperature"] is None
    assert metrics["config"]["service_tier"] == client.calls[0]["service_tier"] == "flex"
    assert "temperature" not in client.calls[0] and "max_tokens" not in client.calls[0]


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
    attempts = tmp_path / "attempts.jsonl"
    before = attempts.read_bytes()
    audits = [json.loads(line) for line in before.splitlines()]
    assert len({(r["task_id"], r["attempt_index"]) for r in audits}) == 4
    assert all(r["verified"] and r["trajectory_available"] and r["tokens"] == 1037 for r in audits)
    pool.collect(tmp_path, workers=2)
    assert len(teacher.calls) == 4 and attempts.read_bytes() == before
    adapter._import_pool(tmp_path)
    adapter._import_pool(tmp_path)
    assert len(ledger.read_records("hotpotqa")) == 4
    assert all(r["demo"]["turns"][0]["prompt"].endswith("<GEN>") for r in ledger.read_records("hotpotqa"))
    identity = json.loads((tmp_path / "identity.json").read_text())
    identity["teacher"] = "other"
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="resume"):
        pool.collect(tmp_path)


def test_pool_archives_failed_and_verified_trajectories(small_pool, teacher, tmp_path):
    replies = ["Thought 1: Find the clock.\nAction 1: Search[clock]",
               "Thought 2: Check the city.\nAction 2: Lookup[city]",
               "Thought 3: Name both.\nAction 3: Finish[Paris France]", "finish[Paris]"]
    teacher.replies = iter(replies * len(small_pool))
    summary = pool.collect(tmp_path)
    path = tmp_path / "attempts.jsonl"
    before = path.read_bytes()
    rows = [json.loads(line) for line in before.splitlines()]
    assert len(rows) == summary["attempts"] == 8
    for failed, verified in zip(rows[::2], rows[1::2]):
        assert failed["task_id"] == verified["task_id"]
        assert (failed["attempt_index"], verified["attempt_index"]) == (0, 1)
        assert failed["question"] == Q["question"] and failed["gold"] == Q["answer"]
        assert failed["prediction"] == "Paris France" and failed["termination_reason"] == "finish"
        assert not failed["verified"] and failed["em"] == 0 and failed["f1"] == pytest.approx(2 / 3)
        assert failed["history"][0]["thought"] == "Find the clock."
        assert failed["history"][1]["action"] == "lookup[city]"
        assert "Paris" in failed["history"][1]["observation"]
        assert failed["steps"] == len(failed["history"]) == 3
        assert failed["response_texts"] == replies[:3]
        assert failed["tokens"] == 3111 and failed["tokens_spent"] == 111
        assert failed["usage"] == {"prompt_tokens": 3000, "completion_tokens": 111, "cached_tokens": 2400}
        assert failed["usage_status"] == "reported" and failed["trajectory_available"]
        assert verified["verified"] and verified["prediction"] == "Paris"
    demos = json.loads((tmp_path / "demos.json").read_text())["demos"]
    assert len(demos) == 4 and all("Paris France" not in demo["worked_example"] for demo in demos.values())
    pool.collect(tmp_path)
    assert len(teacher.calls) == 16 and path.read_bytes() == before


@pytest.mark.parametrize("replies,reason,last_thought", [
    (["Thought 1: Search first.\nAction 1: Search[clock]", RuntimeError("provider failed")], "error", ""),
    (["Thought 1: Need an action.", RuntimeError("fallback failed")], "error", "Need an action."),
    (["No action", "still invalid"] * hp.MAX_STEPS, "step_limit", "No action"),
])
def test_pool_archives_errors_and_step_limits(small_pool, teacher, tmp_path, replies, reason, last_thought):
    teacher.replies = iter((replies + ["finish[Paris]"]) * len(small_pool))
    pool.collect(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    for row in rows[::2]:
        assert not row["verified"] and row["termination_reason"] == reason
        assert row["prediction"] == "" and row["em"] == row["f1"] == 0
        assert row["history"][-1]["thought"] == last_thought
        assert row["history"][-1]["observation"]
        assert row["tokens"] == len(replies) * 1037
        assert row["response_texts"] == [r for r in replies if isinstance(r, str)]
    assert len(rows) == 8


def test_pool_recovers_missing_audits_without_repurchase(small_pool, teacher, tmp_path):
    pool.collect(tmp_path)
    path = tmp_path / "attempts.jsonl"
    # Simulate interruption after the durable ledger write, before the audit append.
    prefix = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(prefix)
    pool.collect(tmp_path)
    before = path.read_bytes()
    assert before.startswith(prefix) and len(teacher.calls) == 4
    rows = [json.loads(line) for line in before.splitlines()]
    assert len(rows) == 4 and rows[0]["trajectory_available"]
    for row in rows[1:]:
        assert row["termination_reason"] == "interrupted_before_trajectory_saved"
        assert not row["trajectory_available"] and row["prediction"] is None
        assert row["question"] == Q["question"] and row["gold"] == Q["answer"]
        assert row["tokens"] == 1037
    pool.collect(tmp_path)
    assert len(teacher.calls) == 4 and path.read_bytes() == before


def test_pool_resumes_after_failed_attempt_without_repurchase(small_pool, teacher, tmp_path):
    teacher.replies = iter(["finish[Paris France]"])
    limit = prompt_bound(hp.build_messages(Q["question"])) + TEACHER_MAX_TOKENS
    summary = pool.collect(tmp_path, limits=Limits(max_tokens=limit))
    assert summary["attempts"] == 1 and not summary["complete"]
    path = tmp_path / "attempts.jsonl"
    before = path.read_bytes()
    assert json.loads(before)["attempt_index"] == 0 and not json.loads(before)["verified"]
    teacher.replies = None
    summary = pool.collect(tmp_path)
    assert summary["complete"] and summary["attempts"] == len(teacher.calls) == 5
    assert path.read_bytes().startswith(before)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["task_id"], r["attempt_index"]) for r in rows[:2]] == [("task-0", 0), ("task-0", 1)]


def test_pool_rejects_duplicate_audit_keys(small_pool, teacher, tmp_path):
    pool.collect(tmp_path)
    path = tmp_path / "attempts.jsonl"
    ledger.append_record(path, json.loads(path.read_text().splitlines()[0]))
    with pytest.raises(ValueError, match="duplicate"):
        pool.collect(tmp_path)
    assert len(teacher.calls) == 4


@pytest.mark.parametrize("limits", [Limits(max_tokens=0), Limits(max_usd=0)])
def test_pool_zero_cap_makes_no_calls_or_attempts(small_pool, teacher, tmp_path, limits):
    summary = pool.collect(tmp_path, workers=4, limits=limits)
    assert summary["tokens"] == summary["attempts"] == 0
    assert not summary["complete"] and not teacher.calls
    assert not (tmp_path / "attempts.jsonl").exists()


def test_pool_unknown_usage_charges_retries_continues_and_resumes(small_pool, teacher, tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(Budget, "wait_for_retry", lambda self, delay: delays.append(delay))
    teacher.missing = True
    summary = pool.collect(tmp_path, workers=1)
    calls = len(small_pool) * REQUEST_ATTEMPTS
    envelope = prompt_bound(hp.build_messages(Q["question"])) + TEACHER_MAX_TOKENS
    assert summary["uncertain_calls"] == len(teacher.calls) == calls
    assert summary["verified"] == 0 and summary["attempts"] == len(small_pool)
    assert summary["tokens"] == envelope * calls
    assert summary["complete"] and summary["stop_reason"] == "complete"
    assert delays == [1., 2.] * len(small_pool)
    rows = ledger.read_records(tmp_path / "teacher_ledger.jsonl")
    archives = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    for row, archive in zip(rows, archives):
        assert row["usage_status"] == archive["usage_status"] == "estimated"
        assert row["attempt_index"] == 0 and not row["verified"]
        assert row["tokens_spent"] == REQUEST_ATTEMPTS * TEACHER_MAX_TOKENS
        assert len(archive["response_texts"]) == REQUEST_ATTEMPTS
        assert archive["termination_reason"] == "error"
        assert archive["tokens"] == REQUEST_ATTEMPTS * envelope
    budget = Budget(tmp_path / "usage.jsonl", Limits())
    assert budget.stopped is None and not budget.pending
    assert sum(r.get("retry_exhausted", False) for r in budget.calls.values()) == len(small_pool)
    before = {p: p.read_bytes() for p in tmp_path.glob("*.jsonl")}
    pool.collect(tmp_path, workers=2)
    assert len(teacher.calls) == calls
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("failure", ["missing_usage", "timeout", "rate_limit"])
def test_pool_unknown_usage_retry_recovers_with_separate_charges(small_pool, teacher, tmp_path, monkeypatch, failure):
    delays = []
    monkeypatch.setattr(Budget, "wait_for_retry", lambda self, delay: delays.append(delay))
    reply = "finish[Paris]"
    if failure == "timeout":
        reply = TimeoutError("unknown provider usage")
    elif failure == "rate_limit":
        reply = appworld_teacher.TeacherAPIError("rate limited")
        reply.status_code, reply.retry_after = 429, "5"
    teacher.replies = iter([reply, reply] + ["finish[Paris]"] * len(small_pool))
    generate = appworld_teacher.generate_reply
    def recovering(*args, **kwargs):
        teacher.missing = len(teacher.calls) < 2
        return generate(*args, **kwargs)
    monkeypatch.setattr(appworld_teacher, "generate_reply", recovering)
    summary = pool.collect(tmp_path)
    envelope = prompt_bound(hp.build_messages(Q["question"])) + TEACHER_MAX_TOKENS
    assert summary["complete"] and summary["stop_reason"] == "complete"
    assert summary["verified"] == summary["attempts"] == len(small_pool)
    assert summary["uncertain_calls"] == 2 and len(teacher.calls) == len(small_pool) + 2
    assert summary["tokens"] == 2 * envelope + len(small_pool) * 1037
    assert delays == ([5., 5.] if failure == "rate_limit" else [1., 2.])
    row = ledger.read_records(tmp_path / "teacher_ledger.jsonl")[0]
    assert row["verified"] and row["usage_status"] == "estimated"
    assert row["tokens_spent"] == 2 * TEACHER_MAX_TOKENS + 37
    budget = Budget(tmp_path / "usage.jsonl", Limits())
    assert [r["status"] for r in budget.calls.values()] == ["estimated"] * 2 + ["reported"] * len(small_pool)
    assert not budget.pending and not budget.exhausted("task-0")
    pool.collect(tmp_path)
    assert len(teacher.calls) == len(small_pool) + 2


def test_pool_401_stops_without_retry_and_keeps_estimated_charge(small_pool, teacher, tmp_path, monkeypatch):
    monkeypatch.setattr(Budget, "wait_for_retry", lambda *a: pytest.fail("401 must not retry"))
    error = appworld_teacher.TeacherAPIError("unauthorized")
    error.status_code = 401
    teacher.replies, teacher.missing = iter([error]), True
    summary = pool.collect(tmp_path)
    assert summary["attempts"] == summary["uncertain_calls"] == len(teacher.calls) == 1
    assert not summary["complete"] and summary["verified"] == 0
    assert "401" in summary["stop_reason"]
    assert summary["tokens"] == prompt_bound(hp.build_messages(Q["question"])) + TEACHER_MAX_TOKENS
    row = ledger.read_records(tmp_path / "teacher_ledger.jsonl")[0]
    assert row["usage_status"] == "estimated" and not row["verified"]


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


def test_teacher_pool_offline_miss_charged_and_continues(small_pool, teacher, tmp_path, monkeypatch):
    monkeypatch.setattr(HotpotQAAdapter, "_wiki", lambda self: hp.Wikipedia(tmp_path / "cache", offline=True))
    teacher.replies = iter(["search[uncached]", "finish[Paris]"] * len(small_pool))
    summary = pool.collect(tmp_path / "pool", offline=True)
    assert summary["attempts"] == len(teacher.calls) == 2 * len(small_pool)
    assert summary["tokens"] == 1037 * summary["attempts"]
    assert summary["complete"] and summary["stop_reason"] == "complete"
    assert summary["verified"] == len(small_pool) and summary["uncertain_calls"] == 0
    rows = ledger.read_records(tmp_path / "pool/teacher_ledger.jsonl")
    path = tmp_path / "pool/attempts.jsonl"
    before = path.read_bytes()
    archives = [json.loads(line) for line in before.splitlines()]
    for row, archive in zip(rows[::2], archives[::2]):
        assert not row["verified"] and row["usage_status"] == "reported"
        assert archive["termination_reason"] == "offline_cache_miss"
        assert archive["history"][0]["action"] == "search[uncached]"
        assert "not cached" in archive["history"][0]["observation"]
        assert archive["response_texts"] == ["search[uncached]"] and archive["tokens"] == 1037
    assert all(r["verified"] and r["attempt_index"] == 1 for r in rows[1::2])
    pool.collect(tmp_path / "pool", offline=True)
    assert len(teacher.calls) == 2 * len(small_pool) and path.read_bytes() == before


def test_budget_concurrent_reservations_and_crash_recovery(tmp_path, monkeypatch, small_pool, teacher):
    messages = hp.build_messages(Q["question"])
    envelope = prompt_bound(messages) + TEACHER_MAX_TOKENS
    budget = Budget(tmp_path / "journal.jsonl", Limits(max_tokens=envelope))
    call = budget.reserve("task-0", 0, messages)
    waiting = threading.Event()
    wait = budget.condition.wait
    def signal_wait(timeout=None):
        waiting.set()
        return wait(timeout)
    monkeypatch.setattr(budget.condition, "wait", signal_wait)
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocked = executor.submit(budget.reserve, "task-1", 0, messages)
        try:
            assert waiting.wait(timeout=5), "second request never waited for the reservation"
        finally:
            budget.finish(call)
        with pytest.raises(BudgetStopped):
            blocked.result(timeout=5)
    reopened = Budget(tmp_path / "journal.jsonl", Limits(max_tokens=100000))
    assert reopened.stopped is None and reopened.usage()["completion_tokens"] == TEACHER_MAX_TOKENS
    out = tmp_path / "pool"
    out.mkdir()
    hp.write_json(out / "identity.json", pool.collection_identity(HotpotQAAdapter.teacher_name(), Limits()))
    # Simulate a crash with a durable reservation but no response or episode saved.
    interrupted = Budget(out / "usage.jsonl", Limits())
    interrupted.reserve("task-0", 0, messages)
    reserved = (out / "usage.jsonl").read_bytes()
    summary = pool.collect(out)
    assert summary["attempts"] == 1 + len(small_pool) and len(teacher.calls) == len(small_pool)
    assert summary["complete"] and summary["verified"] == len(small_pool)
    assert summary["uncertain_calls"] == 1 and summary["tokens"] == envelope + len(small_pool) * 1037
    assert (out / "usage.jsonl").read_bytes().startswith(reserved)
    before = (out / "attempts.jsonl").read_bytes()
    archives = [json.loads(line) for line in before.splitlines()]
    archive = archives[0]
    assert archive["task_id"] == "task-0" and archive["attempt_index"] == 0
    assert not archive["trajectory_available"] and archive["usage_status"] == "estimated"
    assert archive["tokens"] == envelope
    assert (archives[1]["task_id"], archives[1]["attempt_index"]) == ("task-0", 1)
    assert all(r["verified"] and r["usage_status"] == "reported" for r in archives[1:])
    pool.collect(out)
    assert len(teacher.calls) == len(small_pool) and (out / "attempts.jsonl").read_bytes() == before


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
