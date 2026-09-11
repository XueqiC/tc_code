from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.adapters.tau2 import (  # noqa: E402
    NativeRun,
    Tau2Adapter,
    extract_verdicts,
    official_pass_one,
)
import bfas.adapters.tau2 as tau2  # noqa: E402
from bfas import ledger  # noqa: E402
from bfas import run as bfas_run  # noqa: E402


FIXTURE = ROOT / "tests/fixtures/tau2_cli"


@pytest.fixture(autouse=True)
def isolated_tau2_data(tmp_path, monkeypatch):
    """Unit tests need neither an installation nor ambient model credentials."""
    for name in tuple(tau2.os.environ):
        if name.startswith("BFAS_TAU2_"):
            monkeypatch.delenv(name)
    data = tmp_path / "tau2"
    (data / "user_simulator").mkdir(parents=True)
    for domain, count in (("airline", 30), ("retail", 74), ("telecom", 74)):
        directory = data / "domains" / domain
        directory.mkdir(parents=True)
        train = [str(n) for n in range(count)]
        test = [str(n) for n in range(count, count + 20)]
        (directory / "split_tasks.json").write_text(json.dumps({"train": train, "test": test}))
        (directory / "tasks.json").write_text(json.dumps([{"id": n} for n in train + test]))
        (directory / Tau2Adapter._policy_filename(domain)).write_text("native policy")
    binary = tmp_path / "tau2-cli-stub"
    binary.touch()
    monkeypatch.setattr(tau2, "TAU2_DATA", data)
    monkeypatch.setattr(tau2, "TAU2_BIN", binary)


class FixtureTokenizer:
    """Small deterministic chat renderer used instead of a model download."""

    @staticmethod
    def _assistant_text(message: dict[str, Any]) -> str:
        calls = message.get("tool_calls")
        if calls:
            function = calls[0]["function"]
            return (
                "CALL "
                + function["name"]
                + " "
                + json.dumps(
                    function["arguments"], sort_keys=True, separators=(",", ":")
                )
            )
        return str(message.get("content") or "")

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        tools=None,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        assert tokenize is False
        rendered = ""
        for message in messages:
            role = message["role"]
            content = (
                self._assistant_text(message)
                if role == "assistant"
                else str(message.get("content") or "")
            )
            rendered += f"<{role}>{content}<END>"
        if continue_final_message:
            assert messages[-1]["role"] == "assistant"
            return rendered[: -len("<END>")]
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def _fixture_results() -> dict[str, Any]:
    return json.loads((FIXTURE / "results.json").read_text())


def test_tau2_verdicts_require_positive_official_reward() -> None:
    results = _fixture_results()

    assert extract_verdicts(results, ["2", "6", "missing"]) == {
        "2": True,
        "6": False,
        "missing": False,
    }
    assert official_pass_one(results) == pytest.approx(0.5)


def test_tau2_pool_uses_official_train_splits() -> None:
    adapter = Tau2Adapter()

    tasks = adapter.task_pool()

    assert len(tasks) == 178
    assert {task.category for task in tasks} == {"airline", "retail", "telecom"}
    assert sum(task.category == "airline" for task in tasks) == 30
    assert sum(task.category == "retail" for task in tasks) == 74
    assert sum(task.category == "telecom" for task in tasks) == 74
    assert adapter.official_eval_split_disjoint() is True


def test_tau2_rollout_renders_native_logs_byte_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = Tau2Adapter(seed=3, port=9124)
    adapter._tokenizer = FixtureTokenizer()
    adapter._loaded_policy = "fixture-policy"
    monkeypatch.setattr(adapter, "_record_user_sim_usage", lambda *args: None)
    monkeypatch.setattr(
        adapter,
        "_run_cli",
        lambda **kwargs: NativeRun(_fixture_results(), FIXTURE),
    )

    rollouts = adapter.rollout(
        "fixture-policy", ["airline:2", "airline:6"], temperature=0.7
    )

    assert [rollout.verified for rollout in rollouts] == [True, False]
    passed_turn = rollouts[0].turns[0]
    assert passed_turn.prompt == (
        "<system>native\npolicy<END>"
        "<assistant>Hi! How can I help you today?<END>"
        "<user>Find my booking.\nThe id is ABC123.<END>"
        "<assistant>"
    )
    assert passed_turn.target == 'CALL lookup_booking {"booking_id":"ABC123"}'
    assert adapter.rerender({"_render_context": passed_turn.context}) == (
        passed_turn.prompt
    )
    assert rollouts[1].turns[0].target == "I cannot help with that."
    assert adapter.generation_suffix() == "<assistant>"


def test_tau2_cli_points_each_side_at_its_configured_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Tau2Adapter(seed=1, port=9444)
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://teacher.example")
    monkeypatch.setenv("OLLAMA_API_KEY", "teacher-secret")
    monkeypatch.delenv("BFAS_TAU2_USER_MODEL", raising=False)

    def run(command, **kwargs):
        calls.append((list(command), kwargs))
        data_root = Path(kwargs["env"]["TAU2_DATA_DIR"])
        output = data_root / "simulations/bfas_native/results.json"
        output.parent.mkdir(parents=True)
        output.write_text((FIXTURE / "results.json").read_text())

    monkeypatch.setattr(tau2.subprocess, "run", run)
    model, args = adapter._student_args(0.7)
    native_run = adapter._run_cli(
        domain="airline",
        task_ids=["2"],
        split="test",
        agent_model=model,
        agent_args=args,
    )
    try:
        command, kwargs = calls[0]
        agent_args = json.loads(command[command.index("--agent-llm-args") + 1])
        user_args = json.loads(command[command.index("--user-llm-args") + 1])
        assert command[command.index("--agent-llm") + 1] == "openai/bfas-policy"
        assert agent_args["base_url"] == "http://localhost:9444/v1"
        assert command[command.index("--user-llm") + 1] == (
            "openai/gpt-5.4"
        )
        assert user_args == {
            "temperature": 0.0,
            "base_url": "https://teacher.example/v1",
        }
        assert kwargs["env"]["OPENAI_API_KEY"] == "teacher-secret"
        assert kwargs["cwd"] == tau2.TAU2_ROOT
        assert kwargs["check"] is True
        assert native_run.results["simulations"][0]["task_id"] == "2"
    finally:
        native_run.cleanup()


@pytest.mark.parametrize("user_model", [None, "azure/gpt-5.4"])
def test_tau2_user_sim_usage_is_ledgered_with_its_own_purpose(
    monkeypatch: pytest.MonkeyPatch, user_model: str | None,
) -> None:
    if user_model is None:
        monkeypatch.delenv("BFAS_TAU2_USER_MODEL", raising=False)
    else:
        monkeypatch.setenv("BFAS_TAU2_USER_MODEL", user_model)
    adapter = Tau2Adapter()
    adapter._run_serial = 9
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(
        ledger,
        "append_episode",
        lambda benchmark, **record: records.append(
            {"benchmark": benchmark, **record}
        ),
    )

    adapter._record_user_sim_usage("airline", _fixture_results()["simulations"])

    assert [record["purpose"] for record in records] == ["user_sim", "user_sim"]
    assert [record["tokens_spent"] for record in records] == [18, 13]
    assert all(record["teacher"] == (user_model or "gpt-5.4") for record in records)
    assert all(record["attempt_index"] == 9 for record in records)


def test_tau2_serving_probe_makes_a_chat_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"choices":[{"message":{"content":"OK"}}]}'

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(tau2.urllib.request, "urlopen", urlopen)

    Tau2Adapter(port=9555).serving_probe()

    request, timeout = requests[0]
    assert request.full_url == "http://localhost:9555/v1/chat/completions"
    assert json.loads(request.data)["model"] == "bfas-policy"
    assert timeout == 30


def test_tau2_uses_pipeline_vllm_serving_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[Any] = []

    class Registry:
        def __init__(self, port):
            events.append(("registry", port))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Server:
        def __init__(self, model, gpu, port, log_path, served_model_name):
            events.append(
                ("server", model, gpu, port, log_path, served_model_name)
            )

        def start(self):
            events.append("start")

        def close(self):
            events.append("close")

    adapter = Tau2Adapter(port=9666)
    monkeypatch.setattr(bfas_run, "PortRegistry", Registry)
    monkeypatch.setattr(bfas_run, "VLLMServer", Server)
    monkeypatch.setattr(adapter, "serving_probe", lambda: events.append("probe"))

    with bfas_run.serving_lane(
        adapter, "student-model", "0", 9666, tmp_path / "vllm.log"
    ):
        events.append("body")

    assert events == [
        ("registry", 9666),
        (
            "server",
            "student-model",
            "0",
            9666,
            tmp_path / "vllm.log",
            "bfas-policy",
        ),
        "start",
        "probe",
        "body",
        "close",
    ]


def test_tau2_tool_only_turn_with_empty_string_content_renders_the_call() -> None:
    # deepseek logs tool-only assistant turns with content "" (not null);
    # before the fix the renderer took "" as prose and raised "empty target",
    # discarding every verified demo (2026-09-01 first tau2 collection).
    call_log = {
        "request": {
            "messages": [
                {"role": "system", "content": "native\npolicy"},
                {"role": "user", "content": "Find my booking."},
            ],
            "tools": [],
        },
        "response": {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "lookup_booking",
                 "arguments": '{"booking_id": "ABC123"}'}
            ],
        },
    }
    turn = tau2.render_logged_turn(FixtureTokenizer(), call_log)
    assert turn.target
    assert "lookup_booking" in turn.target
    assert "ABC123" in turn.target


def test_tau2_azure_user_simulator_uses_litellm_azure_with_env_only_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = Tau2Adapter(seed=1, port=9445)
    monkeypatch.setenv("BFAS_TAU2_USER_MODEL", "azure/gpt-5.4")
    monkeypatch.setenv("AZURE_LLM_ENDPOINT", "https://apim.example/")
    monkeypatch.setenv("AZURE_LLM_KEY", "azure-secret")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    model, args = adapter._user_args(0.0)
    assert model == "azure/gpt-5.4" and args == {"temperature": 0.0}
    env = adapter._azure_env()
    assert env["AZURE_API_KEY"] == "azure-secret"
    assert env["AZURE_API_BASE"] == "https://apim.example"
    # the ollama-backed teacher still resolves through the OpenAI provider
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://teacher.example")
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    monkeypatch.delenv("BFAS_TEACHER", raising=False)
    monkeypatch.delenv("BFAS_TAU2_TEACHER", raising=False)
    teacher_model, teacher_args = adapter._teacher_args(0.7)
    assert teacher_model == "openai/gpt-5.4"
    assert teacher_args["base_url"] == "https://teacher.example/v1"


def test_official_luna_cli_credentials_and_parameters(monkeypatch):
    monkeypatch.setenv("BFAS_OPENAI_SERVICE_TIER", "flex")
    monkeypatch.setenv("BFAS_TAU2_USER_MODEL", tau2.LUNA_MODEL)
    monkeypatch.setenv("BFAS_TAU2_TEACHER_MODEL", tau2.LUNA_MODEL)
    monkeypatch.setenv("BFAS_TAU2_TEACHER", "azure/obsolete")
    monkeypatch.setenv("OPENAI_API_KEY", "official-secret")
    monkeypatch.setenv("OLLAMA_API_KEY", "wrong-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example/v1")
    adapter = Tau2Adapter(port=8950, served_model_name="gemma4-12b-base")
    expected = {"base_url": tau2.OPENAI_BASE, "service_tier": "flex"}
    assert adapter.teacher_name() == tau2.LUNA_MODEL
    assert adapter._teacher_args(0.7) == (tau2.LUNA_MODEL, expected)
    assert adapter._user_args(0.7) == (tau2.LUNA_MODEL, expected)

    def run(command, **kwargs):
        assert kwargs["env"]["OPENAI_API_KEY"] == "official-secret"
        assert "official-secret" not in str(command)
        assert "wrong-secret" not in str(command)
        agent = json.loads(command[command.index("--agent-llm-args") + 1])
        user = json.loads(command[command.index("--user-llm-args") + 1])
        assert user == expected
        assert agent["temperature"] == 0 and agent["api_key"] == "EMPTY"
        assert agent["base_url"] == "http://localhost:8950/v1"
        output = Path(kwargs["env"]["TAU2_DATA_DIR"]) / "simulations/bfas_native/results.json"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps(_fixture_results()))

    monkeypatch.setattr(tau2.subprocess, "run", run)
    model, args = adapter._student_args(0)
    result = adapter._run_cli(domain="airline", task_ids=["2"], split="test",
                              agent_model=model, agent_args=args)
    result.cleanup()


def test_official_channel_requires_official_key(monkeypatch):
    monkeypatch.setenv("BFAS_TAU2_USER_MODEL", tau2.LUNA_MODEL)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "cannot-substitute")
    monkeypatch.setattr(tau2.subprocess, "run", lambda *a, **k: pytest.fail("no CLI call"))
    adapter = Tau2Adapter()
    model, args = adapter._student_args(0)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        adapter._run_cli(domain="airline", task_ids=["2"], split="test", agent_model=model, agent_args=args)


def test_native_judge_usage_is_separate_from_demo_purchases(monkeypatch):
    records = []
    monkeypatch.setenv("BFAS_TAU2_TEACHER_MODEL", tau2.LUNA_MODEL)
    monkeypatch.setattr(ledger, "append_episode", lambda benchmark, **row: records.append(row))
    simulation = _fixture_results()["simulations"][0]
    simulation["bfas_judge_usage"] = [{"prompt_tokens": 40, "completion_tokens": 2, "cached_tokens": 20}]
    Tau2Adapter()._record_user_sim_usage("airline", [simulation])
    assert [r["purpose"] for r in records] == ["user_sim", "teacher_judge"]
    assert records[1]["tokens_spent"] == 42
    assert records[1]["usage"]["cached_tokens"] == 20


@pytest.mark.parametrize("passed", [True, False])
def test_teacher_and_simulator_usage_reaches_ledger(tmp_path, monkeypatch, passed):
    monkeypatch.setenv("BFAS_TAU2_USER_MODEL", tau2.LUNA_MODEL)
    monkeypatch.setenv("BFAS_TAU2_TEACHER_MODEL", tau2.LUNA_MODEL)
    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "ledger")
    adapter = Tau2Adapter()
    adapter._tokenizer = FixtureTokenizer()
    results = _fixture_results()
    results["simulations"] = results["simulations"][:1]
    simulation = results["simulations"][0]
    simulation["reward_info"]["reward"] = int(passed)
    for message in simulation["messages"][1:]:
        message["raw_data"]["usage"] = {**message["usage"], "prompt_tokens_details": {"cached_tokens": 4}}
    monkeypatch.setattr(adapter, "_run_cli", lambda **kwargs: NativeRun(results, FIXTURE))
    episode = adapter.teacher_episode("airline:2", 0, 0.7)
    ledger.append_episode("tau2", task_id=episode.task_id, teacher=adapter.teacher_name(),
                          attempt_index=0, temperature=0.7, verified=episode.verified,
                          demo=episode.demo, tokens_spent=episode.tokens_spent, usage=episode.usage)
    records = [json.loads(line) for line in (tmp_path / "ledger/tau2.jsonl").read_text().splitlines()]
    assert [r["purpose"] for r in records] == ["user_sim", "teacher"]
    assert records[0]["usage"] == {"prompt_tokens": 10, "cached_tokens": 4, "completion_tokens": 8}
    assert records[1]["usage"] == {"prompt_tokens": 20, "cached_tokens": 4, "completion_tokens": 5}
    assert records[1]["verified"] == passed


def test_estimated_actor_and_judge_usage_survives_purchase_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("BFAS_TAU2_USER_MODEL", tau2.LUNA_MODEL)
    monkeypatch.setenv("BFAS_TAU2_TEACHER_MODEL", tau2.LUNA_MODEL)
    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "ledger")
    results = _fixture_results()
    results["simulations"] = results["simulations"][:1]
    simulation = results["simulations"][0]
    simulation["reward_info"]["reward"] = 0
    usage = {"prompt_tokens": 8000, "completion_tokens": 2048, "cached_tokens": 0}
    for message in simulation["messages"][1:]:
        message["usage"] = None
        message["raw_data"] = {"bfas_charge": {"status": "estimated", "usage": usage}}
    simulation["bfas_judge_usage"] = [{"status": "estimated", "usage": usage}]
    adapter = Tau2Adapter()
    adapter._tokenizer = FixtureTokenizer()
    monkeypatch.setattr(adapter, "_run_cli", lambda **kwargs: NativeRun(results, FIXTURE))
    ledger.acquire_demos("tau2", adapter, ["airline:2"], 1)
    records = [json.loads(line) for line in (tmp_path / "ledger/tau2.jsonl").read_text().splitlines()]
    assert [r["purpose"] for r in records] == ["user_sim", "teacher_judge", "teacher"]
    assert all(r["usage_status"] == "estimated" and r["usage"] == usage for r in records)
    assert all(r["tokens_spent"] == 10048 for r in records)
