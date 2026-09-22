"""Server contract and launch/shard tests: CPU only, HTTP and processes stubbed."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from bfas.rtd.benchmarks import alfworld_evaluation as evaluation
from bfas.rtd.benchmarks.alfworld_server import SERVER_VERSION, checked_server_identity
from bfas.rtd.persistence import ComputeJournal, digest
from rtd_alfworld_evaluation_fixtures import FakeBackend, campaign, put, run
from test_alf_eval_shard import finalise, task_bytes
from tools import alf_eval_shard as shards
from tools import alf_vllm_server as launcher
from tools import rtd_alfworld_evaluate as serial_cli


def launch_identity(c):
    value = dict(version=SERVER_VERSION, vllm_version="0.27.1", host="127.0.0.1",
        port=19327, served_model_name="alfworld-fixture-launch",
        hardware=deepcopy(c.hardware), hardware_hash=c.manifest["hardware_hash"],
        model=c.manifest["checkpoint"], tokenizer=c.manifest["evaluation_harness"]["tokenizer"],
        max_context_tokens=c.config["max_context_tokens"])
    return signed(value)


def signed(value):
    value["identity_hash"] = digest({k: v for k, v in value.items() if k != "identity_hash"})
    return value


class Tokenizer:
    eos_token_id, pad_token_id = 2, 0

    def __call__(self, prompt, *, add_special_tokens, return_tensors=None):
        assert add_special_tokens is False
        if return_tensors:
            import torch
            return dict(input_ids=torch.tensor([[10, 11]]))
        return dict(input_ids=[10, 11])

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join({20: "ACTION: look", 21: "<special>", 22: " x"}[i] for i in ids)


def renderer(monkeypatch, tokenizer=None):
    fake = FakeBackend()
    fake.adapter = SimpleNamespace(_tokenizer=tokenizer or Tokenizer())
    class Renderer:
        adapter = fake.adapter
        def __call__(self, request, history):
            return fake.renderer(request, history)
    monkeypatch.setattr(evaluation, "FrozenRenderer", lambda path: Renderer())


def stub_http(monkeypatch, server, ids, *, mutate=None, status=200, stop_ids=(2,)):
    calls, connections = [], []
    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port) == ("127.0.0.1", server["port"])
            self.closed = False
            connections.append(self)
        def request(self, method, path, *, body, headers):
            assert (method, path) == ("POST", "/v1/completions")
            assert headers == {"Content-Type": "application/json"}
            payload = json.loads(body)
            calls.append(payload)
            assert payload == dict(model=server["served_model_name"], prompt=[10, 11],
                temperature=0.0, max_tokens=256, n=1, stream=False, echo=False,
                use_beam_search=False, top_p=1.0, top_k=-1, min_p=0.0,
                repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
                min_tokens=0, stop=[], stop_token_ids=list(stop_ids), ignore_eos=True,
                include_stop_str_in_output=False, skip_special_tokens=False,
                add_special_tokens=False, truncate_prompt_tokens=None, return_token_ids=True)
        def getresponse(self):
            body = dict(model=server["served_model_name"], choices=[dict(
                text="server detokenization is not the HF decode authority",
                token_ids=ids, finish_reason="stop" if ids[-1] in stop_ids else "length",
                stop_reason=ids[-1] if ids[-1] in stop_ids else None)],
                usage=dict(prompt_tokens=2, completion_tokens=len(ids)))
            if mutate:
                mutate(body)
            return SimpleNamespace(status=status, read=lambda: json.dumps(body).encode())
        def close(self):
            self.closed = True
    monkeypatch.setattr(evaluation.http.client, "HTTPConnection", Connection)
    return calls, connections


@pytest.mark.parametrize("ids", [[20, 21, 2], [20] + [22] * 255, [20] + [22] * 254 + [2], [2]])
def test_server_matches_hf_generation_contract(campaign, monkeypatch, ids):
    import torch
    c, server = campaign, launch_identity(campaign)
    renderer(monkeypatch)
    class Model:
        def to(self, device):
            assert device == "cpu"
            return self
        def eval(self):
            return self
        def generate(self, **kwargs):
            return torch.tensor([[10, 11] + ids])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model()),
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs)))
    calls, connections = stub_http(monkeypatch, server, ids)
    hf = evaluation.HFBackend(c.manifest, device="cpu")
    remote = evaluation.VLLMBackend(c.manifest, server=server)
    actual = remote.generate("prompt", temperature=0, max_new_tokens=256)
    assert actual == hf.generate("prompt", temperature=0, max_new_tokens=256)
    assert actual.token_count == len(ids) and actual.truncated == (ids[-1] != 2)
    assert len(calls) == 1 and all(c.closed for c in connections)
    hf.close()
    remote.close()


@pytest.mark.parametrize("ids", [[20, 106], [20, 1], [20, 50], [20]*256, [20]*255+[106]])
def test_native_turn_stop_contract_in_both_backends(campaign, monkeypatch, ids):
    import torch
    class NativeTokenizer(Tokenizer):
        eos_token_id, unk_token_id = 1, 3
        all_special_tokens = ["<turn|>"]
        def convert_tokens_to_ids(self, token):
            assert token == "<turn|>"
            return 106
        def encode(self, text, **kwargs):
            assert text == "<turn|>" and kwargs == dict(add_special_tokens=False)
            return [106]
    renderer(monkeypatch, NativeTokenizer())
    # Sampling defaults must not leak into HF or vLLM requests.
    put(campaign.model/"generation_config.json", dict(eos_token_id=[1, 106, 50],
        do_sample=True, temperature=.7, top_p=.8, max_new_tokens=900))
    class Model:
        def to(self, device):
            assert device == "cpu"
            return self
        def eval(self):
            return self
        def generate(self, **kwargs):
            cfg = kwargs["generation_config"]
            assert cfg.eos_token_id == [1, 106, 50]
            assert cfg.do_sample is False and cfg.num_beams == 1 and cfg.max_new_tokens == 256
            assert not hasattr(cfg, "temperature")
            return torch.tensor([[10, 11]+ids])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model()),
        GenerationConfig=lambda **kw: SimpleNamespace(**kw)))
    server = launch_identity(campaign)
    stub_http(monkeypatch, server, ids, stop_ids=(1, 106, 50))
    hf = evaluation.HFBackend(campaign.manifest, device="cpu")
    remote = evaluation.VLLMBackend(campaign.manifest, server=server)
    got = remote.generate("prompt", temperature=0, max_new_tokens=256)
    assert got == hf.generate("prompt", temperature=0, max_new_tokens=256)
    stopped = ids[-1] in (1, 106, 50)
    assert got == evaluation.Generation("ACTION: look"*(len(ids)-int(stopped)), len(ids), not stopped)


def test_context_guard_before_http(campaign, monkeypatch):
    server = launch_identity(campaign)
    renderer(monkeypatch)
    calls, _ = stub_http(monkeypatch, server, [20, 2])
    remote = evaluation.VLLMBackend(campaign.manifest, server=server)
    remote.max_context = 257
    with pytest.raises(ValueError, match="context overflow"):
        remote.generate("prompt", temperature=0, max_new_tokens=256)
    assert not calls
    remote.max_context = 258
    assert not remote.generate("prompt", temperature=0, max_new_tokens=256).truncated


@pytest.mark.parametrize("temperature,cap", [(1, 256), (0.1, 256), (0, 255), (0, 257)])
def test_only_greedy_256_before_http(campaign, monkeypatch, temperature, cap):
    server = launch_identity(campaign)
    renderer(monkeypatch)
    calls, _ = stub_http(monkeypatch, server, [20, 2])
    remote = evaluation.VLLMBackend(campaign.manifest, server=server)
    with pytest.raises(ValueError, match="greedy/256"):
        remote.generate("prompt", temperature=temperature, max_new_tokens=cap)
    assert not calls


@pytest.mark.parametrize("fault", ["short_length", "stop_without_eos", "length_with_eos",
    "count", "prompt_count", "early_eos", "other_stop", "error", "foreign_model", "http"])
def test_inconsistent_server_responses_fail_closed(campaign, monkeypatch, fault):
    server = launch_identity(campaign)
    renderer(monkeypatch)
    ids = [20, 2]
    def mutate(body):
        choice = body["choices"][0]
        if fault == "short_length":
            choice.update(token_ids=[20, 21], finish_reason="length", stop_reason=None)
        elif fault == "stop_without_eos":
            choice["token_ids"] = [20, 21]
        elif fault == "length_with_eos":
            choice.update(token_ids=[20] * 255 + [2], finish_reason="length")
            body["usage"]["completion_tokens"] = 256
        elif fault == "count":
            body["usage"]["completion_tokens"] = 1
        elif fault == "prompt_count":
            body["usage"]["prompt_tokens"] = 3
        elif fault == "early_eos":
            choice["token_ids"] = [2, 2]
        elif fault == "other_stop":
            choice["stop_reason"] = 999
        elif fault == "error":
            choice["finish_reason"] = "error"
        elif fault == "foreign_model":
            body["model"] = "another-job"
    _, connections = stub_http(monkeypatch, server, ids, mutate=mutate, status=503 if fault == "http" else 200)
    remote = evaluation.VLLMBackend(campaign.manifest, server=server)
    with pytest.raises(ValueError):
        remote.generate("prompt", temperature=0, max_new_tokens=256)
    assert connections[0].closed


@pytest.mark.parametrize("fault", ["port", "host", "hardware", "model", "tokenizer", "context", "tamper"])
def test_server_binding_rejected_before_shard_writes(campaign, fault):
    c, server = campaign, launch_identity(campaign)
    if fault == "port":
        del server["port"]
    elif fault == "host":
        server["host"] = "elsewhere"
    elif fault == "hardware":
        server["hardware"]["hard"]["memory"] += 1
        server["hardware_hash"] = digest(server["hardware"]["hard"])
    elif fault in ("model", "tokenizer"):
        server[fault] = {}
    elif fault == "context":
        server["max_context_tokens"] += 1
    elif fault == "tamper":
        server["port"] += 1
    if fault != "tamper":
        signed(server)
    with pytest.raises(ValueError):
        shards.evaluate_shard(c.root, c.manifest, output_root=c.output, tag=c.tag,
                             shard=0, of=2, server=server)
    assert not c.output.exists()


def test_server_shards_byte_identical_to_cpu_fixture(campaign, monkeypatch):
    c, server = campaign, launch_identity(campaign)
    reference = run(c, tag="serial")
    class FixtureTokenizer(Tokenizer):
        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens and ids == [20] * 11
            return "THOUGHT: move.\nACTION: go to desk 1"
    renderer(monkeypatch, FixtureTokenizer())
    calls, _ = stub_http(monkeypatch, server, [20] * 11 + [2])
    monkeypatch.setattr(shards, "hardware_identity", lambda: pytest.fail("client probed hardware"))
    # Exercise the production VLLM dispatch with an injected fake HTTP server.
    binding, server_file = c.root / "binding.json", c.root / "server.json"
    put(binding, c.manifest)
    put(server_file, server)
    monkeypatch.setattr(evaluation, "EvaluationEnvBridge", lambda tid, **kw: c.env_factory(tid))
    for index in range(2):
        assert shards.main(["--root", str(c.root), "--binding", str(binding),
            "--output-root", str(c.output), "--tag", c.tag, "--shard", str(index),
            "--of", "2", "--server-json", str(server_file)]) == 0
        assert not (c.directory / "campaign.json").exists()
    assert len(calls) == 280
    assert task_bytes(c.directory) == task_bytes(c.output / "serial")
    assert finalise(c, monkeypatch) == reference
    for name in ("artifacts/binding.json", "artifacts/aggregate.json", "campaign.json"):
        assert (c.directory / name).read_bytes() == (c.output / "serial" / name).read_bytes()
    events = ComputeJournal(c.directory / "audit.jsonl").events
    assert all(e["gpu_seconds"] == 0 for e in events if e["kind"] == "compute_end")
    assert all(e["server_identity_hash"] == server["identity_hash"]
               for e in events if e["kind"] == "compute_begin")


def test_prepare_consumes_server_hardware_without_client_cuda(campaign, capsys):
    c = campaign
    server_file, binding = c.root / "server.json", c.root / "new-binding.json"
    put(server_file, launch_identity(c))
    assert serial_cli.main(["--root", str(c.root), "prepare", "--model", str(c.model),
        "--data-root", str(c.data), "--server-json", str(server_file), "--out", str(binding)]) == 0
    assert json.loads(binding.read_text()) == c.manifest


def test_launcher_occupied_port_fails_before_process_or_gpu(monkeypatch):
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            raise OSError("already in use")
    monkeypatch.setattr(launcher.socket, "socket", lambda *a: Socket())
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("launched process"))
    with pytest.raises(ValueError, match="already listening"):
        launcher.launch(SimpleNamespace(port=19327))


@pytest.mark.parametrize("state", ["ready", "died", "timeout"])
def test_launcher_records_only_ready_identity(campaign, monkeypatch, state):
    from bfas.rtd import hardware
    c = campaign
    directory = c.root / "server-state"
    monkeypatch.setattr(launcher, "check_port", lambda p: None)
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda p: "0.27.1")
    captures, processes, probes, signals = [], [], [], []
    def capture(**kwargs):
        assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
        captures.append(kwargs)
        return c.hardware
    monkeypatch.setattr(hardware, "hardware_identity", capture)
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "must-be-removed")
    # launch edits os.environ in production before torch's first import.
    monkeypatch.setattr(launcher.os, "environ", dict(launcher.os.environ))
    class Process:
        pid = 424242
        finished = False
        def __init__(self, command, **kwargs):
            processes.append((command, kwargs))
        def poll(self):
            return 1 if state == "died" or self.finished else None
        def wait(self, **kwargs):
            self.finished = True
            return 0
    def readiness(port, name):
        probes.append((port, name))
        assert not (directory / "server.json").exists()
        return state == "ready"
    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher, "ready", readiness)
    monkeypatch.setattr(launcher.os, "killpg", lambda *a: signals.append(a))
    times = iter([0, 5])
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(times))
    args = SimpleNamespace(port=19327, gpu="GPU-fixture", max_context_tokens=32768,
        ready_timeout=1, model=c.model, tokenizer=None, state_dir=directory)
    if state == "ready":
        assert launcher.launch(args) == 0
        saved = checked_server_identity(json.loads((directory / "server.json").read_text()), manifest=c.manifest)
        assert saved["hardware"] == c.hardware and saved["port"] == 19327
        assert json.loads((directory / "hardware.json").read_text()) == c.hardware
        assert (directory / "port").read_text() == "19327\n"
        assert probes == [(19327, saved["served_model_name"])]
    else:
        with pytest.raises(RuntimeError if state == "died" else TimeoutError):
            launcher.launch(args)
        assert not (directory / "server.json").exists()
        assert bool(signals) == (state == "timeout")
    assert captures == [dict(optional_packages=("peft",))]
    command, options = processes[0]
    assert options["cwd"] == directory and options["start_new_session"] is True
    assert "--no-enable-log-requests" in command and "--disable-log-requests" not in command
    assert command[command.index("--generation-config") + 1] == "vllm"
    assert command[command.index("--port") + 1] == "19327"
    assert options["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert "VLLM_ATTENTION_BACKEND" not in options["env"]


def test_readiness_rejects_another_launch(monkeypatch):
    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port) == ("127.0.0.1", 19327)
        def request(self, method, path):
            assert (method, path) == ("GET", "/v1/models")
        def getresponse(self):
            return SimpleNamespace(status=200, read=lambda: b'{"data":[{"id":"another-job"}]}')
        def close(self):
            pass
    monkeypatch.setattr(launcher.http.client, "HTTPConnection", Connection)
    assert not launcher.ready(19327, "alfworld-this-launch")


def test_server_hardware_records_missing_peft_and_still_requires_one_gpu(monkeypatch):
    import torch
    from bfas.rtd import hardware
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("real CUDA initialization"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda n: SimpleNamespace(
        name="Fixture GPU", major=8, minor=0, total_memory=12345, uuid="GPU-fixture"))
    monkeypatch.setattr(hardware, "device_inventory", lambda: [])
    monkeypatch.setattr(hardware, "driver_version", lambda: "fixture-driver")
    def version(name):
        if name == "peft":
            raise hardware.importlib.metadata.PackageNotFoundError(name)
        return "fixture-server-version"
    monkeypatch.setattr(hardware.importlib.metadata, "version", version)
    saved = hardware.checked_hardware(hardware.hardware_identity(optional_packages=("peft",)))
    assert saved["hard"]["versions"]["peft"] == "not-installed"
    assert saved["hard"]["versions"]["torch"] == "fixture-server-version"
    with pytest.raises(hardware.importlib.metadata.PackageNotFoundError):
        hardware.hardware_identity()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="exactly one CUDA GPU"):
        hardware.hardware_identity(optional_packages=("peft",))
