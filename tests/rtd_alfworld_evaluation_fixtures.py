"""Self-contained C26-D CPU fixtures; never read an existing run or bank."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bfas.rtd.benchmarks import alfworld_evaluation as evaluation
from bfas.rtd.benchmarks import alfworld_identity as identity
from bfas.rtd.benchmarks.alfworld_support import prompt_messages
from bfas.rtd.hardware import PACKAGES, VERSION
from bfas.rtd.persistence import atomic_json, digest, file_hash, tree_hash


ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    else:
        path.write_text(value)


def hardware():
    return dict(version=VERSION, hard=dict(gpu="Fixture CPU-only simulated A100", capability=[8, 0],
        memory=85094825824, cuda="fixture", driver="fixture-driver", host_class="fixture-host-class",
        versions={p: "fixture" for p in PACKAGES}, python="3.12", machine="x86_64"),
        metadata=dict(hostname="fixture-host", uuid="fixture-gpu", pci_bus_id=None, cuda_device_order="FASTEST_FIRST"))


class FakeBackend:
    def __init__(self):
        self.calls, self.closed = [], False

    def renderer(self, request, history):
        return json.dumps(prompt_messages(request, history), sort_keys=True)

    def generate(self, prompt, *, temperature, max_new_tokens):
        assert temperature == 0.0 and max_new_tokens == 256
        self.calls.append(prompt)
        return evaluation.Generation("THOUGHT: move.\nACTION: go to desk 1", 12, False)

    def close(self):
        self.closed = True


class FakeEnv:
    def __init__(self, tid, *, won=True, terminal_step=2):
        self.tid, self.won, self.terminal_step = tid, won, terminal_step
        self.index, self.closed = 0, False

    def _read(self):
        return dict(op="state", observation="Your task is to: put the apple on the desk.",
                    admissible=["go to desk 1", "look"], done=False, won=False)

    def step(self, command):
        self.index += 1
        assert command in ("go to desk 1", "look")
        done = self.index == self.terminal_step
        return dict(op="state", observation=f"step {self.index}", admissible=["go to desk 1", "look"],
                    done=done, won=self.won if done else False)

    def close(self):
        self.closed = True


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch
    def forbidden(*args, **kwargs):
        pytest.fail("C26-D tests must not query or initialize CUDA")
    for name in ("is_available", "device_count", "get_device_properties", "init", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    root = tmp_path / "isolated"
    for name in [*identity.SCOPES, "src/bfas/rtd/benchmarks/alfworld_identity.py", "src/bfas/adapter.py"]:
        put(root / name, (ROOT / name).read_text())
    data = root / "envs/alfworld/data/json_2.1.1"
    for i in range(140):
        parent = i if i < 137 else i - 137
        tid = f"pick_and_place_simple-fixture-{parent:03d}/trial_{i:03d}"
        for name, contents in {
            "game.tw-pddl": dict(solvable=True),
            "traj_data.json": dict(task_type="pick_and_place_simple"),
            "initial_state.pddl": f"fixture world {i}",
        }.items():
            put(data / "valid_seen" / tid / name, contents)
    environment = root / "envs/alfworld"
    put(environment / ".venv/bin/python", "fixture interpreter bytes")
    site = environment / ".venv/lib/python3.12/site-packages"
    for package in ("alfworld", "textworld"):
        put(site / package / "__init__.py", "# fixture package\n")
        put(site / f"{package}-1.dist-info/METADATA", "Version: 1\n")
        put(site / f"{package}-1.dist-info/RECORD", "fixture record\n")
    model = root / "model"
    put(model / "config.json", dict(model_type="fixture"))
    put(model / "model.safetensors", "fixture model bytes, never deserialized")
    put(model / "tokenizer.json", dict(fixture=True))
    put(model / "tokenizer_config.json", dict(chat_template="fixture {{messages}}"))
    config = deepcopy(identity.OFFICIAL_CONFIG)
    hw = hardware()
    manifest = identity.make_manifest(root, config, data_root=data, model_path=model, hardware=hw)
    c = SimpleNamespace(root=root, data=data, model=model, config=config, hardware=hw,
        manifest=manifest, output=root / "campaigns", tag="round1", backends=[], envs=[])
    def backend_factory(manifest):
        backend = FakeBackend()
        c.backends.append(backend)
        return backend
    def env_factory(tid):
        env = FakeEnv(tid, won=int(tid.rsplit("_", 1)[1]) < 109)
        c.envs.append(env)
        return env
    c.backend_factory, c.env_factory = backend_factory, env_factory
    c.directory = c.output / c.tag
    c.kwargs = dict(output_root=c.output, tag=c.tag, hardware=hw,
                    backend_factory=backend_factory, env_factory=env_factory, lock_timeout=0)
    return c


def run(c, **kwargs):
    return evaluation.evaluate(c.root, c.manifest, **dict(c.kwargs, **kwargs))


def audit(c, previous, current, *, supplement=None):
    note = dict(manifest_hash=digest(c.manifest),
        previous_identity=dict(harness_hash=digest(previous), evaluation_harness=previous),
        new_identity=dict(harness_hash=digest(current), evaluation_harness=current),
        audit=dict(method="fixture-explicit-scoring-continuity", evidence="fixture reviewed projection migration"))
    return dict(version=identity.SUPPLEMENT_VERSION, manifest_hash=digest(c.manifest),
        legacy_harness_hash=c.manifest["harness_hash"], harness_hash=digest(current), evaluation_harness=current,
        identity_updates=[*(supplement or {}).get("identity_updates", []), note])


def changed_harness(c):
    path = c.root / "src/bfas/rtd/benchmarks/alfworld_identity.py"
    path.write_text(path.read_text() + "\n# audited fixture migration\n")
    return identity.make_manifest(c.root, c.config, hardware=c.hardware, **c.manifest["paths"])["evaluation_harness"]


def round_checkpoint(c):
    run_dir = c.root / "training"
    training = dict(config=c.config, config_hash=digest(c.config), base_checkpoint_hash=tree_hash(c.model),
                    tokenizer_hash=identity.tokenizer_identity(c.model)["hash"], hardware_hash=digest(c.hardware["hard"]))
    put(run_dir / "manifest.json", training)
    ckpt = run_dir / "round-1"
    put(ckpt / "lora/adapter_config.json", dict(peft_type="LORA"))
    put(ckpt / "lora/adapter_model.safetensors", "fixture adapter bytes")
    put(ckpt / "round_state.pt", "fixture state bytes")
    meta = dict(round=1, manifest_hash=digest(training), config_hash=training["config_hash"],
                adapter_hash=tree_hash(ckpt / "lora"), round_state_hash=file_hash(ckpt / "round_state.pt"))
    put(ckpt / "checkpoint.json", meta)
    return run_dir
