import copy
from contextlib import contextmanager
import random

import pytest
import torch

import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl import training
from bfas.mech_bfcl.training import chunked_hidden_gradient, frozen_hidden, mixture_loss, pass_schedules, token_schedule
from tools.mech_bfcl import parser


GPU_UUIDS = [
    "GPU-11111111-1111-1111-1111-111111111111",
    "GPU-22222222-2222-2222-2222-222222222222",
    "GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0",
    "GPU-44444444-4444-4444-4444-444444444444",
    "GPU-55555555-5555-5555-5555-555555555555",
]


@pytest.fixture
def gpu_guard(monkeypatch):
    """Every subprocess and /proc read is a stub, even on a host with no GPUs."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUIDS[2])
    outputs = {
        "--query-gpu=index,uuid": "\n".join(f"{i}, {GPU_UUIDS[i]}" for i in (4, 2, 0, 3, 1)),
        "--query-gpu=uuid,memory.free": "\n".join(f"{uuid}, {i + 10} MiB" for i, uuid in enumerate(GPU_UUIDS)),
        "--query-compute-apps=pid,gpu_uuid,used_memory": "",
    }
    cmdlines, reads, calls = {}, [], []

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert command[2:] == ["--format=csv,noheader"]
        assert kwargs == dict(capture_output=True, text=True, check=True)
        calls.append(command[1])
        return SimpleNamespace(stdout=outputs[command[1]])

    def read_bytes(path):
        assert path.parent.parent == Path("/proc") and path.name == "cmdline"
        pid = int(path.parent.name)
        reads.append(pid)
        result = cmdlines[pid]
        if isinstance(result, OSError):
            raise result
        return result

    def processes(rows):
        outputs["--query-compute-apps=pid,gpu_uuid,used_memory"] = "\n".join(
            f"{pid}, {GPU_UUIDS[index]}, {memory}" for pid, index, memory in rows)

    monkeypatch.setattr(training.subprocess, "run", run)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return SimpleNamespace(outputs=outputs, cmdlines=cmdlines, reads=reads, calls=calls,
                           processes=processes)


@pytest.mark.parametrize("visible", [GPU_UUIDS[2], "2", "GPU-8b270cf8", " 2 "])
def test_gpu_guard_ignores_other_gpus_and_holder(gpu_guard, monkeypatch, capsys, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    gpu_guard.processes([(100 + i, i, "20000 MiB") for i in range(5)])
    gpu_guard.cmdlines[102] = (
        b"python\0/home/xueqi/hq/projects/tc-alignment/tools/gpu_hold.py\0--mode\0guard\0")
    training.check_gpu_residency()
    assert gpu_guard.reads == [102]
    assert capsys.readouterr().out == f"Training GPU {GPU_UUIDS[2]}: 12 MiB free\n"


@pytest.mark.parametrize("visible", ["4,2", f"{GPU_UUIDS[2]},4", None])
def test_gpu_guard_checks_all_selected_gpus(gpu_guard, monkeypatch, capsys, visible):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        selected = set(range(5))
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        selected = {2, 4}
    gpu_guard.processes([(100 + i, i, "2048 MiB") for i in range(5)])
    gpu_guard.cmdlines.update({100 + i: b"python\0worker.py\0" for i in selected})
    with pytest.raises(RuntimeError, match="GPU compute processes are still resident") as exc:
        training.check_gpu_residency()
    output = capsys.readouterr().out
    for i in range(5):
        assert (f"pid={100 + i}" in str(exc.value)) == (i in selected)
        assert (GPU_UUIDS[i] in output) == (i in selected)
    assert set(gpu_guard.reads) == {100 + i for i in selected}


@pytest.mark.parametrize("occupancy", ["", "\n  \n"])
def test_gpu_guard_empty_gpu_logs_free_memory(gpu_guard, capsys, occupancy):
    gpu_guard.outputs["--query-compute-apps=pid,gpu_uuid,used_memory"] = occupancy
    training.check_gpu_residency()
    assert not gpu_guard.reads
    assert capsys.readouterr().out == f"Training GPU {GPU_UUIDS[2]}: 12 MiB free\n"


def test_gpu_guard_lists_foreign_processes_even_alongside_holder(gpu_guard):
    gpu_guard.processes([(10, 2, "70000 MiB"), (20, 2, "4096 MiB"), (30, 2, "0 MiB")])
    # Match the full command before shortening the diagnostic.
    gpu_guard.cmdlines[10] = b"python\0" + b"x" * 200 + b"\0gpu_hold.py\0--mode\0guard\0"
    gpu_guard.cmdlines[20] = b"python\0-m\0vllm.entrypoints.openai.api_server\0"
    gpu_guard.cmdlines[30] = b"python\0another_worker.py\0" + b"x" * 300
    with pytest.raises(RuntimeError) as exc:
        training.check_gpu_residency()
    message, *processes = str(exc.value).splitlines()
    assert message == "GPU compute processes are still resident; stop serving and wait before training"
    assert len(processes) == 2
    assert processes[0] == (f"pid=20 gpu={GPU_UUIDS[2]} memory=4096 MiB "
                            "cmd=python -m vllm.entrypoints.openai.api_server")
    assert f"pid=30 gpu={GPU_UUIDS[2]} memory=0 MiB cmd=python another_worker.py" in processes[1]
    assert len(processes[1].split("cmd=", 1)[1]) == 160
    assert processes[1].endswith("...")


@pytest.mark.parametrize("cmdline,diagnostic", [
    (PermissionError(), "<cmdline unavailable>"),
    (FileNotFoundError(), "<cmdline unavailable>"),
    (b"", "<empty cmdline>"),
    (b"python\0worker\xff.py\0", "python worker\ufffd.py"),
])
def test_gpu_guard_cannot_exempt_unknown_process(gpu_guard, cmdline, diagnostic):
    gpu_guard.processes([(20, 2, "[N/A]")])
    gpu_guard.cmdlines[20] = cmdline
    with pytest.raises(RuntimeError, match="GPU compute processes are still resident") as exc:
        training.check_gpu_residency()
    assert f"pid=20 gpu={GPU_UUIDS[2]} memory=[N/A] cmd={diagnostic}" in str(exc.value)


@pytest.mark.parametrize("visible", ["", "-1", "99", "GPU-missing", "GPU-", "2,99"])
def test_gpu_guard_rejects_unresolved_selection(gpu_guard, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError, match="Cannot resolve CUDA_VISIBLE_DEVICES"):
        training.check_gpu_residency()


def test_gpu_guard_propagates_query_failure(gpu_guard, monkeypatch):
    def fail(*args, **kwargs):
        raise training.subprocess.CalledProcessError(1, "nvidia-smi")
    monkeypatch.setattr(training.subprocess, "run", fail)
    with pytest.raises(training.subprocess.CalledProcessError):
        training.check_gpu_residency()


@pytest.mark.parametrize("foreign", [False, True])
def test_train_checks_gpu_and_logs_memory_before_model_load(gpu_guard, monkeypatch, tmp_path, capsys, foreign):
    monkeypatch.delenv("MECH_SERVING_PID", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    gpu_guard.processes([(20, 2, "1024 MiB"), (30, 0, "40000 MiB")])
    gpu_guard.cmdlines[20] = b"python\0worker.py\0" if foreign else b"python\0gpu_hold.py\0guard\0"
    loads = []

    class ModelLoadReached(Exception):
        pass

    def load_model(*args, **kwargs):
        loads.append(True)
        assert capsys.readouterr().out == f"Training GPU {GPU_UUIDS[2]}: 12 MiB free\n"
        raise ModelLoadReached

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: None),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(LoraConfig=None, get_peft_model=None))
    monkeypatch.setattr(training, "encode_rows", lambda *a: ([dict(id="row", prompt_ids=[1], target_ids=[2])], []))
    for arm in ("C", "D"):
        training.write_json(tmp_path / arm / "exercises.json", [dict(task_id="task", arm=arm, layer=0)])
        training.write_json(tmp_path / arm / "generation.json", dict(contexts_hash="shared", ready=True))
    args = SimpleNamespace(run_dir=tmp_path, arm="C", tokenizer="stub", model_path="stub",
                           tokens_per_step=1, passes=1, learning_rate=1e-5)
    with pytest.raises(RuntimeError if foreign else ModelLoadReached) as exc:
        training.train(args, dict(seed=0, support=["task"]))
    assert loads == ([] if foreign else [True])
    if foreign:
        assert "GPU compute processes are still resident" in str(exc.value)
        assert "pid=20" in str(exc.value)
    assert not (tmp_path / "C" / "training").exists()


@pytest.mark.parametrize("chunk", [1, 3, 20])
@pytest.mark.parametrize("softcap", [None, 2.0])
def test_full_vocab_chunked_value_and_gradient_equal_dense(chunk, softcap):
    torch.manual_seed(19)
    head = torch.nn.Linear(5, 17, bias=False)
    head.requires_grad_(False)
    seen = []
    hook = head.register_forward_hook(lambda m, a, o: seen.append(a[0].shape[0]))
    hidden = torch.randn(11, 5, requires_grad=True)
    ref = torch.randn_like(hidden)
    targets = torch.randint(0,17,(11,))
    def project(h):
        x = head(h)
        return x if softcap is None else (x/softcap).tanh()*softcap
    dense = mixture_loss(project(hidden), project(ref), targets)
    expected, = torch.autograd.grad(dense, hidden)
    seen.clear()
    gradient, loss = chunked_hidden_gradient(hidden, ref, head, targets, chunk_size=chunk, softcap=softcap)
    assert max(seen) <= chunk
    torch.testing.assert_close(gradient, expected)
    assert loss == pytest.approx(dense.item(), rel=1e-6)
    assert head.weight.grad is None
    hook.remove()


def test_non_topk_reference_tail_affects_loss():
    logits = torch.tensor([[4., 0., -2., -6.]], requires_grad=True)
    a = torch.tensor([[8., 1., 0., -1.]])
    b = torch.tensor([[8., 1., -1., 0.]])
    assert mixture_loss(logits,a,torch.tensor([0])) != mixture_loss(logits,b,torch.tensor([0]))


def test_reference_detached_and_explicit_half_mixture():
    live = torch.tensor([[.3, -.5]], requires_grad=True)
    ref = torch.tensor([[.2, .4]], requires_grad=True)
    target = torch.tensor([1])
    loss = mixture_loss(live, ref, target)
    logp = live.log_softmax(-1)
    expected = -.5*logp[0,1] - .5*(ref.detach().softmax(-1)*logp).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert ref.grad is None


def test_token_accumulation_exact_one_pass_partial_last_row():
    rows = [dict(target_ids=list(range(n))) for n in (3,7,2)]
    steps = token_schedule(rows, 11, 4)
    assert [sum(s['end']-s['start'] for s in b) for b in steps] == [4,4,3]
    consumed = [(s['row'], p) for b in steps for s in b for p in range(s['start'],s['end'])]
    assert len(consumed) == len(set(consumed)) == 11
    assert consumed[-1] == (2,0)
    with pytest.raises(ValueError, match="Insufficient"):
        token_schedule(rows,13,4)


def test_multi_pass_schedules_are_reproducible_and_reset_exposure():
    rows = [dict(id=str(i), target_ids=list(range(n))) for i, n in enumerate((3, 7, 2, 5, 6))]
    original = copy.deepcopy(rows)
    schedules = pass_schedules(rows, 11, 4, passes=4, seed=19)
    assert schedules == pass_schedules(rows, 11, 4, passes=4, seed=19)
    assert schedules[:1] == pass_schedules(rows, 11, 4, seed=19)
    assert schedules != pass_schedules(rows, 11, 4, passes=4, seed=20)
    assert rows == original
    assert len({tuple(s["row_order"]) for s in schedules}) > 1
    for i, schedule in enumerate(schedules):
        assert schedule["pass_number"] == i + 1
        assert schedule["seed"] == 19 + i
        order = list(range(len(rows)))
        random.Random(19 + i).shuffle(order)
        assert schedule["row_order"] == order
        steps = schedule["steps"]
        assert [sum(s["end"] - s["start"] for s in step) for step in steps] == [4, 4, 3]
        consumed = [(s["row"], p) for step in steps for s in step for p in range(s["start"], s["end"])]
        expected = [(r, p) for r in order for p in range(len(rows[r]["target_ids"]))][:11]
        assert consumed == expected
        assert len(consumed) == len(set(consumed)) == 11


def test_train_cli_dose_defaults_and_overrides():
    defaults = parser().parse_args(["train", "--arm", "C"])
    assert (defaults.passes, defaults.learning_rate, defaults.tokens_per_step) == (1, 1e-5, 512)
    custom = parser().parse_args(["train", "--arm", "D", "--passes", "20", "--learning-rate", "1e-4"])
    assert (custom.passes, custom.learning_rate) == (20, 1e-4)


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_train_cli_rejects_invalid_passes(value):
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--arm", "C", "--passes", value])


@pytest.mark.parametrize("passes", [0, -1])
def test_schedules_reject_nonpositive_passes(passes):
    with pytest.raises(ValueError, match="Positive number of training passes"):
        pass_schedules([dict(target_ids=[1])], 1, passes=passes)


@pytest.fixture
def tiny_training(monkeypatch, tmp_path):
    """Run the real loss and AdamW on CPU with a tiny adapter/backbone stub."""
    monkeypatch.delenv("MECH_SERVING_PID", raising=False)
    monkeypatch.setattr(training, "check_gpu_residency", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    if hasattr(torch, "accelerator"):
        monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda **kwargs: None)
    tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    models, optimizers = [], []
    adamw = torch.optim.AdamW

    def optimizer(*args, **kwargs):
        instance = adamw(*args, **kwargs)
        optimizers.append(instance)
        return instance

    monkeypatch.setattr(torch.optim, "AdamW", optimizer)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(9, 3)
            for name in training.TARGET_MODULES:
                self.add_module(name, torch.nn.Linear(3, 3, bias=False))
            self.adapter = torch.nn.Parameter(torch.zeros(3))
            self.enabled = True
            self.inputs = []

        def forward(self, input_ids, use_cache):
            assert input_ids.device.type == "cpu"
            if self.enabled:
                self.inputs.append(input_ids.clone())
            hidden = self.q_proj(self.embedding(input_ids))
            return SimpleNamespace(last_hidden_state=hidden + (self.adapter if self.enabled else 0))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()
            self.lm_head = torch.nn.Linear(3, 9, bias=False)
            self.config = SimpleNamespace()
            self.requires_grad_(False)
            self.model.adapter.requires_grad_(True)

        def get_base_model(self):
            return self

        @contextmanager
        def disable_adapter(self):
            self.model.enabled = False
            try:
                yield
            finally:
                self.model.enabled = True

        def gradient_checkpointing_enable(self, **kwargs):
            pass

        def enable_input_require_grads(self):
            pass

        def save_pretrained(self, path, **kwargs):
            path.mkdir(parents=True)

    def load_model(*args, **kwargs):
        model = Model()
        models.append(model)
        return model

    tokenizer = SimpleNamespace(save_pretrained=lambda *a: None)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        LoraConfig=lambda **kwargs: None, get_peft_model=lambda model, config: model))
    encoded = {arm: [dict(id=f"{arm}_{i}", prompt_ids=[1, 2], target_ids=[3 + i] * n)
                     for i, n in enumerate(lengths)]
               for arm, lengths in (("C", (1, 2, 3, 4)), ("D", (3, 4, 5, 6)))}
    monkeypatch.setattr(training, "encode_rows", lambda bank, tokenizer: (
        copy.deepcopy(encoded[bank[0]["arm"]]), []))
    for arm in ("C", "D"):
        training.write_json(tmp_path / arm / "exercises.json", [dict(task_id="task", arm=arm, layer=0)])
        training.write_json(tmp_path / arm / "generation.json", dict(contexts_hash="shared", ready=True))

    def run(arm="C", **dose):
        args = parser().parse_args(["train", "--run-dir", str(tmp_path), "--arm", arm])
        for key, value in {"tokens_per_step": 4, "passes": 3, "learning_rate": 1e-3, **dose}.items():
            setattr(args, key, value)
        args.position_chunk = 2
        training.train(args, dict(seed=0, support=["task"]))

    return SimpleNamespace(run=run, models=models, optimizers=optimizers, encoded=encoded)


@pytest.mark.parametrize("passes", [1, 3])
def test_train_accounts_for_all_passes_and_matches_arms(tiny_training, tmp_path, passes):
    for arm in ("C", "D"):
        tiny_training.run(arm, passes=passes)
        plan_bytes = (tmp_path / "training_plan.json").read_bytes()
        if arm == "C":
            frozen = plan_bytes
        else:
            assert plan_bytes == frozen
        plan = training.read_json(tmp_path / "training_plan.json")
        directory = tmp_path / arm / "training"
        config = training.read_json(directory / "config.json")
        metrics = training.read_json(directory / "metrics.json")
        steps = training.read_json(directory / "steps.json")
        for artifact in (plan, config, metrics):
            assert artifact["passes"] == passes
            assert artifact["learning_rate"] == 1e-3
            assert artifact["tokens_per_step"] == 4
        assert metrics["supervised_tokens_per_pass"] == plan["common_supervised_cap"] == 10
        assert metrics["supervised_tokens"] == 10 * passes
        assert metrics["optimizer_steps"] == len(steps) == 3 * passes
        assert [s["step"] for s in steps] == list(range(1, 3 * passes + 1))
        for p in range(1, passes + 1):
            assert [s["supervised_tokens"] for s in steps if s["pass_number"] == p] == [4, 4, 2]
        model = tiny_training.models[-1]
        optimizer = tiny_training.optimizers[-1]
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.state[model.model.adapter]["step"].item() == 3 * passes
        assert model.model.adapter.detach().abs().sum() > 0
        assert metrics["input_tokens"] == sum(ids.numel() for ids in model.model.inputs)
        assert metrics["unique_input_tokens"] == sum(2 + len(r["target_ids"]) for r in tiny_training.encoded[arm])
        schedules = plan["schedules"][arm]
        assert len(schedules) == passes
        exposed_rows = {s["row"] for schedule in schedules for step in schedule["steps"] for s in step}
        assert metrics["data_examples"] == len(exposed_rows)
        assert plan["row_ids"][arm] == [r["id"] for r in tiny_training.encoded[arm]]


@pytest.mark.parametrize("first_arm", ["C", "D"])
@pytest.mark.parametrize("changed", [dict(passes=4), dict(learning_rate=2e-3), dict(tokens_per_step=5)])
def test_train_rejects_cross_arm_dose_changes(tiny_training, tmp_path, first_arm, changed):
    tiny_training.run(first_arm)
    frozen = (tmp_path / "training_plan.json").read_bytes()
    other_arm = "D" if first_arm == "C" else "C"
    with pytest.raises(ValueError, match="C/D exposure plan changed after it was frozen"):
        tiny_training.run(other_arm, **changed)
    assert (tmp_path / "training_plan.json").read_bytes() == frozen
    assert len(tiny_training.models) == len(tiny_training.optimizers) == 1
    assert not (tmp_path / other_arm / "training").exists()


@pytest.mark.parametrize("rate", [0, -1e-5, float("nan"), float("inf"), -float("inf")])
def test_train_rejects_invalid_learning_rate(tiny_training, tmp_path, rate):
    with pytest.raises(ValueError, match="Learning rate must be positive and finite"):
        tiny_training.run(learning_rate=rate)
    assert not tiny_training.models
    assert not (tmp_path / "training_plan.json").exists()


def test_reference_uses_same_backbone_disabled_adapter_and_restores_mode():
    from contextlib import contextmanager
    from types import SimpleNamespace
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor(2.),requires_grad=False)
            self.adapter=torch.nn.Parameter(torch.tensor(3.))
            self.enabled=True
            self.calls=[]
        def get_base_model(self):
            return SimpleNamespace(model=self)
        @contextmanager
        def disable_adapter(self):
            self.enabled=False
            try:
                yield
            finally:
                self.enabled=True
        def forward(self,input_ids,use_cache):
            self.calls.append((self.enabled,self.training,torch.is_grad_enabled()))
            out=input_ids.float()[...,None]*(self.weight+(self.adapter if self.enabled else 0))
            return SimpleNamespace(last_hidden_state=out)
    model=Model().train()
    ids=torch.tensor([[1,2,3]])
    reference=frozen_hidden(model,ids,slice(1,3))
    torch.testing.assert_close(reference,torch.tensor([[4.],[6.]]))
    assert not reference.requires_grad
    assert model.calls == [(False,False,False)]
    assert model.training and model.enabled
    assert model.adapter.item() == 3 and model.weight.item() == 2
