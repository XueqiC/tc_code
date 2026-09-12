"""CPU-only objective audit, causal alignment, provenance, and read-only CLI tests."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl import training
from bfas.mech_bfcl.common import STUDENT, digest, read_json, training_directory, write_json
from tools import mech_bfcl as cli


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4)
        self.adapter = torch.nn.Parameter(torch.tensor([1.2, -0.7, 0.3, -1.0]))
        self.enabled = True
        self.calls = []

    def forward(self, input_ids, use_cache):
        assert input_ids.device.type == "cpu" and use_cache is False
        self.calls.append((self.enabled, self.training, torch.is_grad_enabled(), input_ids.tolist()))
        hidden = self.embedding(input_ids).cumsum(1) / 10
        return SimpleNamespace(last_hidden_state=hidden + (self.adapter if self.enabled else 0))


class TinyModel(torch.nn.Module):
    def __init__(self, softcap=2.0):
        super().__init__()
        torch.manual_seed(27)
        self.model = TinyDecoder()
        self.lm_head = torch.nn.Linear(4, 7, bias=False)
        self.config = SimpleNamespace(final_logit_softcapping=softcap, _commit_hash="saved-revision")

    def get_base_model(self):
        return self

    @contextmanager
    def disable_adapter(self):
        enabled = self.model.enabled
        self.model.enabled = False
        try:
            yield
        finally:
            self.model.enabled = enabled


ROWS = [dict(id="a", prompt_ids=[1, 2], target_ids=[3]),
        dict(id="b", prompt_ids=[2], target_ids=[1, 2, 3]),
        dict(id="c", prompt_ids=[3, 1, 2], target_ids=[2, 1])]


def dense_expected(model, row):
    """Independent dense forward and explicit probability formulas, including softcap."""
    p, y = row["prompt_ids"], row["target_ids"]
    with torch.no_grad():
        inputs = torch.tensor([p + y[:-1]])
        reference = model.model.embedding(inputs).cumsum(1)[0, len(p)-1:] / 10
        logits = [model.lm_head(reference), model.lm_head(reference + model.model.adapter)]
        softcap = model.config.final_logit_softcapping
        if softcap is not None:
            logits = [(x / softcap).tanh() * softcap for x in logits]
        q = logits[0].float().softmax(-1)
        logq = logits[0].float().log_softmax(-1)
        entropy = -(q * logq).sum(-1)
        expected = {}
        for state, scores in zip(("before", "after"), logits):
            logp = scores.float().log_softmax(-1)
            nll = -logp[torch.arange(len(y)), torch.tensor(y)]
            ce = -(q * logp).sum(-1)
            expected[state] = dict(teacher_nll=nll, reference_ce=ce, base_entropy=entropy,
                                   kl=(q * (logq - logp)).sum(-1), mixture=(nll + ce) / 2,
                                   target_argmax=scores.argmax(-1).eq(torch.tensor(y)))
        return expected, logits


@pytest.mark.parametrize("chunk", [1, 2, 9])
@pytest.mark.parametrize("softcap", [None, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_audit_matches_dense_objective_and_token_weighted_means(chunk, softcap, dtype):
    model = TinyModel(softcap).to(dtype).train()
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    projections = []
    hook = model.lm_head.register_forward_hook(
        lambda m, args, output: projections.append((len(args[0]), m.training, torch.is_grad_enabled())))
    report = training.audit_encoded_rows(model, ROWS, chunk_size=chunk)
    hook.remove()
    assert projections and all(n <= chunk and not mode and not grad for n, mode, grad in projections)
    assert model.training and model.model.enabled
    assert len(model.model.calls) == 2 * len(ROWS)
    assert all(not mode and not grad for _, mode, grad, _ in model.model.calls)
    for i, row in enumerate(ROWS):
        before_call, after_call = model.model.calls[2*i:2*i+2]
        assert before_call[0] is False and after_call[0] is True
        assert before_call[3] == after_call[3] == [row["prompt_ids"] + row["target_ids"][:-1]]
        actual = report["exercises"][i]
        assert actual["id"] == row["id"]
        assert actual["supervised_tokens"] == len(row["target_ids"])
        expected, logits = dense_expected(model, row)
        for index, token in enumerate(actual["tokens"]):
            assert token["target_index"] == index
            assert token["target_id"] == row["target_ids"][index]
            assert token["prediction_position"] == len(row["prompt_ids"])-1+index
            for state in expected:
                for key, values in expected[state].items():
                    assert token[state][key] == pytest.approx(values[index].item(), abs=1e-6)
                assert isinstance(token[state]["target_argmax"], bool)
        for state in expected:
            for key, values in expected[state].items():
                summary_key = "target_argmax_rate" if key == "target_argmax" else key
                assert actual[state][summary_key] == pytest.approx(values.float().mean().item(), abs=1e-6)
        optimizer_loss = training.mixture_loss(logits[1], logits[0], torch.tensor(row["target_ids"]))
        assert actual["after"]["mixture"] * len(row["target_ids"]) == pytest.approx(optimizer_loss.item())
        assert actual["before"]["kl"] == 0
        assert actual["before"]["reference_ce"] == actual["before"]["base_entropy"]
        assert actual["after"]["base_entropy"] == actual["before"]["base_entropy"]
        assert actual["base_target_argmax_rate"] == actual["before"]["target_argmax_rate"]
    overall = report["overall"]
    assert overall["supervised_tokens"] == 6
    for state in ("before", "after"):
        for key, actual in overall[state].items():
            weighted = sum(row[state][key] * row["supervised_tokens"] for row in report["exercises"]) / 6
            assert actual == pytest.approx(weighted)
    unweighted = sum(row["after"]["teacher_nll"] for row in report["exercises"]) / 3
    assert overall["after"]["teacher_nll"] != pytest.approx(unweighted, abs=1e-6)
    assert overall["base_target_argmax_rate"] == overall["before"]["target_argmax_rate"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


def test_uniform_base_has_entropy_floor_even_when_student_fits_target():
    head = torch.nn.Identity()
    base = torch.zeros(2, 4, requires_grad=True)
    student = torch.tensor([[12., 0., 0., 0.], [12., 0., 0., 0.]], requires_grad=True)
    values = training.chunked_loss_audit(student, base, head, torch.tensor([0, 1]), chunk_size=1)
    assert values["before"]["base_entropy"] == pytest.approx([torch.log(torch.tensor(4.)).item()] * 2)
    assert values["before"]["target_argmax"] == [True, False]
    assert values["after"]["teacher_nll"][0] < 0.001
    assert values["after"]["reference_ce"][0] > 8
    assert values["after"]["mixture"][0] > 4
    assert student.grad is base.grad is None


def test_audit_restores_model_mode_on_failure():
    model = TinyModel().train()
    with pytest.raises(ValueError, match="Invalid position chunk"):
        training.audit_encoded_rows(model, ROWS, chunk_size=0)
    assert model.training and model.model.enabled


@pytest.fixture
def audit_run(tmp_path, monkeypatch):
    """Real native_pair/encode_rows; only the native renderer and loaders are fakes."""
    monkeypatch.setitem(sys.modules, "tools.bfcl_pool_render_gemma4", SimpleNamespace(
        render_pair=lambda tokenizer, messages, functions, demo:
        (messages[0]["content"], demo["content"] + " <turn|>")))

    def encode(text, add_special_tokens):
        assert add_special_tokens is False
        return [int(token) for token in text.split()]

    tokenizer = SimpleNamespace(encode=encode)
    loads, models = [], []

    def load_tokenizer(path, **kwargs):
        loads.append(("tokenizer", path, kwargs))
        return tokenizer

    def load_model(path, **kwargs):
        assert not torch.is_grad_enabled()
        loads.append(("base", path, kwargs))
        model = TinyModel()
        models.append(model)
        return model

    def load_adapter(base, path, **kwargs):
        assert not torch.is_grad_enabled()
        loads.append(("adapter", path, kwargs))
        return base

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=load_adapter)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-caller-selection")

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only audit invoked training, optimizer, harness, lock, or run binding")

    for name in ("setup_harness", "bind_run", "lock"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(training, "train", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)

    def prepare(arm="C", split_seed=0, train_seed=None):
        splits = dict(seed=split_seed, support=["task"])
        write_json(tmp_path / "splits.json", splits)
        bank = [dict(id=row["id"], task_id="task", arm=arm, layer=0,
                     validation=dict(valid=True), functions=[],
                     messages=[dict(role="user", content=" ".join(map(str, row["prompt_ids"])))],
                     demo=dict(kind="abstain", text=" ".join(map(str, row["target_ids"]))))
                for row in reversed(ROWS)]
        bank += [dict(bank[0], id="invalid", validation=dict(valid=False)),
                 dict(bank[0], id="overlong", messages=[dict(role="user", content="1 " * 8193)]),
                 dict(bank[0], id="empty", messages=[dict(role="user", content="")])]
        rows, excluded = training.encode_rows(bank, tokenizer)
        assert rows == [dict(row, context_bucket=4096) for row in ROWS]
        write_json(tmp_path / arm / "exercises.json", bank)
        write_json(tmp_path / "training_plan.json", dict(
            seed=split_seed, tokenizer="saved-tokenizer", bank_hashes={arm: digest(bank)},
            encoded_hashes={arm: digest(rows)}, row_ids={arm: [row["id"] for row in rows]},
            common_supervised_cap=2, passes=5, excluded={arm: excluded}))
        seed = split_seed if train_seed is None else train_seed
        directory = training_directory(tmp_path, arm, seed, split_seed)
        write_json(directory / "config.json", dict(train_seed=seed, model_path="saved-base",
                                                    model_commit="saved-revision", position_chunk=2))
        write_json(directory / "adapter/adapter_config.json", {})
        write_json(directory / "steps.json", [dict(loss=0.001)])
        write_json(tmp_path / "protocol.json", dict(untouched=True))
        return directory

    def run(arm="C", split_seed=0, train_seed=None, extra=()):
        argv = ["audit-loss", "--run-dir", str(tmp_path), "--splits", str(tmp_path / "splits.json"),
                "--seed", str(split_seed), "--arm", arm]
        if train_seed is not None:
            argv += ["--train-seed", str(train_seed)]
        return cli.main(argv + list(extra))

    return SimpleNamespace(prepare=prepare, run=run, loads=loads, models=models)


@pytest.mark.parametrize("arm", ["C", "D"])
@pytest.mark.parametrize("split_seed,train_seed", [(0, None), (19, None), (19, 19), (19, 0), (0, -3)])
@pytest.mark.parametrize("override", [False, True])
def test_cli_audit_only_writes_report_and_selects_saved_adapter(
        audit_run, tmp_path, capsys, monkeypatch, arm, split_seed, train_seed, override):
    directory = audit_run.prepare(arm, split_seed, train_seed)
    checkpoint, extra = directory / "adapter", []
    if override:
        checkpoint = tmp_path / "custom-adapter"
        write_json(checkpoint / "adapter_config.json", {})
        extra = ["--checkpoint", str(checkpoint), "--model-path", "relocated-base",
                 "--tokenizer", "relocated-tokenizer", "--position-chunk", "1"]
    # Cover initial output and replacement of an old audit alike.
    if override:
        write_json(directory / "loss_audit.json", dict(old=True))
    original = {p.relative_to(tmp_path): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in tmp_path.rglob("*") if p.is_file()}
    entries = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    writes = []

    # Path.write_text observes attempted temporary-file writes as well as final files.
    write_text = Path.write_text

    def tracked_write(path, *args, **kwargs):
        writes.append(path)
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", tracked_write)
    assert audit_run.run(arm, split_seed, train_seed, extra) == 0
    output = directory / "loss_audit.json"
    assert writes == [output]
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*")} == entries | {output.relative_to(tmp_path)}
    for relative, (content, mtime) in original.items():
        if relative != output.relative_to(tmp_path):
            assert (tmp_path / relative).read_bytes() == content
            assert (tmp_path / relative).stat().st_mtime_ns == mtime
    report = read_json(output)
    assert report["arm"] == arm
    assert report["train_seed"] == (split_seed if train_seed is None else train_seed)
    assert report["checkpoint"] == str(checkpoint)
    assert report["model_commit"] == "saved-revision"
    assert report["overall"]["supervised_tokens"] == 6  # Every exercise once, regardless of cap/passes.
    assert [row["id"] for row in report["exercises"]] == [row["id"] for row in ROWS]
    assert [row["id"] for row in report["excluded"]] == ["empty", "overlong"]
    assert report["invalid_row_ids"] == ["invalid"]
    assert report["training_plan_hash"] == digest(read_json(tmp_path / "training_plan.json"))
    assert report["mixture_weights"] == [0.5, 0.5]
    assert audit_run.loads == [
        ("tokenizer", "relocated-tokenizer" if override else "saved-tokenizer", dict(local_files_only=True)),
        ("base", "relocated-base" if override else "saved-base",
         dict(torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa",
              device_map={"": 0}, revision="saved-revision")),
        ("adapter", str(checkpoint), dict(is_trainable=False, local_files_only=True)),
    ]
    assert report["position_chunk"] == (1 if override else 2)
    assert training.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-caller-selection"
    assert len(audit_run.models) == 1
    assert all(not p.requires_grad and p.grad is None for p in audit_run.models[0].parameters())
    output_text = capsys.readouterr().out
    for label in ("teacher NLL", "ref CE", "entropy floor", "KL", "mixture", "target argmax",
                  "before", "after", "Base already predicts target"):
        assert label in output_text
    exercise_lines = output_text.split("Per exercise (before -> after):\n", 1)[1].splitlines()
    assert len(exercise_lines) == 4
    assert [line.split()[0] for line in exercise_lines[1:]] == ["a", "b", "c"]


@pytest.mark.parametrize("changed", ["bank", "encoding", "split_seed", "train_seed", "checkpoint", "cuda"])
def test_audit_rejects_invalid_inputs_without_writes(audit_run, tmp_path, monkeypatch, changed):
    directory = audit_run.prepare()
    extra = []
    if changed == "bank":
        path = tmp_path / "C/exercises.json"
        value = read_json(path)
        value[0]["demo"]["text"] = "1"
        write_json(path, value)
    elif changed in ("encoding", "split_seed"):
        path = tmp_path / "training_plan.json"
        value = read_json(path)
        if changed == "encoding":
            value["encoded_hashes"]["C"] = "changed"
        else:
            value["seed"] = 123
        write_json(path, value)
    elif changed == "train_seed":
        path = directory / "config.json"
        value = read_json(path)
        write_json(path, dict(value, train_seed=123))
    elif changed == "checkpoint":
        extra = ["--checkpoint", str(tmp_path / "missing")]
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        audit_run.run(extra=extra)
    assert not audit_run.models
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == original


@pytest.mark.parametrize("extra", [["--arm", "base"], ["--train-seed", "1.5"],
                                  ["--position-chunk", "0"], ["--position-chunk", "-1"]])
def test_audit_cli_rejects_invalid_options(extra):
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["audit-loss", "--arm", "C", *extra])


def test_audit_saved_tokenizer_default_does_not_change_training_default():
    assert cli.parser().parse_args(["audit-loss", "--arm", "C"]).tokenizer is None
    assert cli.parser().parse_args(["train", "--arm", "C"]).tokenizer == STUDENT
