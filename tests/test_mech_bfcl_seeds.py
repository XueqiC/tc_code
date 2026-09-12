"""CPU-only CLI routing and served-adapter identity for training-seed repeats."""
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl import pipeline, training
from bfas.mech_bfcl.common import STUDENT, write_json
from tools import mech_bfcl as cli


@pytest.mark.parametrize("command,arm", [("train", "C"), ("evaluate", "C"), ("evaluate", "D"), ("evaluate", "base")])
@pytest.mark.parametrize("split_seed,train_seed", [(0, None), (19, None), (19, 19), (19, 0), (0, -3)])
def test_cli_resolves_training_seed_without_changing_split(tmp_path, monkeypatch, command, arm, split_seed, train_seed):
    split = dict(seed=split_seed)
    path = tmp_path / "splits.json"
    write_json(path, split)
    calls = []
    monkeypatch.setattr(cli, "setup_harness", lambda *args: None)
    monkeypatch.setattr(cli, "bind_run", lambda directory, splits: None)
    monkeypatch.setattr(training if command == "train" else pipeline, command,
                        lambda args, splits: calls.append((args, copy.deepcopy(splits))))
    argv = [command, "--run-dir", str(tmp_path), "--splits", str(path), "--arm", arm, "--seed", str(split_seed)]
    if train_seed is not None:
        argv += ["--train-seed", str(train_seed)]
    assert cli.main(argv) == 0
    args, actual_split = calls[0]
    seed = split_seed if train_seed is None else train_seed
    assert args.train_seed == seed
    assert args.seed == split_seed
    assert actual_split == split
    alias = f"mech-{arm}" + (f"-s{seed}" if seed != split_seed else "")
    assert args.served_model == (alias if command == "evaluate" and arm != "base" else STUDENT)


@pytest.mark.parametrize("command", ["train", "evaluate"])
def test_cli_train_seed_requires_integer(command):
    with pytest.raises(SystemExit):
        cli.parser().parse_args([command, "--arm", "C", "--train-seed", "1.5"])


@pytest.mark.parametrize("arm", ["C", "D"])
@pytest.mark.parametrize("split_seed,train_seed,folder,alias_suffix", [
    (0, None, "training", ""), (19, 19, "training", ""),
    (19, 0, "training_s0", "-s0"), (0, -3, "training_s-3", "-s-3"),
])
def test_server_checks_seed_specific_alias_and_adapter(tmp_path, monkeypatch, arm, split_seed, train_seed, folder, alias_suffix):
    artifact = tmp_path / arm / folder / "adapter"
    write_json(artifact / "adapter_config.json", {})
    alias = f"mech-{arm}{alias_suffix}"
    model = dict(id=alias, root=str(artifact))
    monkeypatch.setattr(pipeline.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.StringIO(json.dumps(dict(data=[model]))))
    args = SimpleNamespace(run_dir=tmp_path, arm=arm, seed=split_seed, train_seed=train_seed,
                           served_model=alias, base_url="http://stub.invalid/v1")
    assert pipeline.check_server(args, arm) == model
    model["root"] = str(tmp_path / arm / "wrong/adapter")
    with pytest.raises(ValueError, match="trained adapter path"):
        pipeline.check_server(args, arm)
    model.update(root=str(artifact), id="custom-alias")
    args.served_model = "custom-alias"
    if alias_suffix:
        with pytest.raises(ValueError, match=f"must be served as {alias}"):
            pipeline.check_server(args, arm)
    else:
        assert pipeline.check_server(args, arm) == model  # Existing default-seed overrides still work.


def test_base_server_is_independent_of_training_seed(tmp_path, monkeypatch):
    model = dict(id=STUDENT, root="base")
    monkeypatch.setattr(pipeline.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.StringIO(json.dumps(dict(data=[model]))))
    args = SimpleNamespace(run_dir=tmp_path, seed=0, train_seed=7, served_model=STUDENT,
                           base_url="http://stub.invalid/v1")
    assert pipeline.check_server(args, "base") == model


@pytest.mark.parametrize("seed", [0, 7])
def test_midpoint_cli_and_server_require_mid_alias_and_adapter(tmp_path, monkeypatch, seed):
    split_path = tmp_path / "splits.json"
    write_json(split_path, dict(seed=0))
    calls = []
    monkeypatch.setattr(cli, "setup_harness", lambda *a: None)
    monkeypatch.setattr(cli, "bind_run", lambda *a: None)
    monkeypatch.setattr(pipeline, "evaluate", lambda args, splits: calls.append(args))
    cli.main(["evaluate", "--arm", "C", "--run-dir", str(tmp_path), "--splits", str(split_path),
              "--train-seed", str(seed), "--checkpoint", "mid", "--base-url", "http://stub/v1"])
    args = calls[0]
    alias = "mech-C" + ("-s7" if seed else "") + "-mid"
    assert args.served_model == alias
    folder = "training_s7" if seed else "training"
    artifact = tmp_path / "C" / folder / "adapter_mid"
    write_json(artifact / "adapter_config.json", {})
    model = dict(id=alias, root=str(artifact))
    monkeypatch.setattr(pipeline.urllib.request, "urlopen", lambda *a, **kw: io.StringIO(json.dumps(dict(data=[model]))))
    assert pipeline.check_server(args, "C") == model
    model["root"] = str(artifact.with_name("adapter"))
    with pytest.raises(ValueError, match="trained adapter path"):
        pipeline.check_server(args, "C")
    model.update(root=str(artifact), id="mech-C")
    args.served_model = "mech-C"
    with pytest.raises(ValueError, match="must be served as"):
        pipeline.check_server(args, "C")
