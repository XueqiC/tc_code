"""C26-D portable scoring identities and explicitly audited ALFWorld continuity.

No git command, environment startup, model import or CUDA query is needed.
Hardware facts are supplied by the caller and checked by RTD's device-class
validator. Supplements are read-only inputs: evaluation never invents an audit.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

from ..hardware import checked_hardware
from ..identity import verified_checkpoint
from ..persistence import manifest_digest, digest, file_hash, tree_hash
from ..scoring_scope import _ScientificAST


VERSION = "alfworld-evaluation-scoring-c26d-v1"
SUPPLEMENT_VERSION = "alfworld-audited-scoring-continuity-v1"
OFFICIAL_CONFIG = dict(benchmark="alfworld", alfworld_evaluation_split="valid_seen",
    alfworld_expected_eval_tasks=140, alfworld_max_episode_steps=40,
    alfworld_student_react=True, evaluation_temperature=0.0,
    max_action_tokens_by_benchmark={"alfworld": {"agent_action": 256}},
    max_context_tokens=32768)
WORLD_NAMES = ("game.tw-pddl", "traj_data.json", "initial_state.pddl")

# As in BFCL scoring_scope, project only mixed operational/scientific modules.
# Missing/duplicate selectors fail closed. Imports referenced by selectors bind
# too. No existing BFCL projection or version is extended or reinterpreted.
SCOPES = {
    "src/bfas/adapters/alfworld.py": (
        "TEACHER_REACT_INSTRUCTION", "TEACHER_REACT_PROMPT", "TEACHER_REACT_EXAMPLES",
        "PROMPT", "_ACTION_MARKER_RE", "_GOAL_RE", "_goal_line", "_obs_with_goal",
        "_category", "_game_ids", "_worker_config", "_worker", "_worker_episode",
        "_rpc_timeout", "ALFWorldRPCError", "_EnvBridge._rpc_error_type",
        "_EnvBridge._configure_rpc", "_EnvBridge._begin_rpc", "_EnvBridge._rpc_failure",
        "_EnvBridge._read_lines", "_EnvBridge._read", "_EnvBridge.step",
        "ALFWorldAdapter._render", "ALFWorldAdapter._pick_command",
        "ALFWorldAdapter._teacher_command_text", "ALFWorldAdapter._teacher_command"),
    "src/alfworld_eval.py": ("pick_command",),
    "src/bfas/adapter.py": ("BenchmarkAdapter.evaluate", "BenchmarkAdapter.prepare_renderer"),
    "src/bfas/cc_pairs.py": ("thinking_off",),
    "tools/behavior_atom/gpu_driver.py": ("_thinking_off",),
    "src/bfas/rtd/benchmarks/alfworld_support.py": (
        "prompt_messages", "FrozenRenderer", "EnvironmentUnavailable", "BoundedEnvBridge._rpc_error_type"),
    "src/bfas/rtd/benchmarks/alfworld_evaluation.py": (
        "Generation", "_checked_state", "official_episode", "validate_records",
        "aggregate_records", "compare_base", "HFBackend", "VLLMBackend", "EvaluationEnvBridge.__init__"),
    "src/bfas/rtd/benchmarks/alfworld_server.py": ("SERVER_VERSION", "checked_server_identity"),
}


def scoring_projection(name, content):
    if name not in SCOPES:
        return None
    tree = ast.parse(content, filename=name)
    selected = []
    for symbol in SCOPES[name]:
        parts, body = symbol.split("."), tree.body
        if len(parts) == 2:
            classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == parts[0]]
            if len(classes) != 1:
                raise ValueError(f"scoring scope requires one class: {name}:{parts[0]}")
            # Base-class changes can affect inherited parsing/render behavior.
            selected.extend(deepcopy(classes[0].bases))
            body = classes[0].body
        nodes = [n for n in body if
            (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == parts[-1]) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == parts[-1]
                                              for t in n.targets))]
        if len(nodes) != 1:
            raise ValueError(f"scoring scope requires one definition: {name}:{symbol}")
        selected.append(_ScientificAST().visit(deepcopy(nodes[0])))
    if name == "src/bfas/rtd/benchmarks/alfworld_evaluation.py":
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate"]
        if len(functions) != 1:
            raise ValueError("scoring scope requires one campaign coordinator")
        nodes = list(ast.walk(functions[0]))
        # Bind scientific dispatch as BFCL binds the merge/evaluator invocation,
        # while tag leases, accounting and reuse logging remain operational.
        for symbol in ("HFBackend", "official_episode", "aggregate_records"):
            calls = [n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == symbol]
            if len(calls) != 1:
                raise ValueError(f"scoring scope requires one {symbol} invocation")
            selected.append(deepcopy(calls[0]))
        factories = [n for n in nodes if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == "factory" for t in n.targets)]
        if len(factories) != 1:
            raise ValueError("scoring scope requires one environment factory selection")
        selected.append(deepcopy(factories[0]))
    used = {n.id for item in selected for n in ast.walk(item) if isinstance(n, ast.Name)}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            node = deepcopy(node)
            node.names = [a for a in node.names if (a.asname or a.name.split(".")[0]) in used]
            if node.names:
                imports.append(node)
    return [ast.dump(n, include_attributes=False) for n in imports + selected]


def checked_config(config):
    for key, expected in OFFICIAL_CONFIG.items():
        actual = config.get(key)
        if key == "max_context_tokens":
            if type(actual) is not int or actual < 257:
                raise ValueError("explicit max_context_tokens >= 257 required")
        elif (actual != expected or
              isinstance(expected, (int, bool)) and type(actual) is not type(expected) or
              key == "evaluation_temperature" and type(actual) not in (int, float)):
            raise ValueError(f"invalid official ALFWorld config: {key}")
    return deepcopy(config)


def official_expectations(data_root):
    """Freeze the adapter's population, with every selected world byte pinned."""
    from ...adapters.alfworld import _game_ids
    split = Path(data_root) / "valid_seen"
    ids = _game_ids(split)
    if len(ids) != 140 or len(set(ids)) != 140:
        raise ValueError("official valid_seen requires exactly 140 distinct task IDs")
    files = {}
    for tid in ids:
        task_dir = split / tid
        for name in WORLD_NAMES:
            if not (task_dir / name).is_file():
                raise ValueError(f"missing valid_seen world file: {tid}/{name}")
        for path in sorted(task_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(split).as_posix()] = file_hash(path)
    return dict(split="valid_seen", task_ids=ids, count=140, task_ids_hash=digest(ids),
                data_manifest=files, data_manifest_hash=digest(files))


def checked_expectations(expected):
    ids = expected.get("task_ids", [])
    if (expected.get("split") != "valid_seen" or expected.get("count") != 140 or
            len(ids) != 140 or not all(isinstance(t, str) and len(t.split("/")) == 2
                and all(p not in ("", ".", "..") for p in t.split("/")) for t in ids) or
            len(set(ids)) != 140 or ids != sorted(ids) or
            expected.get("task_ids_hash") != digest(ids)):
        raise ValueError("invalid frozen valid_seen task list")
    files = expected.get("data_manifest", {})
    if (expected.get("data_manifest_hash") != digest(files) or
            any(f"{tid}/{name}" not in files for tid in ids for name in WORLD_NAMES) or
            any("/".join(n.split("/")[:2]) not in ids for n in files)):
        raise ValueError("invalid valid_seen data manifest")
    return expected


def tokenizer_identity(directory):
    directory = Path(directory)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (directory / name).is_file():
            raise ValueError(f"missing tokenizer file: {name}")
    config = json.loads((directory / "tokenizer_config.json").read_text())
    if not (directory / "chat_template.jinja").is_file() and not config.get("chat_template"):
        raise ValueError("missing tokenizer chat template")
    # Same hash convention as RTD cli.make_manifest (including chat templates).
    files = [(p.name, file_hash(p)) for p in sorted(directory.glob("*")) if p.is_file()
             and any(k in p.name for k in ("token", "vocab", "merges", "chat_template"))]
    return dict(files=[list(row) for row in files], hash=digest(files))


def model_identity(model_path, checkpoint=None):
    model_path = Path(model_path)
    if not (model_path / "config.json").is_file() or not list(model_path.glob("*.safetensors")):
        raise ValueError("local model config and safetensors weights required")
    value = dict(base_checkpoint_hash=tree_hash(model_path),
                 model_config_hash=file_hash(model_path / "config.json"), loading="merged")
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        lora = checkpoint / "lora" if (checkpoint / "lora").is_dir() else checkpoint
        if not (lora / "adapter_config.json").is_file() or not (lora / "adapter_model.safetensors").is_file():
            raise ValueError("checkpoint requires RTD round-N/lora or a PEFT adapter directory")
        value.update(loading="adapter_overlay", adapter_hash=tree_hash(lora))
    return value


def environment_identity(environment_root):
    """Read source/assets and dependency records without importing TextWorld."""
    root = Path(environment_root)
    python = root / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"ALFWorld environment Python unavailable: {python}")
    files, versions = {}, {}
    for package in ("alfworld", "textworld"):
        dirs = list((root / ".venv/lib").glob(f"python*/site-packages/{package}"))
        if len(dirs) != 1:
            raise FileNotFoundError(f"installed {package} unavailable or ambiguous")
        for path in sorted(dirs[0].rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
                files[path.relative_to(dirs[0].parent).as_posix()] = file_hash(path)
        metadata = list(dirs[0].parent.glob(f"{package}-*.dist-info/METADATA"))
        if len(metadata) != 1:
            raise ValueError(f"ambiguous {package} version")
        versions[package] = file_hash(metadata[0])
        for record in sorted(dirs[0].parent.glob("*.dist-info/RECORD")):
            files[record.relative_to(dirs[0].parent).as_posix()] = file_hash(record)
    return dict(files=files, versions=versions, python_hash=file_hash(python),
                environment_seed="existing adapter default", python_hash_seed=0)


def evaluation_harness_identity(root, config, *, data_root, model_path, tokenizer_path,
                                environment_root=None):
    root = Path(root)
    checked_config(config)
    scoring_files = {name: digest(scoring_projection(name, (root / name).read_text())) for name in SCOPES}
    # This module defines the inventory, population and continuity rules. Pin its
    # bytes conservatively; no permissive automatic projection of identity code.
    name = "src/bfas/rtd/benchmarks/alfworld_identity.py"
    scoring_files[name] = file_hash(root / name)
    return dict(version=VERSION, scoring_files=scoring_files,
        expected=official_expectations(data_root), model=model_identity(model_path),
        tokenizer=tokenizer_identity(tokenizer_path),
        environment=environment_identity(environment_root or root / "envs/alfworld"),
        config={k: config[k] for k in OFFICIAL_CONFIG},
        prompt_format="C26-B FrozenRenderer: adapter ReAct + existing thinking_off")


def make_manifest(root, config, *, data_root, model_path, hardware, tokenizer_path=None,
                  checkpoint=None, run_directory=None, round_number=None, environment_root=None):
    """CPU preparation for a new campaign; never mutates a training manifest."""
    config = checked_config(config)
    hardware = deepcopy(checked_hardware(hardware))
    tokenizer_path = tokenizer_path or model_path
    if run_directory is not None:
        training = json.loads((Path(run_directory) / "manifest.json").read_text())
        meta = verified_checkpoint(run_directory, training, round_number)
        checkpoint = Path(run_directory) / f"round-{round_number}"
        if (training["config"] != config or training["base_checkpoint_hash"] != tree_hash(model_path)
                or training.get("tokenizer_hash") != tokenizer_identity(tokenizer_path)["hash"]
                or training["hardware_hash"] != digest(hardware["hard"])):
            raise ValueError("training config/model/tokenizer/hardware binding mismatch")
    else:
        meta = None
    harness = evaluation_harness_identity(root, config, data_root=data_root, model_path=model_path,
        tokenizer_path=tokenizer_path, environment_root=environment_root)
    return dict(version=VERSION, config=config, config_hash=digest(config),
        evaluation_harness=harness, harness_hash=digest(harness),
        checkpoint=model_identity(model_path, checkpoint), round_checkpoint=meta,
        hardware=hardware, hardware_hash=digest(hardware["hard"]),
        paths=dict(data_root=str(Path(data_root).resolve()), model_path=str(Path(model_path).resolve()),
            tokenizer_path=str(Path(tokenizer_path).resolve()),
            checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
            run_directory=str(Path(run_directory).resolve()) if run_directory else None,
            round_number=round_number,
            environment_root=str(Path(environment_root or Path(root) / "envs/alfworld").resolve())))


def audited_harness_hashes(manifest, current, supplement=None):
    """Only a connected ALFWorld audit bound to the FULL manifest authorizes reuse.

    Each note carries previous/new content, a method and evidence. This checks
    an explicit operator's audit, not an assertion that arbitrary edits preserve
    scores. C25/BFCL supplements and unaudited historical hashes are refused.
    """
    original = manifest["evaluation_harness"]
    if manifest["harness_hash"] != digest(original):
        raise ValueError("original harness binding mismatch")
    if supplement is None:
        if current != original:
            raise ValueError("harness changed without an audited ALFWorld supplement")
        return [digest(current)]
    if (supplement.get("version") != SUPPLEMENT_VERSION or
            supplement.get("manifest_hash") != manifest_digest(manifest) or
            supplement.get("legacy_harness_hash") != digest(original)):
        raise ValueError("ALFWorld supplement binding mismatch")
    prior, hashes = original, [digest(original)]
    updates = supplement.get("identity_updates", [])
    if not updates:
        raise ValueError("empty audited identity chain")
    for note in updates:
        new = note.get("new_identity", {})
        if (note.get("manifest_hash") != manifest_digest(manifest) or
                note.get("previous_identity") != dict(harness_hash=digest(prior), evaluation_harness=prior) or
                new.get("harness_hash") != digest(new.get("evaluation_harness")) or
                not note.get("audit", {}).get("method") or not note.get("audit", {}).get("evidence")):
            raise ValueError("audited identity chain binding mismatch")
        candidate = new["evaluation_harness"]
        # Continuity cannot migrate data, weights, tokenizer, or science config.
        if any(candidate.get(k) != original.get(k) for k in ("expected", "model", "tokenizer", "config")):
            raise ValueError("identity audit changed data/model/tokenizer/config")
        prior = candidate
        hashes.append(digest(prior))
    if (prior != current or supplement.get("evaluation_harness") != current or
            supplement.get("harness_hash") != digest(current)):
        raise ValueError("audited identity chain endpoint mismatch")
    return list(dict.fromkeys(reversed(hashes)))


def guard_manifest(root, manifest, *, hardware, supplement=None):
    if manifest.get("version") != VERSION or digest(manifest["config"]) != manifest["config_hash"]:
        raise ValueError("evaluation config hash mismatch")
    checked_hardware(manifest["hardware"])
    checked_hardware(hardware)
    if (manifest["hardware_hash"] != digest(manifest["hardware"]["hard"]) or
            hardware["hard"] != manifest["hardware"]["hard"]):
        raise ValueError("evaluation hardware class mismatch")
    current = make_manifest(root, manifest["config"], hardware=hardware, **manifest["paths"])
    hashes = audited_harness_hashes(manifest, current["evaluation_harness"], supplement)
    for key in ("config_hash", "checkpoint", "round_checkpoint", "hardware_hash"):
        if current[key] != manifest[key]:
            raise ValueError(f"evaluation {key} binding mismatch")
    return current, hashes


def campaign_identity(manifest):
    harness = manifest["evaluation_harness"]
    checked_expectations(harness["expected"])
    return dict(benchmark="alfworld", checkpoint=manifest["checkpoint"],
        round_checkpoint=manifest["round_checkpoint"], config_hash=manifest["config_hash"],
        base_checkpoint_hash=harness["model"]["base_checkpoint_hash"],
        tokenizer_hash=harness["tokenizer"]["hash"], expected_hash=digest(harness["expected"]),
        data_hash=harness["expected"]["data_manifest_hash"], hardware_hash=manifest["hardware_hash"],
        evaluation_harness_hash=manifest["harness_hash"], evaluation_temperature=0.0,
        split="valid_seen", max_steps=40, max_action_tokens=256)
