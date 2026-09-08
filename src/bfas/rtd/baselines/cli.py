"""prepare / tune / run without modifying the existing RTD command boundary."""
import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import yaml

from ..persistence import atomic_json, digest, exclusive_run, file_hash
from .config import BaselineConfig, GRIDS, RECIPES, RUNTIME_KEYS, load_config, strict_mapping
from .evidence import load_evidence, prepare
from .journal import ResourceJournal

ROOT = Path(__file__).resolve().parents[4]


def new_destination(path, *, protected=()):
    path = Path(path).resolve()
    for parent in [ROOT/"results/rtd_v1", ROOT/"data", ROOT/"logs", *map(Path, protected)]:
        parent = parent.resolve()
        if path == parent or path.is_relative_to(parent) or parent.is_relative_to(path):
            raise ValueError(f"output overlaps a protected/source path: {path}")
    if path.exists():
        raise FileExistsError(f"new output directory required: {path}")
    return path


def prepare_suite(config_path, out):
    spec = yaml.safe_load(Path(config_path).read_text())
    strict_mapping(spec, {"version", "bank", "hardware_class_hash", "pools", "recipes", "views", "common"},
                   required={"version", "bank", "hardware_class_hash", "pools", "recipes", "views", "common"}, label="suite")
    if spec["version"] != "c27-suite-v1" or not spec["pools"] or not spec["hardware_class_hash"]:
        raise ValueError("suite needs finished source paths and a declared common hardware class")
    strict_mapping(spec["pools"], {"A0", "A1"}, label="pools")
    if (not isinstance(spec["recipes"], dict) or not spec["recipes"] or set(spec["recipes"]) - set(RECIPES)
            or not spec["views"] or set(spec["views"]) - {"slots", "tokens"}):
        raise ValueError("stage 1 suite supports B1--B4 and slots/tokens")
    common = yaml.safe_load(Path(spec["common"]).read_text())
    strict_mapping(common, {"runtime", "seed", "new_teacher_calls", "evaluate_after_round"}, label="common")
    strict_mapping(common.get("runtime", {}), RUNTIME_KEYS, label="common.runtime")
    if common.get("seed") != 0 or common.get("new_teacher_calls") is not False or common.get("evaluate_after_round") is not False:
        raise ValueError("common requires seed0, no new teachers, separate evaluation")
    templates = {recipe: load_config(path) for recipe, path in spec["recipes"].items()}
    if any(config.recipe != name for name, config in templates.items()):
        raise ValueError("suite recipe path/name mismatch")
    destination = new_destination(out, protected=[spec["bank"], *spec["pools"].values()])
    destination.mkdir(parents=True)
    for pool, source in spec["pools"].items():
        evidence_path = destination/f"{pool}_evidence.json"
        evidence = prepare(source, spec["bank"], evidence_path, root=ROOT,
                           expected_hardware_hash=spec["hardware_class_hash"])
        if evidence["source_manifest"]["arm"] != {"A0": "R0", "A1": "R1"}[pool]:
            raise ValueError("A0/A1 source arm mapping differs")
        source_config = evidence["source_manifest"]["config"]
        runtime = {k: v for k, v in source_config.items() if k in RUNTIME_KEYS}
        # Preserve effective source settings. Common declarations assert equality;
        # they never fill old manifests from today's live YAML defaults.
        for key, value in common.get("runtime", {}).items():
            if runtime.get(key) != value:
                raise ValueError(f"common/source runtime mismatch: {key}")
        for recipe, view in product(spec["recipes"], spec["views"]):
            cfg = replace(templates[recipe], view=view, pool=pool,
                evidence_path=str(evidence_path), evidence_hash=file_hash(evidence_path), runtime=runtime,
                update_tokens=evidence["budgets"]["update_tokens"],
                pg_reserved_tokens=evidence["budgets"]["pg_reserved_tokens"])
            with (destination/f"{pool}_{recipe}_{view}.yaml").open("x") as stream:
                yaml.safe_dump(cfg.to_dict(), stream, sort_keys=False)
    atomic_json(destination/"prepared.json", dict(version="c27-prepared-v1", suite=spec,
        suite_hash=file_hash(config_path), files={p.name: file_hash(p) for p in sorted(destination.iterdir())}))
    return str(destination)


def selected_config(config, selection):
    """Frozen A1/V-S support selection transfers across pools and views."""
    if selection is None:
        return config
    data = json.loads(Path(selection).read_text())
    if (data.get("version") != "c27-selection-v1" or data.get("seed") != 0
            or data.get("criterion") != "rotating_support_round_mean" or data.get("source_pool") != "A1"
            or data.get("source_view") != "slots"):
        raise ValueError("selection must come from the declared A1/V-S support grid")
    evidence = load_evidence(config.evidence_path, config.evidence_hash)
    if data.get("transfer_identity") != transfer_identity(evidence):
        raise ValueError("tuning transfer requires the same base/tokenizer/data/hardware class")
    trial = data["selected"].get(config.recipe)
    if trial is None:
        # A B1/B2-only stage-1 grid has not tuned the B3/B4 recipes.
        if config.recipe in {"b1", "b2"}:
            raise ValueError("missing recipe selection")
        return config
    allowed = GRIDS[config.recipe]
    hp = trial["hyperparameters"]
    if set(hp) != set(allowed) or any(value not in allowed[key] for key, value in hp.items()):
        raise ValueError("selection hyperparameters outside the declared grid")
    return replace(config, **hp)


def transfer_identity(evidence):
    source = evidence["source_manifest"]
    return dict(hardware_class_hash=evidence["hardware_class_hash"],
                **{key: source[key] for key in ("base_checkpoint_hash", "tokenizer_hash", "data_hash")})


def make_baseline_manifest(config, evidence):
    from ..cli import make_manifest
    from ..identity import evaluation_harness_identity, saved_identities
    effective = dict(config.runtime, mode="fixed_evidence", baseline=config.to_dict(),
        fixed_evidence_ids=evidence["owned"], fixed_source_data_hash=evidence["source_manifest"]["data_hash"],
        fixed_source_base_checkpoint_hash=evidence["source_manifest"]["base_checkpoint_hash"],
        fixed_source_hardware_hash=evidence["hardware_class_hash"],
        evaluate_after_round=False, new_teacher_calls=False)
    source = evidence["source_manifest"]
    if config.runtime != {k: v for k, v in source["config"].items() if k in RUNTIME_KEYS}:
        raise ValueError("runtime must preserve the source's full effective backend/LoRA settings")
    audit = {k: source[k] for k in ("bank_public_cap_sum", "budget_ceilings", "recorded_bank_usage", "available_packages", "m")}
    audit["bank_path"] = evidence["bank_path"]
    manifest = make_manifest(effective, config.recipe.upper(), audit)
    if manifest["tokenizer_hash"] != source["tokenizer_hash"]:
        raise ValueError("source tokenizer differs")
    # Read an already-audited continuity supplement when present; never create one.
    identity = saved_identities(ROOT, Path(evidence["source_run"]), source)
    if evaluation_harness_identity(ROOT, effective) != identity["evaluation_harness"]:
        raise ValueError("source/baseline evaluation harness differs; needs an external continuity audit")
    manifest.update(version="c27-baselines-run-v1", baseline_source_evidence_hash=config.evidence_hash,
        checkpoint_schedule="three training-progress checkpoints using the complete final purchased pool",
        baseline_label="B1-replay-L1" if config.recipe == "b1" else config.recipe,
        baseline_method=dict(features_off=True, full_action_sum=True, source_refresh="round", source_samples=2,
                             transport_term=config.recipe == "b3_mix"),
        resources=dict(manifest["resources"], matched_teacher_tokens=evidence["budgets"]["teacher_tokens"],
                       feedback_schedule_hash=digest(evidence["feedback"]),
                       support_feedback="rotating meta-training; development is not independent validation"))
    return manifest


def run_config(config, directory, *, selection=None, development_schedule=None):
    from ..experiment import BFCLSupport
    from ..runtime import load_backend
    from tools.behavior_atom.checker_bridge import CheckerBridge
    from .runner import BaselineRunner
    config = selected_config(config, selection).validate_ready()
    evidence = load_evidence(config.evidence_path, config.evidence_hash)
    directory = new_destination(directory, protected=[evidence["source_run"], evidence["bank_path"]])
    manifest = make_baseline_manifest(config, evidence)
    manifest["baseline_selection"] = (dict(path=str(Path(selection).resolve()), sha256=file_hash(selection),
        recipe_selected=config.recipe in json.loads(Path(selection).read_text())["selected"]) if selection else None)
    if development_schedule:
        manifest["development_schedule"] = development_schedule
        manifest["development_seed"] = 20000
    directory.mkdir(parents=True)
    with exclusive_run(directory):
        atomic_json(directory/"manifest.json", manifest)
        with (directory/"teacher.jsonl").open("x") as stream:
            for event in evidence["ledger_events"]:
                stream.write(json.dumps(event, allow_nan=False)+"\n")
        atomic_json(directory/"effective_config.json", config.to_dict())
        journal = ResourceJournal(directory/"compute.jsonl", cuda=True)
        backend = load_backend(config.runtime, manifest, journal)
        support = BFCLSupport(ROOT, config.runtime)
        with CheckerBridge() as checker:
            runner = BaselineRunner(config, evidence, manifest, directory, backend, support, journal,
                                    checker=checker, development_schedule=development_schedule)
            result = runner.run()
        return result


def grid_trials(recipes):
    for recipe in recipes:
        grid = GRIDS[recipe]
        for values in product(*grid.values()):
            yield recipe, dict(zip(grid, values))


def choose_trials(trials):
    selected = {}
    for recipe in sorted({t["recipe"] for t in trials}):
        candidates = [t for t in trials if t["recipe"] == recipe]
        def tie(t):
            hp = t["hyperparameters"]
            return (-t["score"], hp["commits"], hp.get("lr", hp.get("eta_multiplier", 0)), hp.get("meta_lr", 0))
        winner = min(candidates, key=tie)
        selected[recipe] = dict(winner, tied_best=sum(t["score"] == winner["score"] for t in candidates),
                               identifiable=len({t["score"] for t in candidates}) > 1)
    return selected


def tune(config_path, prepared, out, *, trial_runner=run_config):
    spec = yaml.safe_load(Path(config_path).read_text())
    strict_mapping(spec, {"version", "recipes", "grids", "seed", "pool", "view", "development_seed"},
        required={"version", "recipes", "grids", "seed", "pool", "view", "development_seed"}, label="tuning")
    if (spec["version"] != "c27-tuning-v1" or spec["seed"] != 0 or spec["pool"] != "A1"
            or spec["view"] != "slots" or spec["development_seed"] != 20000
            or not spec["recipes"] or set(spec["recipes"]) - set(RECIPES)
            or spec["grids"] != {r: GRIDS[r] for r in spec["recipes"]}):
        raise ValueError("tuning requires the exact predeclared small grids, seed0, A1/V-S")
    configs = {r: load_config(Path(prepared)/f"A1_{r}_slots.yaml").validate_ready() for r in spec["recipes"]}
    if len({c.evidence_hash for c in configs.values()}) != 1:
        raise ValueError("tuning trials must share one final evidence set")
    first = next(iter(configs.values()))
    evidence = load_evidence(first.evidence_path, first.evidence_hash)
    # Predeclared public-parent table: first reference window each round. This
    # is fixed before any trial, and is an EXTRA independent seed stream.
    development = [dict(g, step=12, role="support_development", reused=False)
                   for g in evidence["feedback"] if g["step"] == 1 and g["role"] == "reference_feedback"]
    directory = new_destination(out, protected=[evidence["source_run"], evidence["bank_path"]])
    directory.mkdir(parents=True)
    atomic_json(directory/"tuning_protocol.json", dict(spec=spec, evidence_hash=first.evidence_hash,
        development_schedule=development, development_schedule_hash=digest(development)))
    trials = []
    for i, (recipe, hp) in enumerate(grid_trials(spec["recipes"])):
        config = replace(configs[recipe], **hp)
        result = trial_runner(config, directory/f"trial-{i:02d}-{recipe}", development_schedule=development)
        if not result["passed"] or not result["exposure"]["matched"] or len(result["development_scores"]) != 3:
            raise ValueError("incomplete/unmatched tuning trial")
        score = sum(block["score"] for block in result["development_scores"])/3
        trials.append(dict(recipe=recipe, hyperparameters=hp, score=score, resources=result["resources"],
                           config_hash=digest(config.to_dict()), evidence_hash=config.evidence_hash))
        atomic_json(directory/"trials.json", trials)
    selection = dict(version="c27-selection-v1", seed=0, criterion="rotating_support_round_mean",
        source_pool="A1", source_view="slots", evidence_hash=first.evidence_hash,
        transfer_identity=transfer_identity(evidence),
        development_schedule_hash=digest(development), selected=choose_trials(trials),
        trial_count=len(trials), trials=trials, independent_validation=False,
        total_resources={k: sum(t["resources"].get(k, 0) for t in trials)
                         for k in {k for t in trials for k in t["resources"]}})
    selection["total_resources"]["teacher_allocated_tokens_across_trials"] = selection["total_resources"].get("teacher_tokens", 0)
    selection["total_resources"]["teacher_tokens"] = evidence["budgets"]["teacher_tokens"]
    atomic_json(directory/"selection.json", selection)
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description="C27 isolated B1/B2/B3/B4, no new teacher calls")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--config", required=True, help="suite with completed R0/R1 paths and common hardware hash")
    p.add_argument("--out", required=True)
    p = subs.add_parser("tune")
    p.add_argument("--config", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--out", required=True)
    p = subs.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--selection")
    p.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_suite(args.config, args.out)
    elif args.command == "tune":
        result = tune(args.config, args.prepared, args.out)
    else:
        result = run_config(load_config(args.config), args.run_dir, selection=args.selection)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
