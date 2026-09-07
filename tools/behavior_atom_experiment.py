#!/usr/bin/env python3
"""Thin, CPU-first behaviour-atom stage entry point.

Saved fit input: JSON or NPZ with X (source, code), M (probe, source),
source_groups (parent-task groups), and fixed folds (one label/source, or JSON
[{train: indices, test: indices}]). Optional: source_ids, probe_ids, probe_groups,
category, length, strata, C (H-orthonormal parameter decoder), response_kind.
X must be fixed H coordinates, not globally fitted/centred PCA or a sketch.
Hierarchical fitting requires source_factor (or aligned source manifests).
Probe metadata: probe_factor, probe_family, probe_margin_base. M_teacher and
M_student store separate likelihood deltas; teacher_target_kind="margin" also
supports independent teacher margins, which cannot be subtracted from student
log-likelihood. --target all fits teacher/student/combined (teacher/student only
for margins). --model hierarchical with the default behavior target also fits
the likelihood sides when both are available. Flat/behavior remains the default.

A JSON descriptor can instead reference {"responses": "paired.jsonl",
"codes": "codes.npz"}; JSONL uses behavior.response's hashed bundle format.
Its codes file supplies source_ids/groups/folds; eval_seeds come from the bundle
provenance or descriptor. Paths are relative to the file containing the path.

Config example (saved-response fit):
  {"run_id":"synthetic", "data":"response.npz",
   "fit":{"ranks":[1,2,4],"lambdas":[0,0.01,0.1,1],
          "permutation_seeds":[11,29,47],"bootstrap_samples":200}}
Scientific thresholds are optional only for analysis: without the complete
DecisionConfig in "decision", the stage decision is INCONCLUSIVE. Measurement
evidence is supplied in "measurement": {"stable":true,"resolved":true}.

Pilot/collect manifest: {sources:[{source_id,parent_task_id,trajectory_id,
state_hash,rows:[existing teacherized rows]}], probes:[ProbeRecord fields plus
optional row], dependencies:[{job_id,status,exit_code,...}]}.
Pilot/collect config's "protocol" is the ENTIRE frozen execution configuration:
  {"micro_update": {"model_id":...,"init_state_path":...,"steps":...},
   "eval_seeds":[0], "stochastic":false,
   "evaluator":"package.module:evaluate_update",
   "model_factory":"package.module:make_base"}
An evaluator receives keyword arguments update, source, probes, eval_seeds,
protocol and returns OutcomeRecord objects/dicts. It must use saved supervision
and paired seeds and make no teacher calls. The runner is imported only when
executing: src.bfas.behavior.microupdate.run_micro_update(factory, rows, cfg).
Pilot additionally requires "pilot_assessor": "module:function", returning
{passed,measurement_stable,measurement_resolved,metrics} from keyword arguments
outcomes, updates, protocol. Its thresholds belong inside the frozen protocol.
Only an affirmative measured assessment creates pilot_pass.json. Collect config
references that file via "pilot_pass"; hashes include resolved checkpoint files.

Examples (no GPU job is launched by fit/report/dry-run):
  .venv/bin/python tools/behavior_atom_experiment.py codes --pilot-smoke --dry-run
  .venv/bin/python tools/behavior_atom_experiment.py codes --run-dir results/behavior_atom_v1/merged
  .venv/bin/python tools/behavior_atom_experiment.py fit --config config.json --dry-run
  .venv/bin/python tools/behavior_atom_experiment.py fit --config config.json
  .venv/bin/python tools/behavior_atom_experiment.py fit --config config.json \
      --model hierarchical --target all --family-ranks 1 2 --min-family-sources 8 --residual-rank 0
  .venv/bin/python tools/behavior_atom_experiment.py report --config config.json \
      --manifest results/behavior_atom_v1/synthetic/fit_result.json

Outputs default to results/behavior_atom_v1/<run_id>/. Each stage has an
immutable manifest, immutable artifacts and a hash-checked completion record.
Resume verifies identity/content and skips completed work. Interrupted foreign
micro-update directories are refused by their runner, never silently restarted.
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.bfas.behavior.lowrank import (  # noqa: E402
    DecisionConfig, FitConfig, HierarchicalResponseModel, cross_fit, grouped_folds, stage_decision,
)
from src.bfas.behavior.response import (  # noqa: E402
    OutcomeRecord, ProbeRecord, assemble_matrix, canonical_json, content_hash,
    read_responses, validate_split, write_immutable, write_json, write_responses,
)

STAGES = ("audit", "decompose", "pilot", "collect", "fit", "intervene", "distill", "report")


def _resolve(path, base: Path) -> Path:
    path = Path(path)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _read(path: Path, references: dict) -> dict:
    references[str(path)] = content_hash(path.read_bytes())
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _jsonl_dataset(path: Path, codes: dict, descriptor: dict, references: dict) -> dict:
    probes, outcomes, manifest = read_responses(path)
    for artifact in (path, Path(str(path) + ".manifest.json")):
        references[str(artifact)] = content_hash(artifact.read_bytes())
    if "source_ids" not in codes:
        raise ValueError("JSONL codes input requires explicit source_ids to align matrix columns")
    assembled = assemble_matrix(outcomes, source_ids=list(map(str, codes["source_ids"])),
                                probe_ids=[p.probe_id for p in probes],
                                eval_seeds=descriptor.get("eval_seeds", manifest["provenance"].get("eval_seeds")),
                                run_ids=descriptor.get("run_ids"))
    factors = {}
    for record in outcomes:
        if record.source_factor is not None:
            if record.source_id in factors and factors[record.source_id] != record.source_factor:
                raise ValueError("inconsistent source_factor in response records")
            factors[record.source_id] = record.source_factor
    labels = {}
    if factors:
        if any(s not in factors for s in assembled.source_ids):
            raise ValueError("source_factor is missing for a saved source")
        labels["source_factor"] = [factors[s] for s in assembled.source_ids]
        if "source_factor" in codes and list(codes["source_factor"]) != labels["source_factor"]:
            raise ValueError("source_factor differs between codes and response records")
    return {**codes, **labels, "M": assembled.M, "M_teacher": assembled.teacher, "M_student": assembled.student,
            "probe_ids": assembled.probe_ids, "probe_groups": [p.group for p in probes],
            "probe_factor": [p.probe_factor or p.probe_family or p.group for p in probes],
            "probe_family": [p.probe_family for p in probes], "probe_margin_base": [p.probe_margin_base for p in probes],
            "response_diagnostics": assembled.diagnostics, "response_kind": "behavior"}


def load_dataset(path: Path, config: dict, references: dict) -> dict:
    if path.suffix == ".jsonl":
        if "codes" not in config:
            raise ValueError("direct JSONL input requires config.codes")
        code_path = _resolve(config["codes"], Path(config["_base"]))
        raw = _jsonl_dataset(path, _read(code_path, references), config, references)
    else:
        raw = _read(path, references)
        if "dataset" in raw:
            return {**load_dataset(_resolve(raw["dataset"], path.parent), config, references),
                    **{k: v for k, v in raw.items() if k != "dataset"}}
        if "responses" in raw:
            response_path = _resolve(raw["responses"], path.parent)
            codes = _read(_resolve(raw["codes"], path.parent), references) if "codes" in raw else {}
            response = (_jsonl_dataset(response_path, codes, raw, references) if response_path.suffix == ".jsonl"
                        else {**codes, **_read(response_path, references)})
            raw = {**response, **{k: v for k, v in raw.items() if k not in {"codes", "responses"}}}
    return raw


def _target_dataset(data, target):
    if target == "all":
        target = "teacher"
    if target == "behavior":
        return data
    if target not in {"teacher", "student", "combined"}:
        raise ValueError("target must be behavior, teacher, student, combined, or all")
    required = ("M_teacher", "M_student") if target == "combined" else (f"M_{target}",)
    if any(key not in data for key in required):
        raise ValueError(f"{target} target requires {', '.join(required)}")
    matrix = np.asarray(data[required[0]], dtype=float)
    extra = {}
    if target == "combined":
        student = np.asarray(data["M_student"], dtype=float)
        if student.shape != matrix.shape:
            raise ValueError("teacher/student response axes differ")
        if data.get("teacher_target_kind", "log_likelihood") == "margin":
            raise ValueError("combined requires teacher log-likelihood, not an already differenced teacher margin")
        matrix = matrix - student
        extra["_student_response"] = student
    return {**data, **extra, "M": matrix, "response_kind": "likelihood", "response_target": target}


def _fit_inputs(data: dict, config: dict) -> dict:
    data = _target_dataset(data, config.get("target", "behavior"))
    for key in ("X", "M", "source_groups"):
        if key not in data:
            raise ValueError(f"saved fit input requires {key}")
    if "folds" not in data and "folds" not in config:
        raise ValueError("fixed folds must be supplied, never inferred from response outcomes")
    if data.get("code_provenance", "fixed_h_coordinates") not in {"fixed_h_coordinates", "h_whitened_deltas"}:
        raise ValueError("codes must be fixed H coordinates; global learned bases/sketches are inapplicable")
    X, M = np.asarray(data["X"], dtype=float), np.asarray(data["M"], dtype=float)
    if X.ndim != 2 or M.ndim != 2 or X.shape[0] != M.shape[1] or not min(*X.shape, M.shape[0]):
        raise ValueError("expected nonempty X=(sources,codes), M=(probes,sources)")
    if not np.isfinite(X).all() or np.isinf(M).any():
        raise ValueError("codes must be finite; responses may be finite or NA")
    if len(data["source_groups"]) != X.shape[0]:
        raise ValueError("source group axis differs from codes")
    if "source_ids" in data and (len(data["source_ids"]) != X.shape[0] or len(set(data["source_ids"])) != X.shape[0]):
        raise ValueError("source_ids must be unique and match codes")
    folds = data.get("folds", config.get("folds"))
    grouped_folds(data["source_groups"], folds)
    metadata = {key: data[key] for key in ("category", "length", "source_factor") if key in data}
    if "source_factor" not in metadata and data.get("sources"):
        sources = {s["source_id"]: s for s in data["sources"]}
        ids = data.get("source_ids", [s["source_id"] for s in data["sources"]])
        metadata["source_factor"] = [sources[s].get("source_factor", sources[s].get("family", sources[s].get("factor"))) for s in ids]
    cfg = FitConfig(**config.get("fit", {}))
    if cfg.model == "hierarchical":
        labels = metadata.get("source_factor")
        if labels is None or np.shape(labels) != (len(X),) or any(not isinstance(f, str) or not f for f in labels):
            raise ValueError("hierarchical fit requires manifest source_factor labels")
    probe_factors = data.get("probe_factor", data.get("probe_family"))
    if probe_factors is None and data.get("probes"):
        probes = {p["probe_id"]: p for p in data["probes"]}
        ids = data.get("probe_ids", [p["probe_id"] for p in data["probes"]])
        probe_factors = [probes[p].get("probe_factor") or probes[p].get("probe_family") or probes[p].get("group", "other") for p in ids]
    for key in ("probe_factor", "probe_family", "probe_margin_base"):
        if key in data and np.shape(data[key]) != (M.shape[0],):
            raise ValueError(f"{key} axis differs from response probes")
    return {"X": X, "M": M, "source_groups": data["source_groups"], "folds": folds,
            "probe_groups": data.get("probe_groups"), "strata": data.get("strata"),
            "metadata": metadata, "probe_factors": probe_factors,
            "student_response": data.get("_student_response"), "config": cfg}


def frozen_protocol(config: dict) -> dict:
    """Resolve frozen execution paths and bind checkpoint/init file contents."""
    protocol = json.loads(canonical_json(config.get("protocol", {})))
    if not protocol:
        raise ValueError("pilot/collect require a frozen protocol object")
    if protocol.get("new_teacher_tokens", 0) != 0:
        raise ValueError("new teacher token budget must be zero")
    micro = protocol.setdefault("micro_update", {})
    forbidden = {"source_id", "output_dir", "run_id", "dry_run", "resume", "probe_rows", "data_manifest_hash"}
    if forbidden & micro.keys():
        raise ValueError(f"stage-owned micro-update fields in protocol: {sorted(forbidden & micro.keys())}")
    artifacts = {}
    for key in ("init_state_path",):
        if key in micro:
            path = _resolve(micro[key], Path(config["_base"]))
            micro[key] = str(path)
            artifacts[key] = content_hash(path.read_bytes()) if path.is_file() else None
    protocol["input_artifact_hashes"] = artifacts
    return protocol


def require_pilot_pass(config: dict, protocol: dict, references: dict) -> dict:
    if not config.get("pilot_pass"):
        raise ValueError("collect requires a pilot pass JSON file with a matching frozen config hash")
    path = _resolve(config["pilot_pass"], Path(config["_base"]))
    if not path.is_file():
        raise ValueError(f"collect requires an existing pilot pass file: {path}")
    passed = _read(path, references)
    if (passed.get("status") != "pass" or passed.get("passed") is not True
            or passed.get("frozen_config_hash") != content_hash(protocol)):
        raise ValueError("pilot pass is invalid or its frozen config hash differs from the collection protocol")
    if passed.get("measurement_stable") is not True or passed.get("measurement_resolved") is not True:
        raise ValueError("pilot pass lacks affirmative measurement stability/resolution")
    for name, expected in passed.get("artifact_hashes", {}).items():
        artifact = _resolve(name, path.parent)
        if not artifact.is_file() or content_hash(artifact.read_bytes()) != expected:
            raise ValueError("pilot pass evidence artifact is missing or changed")
        references[str(artifact)] = expected
    return passed


def _probe(record: dict) -> ProbeRecord:
    return ProbeRecord(**{f.name: record[f.name] for f in fields(ProbeRecord) if f.name in record})


def _validate_collection(manifest: dict, stage: str, protocol: dict, *, executing=False):
    sources, probes = manifest.get("sources", []), manifest.get("probes", [])
    ids = [source["source_id"] for source in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("source IDs must be unique")
    if not sources or not probes:
        raise ValueError("pilot/collect require nonempty source and probe manifests")
    count = sum(s.get("role") != "no_op" for s in sources)
    noops = len(sources) - count
    if count > (4 if stage == "pilot" else 24) or noops > 2 or len(probes) > 32:
        raise ValueError("P2 cap exceeded: 4 pilot/24 formal sources, 2 no-ops, 32 development probes")
    parsed = [_probe(p) for p in probes]
    if len({p.probe_id for p in parsed}) != len(parsed):
        raise ValueError("probe IDs must be unique")
    for source in sources:
        if not all(source.get(key) for key in ("parent_task_id", "trajectory_id", "state_hash")):
            raise ValueError("sources require parent_task_id, trajectory_id and state_hash for split audit")
    validate_split(sources, parsed)
    seeds = protocol.get("eval_seeds", [0])
    if (not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds)
            or len(seeds) > (3 if protocol.get("stochastic", False) else 1)):
        raise ValueError("fixed paired seed cap: one deterministic, at most three stochastic")
    if executing:
        if not protocol.get("evaluator"):
            raise ValueError("real pilot/collect requires a paired behaviour evaluator callable")
        if stage == "pilot" and not protocol.get("pilot_assessor"):
            raise ValueError("real pilot requires a measurement assessor with thresholds frozen in protocol")
        if any(not source.get("rows") for source in sources):
            raise ValueError("each source must supply its existing teacherized rows")
        for dependency in manifest.get("dependencies", []):
            if not isinstance(dependency, dict) or dependency.get("status") != "complete" or dependency.get("exit_code") != 0:
                raise ValueError("unresolved dependency: require job_id, complete status and zero exit_code")
            if not dependency.get("job_id"):
                raise ValueError("dependency completion requires a job ID")
    return parsed, seeds


def plan(stage: str, config: dict, data: dict, protocol: dict | None = None) -> dict:
    cost = config.get("cost_estimate", {})
    if stage in {"pilot", "collect"}:
        sources, probes = len(data.get("sources", [])), len(data.get("probes", []))
        seeds = len((protocol or {}).get("eval_seeds", [0]))
        models = sources + 1  # common base plus independent updated models
        rollouts = 2 * sources * probes * seeds  # conservative: base repeated per source
        gpu_train = cost.get("gpu_hours_per_update")
        gpu_eval = cost.get("gpu_seconds_per_rollout")
        gpu_hours = (sources * gpu_train + rollouts * gpu_eval / 3600
                     if gpu_train is not None and gpu_eval is not None else None)
        dependencies = data.get("dependencies", []) + [
            "src.bfas.behavior.microupdate.run_micro_update",
            (protocol or {}).get("evaluator", "paired behaviour evaluator (required for execution)"),
            (protocol or {}).get("micro_update", {}).get("init_state_path", "saved common initial state"),
        ]
        if stage == "collect":
            dependencies.append(config.get("pilot_pass", "pilot pass JSON with frozen config hash"))
        environment_seconds = (rollouts * cost["environment_seconds_per_rollout"]
                               if "environment_seconds_per_rollout" in cost else None)
    else:
        X, M = np.asarray(data.get("X", [])), np.asarray(data.get("M", []))
        sources = len(X) if X.ndim == 2 else len(data.get("sources", []))
        probes = M.shape[0] if M.ndim == 2 else len(data.get("probes", []))
        models, rollouts, gpu_hours, environment_seconds = 0, 0, 0, 0
        dependencies = data.get("dependencies", [])
        if stage in {"fit", "report"} and X.ndim == 2:
            cfg = FitConfig(**config.get("fit", {}))
            models = 5 + len(cfg.permutation_seeds) + sum(k in data for k in ("category", "length"))
            models += 2 if cfg.model == "hierarchical" else 0
            if config.get("target") == "all":
                models *= 2 if data.get("teacher_target_kind") == "margin" else 3
            elif cfg.model == "hierarchical" and config.get("target", "behavior") == "behavior" and all(
                    k in data and np.isfinite(np.asarray(data[k], dtype=float)).any() for k in ("M_teacher", "M_student")):
                models *= 3 if data.get("teacher_target_kind") == "margin" else 4
            target_data = _target_dataset(data, config.get("target", "behavior"))
            probes = np.shape(target_data["M"])[0]
            dependencies = ["numpy", "saved responses", "fixed source-group folds", "fixed H-coordinate codes"]
        if stage in {"intervene", "distill"}:
            models = config.get("planned_models")
            rollouts = config.get("planned_max_rollouts")
            gpu_hours, environment_seconds = cost.get("gpu_hours"), cost.get("environment_seconds")
    future_plans = {
        "intervene": {"directions": ["common", "B2 predicted beneficial", "B2/B1 disagreement"],
                      "dose_multipliers": [0.5, 1, 2], "dose_constraint": "within frozen pilot safety range",
                      "controls": ["no-op", "random realizable direction at matched H norm", "two-direction combination"],
                      "evaluation": "independent confirmation probes; paired seeds; no automatic execution"},
        "distill": {"data": "one fixed existing training pool, shared order and token/step budget",
                    "arms": ["original trainer", "common direction", "same-rank PCA", "B2 directions"],
                    "controller": "off/moments/fallback rules must be frozen and tested before training",
                    "evaluation": "learning plus retention damage; no automatic execution"},
    }
    return {"stage": stage, "status": "dry-run plan", "number_of_models": models,
            "sources": sources, "probes": probes, "max_rollouts": rollouts,
            "dependencies": dependencies, "new_teacher_tokens": 0,
            "cost_estimate": {"gpu_hours": gpu_hours, "environment_seconds": environment_seconds,
                              "environment_rollouts_upper_bound": rollouts,
                              "basis": "configured throughput rates; null means not measured",
                              "rates": cost},
            "plan": config.get("plan", future_plans.get(stage, f"execute {stage} within the specified source/probe budget")),
            "frozen_config_hash": content_hash(protocol) if protocol else None}


def _symbol(name: str):
    module, separator, attr = name.partition(":")
    if not separator:
        module, _, attr = name.rpartition(".")
    if not module or not attr:
        raise ValueError(f"callable must have module:attribute syntax: {name}")
    return getattr(importlib.import_module(module), attr)


def _fit_one(data: dict, config: dict) -> tuple[dict, dict]:
    data = _target_dataset(data, config.get("target", "behavior"))
    if data.get("probes"):
        probes = {p["probe_id"]: p for p in data["probes"]}
        ids = data.get("probe_ids", [p["probe_id"] for p in data["probes"]])
        data = {**data, **{key: [probes[p].get(key) for p in ids] for key in ("probe_family", "probe_margin_base") if key not in data}}
    inputs = _fit_inputs(data, config)
    result = cross_fit(**inputs)
    measured = config.get("measurement", {})
    decision = stage_decision(result, DecisionConfig(**config["decision"]) if config.get("decision") else None,
                              measurement_stable=measured.get("stable"), measurement_resolved=measured.get("resolved"),
                              response_kind=str(data.get("response_kind", "behavior")))
    arrays = {f"prediction_{name}": value for name, value in result.predictions.items()}
    arrays.update({f"prediction_level_{name}": value for name, value in result.level_predictions.items()})
    arrays["target_matrix"] = result.target_matrix
    if "student_common_mode" in result.diagnostics:
        arrays["student_common_prediction"] = result.diagnostics["student_common_mode"]["prediction"]
    serialized_folds = []
    C = np.asarray(data["C"], dtype=float) if "C" in data else None
    if C is not None and (C.ndim != 2 or C.shape[1] != np.shape(data["X"])[1] or not np.isfinite(C).all()):
        raise ValueError("C must be a finite (parameter_dimension, code_dimension) decoder")
    def serialize(model, prefix):
        if isinstance(model, HierarchicalResponseModel):
            return {"name": model.name, "rank": model.rank, "requested_rank": model.requested_rank,
                    "factor_mapping": model.factor_mapping, "diagnostics": model.diagnostics,
                    "blocks": {name: serialize(block, prefix + "_" + name) for name, block in model.blocks().items()}}
        record = asdict(model)
        arrays[prefix + "_R"], arrays[prefix + "_basis"] = model.R, model.basis
        if not model.feature_kind:
            modes = model.modes(C)
            record["modes"] = modes
            arrays[prefix + "_V"], arrays[prefix + "_B"] = modes["V"], modes["B"]
            if modes["U"] is not None:
                arrays[prefix + "_U"] = modes["U"]
        return record

    for fold in result.folds:
        models = {name: serialize(model, f"fold{fold['fold']}_{name}") for name, model in fold["models"].items()}
        serialized_folds.append({**fold, "models": models})
    return {"schema_version": "behavior-fit-v1", "stage": "P2", **decision,
            "metrics": result.metrics, "confidence_intervals": result.confidence_intervals,
            "diagnostics": result.diagnostics, "folds": serialized_folds,
            "source_ids": data.get("source_ids", [str(i) for i in range(np.shape(data["X"])[0])]),
            "probe_ids": data.get("probe_ids", [str(i) for i in range(np.shape(data["M"])[0])]),
            "source_factor": inputs["metadata"].get("source_factor"), "probe_factor": inputs["probe_factors"],
            "probe_family": data.get("probe_family"), "probe_margin_base": data.get("probe_margin_base"),
            "response_diagnostics": data.get("response_diagnostics", {}),
            "fit_config": asdict(FitConfig(**config.get("fit", {}))),
            "decision_config": config.get("decision"), "measurement": measured,
            "response_target": data.get("response_target", "behavior"),
            "response_kind": str(data.get("response_kind", "behavior"))}, arrays


def _fit(data: dict, config: dict) -> tuple[dict, dict]:
    """Keep the primary competition API and attach independent likelihood sides."""
    target = config.get("target", "behavior")
    primary = "teacher" if target == "all" else target
    fitted, arrays = _fit_one(data, {**config, "target": primary})
    sides_present = all(key in data and np.isfinite(np.asarray(data[key], dtype=float)).any()
                        for key in ("M_teacher", "M_student"))
    if target == "all" or (target == "behavior" and config.get("fit", {}).get("model") == "hierarchical" and sides_present):
        targets = {}
        side_names = ("teacher", "student") if data.get("teacher_target_kind") == "margin" else ("teacher", "student", "combined")
        for side in side_names:
            value, side_arrays = ((dict(fitted), dict(arrays)) if side == primary else
                                 _fit_one(data, {**config, "target": side}))
            targets[side] = value
            arrays.update({f"target_{side}_{key}": value for key, value in side_arrays.items()})
        fitted["targets"] = targets
    return fitted, arrays


def _report(fit: dict, manifest: dict, artifacts: dict, costs: dict) -> tuple[dict, str]:
    decision = fit.get("decision", "INCONCLUSIVE")
    report = {"stage": fit.get("stage", "P2"), "decision": decision,
              "reason": fit.get("reason", "no response model fitted"),
              "hypothesis": "training-update codes predict behaviour beyond common/PCA directions and stratified permutations",
              "fixed": ["source/probe identities", "grouped folds", "rank/lambda grid", "paired seeds", "H coordinates"],
              "changed": ["response model family", "training-only selected rank/lambda"],
              "valid_runs": [manifest["run_id"]], "invalid_runs": [],
              "metrics": fit.get("metrics", {}), "confidence_intervals": fit.get("confidence_intervals", {}),
              "response_target": fit.get("response_target", "behavior"), "targets": fit.get("targets", {}),
              "hierarchy": {key: fit.get("diagnostics", {}).get(key) for key in
                            ("explained_variance_by_level", "matched_rank_comparison", "matched_rank_note", "student_common_mode")},
              "effective_samples": fit.get("diagnostics", {}).get("observations", {}),
              "costs": costs, "artifacts": artifacts, "execution": manifest,
              "blocking_issues": [] if decision in {"GO-candidate", "SIMPLIFY"} else [fit.get("reason", "insufficient evidence")],
              "checks": {"numerical_measurement": fit.get("measurement", {}),
                         "missingness": fit.get("response_diagnostics", {}),
                         "fit_diagnostics": fit.get("diagnostics", {})},
              "control_interpretation": "Use paired candidate-vs-B0/B1 intervals and every fixed B3 seed; likelihood sides are auxiliary, never gate substitutes.",
              "conclusion_boundary": "fixed probes and supplied updates only; CPU synthetic validation does not establish measured behaviour or intervention benefit",
              "next_actions": [{"action": "inspect group influence, missingness and frozen measurement evidence before the next gate",
                                "gpu_hours": 0, "new_teacher_tokens": 0}],
              "metric_evidence": "fit_result.json:/metrics and /confidence_intervals; rows indexed by source_ids/probe_ids"}
    lines = [f"{decision}: {report['reason']}", "", report["hypothesis"], "",
             "| Model | selected ranks by fold | held-out MSE | gain vs zero | net-benefit Spearman | damage AUROC | MSE CI |",
             "|---|---|---:|---:|---:|---:|---|"]
    def number(value):
        return "NA" if value is None or not np.isfinite(value) else f"{value:.6g}"
    for name, metric in report["metrics"].items():
        ranks = [f["models"][name]["rank"] for f in fit.get("folds", [])]
        interval = report["confidence_intervals"].get("models", {}).get(name, {}).get("mse", {})
        lines.append(f"| {name} | {ranks} | {number(metric['mse'])} | {number(metric['improvement_vs_zero'])} | "
                     f"{number(metric['net_benefit_spearman'])} | {number(metric['damage_auroc'])} | "
                     f"[{number(interval.get('low'))}, {number(interval.get('high'))}] |")
    primary = fit.get("response_target", "behavior")
    for side, result in [(primary, fit), *[(k, v) for k, v in fit.get("targets", {}).items() if k != primary]]:
        diagnostic = result.get("diagnostics", {})
        candidate = diagnostic.get("candidate", "B2")
        lines += ["", f"Response target: **{side}**; candidate: `{candidate}`."]
        if side in fit.get("targets", {}):
            metric = result["metrics"][candidate]
            lines.append(f"Held-out MSE: {number(metric['mse'])}; net-benefit Spearman: {number(metric['net_benefit_spearman'])}.")
        if diagnostic.get("student_common_mode"):
            lines += ["", "Student common mode: one explicit erasure coordinate fitted using student training responses only.",
                      "Combined target = teacher − student + fitted student common mode. The removed raw-margin contribution is its negative; raw-margin predictions subtract it again."]
        if diagnostic.get("explained_variance_by_level"):
            lines += ["", "| Level | incremental held-out explained variance | cumulative |", "|---|---:|---:|"]
            for level, values in diagnostic["explained_variance_by_level"].items():
                lines.append(f"| {level} | {number(values['incremental_explained_variance'])} | {number(values['cumulative_explained_variance'])} |")
            lines.append("Variance is measured about zero on shared held-out cells; negative increments are retained.")
            lines += ["", "Hierarchical vs flat B2 at matched total rank:", "",
                      "| Fold | rank budget | hierarchical fitted rank | flat fitted rank | hierarchical MSE | flat B2 MSE |",
                      "|---|---:|---:|---:|---:|---:|"]
            for row in diagnostic["matched_rank_comparison"]:
                lines.append(f"| {row['fold']} | {row['total_rank_budget']} | {row['hierarchical_fitted_rank']} | {row['flat_fitted_rank']} | {number(row['hierarchical_mse'])} | {number(row['flat_B2_mse'])} |")
            lines.append(diagnostic["matched_rank_note"])
        factors = result.get("metrics", {}).get(candidate, {}).get("per_factor", {})
        if factors:
            lines += ["", "| Factor axis | factor | net-benefit Spearman | source ranking (indices) |", "|---|---|---:|---|"]
            for axis, labels in factors.items():
                for label, metric in labels.items():
                    lines.append(f"| {axis} | {label} | {number(metric['net_benefit_spearman'])} | {metric['source_ranking']} |")
    lines += ["", "Effective samples: `" + canonical_json(report["effective_samples"]) + "`.", "",
              report["control_interpretation"], "", "Measurement/implementation evidence: `" + canonical_json(fit.get("measurement", {})) + "`.",
              "Unreported floor, KL, update norm, optimizer or checker checks remain unverified.", "",
              "Costs: `" + canonical_json(costs) + "`. Existing teacher supervision is not recounted as new teacher tokens.", "",
              "Commit: `" + manifest["commit"] + "`; config hash: `" + manifest["config_hash"] + "`.",
              "Input/split content hashes, seeds and artifact hashes are recorded in the stage manifest and stage_report.json.", "",
              "Executed command:", "```bash", manifest["command"], "```", "",
              "Evidence: `fit_result.json`, `fit_arrays.npz`, and `stage_report.json` (NA is JSON null).", "",
              report["conclusion_boundary"], "", report["next_actions"][0]["action"] + "."]
    return report, "\n".join(lines) + "\n"


def _collect(stage, config, protocol, data, out, resume):
    probes, seeds = _validate_collection(data, stage, protocol, executing=True)
    # The concurrent implementation is an optional dependency. Never import it
    # during module import, fit/report, or a resource-plan dry run.
    runner = getattr(importlib.import_module("src.bfas.behavior.microupdate"), "run_micro_update")
    factory = _symbol(protocol["model_factory"]) if protocol.get("model_factory") else None
    evaluator = _symbol(protocol["evaluator"])
    assessor = _symbol(protocol["pilot_assessor"]) if stage == "pilot" else None
    outcomes, updates = [], []
    for source in data["sources"]:
        sid = source["source_id"]
        # IDs are hashed for artifact paths, never interpreted as filenames.
        source_run = f"{stage}-{content_hash(sid)[:16]}"
        source_path = out / f"{source_run}.responses.jsonl"
        if resume and source_path.exists():
            saved_probes, saved_outcomes, saved_manifest = read_responses(source_path)
            expected = {"protocol_hash": content_hash(protocol), "source_hash": content_hash(source),
                        "eval_seeds": seeds}
            if any(saved_manifest["provenance"].get(k) != v for k, v in expected.items()) or saved_probes != probes:
                raise ValueError("source response resume identity mismatch")
            outcomes.extend(saved_outcomes)
            updates.append(saved_manifest["provenance"]["update"])
            continue
        cfg = {**protocol["micro_update"], "source_id": sid, "run_id": source_run,
               "output_dir": str(out / "updates"), "dry_run": False, "resume": resume,
               "data_manifest_hash": content_hash(data),
               "probe_rows": [p["row"] for p in data["probes"] if "row" in p]}
        if source.get("role") == "no_op":
            cfg["steps"] = 0
        update = runner(factory, source["rows"], cfg)
        update_manifest = update.manifest
        if update_manifest.get("status") != "complete":
            raise ValueError(f"micro-update did not complete for {sid}")
        records = evaluator(update=update, source=source, probes=data["probes"], eval_seeds=seeds, protocol=protocol)
        records = [r if isinstance(r, OutcomeRecord) else OutcomeRecord(**r) for r in records]
        factor = source.get("source_factor", source.get("family", source.get("factor")))
        if factor is not None:
            if any(r.source_factor not in {None, factor} for r in records):
                raise ValueError("evaluator source_factor differs from manifest")
            records = [replace(r, source_factor=factor) for r in records]
        if any(r.source_id != sid or r.run_id != update_manifest["run_id"] for r in records):
            raise ValueError("evaluator returned mismatched source or update run identity")
        assemble_matrix(records, source_ids=[sid], probe_ids=[p.probe_id for p in probes], eval_seeds=seeds)
        provenance = {"protocol_hash": content_hash(protocol), "source_hash": content_hash(source),
                      "eval_seeds": seeds, "update": update_manifest}
        write_responses(source_path, probes, records, provenance, resume=resume)
        outcomes.extend(records)
        updates.append(update_manifest)
    combined = out / "responses.jsonl"
    assembled = assemble_matrix(outcomes, source_ids=[s["source_id"] for s in data["sources"]],
                                probe_ids=[p.probe_id for p in probes], eval_seeds=seeds)
    write_responses(combined, probes, outcomes, {"eval_seeds": seeds, "frozen_config_hash": content_hash(protocol)}, resume=resume)
    result = {"stage": stage, "status": "complete", "updates": updates,
              "response_diagnostics": assembled.diagnostics, "new_teacher_tokens": 0}
    if stage == "pilot":
        assessed = assessor(outcomes=outcomes, updates=updates, protocol=protocol)
        if not isinstance(assessed, dict) or not isinstance(assessed.get("passed"), bool):
            raise ValueError("pilot assessor must return a machine-readable boolean passed and measurement evidence")
        result["assessment"] = assessed
        if assessed["passed"]:
            if assessed.get("measurement_stable") is not True or assessed.get("measurement_resolved") is not True:
                raise ValueError("pilot cannot pass without stable/resolved behaviour measurements")
            if not np.isfinite(assembled.M).any() or not np.any(np.isfinite(assembled.M) & (assembled.M != 0)):
                raise ValueError("pilot cannot pass with no measured behaviour changes")
            write_json(out / "pilot_pass.json", {**assessed, "status": "pass", "frozen_config_hash": content_hash(protocol),
                       "artifact_hashes": {p.name: content_hash(p.read_bytes()) for p in
                                           (combined, Path(str(combined) + ".manifest.json"))}}, resume=resume)
    return result


def main(argv=None) -> int:
    # GPU shard assembly is CPU-only and needs no training configuration.
    command_line = list(sys.argv[1:] if argv is None else argv)
    if command_line and command_line[0] == "merge":
        from tools.behavior_atom.gpu_driver import main as gpu_main
        return gpu_main(command_line)
    if command_line and command_line[0] == "codes":
        from tools.behavior_atom.build_codes import main as codes_main
        return codes_main(command_line[1:])
    if command_line and command_line[0] == "margin":
        from src.bfas.behavior.margin import main as margin_main
        return margin_main(command_line[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("codes", help="encode saved deltas in training-fold Fisher bases (codes --help)")
    subparsers.add_parser("margin", help="checked decode pairs, saved-state likelihoods and Q1–Q3 (margin --help)")
    for stage in STAGES:
        child = subparsers.add_parser(stage)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--manifest", type=Path, help="saved input or data manifest; overrides config.data")
        child.add_argument("--output-dir", type=Path, help="run directory; defaults to results/behavior_atom_v1/<run_id>")
        child.add_argument("--dry-run", action="store_true")
        child.add_argument("--resume", action="store_true")
        if stage in {"pilot", "collect"}:
            child.add_argument("--sources", type=Path, help="GPU source-unit manifest (alias for --manifest)")
            child.add_argument("--probes", type=Path, help="GPU BFCL probe manifest")
            child.add_argument("--pool", type=Path, help="existing teacherized pool JSONL")
            child.add_argument("--shard", help="collect source partition, zero-based i/n")
            child.add_argument("--profile", action="store_true", help="GPU driver: save phase timings in manifest.json")
            child.add_argument("--cpu-threads", type=int, help="GPU driver: cap torch CPU threads (default: 8)")
            if stage == "pilot":
                child.add_argument("--max-sources", type=int, help="GPU driver: run at most N selected sources")
                child.add_argument("--smoke", action="store_true", help="timed diagnostic: one source, one step, one probe, at most 32 new tokens")
        if stage in {"fit", "report"}:
            child.add_argument("--model", choices=("flat", "hierarchical"), help="response model; defaults to flat or config.fit.model")
            child.add_argument("--target", choices=("behavior", "teacher", "student", "combined", "all"),
                               help="response target; all fits teacher/student/combined independently")
            child.add_argument("--family-ranks", nargs="+", type=int, choices=(1, 2))
            child.add_argument("--min-family-sources", type=int)
            child.add_argument("--residual-rank", type=int)
    args = parser.parse_args(argv)
    try:
        if args.resume and (args.dry_run or args.stage in {"intervene", "distill"}):
            raise ValueError("--resume is inapplicable to dry-run plans and unimplemented execution stages")
        references = {}
        config_path = args.config.resolve()
        config = _read(config_path, references)
        if args.stage in {"fit", "report"}:
            fit_config = dict(config.get("fit", {}))
            for key in ("model", "family_ranks", "min_family_sources", "residual_rank"):
                value = getattr(args, key)
                if value is not None:
                    fit_config[key] = value
            config["fit"] = fit_config
            if args.target is not None:
                config["target"] = args.target
        config["_base"] = str(config_path.parent)
        source_arg = getattr(args, "sources", None)
        if source_arg and args.manifest:
            raise ValueError("use only one of --sources and --manifest")
        manifest_arg = source_arg or args.manifest
        input_path = manifest_arg.resolve() if manifest_arg else (_resolve(config["data"], config_path.parent) if config.get("data") else None)
        if args.stage in {"fit", "report", "decompose", "audit", "pilot", "collect"} and input_path is None:
            raise ValueError(f"{args.stage} requires --manifest or config.data")
        data = (load_dataset(input_path, config, references) if args.stage in {"fit", "report", "decompose"}
                else _read(input_path, references)) if input_path else {}
        if args.stage in {"pilot", "collect"}:
            gpu_inputs = ("units" in data or isinstance(data.get("sources"), str)
                          or config.get("driver") == "gpu" or source_arg
                          or args.probes or args.pool)
            if gpu_inputs:
                from tools.behavior_atom.gpu_driver import cli_run
                for key in ("probes", "pool"):
                    value = getattr(args, key)
                    if value:
                        data[key] = str(value.resolve())
                    elif key in config:
                        data.setdefault(key, str(_resolve(config[key], config_path.parent)))
                return cli_run(args.stage, config, data, input_path, args.output_dir,
                               resume=args.resume, dry_run=args.dry_run, shard=args.shard,
                               profile=args.profile, cpu_threads=args.cpu_threads,
                               max_sources=getattr(args, "max_sources", None), smoke=getattr(args, "smoke", False))
            if args.profile or args.cpu_threads is not None or getattr(args, "max_sources", None) is not None or getattr(args, "smoke", False):
                raise ValueError("profiling/thread/source/smoke flags require GPU source-unit manifests")
            if args.shard:
                raise ValueError("--shard requires GPU source-unit manifests")
        if args.stage == "report" and data.get("schema_version") == "behavior-fit-v1":
            if args.model is not None and args.model != data.get("fit_config", {}).get("model", "flat"):
                raise ValueError("--model differs from the saved fit; report cannot refit without responses/codes")
            for key in ("family_ranks", "min_family_sources", "residual_rank"):
                value = getattr(args, key)
                if value is not None and value != data.get("fit_config", {}).get(key):
                    raise ValueError(f"--{key.replace('_', '-')} differs from the saved fit")
            if args.target is not None and args.target != data.get("response_target", "behavior"):
                if args.target in data.get("targets", {}):
                    data = data["targets"][args.target]
                elif args.target != "all" or not data.get("targets"):
                    raise ValueError("requested target is absent from the saved fit")
        protocol = frozen_protocol(config) if args.stage in {"pilot", "collect"} else None
        if args.stage == "collect":
            require_pilot_pass(config, protocol, references)  # enforce even for --dry-run
        if args.stage in {"pilot", "collect"}:
            _validate_collection(data, args.stage, protocol)
        if args.stage == "fit":
            _fit_inputs(data, config)
        run_id = config.get("run_id", f"{args.stage}-{content_hash({k: v for k, v in config.items() if k != '_base'})[:12]}")
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a single directory name")
        out = args.output_dir.resolve() if args.output_dir else ROOT / "results" / "behavior_atom_v1" / run_id
        resource_plan = plan(args.stage, config, data, protocol)
        resource_plan.update(run_id=run_id, output_dir=str(out))
        if args.dry_run or args.stage in {"intervene", "distill"}:
            print(canonical_json(resource_plan))
            if not args.dry_run:
                print(f"{args.stage} execution is unavailable; --dry-run is required", file=sys.stderr)
                return 2
            return 0
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        effective = {k: v for k, v in config.items() if k != "_base"}
        # --resume is operational, not part of the immutable scientific identity.
        command_args = [arg for arg in (sys.argv[1:] if argv is None else argv) if arg != "--resume"]
        manifest = {"schema_version": "behavior-stage-v1", "run_id": run_id, "stage": args.stage,
                    "commit": commit, "config": effective, "config_hash": content_hash(effective),
                    "input_hashes": references, "data_manifest_hash": content_hash(references),
                    "split_hash": content_hash(data.get("folds", config.get("folds", data.get("sources", [])))),
                    "code_hashes": {str(path.relative_to(ROOT)): content_hash(path.read_bytes()) for path in
                                    [Path(__file__).resolve(), ROOT / "src/bfas/behavior/response.py", ROOT / "src/bfas/behavior/lowrank.py"]},
                    "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *map(str, command_args)]),
                    "training_seeds": (protocol or {}).get("micro_update", {}).get("seed"),
                    "evaluation_seeds": (protocol or {}).get("eval_seeds", data.get("eval_seeds")),
                    "fit_seeds": {"bootstrap": config.get("fit", {}).get("bootstrap_seed", 0),
                                  "permutations": config.get("fit", {}).get("permutation_seeds", [11, 29, 47])},
                    "job_ids": [d["job_id"] for d in data.get("dependencies", []) if isinstance(d, dict) and d.get("job_id")],
                    "new_teacher_tokens": 0}
        write_json(out / f"{args.stage}.manifest.json", manifest, resume=args.resume)
        complete_path = out / f"{args.stage}.complete.json"
        if complete_path.exists():
            saved = json.loads(complete_path.read_text())
            if not args.resume:
                raise FileExistsError("completed stage exists; use --resume")
            for name, digest in saved["artifact_hashes"].items():
                if content_hash((out / name).read_bytes()) != digest:
                    raise ValueError(f"completed artifact changed: {name}")
            print(canonical_json({"status": "resumed complete", "output_dir": str(out)}))
            return 0
        artifact_paths = []
        if args.stage in {"fit", "report"}:
            if data.get("schema_version") == "behavior-fit-v1":
                if args.stage == "fit":
                    raise ValueError("fit needs saved responses/codes, not an already fitted report")
                fitted, arrays = data, None
            else:
                fitted, arrays = _fit(data, config)
            fit_path = out / "fit_result.json"
            write_json(fit_path, fitted, resume=args.resume or args.stage == "report")
            artifact_paths.append(fit_path)
            if arrays is not None:
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **arrays)
                array_path = out / "fit_arrays.npz"
                write_immutable(array_path, buffer.getvalue(), resume=args.resume or args.stage == "report")
                artifact_paths.append(array_path)
            elif (out / "fit_arrays.npz").exists():
                artifact_paths.append(out / "fit_arrays.npz")
            costs = {"new_teacher_tokens": 0, "gpu_hours": 0, "environment_rollouts": 0,
                     "cpu_only": True, "historical_training_and_teacher_usage": config.get("historical_costs", "not supplied")}
            artifacts = {p.name: content_hash(p.read_bytes()) for p in artifact_paths}
            report, markdown = _report(fitted, manifest, artifacts, costs)
            # A report subcommand may follow fit in the same directory; keep
            # each report immutable while preserving the original stage report.
            prefix = "report" if args.stage == "report" and (out / "stage_report.json").exists() else "stage_report"
            write_json(out / f"{prefix}.json", report, resume=args.resume)
            write_immutable(out / f"{prefix}.md", markdown.encode(), resume=args.resume)
            artifact_paths += [out / f"{prefix}.json", out / f"{prefix}.md"]
        elif args.stage in {"pilot", "collect"}:
            collected = _collect(args.stage, config, protocol, data, out, args.resume)
            write_json(out / f"{args.stage}_result.json", collected, resume=args.resume)
            artifact_paths += [out / f"{args.stage}_result.json", out / "responses.jsonl", out / "responses.jsonl.manifest.json"]
            if (out / "pilot_pass.json").exists():
                artifact_paths.append(out / "pilot_pass.json")
        elif args.stage == "audit":
            problems = []
            try:
                validate_split(data.get("sources", []), [_probe(p) for p in data.get("probes", [])])
            except ValueError as exc:
                problems.append(str(exc))
            runs = data.get("runs", [])
            audit = {"stage": "P0", "decision": "BLOCKED" if problems else "INCONCLUSIVE",
                     "reason": "supplied manifests audited; numerical validity still requires measurement",
                     "valid_runs": [r for r in runs if r.get("status") == "complete" and r.get("exit_code") == 0],
                     "invalid_runs": [r for r in runs if r.get("exit_code") not in {None, 0}],
                     "blocking_issues": problems, "dependencies": data.get("dependencies", []),
                     "costs": config.get("historical_costs", {}), "metrics": {}, "confidence_intervals": {},
                     "artifacts": references, "next_actions": [], "new_teacher_tokens": 0}
            path = out / "audit.json"
            write_json(path, audit, resume=args.resume)
            artifact_paths.append(path)
        elif args.stage == "decompose":
            if "M_teacher" not in data or "M_student" not in data:
                raise ValueError("decompose requires saved M_teacher and M_student with the same score definition")
            teacher, student = np.asarray(data["M_teacher"], dtype=float), np.asarray(data["M_student"], dtype=float)
            if teacher.ndim != 2 or teacher.shape != student.shape or np.isinf(teacher).any() or np.isinf(student).any():
                raise ValueError("teacher/student response axes must match and contain finite values or NA")
            if not data.get("likelihood_definition", data.get("response_diagnostics", {}).get("likelihood_definition")):
                raise ValueError("decompose requires an explicit matched likelihood_definition")
            decomposition = {"M_teacher": teacher, "M_student": student, "M_difference": teacher - student,
                             "likelihood_definition": data.get("likelihood_definition", data.get("response_diagnostics", {}).get("likelihood_definition"))}
            if "X" in data:
                decomposition["competition"] = {side: _fit({**data, "M": matrix, "response_kind": "likelihood"}, config)[0]
                                                 for side, matrix in (("teacher", teacher), ("student", student), ("difference", teacher - student))}
            else:
                decomposition["missing_dependencies"] = ["codes and fixed folds for baseline predictions"]
            path = out / "decomposition.json"
            write_json(path, decomposition, resume=args.resume)
            artifact_paths.append(path)
        write_json(complete_path, {"status": "complete", "exit_code": 0,
                   "manifest_hash": content_hash(manifest),
                   "artifact_hashes": {p.name: content_hash(p.read_bytes()) for p in artifact_paths}}, resume=args.resume)
        print(canonical_json({"status": "complete", "stage": args.stage, "output_dir": str(out), "new_teacher_tokens": 0}))
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        print(f"{args.stage}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
