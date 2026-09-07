"""One resident BFCL student, independent LoRA updates, and checked responses.

The stage CLI accepts --manifest sources.json --probes probes.json --pool pool.jsonl,
or config.data naming {sources: 'sources.json', probes: 'probes.json', pool: 'pool.jsonl'}.
Paths in a descriptor are relative to that descriptor. Configuration lives in
protocol.micro_update and protocol.evaluation; all defaults below are frozen.
Source pool_path references also accept repository-relative paths. Pilot defaults
to --roles pilot noop; collect defaults to --roles formal. Add anchor explicitly.
Use init-state --config CONFIG [--init-state-path PATH] to save only initialization.
Settings differences are recorded warnings unless --strict-init-state is set.
Only greedy decoding is supported: stochastic requests are rejected, never
silently treated as paired probability estimates. Training uses the existing
micro-update runner, with every row of a pack/contrastive unit in every step.

Collect --shard 0/2 and --shard 1/2 write separate shard-00000-of-00002 directories.
Merge (CPU only): python tools/behavior_atom/gpu_driver.py merge --shards DIR0 DIR1
--output-dir MERGED. Complete source directories are transactional; --resume
checks content and restarts only uncommitted work with the same frozen settings.

KL is KL(updated || base), full vocabulary at the first response position of
each capped probe prompt, averaged over probes. It is NOT rollout/sequence KL.
Likelihood uses the trainer's native-logit cross entropy and fp32 reduction,
including EOS; the same ell convention is supplied to the training gradients.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.bfas.behavior.provenance import repo_commit
from src.bfas.behavior.response import (
    OutcomeRecord, ProbeRecord, assemble_matrix, canonical_json, content_hash, read_responses,
    validate_split, write_immutable, write_json, write_responses,
)
from tools.behavior_atom.checker_bridge import CheckerBridge

VERSION = "behavior-gpu-v1"
KL_DEFINITION = "KL(updated||base), full vocabulary, mean probe first-response position"


def _read(path):
    return json.loads(Path(path).read_text())


def _resolve(path, base):
    return (Path(base) / path).resolve()


def _sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


def _check_sha1(text, expected, label):
    if not isinstance(expected, str) or len(expected) not in {16, 40} or _sha1(text)[:len(expected)] != expected:
        raise ValueError(f"{label} content hash mismatch")


def _slug(sid):
    return sid if re.fullmatch(r"[A-Za-z0-9_-]+", sid) else content_hash(sid)


def _thinking_off(prompt):
    # BFCLAdapter._render already renders tools and the assistant prefix. Do not
    # wrap it in another chat template. Complete an empty Qwen thinking prefix.
    if prompt.rstrip().endswith("</think>"):
        return prompt
    if prompt.rstrip().endswith("<think>"):
        return prompt.rstrip() + "\n\n</think>\n\n"
    return prompt + "<think>\n\n</think>\n\n"


def frozen_settings(protocol):
    """Resolve every scientific default before computing the pilot/collect hash."""
    result = json.loads(json.dumps(protocol))
    if result.get("new_teacher_tokens", 0) != 0:
        raise ValueError("new teacher token budget must be zero")
    micro = {"model_id": "Qwen/Qwen3.5-4B", "steps": 3, "lr": 5e-6,
             "beta": 0.1, "seed": 0, "init_seed": 0, "device": "cuda",
             "precision": "native", "ell": "full_sequence", "prompt_cap": int(os.environ.get("AW_MAX_PROMPT_TOKENS", 2048)),
             "prompt_side": "tail", "response_cap": 512, "allow_prompt_truncation": True,
             "weight_decay": 0.01, "adam_betas": [0.9, 0.999], "adam_eps": 1e-8,
             "scheduler": "constant", "max_grad_norm": None, "cpu_threads": 8, **result.get("micro_update", {})}
    evaluation = {"temperature": 0, "do_sample": False, "num_beams": 1,
                  "max_new_tokens": 512, "batch_size": 4, "token_budget": 8192,
                  "head_chunk": 128, "enable_thinking": False,
                  "ell": micro["ell"], "policy_kl": KL_DEFINITION,
                  "checker_model": "Qwen/Qwen3.5-4B-FC", **result.get("evaluation", {})}
    pilot = {"max_policy_kl": 0.1, "max_abstention_shift": 0.25,
             "max_failed_fraction": 0.0, "floor_tolerance": 0.0,
             "min_changed_probes": 1, "min_changed_source_fraction": 0.5,
             **result.get("pilot", {})}
    if result.get("stochastic", False) or evaluation["do_sample"] or evaluation["temperature"] != 0 or evaluation["num_beams"] != 1:
        raise ValueError("GPU driver requires deterministic greedy decoding; stochastic paired seeds are unsupported")
    if evaluation["enable_thinking"] is not False or micro["prompt_side"] != "tail":
        raise ValueError("GPU driver requires thinking off and AW_MAX_PROMPT_TOKENS tail retention")
    if evaluation["ell"] != micro["ell"] or micro["ell"] not in {"full_sequence", "length_normalized"}:
        raise ValueError("gradient and evaluation ell must match")
    if evaluation["policy_kl"] != KL_DEFINITION:
        raise ValueError("unsupported policy KL definition")
    if micro.get("batch_size", 1) != 1 or micro.get("accumulation", 1) != 1:
        raise ValueError("unit batching is driver-owned: all rows once per step")
    if micro["response_cap"] != 512:
        raise ValueError("response_cap must match the trainer (512 plus EOS)")
    for key in ("max_new_tokens", "batch_size", "token_budget", "head_chunk"):
        if type(evaluation[key]) is not int or evaluation[key] < 1:
            raise ValueError(f"invalid evaluation.{key}")
    for key in ("steps", "prompt_cap", "cpu_threads"):
        if type(micro[key]) is not int or micro[key] < 1:
            raise ValueError(f"invalid micro_update.{key}")
    for key, value in pilot.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid pilot.{key}")
    for key in ("max_abstention_shift", "max_failed_fraction", "min_changed_source_fraction"):
        if pilot[key] > 1:
            raise ValueError(f"pilot.{key} must be a fraction")
    seeds = result.get("eval_seeds", [0])
    if len(seeds) != 1 or type(seeds[0]) is not int or not 0 <= seeds[0] < 2**32:
        raise ValueError("greedy outcomes require exactly one fixed evaluation seed")
    result.update(micro_update=micro, evaluation=evaluation, pilot=pilot,
                  eval_seeds=seeds, stochastic=False, new_teacher_tokens=0,
                  driver_version=VERSION, unit_batching="all rows once per optimizer step",
                  model_loading="trainer bf16 base and native PEFT trainable dtypes",
                  gradient_checkpointing={"enabled": True, "use_reentrant": False},
                  loss="-logsigmoid(beta * ((ell_T - ell_S) - initial_margin))",
                  numerical_settings={"deterministic_algorithms": True, "allow_tf32": False,
                                      "cudnn_benchmark": False, "autocast": False},
                  thinking_prefix="append empty <think> block to serialized prompts before tail cap",
                  score_precision="native logits/cross entropy; fp32 sum; EOS included",
                  policy_precision="fp32 log_softmax and KL accumulation")
    return result


def load_inputs(config, data, input_path):
    """Resolve row references and verify actual prompt/teacher/student contents."""
    base = Path(input_path).parent
    inputs = {}
    def read_ref(value, key):
        if isinstance(value, (str, Path)):
            path = _resolve(value, base)
            inputs[key] = content_hash(path.read_bytes())
            return _read(path)
        inputs[key] = content_hash(value)
        return value
    source_ref = data if "units" in data else data.get("sources", config.get("sources"))
    sources = read_ref(source_ref, "sources")
    source_base = _resolve(source_ref, base).parent if isinstance(source_ref, (str, Path)) else base
    probes = read_ref(data.get("probes", config.get("probes")), "probes")
    def source_pool_path(value):
        path = _resolve(value, source_base)
        # Coordinator manifests use data/... paths from the repository root.
        return path if path.exists() or Path(value).is_absolute() else _resolve(value, ROOT)
    default_pool = data.get("pool") or config.get("pool")
    pool_path = (_resolve(default_pool, base) if default_pool else
                 source_pool_path(sources.get("pool_path", "")))
    pools, pool_hashes = {}, {}
    def read_pool(path):
        if path not in pools:
            if not path.is_file():
                raise ValueError(f"GPU driver requires an existing pool JSONL: {path}")
            payload = path.read_bytes()
            pool_hashes[path] = content_hash(payload)
            pools[path] = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        return pools[path]
    read_pool(pool_path)
    inputs["pool"] = pool_hashes[pool_path]
    units = sources["units"]
    raw_probes = probes["probes"] if isinstance(probes, dict) else probes
    # Content identities are independent of descriptor syntax and mount paths.
    inputs["sources"] = content_hash(units)
    inputs["probes"] = content_hash(raw_probes)
    if not units or not raw_probes:
        raise ValueError("nonempty units and probes are required")
    prepared = []
    for unit in units:
        if unit.get("unit_type") not in {"event", "single", "pack", "contrastive", "noop"}:
            raise ValueError("unsupported unit_type")
        if not all(isinstance(unit.get(k), str) and unit[k] for k in ("source_id", "parent_task_id", "group", "role")):
            raise ValueError("source identity/group/role is missing")
        rows = []
        for ref in unit["rows"]:
            row_pool = ref.get("pool_path") or unit.get("pool_path")
            pool = read_pool(source_pool_path(row_pool) if row_pool else pool_path)
            index = ref["pool_row"]
            if type(index) is not int or not 0 <= index < len(pool):
                raise ValueError("pool_row must be a zero-based nonblank JSONL row index")
            row = pool[index]
            for field, key in (("prompt", "prompt_sha1"), ("response", "yT_sha1"), ("_rejected", "yS_sha1")):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"pool row requires {field}")
                _check_sha1(row[field], ref[key], key)
            from tools.behavior_atom.split_check import event_key
            if ref["event_key"] != event_key(row):
                raise ValueError("pool event_key mismatch")
            # Serialized BFCL text is authoritative, even if a legacy row also
            # carries messages. No teacher text or supervision is generated.
            rows.append({**{k: v for k, v in row.items() if k != "messages"},
                         "prompt": _thinking_off(row["prompt"]),
                         "source_id": unit["source_id"], "event_id": ref["event_key"]})
        if not rows or (unit["unit_type"] in {"single", "event"} and len(rows) != 1):
            raise ValueError("empty unit or multiple rows in a single event")
        if len({r["event_key"] for r in unit["rows"]}) != len(rows):
            raise ValueError("duplicate row within source unit")
        prepared.append({**unit, "resolved_rows": rows})
    records = []
    for p in raw_probes:
        _check_sha1(p["prompt"], p["prompt_sha1"], "probe prompt_sha1")
        if p["expected"] not in {"call", "abstain"}:
            raise ValueError("expected must be call or abstain")
        if p["expected"] == "call" and (not p.get("truth") or not p.get("function")):
            raise ValueError("call probes require function and truth")
        records.append(ProbeRecord(
            probe_id=p["probe_id"], parent_task_id=p["parent_task_id"], trajectory_id=p["task_id"],
            state_hash=content_hash(p["prompt"]), target_hash=content_hash(p.get("response", p["truth"])),
            student_output_hash=content_hash(p.get("_rejected")), benchmark="bfcl",
            split=p.get("split", "P_disc"), group=p["group"], probe_factor=p.get("factor"),
            probe_family=p.get("family"), probe_margin_base=p.get("base_margin")))
    if len({u["source_id"] for u in units}) != len(units) or len({p.probe_id for p in records}) != len(records):
        raise ValueError("duplicate source or probe ID")
    if len({_slug(u["source_id"]) for u in units}) != len(units):
        raise ValueError("source artifact path collision")
    validate_split(units, records)
    trajectories = {r["_traj"] for u in prepared for r in u["resolved_rows"]}
    if trajectories & {p.trajectory_id for p in records} or {u["group"] for u in units} & {p.group for p in records}:
        raise ValueError("source/probe trajectory or group overlap")
    if {r["prompt_sha1"][:16] for u in units for r in u["rows"]} & {p["prompt_sha1"][:16] for p in raw_probes}:
        raise ValueError("source/probe prompt overlap")
    if len(records) > 32:
        raise ValueError("P2 permits at most 32 development probes")
    inputs["pools"] = sorted(set(pool_hashes.values()))
    return {"sources": prepared, "probes": raw_probes, "probe_records": records,
            "input_hashes": inputs, "dependencies": data.get("dependencies", [])}


def select_sources(data, stage, shard=None, roles=None, max_sources=None):
    if stage not in {"pilot", "collect"}:
        raise ValueError("source selection requires pilot or collect")
    if roles is None:
        roles = ["pilot", "noop"] if stage == "pilot" else ["formal"]
    if isinstance(roles, str):
        roles = [roles]
    roles = {role.strip() for value in roles for role in re.split(r"[,+]", value)}
    if not roles or roles - {"pilot", "formal", "noop", "no_op", "anchor"}:
        raise ValueError("--roles requires pilot, formal, noop, no_op or anchor")
    if roles & {"noop", "no_op"}:
        roles.update({"noop", "no_op"})
    sources = [u for u in data["sources"] if u["role"] in roles]
    counts = {role: sum(u["role"] == role for u in sources) for role in roles}
    if (not sources or counts.get("pilot", 0) > 4 or counts.get("formal", 0) > 24 or
            counts.get("noop", 0) + counts.get("no_op", 0) > 2):
        raise ValueError("P2 source budget: at most 4 pilots / 24 formal units and 2 noops")
    if stage == "pilot" and not any(u["role"] == "pilot" for u in sources):
        raise ValueError("pilot requires role=pilot sources")
    if max_sources is not None:
        if stage != "pilot" or type(max_sources) is not int or max_sources < 1:
            raise ValueError("--max-sources requires a positive integer and the pilot stage")
        # Always exercise training, even when noops precede pilot units on disk.
        sources = sorted(sources, key=lambda u: u["role"] != "pilot")[:max_sources]
    index, total = 0, 1
    if shard is not None:
        if stage != "collect":
            raise ValueError("--shard is supported only by collect")
        try:
            index, total = map(int, shard.split("/"))
        except (ValueError, AttributeError):
            raise ValueError("--shard must be zero-based i/n") from None
        if not 0 <= index < total or total > len(sources):
            raise ValueError("--shard requires 0 <= i < n <= source count")
    return sources, sources[index::total], (index, total)


def resource_plan(stage, config, protocol, data, shard=None, roles=None, max_sources=None):
    all_sources, sources, partition = select_sources(data, stage, shard, roles, max_sources)
    n, p = len(sources), len(data["probes"])
    rollouts = (n + 1) * p
    checks = 2 * p if stage == "pilot" else 0
    cost = config.get("cost_estimate", {})
    train, inference = cost.get("gpu_hours_per_update"), cost.get("gpu_seconds_per_rollout")
    hours = (sum(u["role"] not in {"noop", "no_op"} for u in sources) * train +
             (rollouts + checks) * inference / 3600) if train is not None and inference is not None else None
    return dict(stage=stage, status="dry-run plan", number_of_models=1, resident_models=1,
                model_states=n + 1, sources=n, total_sources=len(all_sources), probes=p,
                max_rollouts=rollouts, numerical_check_rollouts=checks,
                total_rollouts_including_checks=rollouts + checks, shard=list(partition),
                dependencies=data["dependencies"] + ["torch", "transformers", "peft", "bfcl_eval",
                    "src.bfas.behavior.microupdate.run_micro_update", protocol["micro_update"]["model_id"]] +
                    ([config.get("pilot_pass", "matching pilot_pass.json")] if stage == "collect" else []),
                new_teacher_tokens=0, teacher_tokens=0, frozen_config_hash=content_hash(protocol),
                cost_estimate={"gpu_hours": hours, "rates": cost,
                               "basis": "configured throughput; null means unmeasured; includes pilot checks"})


class Student:
    """Inject a tiny model/tokenizer and fake parse/check for CPU integration tests."""
    def __init__(self, model, tokenizer, checker_version, parse, check, *, checker_mode="injected", close=None):
        self.model, self.tokenizer = model, tokenizer
        self.checker_version, self.parse, self.check = checker_version, parse, check
        self.checker_mode, self._close_checker = checker_mode, close

    def close(self):
        if self._close_checker is not None:
            self._close_checker()

    def generate(self, ids, mask, settings):
        from transformers import GenerationConfig
        cfg = GenerationConfig(do_sample=False, temperature=0.0, num_beams=1,
                               max_new_tokens=settings["max_new_tokens"],
                               pad_token_id=self.tokenizer.pad_token_id,
                               eos_token_id=self.tokenizer.eos_token_id,
                               bos_token_id=getattr(self.tokenizer, "bos_token_id", None),
                               use_cache=True)
        return self.model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)


def load_model(protocol):
    """Build the driver model without tokenizer, checker, or evaluation work."""
    import torch
    import appworld_train as trainer
    micro = protocol["micro_update"]
    if not micro["device"].startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("real GPU driver requires CUDA; inject Student for CPU tests")
    model = trainer.build_lora_model(micro["model_id"], micro["init_seed"])
    model.to(torch.device(micro["device"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    # PEFT may promote A/B to fp32. Keep them so deltas match the Fisher layout.
    return model


def load_student(protocol):
    """Load exactly one bf16 base + trainer LoRA, without changing its layout."""
    import appworld_train as trainer
    from tools.bfcl_event_mine_single import parsed_calls
    micro = protocol["micro_update"]
    model = load_model(protocol)
    tokenizer = trainer.load_tokenizer(micro["model_id"])
    checker = CheckerBridge(protocol["evaluation"]["checker_model"])
    return Student(model, tokenizer, checker.checker_version, parsed_calls, checker.check,
                   checker_mode=checker.mode, close=checker.close)


@contextmanager
def _context(protocol):
    # Reuse the runner's deterministic flags, paired RNG restoration, and caps.
    from src.bfas.behavior import microupdate as micro
    import torch
    cfg = micro.MicroUpdateConfig(**{**protocol["micro_update"], "init_state_path": "unused", "source_id": "evaluation"})
    cfg.seed = protocol["eval_seeds"][0]
    with micro._RUN_LOCK, micro._training_context(cfg):
        previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device_type=torch.device(cfg.device).type, enabled=False):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _batches(lengths, evaluation):
    result, batch = [], []
    for k in sorted(range(len(lengths)), key=lambda k: (lengths[k], k)):
        if batch and (len(batch) >= evaluation["batch_size"] or (len(batch) + 1) * lengths[k] > evaluation["token_budget"]):
            result.append(batch)
            batch = []
        batch.append(k)
    return result + ([batch] if batch else [])


def _padded(sequences, pad, device, left=False):
    import torch
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        start = length - len(sequence) if left else 0
        ids[i, start:start + len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, start:start + len(sequence)] = 1
    return ids, mask


def _position_logits(student, ids, mask, positions, chunk):
    """Only response positions pass through the LM head on the real Qwen model."""
    from src.bfas.fingerprint import _backbone_and_head
    backbone, head = _backbone_and_head(student.model)
    if backbone is not None:
        hidden = backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        for start in range(0, len(positions), chunk):
            pos = positions[start:start + chunk]
            yield head(hidden[pos[:, 0], pos[:, 1]])
    else:
        logits = student.model(input_ids=ids, attention_mask=mask).logits
        for start in range(0, len(positions), chunk):
            pos = positions[start:start + chunk]
            yield logits[pos[:, 0], pos[:, 1]]


def _failure(record, kind, exc):
    record["errors"].append(f"{kind}:{type(exc).__name__}")


def evaluate(student, probes, protocol, *, progress=None, phase="evaluation"):
    """Batched greedy verdicts, optional paired ell, and full-vocabulary policy."""
    import torch
    import torch.nn.functional as F
    import appworld_train as trainer
    from src.bfas.behavior.microupdate import Progress
    progress = progress or Progress(protocol["micro_update"]["device"])
    ev, micro = protocol["evaluation"], protocol["micro_update"]
    tok, device = student.tokenizer, micro["device"]
    rows = [{"prompt": _thinking_off(p["prompt"])} for p in probes]
    prompts = [trainer.prompt_token_ids(tok, row) for row in rows]
    if any(not p for p in prompts):
        raise ValueError("empty encoded probe prompt")
    if not micro["allow_prompt_truncation"] and any(len(p) > micro["prompt_cap"] for p in prompts):
        raise ValueError("probe prompt exceeds cap without truncation authorization")
    counts = [dict(prompt_tokens=len(p), prompt_tokens_dropped=max(0, len(p) - micro["prompt_cap"])) for p in prompts]
    prompts = [p[-micro["prompt_cap"]:] for p in prompts]
    records = [dict(probe_id=p["probe_id"], outcome=None, output=None, called=None,
                    teacher=None, student=None, errors=[], counts=c) for p, c in zip(probes, counts)]
    policy = [None] * len(probes)
    student.model.eval()
    with _context(protocol), torch.no_grad():
        for batch_number, batch in enumerate(_batches(list(map(len, prompts)), ev), 1):
            with progress.phase(f"{phase}/probe_batch/{batch_number}", probes=len(batch)):
                ids, mask = _padded([prompts[i] for i in batch], tok.pad_token_id, device, left=True)
                try:
                    outputs = student.generate(ids, mask, ev)
                    if len(outputs) != len(batch):
                        raise ValueError("generation batch size mismatch")
                    texts = tok.batch_decode(outputs[:, ids.shape[1]:], skip_special_tokens=True)
                    for i, text in zip(batch, texts):
                        records[i]["output"] = text
                except Exception as exc:
                    for i in batch:
                        _failure(records[i], "generate", exc)
                for i in batch:
                    if records[i]["output"] is None:
                        continue
                    try:
                        calls = student.parse(records[i]["output"])
                        records[i]["called"] = bool(calls)
                    except Exception as exc:
                        _failure(records[i], "parse", exc)
                        continue
                    try:
                        if probes[i]["expected"] == "abstain":
                            verdict = {"valid": not calls, "error_type": "unexpected_call" if calls else None}
                        elif not calls:
                            verdict = {"valid": False, "error_type": "missing_call"}
                        else:
                            verdict = student.check(probes[i], calls)
                        if not isinstance(verdict, dict) or type(verdict.get("valid")) is not bool:
                            raise ValueError("checker must return boolean valid")
                        records[i]["outcome"] = int(verdict["valid"])
                        records[i]["verdict_error_type"] = None if verdict["valid"] else str(verdict.get("error_type") or verdict.get("error") or "incorrect_call")
                    except Exception as exc:
                        _failure(records[i], "checker", exc)
                # Use right padding for forward-only passes, fixed for all models.
                ids, mask = _padded([prompts[i] for i in batch], tok.pad_token_id, device)
                positions = torch.tensor([[b, len(prompts[i]) - 1] for b, i in enumerate(batch)], device=device)
                try:
                    logps = torch.cat([F.log_softmax(logits.float(), dim=-1)
                                       for logits in _position_logits(student, ids, mask, positions, ev["head_chunk"])])
                    if not torch.isfinite(logps).all():
                        raise ValueError("nonfinite policy log probabilities")
                    for i, logp in zip(batch, logps):
                        policy[i] = logp
                except Exception as exc:
                    for i in batch:
                        _failure(records[i], "policy", exc)
        encoded, owners = [], []
        for i, p in enumerate(probes):
            for key, field in (("response", "teacher"), ("_rejected", "student")):
                if p.get(key) is None:
                    continue
                try:
                    ids, labels = trainer.encode(tok, {**rows[i], "response": p[key]}, device="cpu")
                    encoded.append((ids[0].tolist(), labels[0].tolist()))
                    owners.append((i, field))
                    records[i]["counts"][field] = {"effective_loss_tokens": int((labels[:, 1:] != -100).sum()),
                        "response_tokens_dropped": max(0, len(tok(p[key], add_special_tokens=False)["input_ids"]) - micro["response_cap"])}
                except Exception as exc:
                    _failure(records[i], "likelihood", exc)
        for batch_number, batch in enumerate(_batches([len(e[0]) for e in encoded], ev), 1):
            with progress.phase(f"{phase}/likelihood_batch/{batch_number}", rows=len(batch)):
                ids, mask = _padded([encoded[i][0] for i in batch], tok.pad_token_id, device)
                positions, targets, counts = [], [], []
                for b, k in enumerate(batch):
                    count = 0
                    for t, label in enumerate(encoded[k][1][1:]):
                        if label != -100:
                            positions.append([b, t]); targets.append(label); count += 1
                    counts.append(count)
                try:
                    positions = torch.tensor(positions, dtype=torch.long, device=device)
                    targets = torch.tensor(targets, dtype=torch.long, device=device)
                    values = []
                    offset = 0
                    for logits in _position_logits(student, ids, mask, positions, ev["head_chunk"]):
                        n = len(logits)
                        values.append(-F.cross_entropy(logits, targets[offset:offset + n], reduction="none").float())
                        offset += n
                    values = torch.cat(values)
                    offset = 0
                    for k, count in zip(batch, counts):
                        score = values[offset:offset + count].sum()
                        offset += count
                        if micro["ell"] == "length_normalized":
                            score = score / count
                        if not count or not torch.isfinite(score):
                            raise ValueError("nonfinite/empty likelihood")
                        i, field = owners[k]
                        records[i][field] = float(score)
                except Exception as exc:
                    for k in batch:
                        i, field = owners[k]
                        records[i][field] = None
                        _failure(records[i], "likelihood", exc)
    return records, policy


def _policy_kl(base, updated, *, device=None):
    import torch
    values = []
    for initial, current in zip(base, updated):
        if initial is None or current is None:
            values.append(None)
        else:
            target = device or (current.device if isinstance(current, torch.Tensor) else "cpu")
            b, u = torch.as_tensor(initial, device=target), torch.as_tensor(current, device=target)
            values.append(float((u.exp() * (u - b)).sum(dtype=torch.float32)))
    return {"definition": KL_DEFINITION, "per_probe": values,
            "mean": float(np.mean(values)) if values and all(v is not None for v in values) else None,
            "max": max((v for v in values if v is not None), default=None)}


def _compare(base, other):
    scores = [abs(a[k] - b[k]) for a, b in zip(base, other) for k in ("teacher", "student")
              if a[k] is not None and b[k] is not None]
    return {"exact_equal": base == other,
            "outcome_changes": sum(a["outcome"] != b["outcome"] for a, b in zip(base, other)),
            "raw_output_changes": sum(a["output"] != b["output"] for a, b in zip(base, other)),
            "max_abs_score_change": max(scores, default=0.0), "scored_sides": len(scores)}


def _outcomes(source, run_id, probes, base, updated, protocol, checker):
    result = []
    for p, b, u in zip(probes, base, updated):
        errors = ["base:" + e for e in b["errors"]] + ["updated:" + e for e in u["errors"]]
        result.append(OutcomeRecord(
            source_id=source["source_id"], run_id=run_id, probe_id=p["probe_id"],
            eval_seed=protocol["eval_seeds"][0], checker_version=checker,
            raw_output=u["output"], raw_output_hash=content_hash(u["output"]) if u["output"] is not None else None,
            base_raw_output_hash=content_hash(b["output"]) if b["output"] is not None else None,
            base_outcome=b["outcome"], updated_outcome=u["outcome"],
            teacher_likelihood=u["teacher"], student_likelihood=u["student"],
            base_teacher_likelihood=b["teacher"], base_student_likelihood=b["student"],
            likelihood_definition=protocol["micro_update"]["ell"] + "; native CE; fp32 reduction; EOS included",
            status="error" if errors else "ok",
            error_type=";".join(errors) if errors else u.get("verdict_error_type"),
            source_factor=source.get("factor", source.get("family"))))
    return result


def _seal(directory, metadata):
    hashes = {str(p.relative_to(directory)): content_hash(p.read_bytes()) for p in sorted(directory.rglob("*"))
              if p.is_file() and p.name != "complete.json"}
    write_json(directory / "complete.json", {**metadata, "status": "complete", "exit_code": 0, "artifact_hashes": hashes})


def _verify(directory):
    saved = _read(directory / "complete.json")
    if saved.get("status") != "complete" or saved.get("exit_code") != 0:
        raise ValueError("incomplete GPU artifact")
    for name, expected in saved["artifact_hashes"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or content_hash(path.read_bytes()) != expected:
            raise ValueError(f"artifact content hash mismatch: {name}")
    return saved


def _save_policy(path, policy):
    buffer = io.BytesIO()
    np.savez(buffer, **{str(i): p.detach().cpu().numpy() if hasattr(p, "detach") else p
                       for i, p in enumerate(policy) if p is not None})
    write_immutable(path, buffer.getvalue())


def _load_policy(path, n):
    with np.load(path, allow_pickle=False) as archive:
        return [archive[str(i)].copy() if str(i) in archive else None for i in range(n)]


def _matrix(out, probes, records, source_ids, protocol, provenance, resume=False):
    write_responses(out / "responses.jsonl", probes, records, provenance, resume=resume)
    matrix = assemble_matrix(records, source_ids=source_ids, probe_ids=[p.probe_id for p in probes],
                             eval_seeds=protocol["eval_seeds"])
    write_json(out / "response_matrix.json", {"source_ids": source_ids, "probe_ids": list(matrix.probe_ids),
               "M": matrix.M, "M_teacher": matrix.teacher, "M_student": matrix.student,
               "diagnostics": matrix.diagnostics}, resume=resume)
    return matrix


def _assess(protocol, base, records, updates, floor):
    thresholds = protocol["pilot"]
    active = [u for u in updates if u["role"] not in {"noop", "no_op"}]
    changed = {r.probe_id for r in records if r.difference not in {None, 0}}
    changed_sources = {r.source_id for r in records if r.difference not in {None, 0}}
    failure_fraction = sum(r.status != "ok" for r in records) / len(records)
    kl = {u["source_id"]: u["kl"] for u in updates}
    stable = (floor["repeat"]["exact_equal"] and floor["zero_delta_equal"] and
              all(u["reload_equal"] and u["noop_equal"] is not False for u in updates) and
              failure_fraction <= thresholds["max_failed_fraction"])
    resolved = bool(active and changed and len(changed) >= thresholds["min_changed_probes"] and
                    len(changed_sources) / len(active) >= thresholds["min_changed_source_fraction"] and
                    all(u["update_norm"] > 0 for u in active))
    if floor["repeat"]["scored_sides"]:
        resolved = resolved and max(u["score_change"] for u in active) > floor["repeat"]["max_abs_score_change"]
    safe = all(u["kl"]["mean"] is not None and -1e-6 <= u["kl"]["mean"] <= thresholds["max_policy_kl"] and
               u["abstention_shift"] is not None and u["abstention_shift"] <= thresholds["max_abstention_shift"] for u in updates)
    passed = bool(stable and resolved and safe and floor["repeat"]["max_abs_score_change"] <= thresholds["floor_tolerance"])
    return {"pass": passed, "passed": passed, "status": "pass" if passed else "measurement-inconclusive",
            "frozen_config_hash": content_hash(protocol), "floor": floor, "kl": kl,
            "n_changed_probes": len(changed), "n_changed_sources": len(changed_sources),
            "measurement_stable": bool(stable), "measurement_resolved": bool(resolved),
            "policy_within_bounds": safe, "failed_fraction": failure_fraction, "thresholds": thresholds}


def _ensure_initial_state(model, path):
    """Publish a complete immutable state, including concurrent shard creation."""
    from src.bfas.behavior import microupdate as micro
    path = Path(path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".initial-") as work:
        temporary = Path(work) / "initial_state.pt"
        micro.save_initial_state(model, temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
    return True


def _validate_initial_state(model, state):
    from src.bfas.behavior import microupdate as micro
    from src.bfas.behavior.deltas import layout_hash, tensor_state_hash
    if state["format"] != "behavior-init-v1" or layout_hash(model) != state["layout_hash"]:
        raise ValueError("initial-state layout mismatch")
    if micro._frozen_hash(model) != state["frozen_hash"]:
        raise ValueError("initial-state frozen checkpoint hash mismatch")
    trainables_hash = tensor_state_hash(state["trainables"])
    if (tensor_state_hash(micro._trainables(model)) != trainables_hash or
            state.get("trainables_hash", trainables_hash) != trainables_hash):
        raise ValueError("initial-state trainables hash mismatch")
    return micro._settings_diff(state["model_settings"], micro._model_settings(model))


def _verify_delta_replay(model, base, delta):
    """Compare a disk-reloaded delta's replay to the live update on its device."""
    import torch
    from src.bfas.behavior.deltas import layout_hash, BLOCK_SIZE
    if layout_hash(model) != delta.manifest["layout_hash"]:
        raise ValueError("saved delta layout mismatch")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            high = delta.tensors[name].to(parameter.device).reshape(-1)
            low = delta.roundoff.get(name)
            low = low.to(parameter.device).reshape(-1) if low is not None else None
            initial, updated = base[name].reshape(-1), parameter.detach().reshape(-1)
            for start in range(0, parameter.numel(), 16 * BLOCK_SIZE):
                sl = slice(start, start + 16 * BLOCK_SIZE)
                replay = initial[sl].double() + high[sl].double()
                if low is not None:
                    replay += low[sl].double()
                if not torch.equal(replay.to(parameter.dtype).view(torch.uint8), updated[sl].contiguous().view(torch.uint8)):
                    raise ValueError("saved delta parameter replay mismatch")
    return True


def _record_profile(out, progress):
    """Atomically finish timing metadata before completion hashes are sealed."""
    path = Path(out) / "manifest.json"
    if progress.profile and path.exists():
        manifest = _read(path)
        manifest["phase_elapsed_seconds"] = {**manifest.get("phase_elapsed_seconds", {}), **progress.elapsed}
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".profile-", delete=False) as stream:
            stream.write(canonical_json(manifest) + "\n")
        Path(stream.name).replace(path)


def create_initial_state(protocol, path, *, dry_run=False):
    """Create only the seeded initialization and its manifest; no checker needed."""
    from src.bfas.behavior import microupdate as micro
    import torch
    path = Path(path)
    result = {"stage": "init-state", "status": "dry-run plan" if dry_run else "complete",
              "init_state_path": str(path), "init_seed": protocol["micro_update"]["init_seed"],
              "number_of_models": 1, "max_rollouts": 0, "new_teacher_tokens": 0}
    if dry_run:
        return result
    if path.exists():
        raise FileExistsError(f"initial state already exists: {path}")
    with _context(protocol):
        micro._seed(result["init_seed"], torch.device(protocol["micro_update"]["device"]).type == "cuda")
        model = load_model(protocol)
        if not _ensure_initial_state(model, path):
            raise FileExistsError(f"initial state already exists: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    result.update(init_state_created=True, init_state_settings_diff=[],
                  layout_hash=state["layout_hash"], frozen_hash=state["frozen_hash"],
                  trainables_hash=state["trainables_hash"], initial_file_hash=content_hash(path.read_bytes()))
    write_json(path.with_suffix(path.suffix + ".manifest.json"), result)
    return result


def run(stage, config, protocol, data, out, *, resume=False, dry_run=False, shard=None, student_factory=None,
        strict_init_state=False, roles=None, profile=False, cpu_threads=None, max_sources=None, smoke=False):
    """Own the checker worker for this run, including failed/resumed runs."""
    if smoke and stage != "pilot":
        raise ValueError("--smoke requires the pilot stage")
    protocol = json.loads(json.dumps(protocol))
    if cpu_threads is not None:
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("--cpu-threads must be a positive integer")
        protocol["micro_update"]["cpu_threads"] = cpu_threads
    if smoke:
        profile, max_sources = True, 1
        protocol["micro_update"]["steps"] = 1
        protocol["evaluation"]["max_new_tokens"] = min(32, protocol["evaluation"]["max_new_tokens"])
        protocol["smoke"] = True
        data = {**data, "probes": data["probes"][:1], "probe_records": data["probe_records"][:1]}
    from src.bfas.behavior import microupdate as micro
    progress = micro.Progress(protocol["micro_update"]["device"], profile)
    if not dry_run:
        micro.install_stack_dump()
        threads = micro.cap_cpu_threads(protocol["micro_update"].get("cpu_threads", 8))
        progress.emit("runtime", "ready", cpu_threads=threads, profile=profile, smoke=smoke)
    with ExitStack() as stack:
        def managed_student(settings):
            student = (student_factory or load_student)(settings)
            stack.callback(student.close)
            return student
        try:
            return _run(stage, config, protocol, data, out, resume=resume, dry_run=dry_run,
                        shard=shard, student_factory=managed_student,
                        strict_init_state=strict_init_state or config.get("strict_init_state", False),
                        roles=roles if roles is not None else config.get("roles"),
                        max_sources=max_sources, progress=progress)
        except BaseException:
            # Keep timings of failed phases too; never modify sealed evidence.
            if getattr(progress, "owns_manifest", False) and not (progress.output_dir / "complete.json").exists():
                _record_profile(progress.output_dir, progress)
            raise


def _run(stage, config, protocol, data, out, *, resume=False, dry_run=False, shard=None, student_factory=None,
         strict_init_state=False, roles=None, max_sources=None, progress=None):
    """Execute prepared inputs; no real model is loaded in dry-run or merge."""
    all_sources, sources, partition = select_sources(data, stage, shard, roles, max_sources)
    out = Path(out)
    if shard is not None:
        out = out / f"shard-{partition[0]:05d}-of-{partition[1]:05d}"
    plan = resource_plan(stage, config, protocol, data, shard, roles, max_sources)
    plan.update(optimizer_steps=protocol["micro_update"]["steps"], smoke=protocol.get("smoke", False))
    plan["output_dir"] = str(out)
    pilot_evidence = None
    if stage == "collect":
        from tools.behavior_atom_experiment import require_pilot_pass
        pilot_evidence = require_pilot_pass(config, protocol, {})
    if dry_run:
        print(json.dumps(plan, sort_keys=True))
        return plan
    for dep in data["dependencies"]:
        if not isinstance(dep, dict) or not dep.get("job_id") or dep.get("status") != "complete" or dep.get("exit_code") != 0:
            raise ValueError("unresolved job dependency (job_id, complete, exit_code=0 required)")
    code_paths = [Path(__file__), ROOT / "tools/behavior_atom/checker_bridge.py",
                  ROOT / "src/appworld_train.py"] + list((ROOT / "src/bfas/behavior").glob("*.py"))
    identity = {"version": VERSION, "stage": stage, "frozen_config_hash": content_hash(protocol),
                "input_hashes": data["input_hashes"], "source_ids": [u["source_id"] for u in all_sources],
                "probe_ids": [p["probe_id"] for p in data["probes"]],
                "code_hashes": {str(p.relative_to(ROOT)): content_hash(p.read_bytes()) for p in sorted(code_paths)}}
    commit = repo_commit(ROOT)
    manifest = {**commit, "identity": identity, "shard": list(partition), "selected_sources": [u["source_id"] for u in sources],
                "protocol": protocol, "resource_plan": {k: v for k, v in plan.items() if k != "output_dir"}}
    if protocol.get("fisher"):
        # Whitening consumes Fisher only at fit/decompose time. Keep its
        # availability out of the frozen pilot/collect execution identity.
        fisher_path = _resolve(protocol["fisher"], config.get("_base", ROOT))
        try:
            manifest["fisher"] = content_hash(fisher_path.read_bytes())
        except FileNotFoundError:
            manifest["fisher"] = "pending"
    saved_manifest = None
    if (out / "manifest.json").exists():
        saved_manifest = _read(out / "manifest.json")
        # Launch provenance may change when outputs dirty the checkout or Git
        # becomes unavailable; code_hashes already validate the executed files.
        commit = {key: saved_manifest.get(key, value) for key, value in commit.items()}
        manifest.update(commit)
        # Preserve the status observed at launch if Fisher arrived meanwhile.
        if "fisher" in manifest and "fisher" in saved_manifest:
            manifest["fisher"] = saved_manifest["fisher"]
        observed = {"checker_mode", "init_state_created", "init_state_settings_diff", "phase_elapsed_seconds"}
        if not resume or {k: v for k, v in saved_manifest.items() if k not in observed} != manifest:
            raise ValueError("resume identity mismatch or --resume missing")
        if strict_init_state:
            from src.bfas.behavior.microupdate import _check_settings_diff
            _check_settings_diff(saved_manifest.get("init_state_settings_diff", []), True)
        if (out / "complete.json").exists():
            _verify(out)
            return _read(out / "result.json")
    if not (out / "launch.json").exists():
        write_json(out / "launch.json", {**commit, "command": shlex.join([sys.executable, *sys.argv]),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    import torch
    from src.bfas.behavior import microupdate as micro
    from src.bfas.behavior.deltas import apply_delta, load_delta, save_delta, tensor_state_hash
    with progress.phase("model_load"), _context(protocol):
        micro._seed(protocol["micro_update"]["init_seed"],
                    torch.device(protocol["micro_update"]["device"]).type == "cuda")
        student = (student_factory or load_student)(protocol)
    manifest["checker_mode"] = student.checker_mode
    model = student.model
    if hasattr(model, "config"):
        model.config.use_cache = False
    with progress.phase("initial_state_load_verify"):
        init = out / "initial_state.pt"
        initial_record = out / "initial_state.json"
        settings_diff = []
        created = False
        if not init.exists():
            external = protocol["micro_update"].get("init_state_path")
            if external:
                created = _ensure_initial_state(model, external)
                state = torch.load(external, map_location="cpu", weights_only=True)
                settings_diff = _validate_initial_state(model, state)
                micro._check_settings_diff(settings_diff, strict_init_state)
                micro._restore(model, state)
            else:
                created = True
            # Normalize legacy settings in the local state used by micro-updates.
            _ensure_initial_state(model, init)
        state = torch.load(init, map_location="cpu", weights_only=True)
        local_diff = _validate_initial_state(model, state)
        micro._check_settings_diff(local_diff, strict_init_state)
    manifest["init_state_settings_diff"] = sorted(set(settings_diff + local_diff +
        (saved_manifest.get("init_state_settings_diff", []) if saved_manifest else [])))
    manifest["init_state_created"] = saved_manifest["init_state_created"] if saved_manifest else created
    if saved_manifest and "phase_elapsed_seconds" in saved_manifest:
        manifest["phase_elapsed_seconds"] = saved_manifest["phase_elapsed_seconds"]
    write_json(out / "manifest.json", manifest, resume=resume)
    progress.owns_manifest = True
    progress.output_dir = out
    progress.on_phase = lambda: _record_profile(out, progress)
    with progress.phase("resident_state_cache"):
        resident = micro.ResidentState(model, state, init, commit)
    base_hash = resident.base_hash
    buffers_hash = tensor_state_hash(state["buffers"])
    def restore(phase="state_restore"):
        with progress.phase(phase):
            resident.restore()
            model.eval()
    restore()
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for package in ("torch", "transformers", "peft", "numpy", "tokenizers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    runtime = {"initial_trainables_hash": base_hash, "initial_buffers_hash": buffers_hash,
               "library_versions": versions,
               "frozen_checkpoint_hash": state["frozen_hash"], "layout_hash": state["layout_hash"],
               "model_settings_hash": content_hash(micro._model_settings(model)),
               "checker_version": student.checker_version,
               "tokenizer_hash": content_hash({"vocab": student.tokenizer.get_vocab(),
                    "template": getattr(student.tokenizer, "chat_template", None),
                    "backend": (student.tokenizer.backend_tokenizer.to_str()
                                if hasattr(student.tokenizer, "backend_tokenizer") else None),
                    "special_ids": [student.tokenizer.pad_token_id, student.tokenizer.eos_token_id]}),
               "initial_file_hash": content_hash(init.read_bytes())}
    shared_runtime = {k: v for k, v in runtime.items() if k != "initial_file_hash"}
    if pilot_evidence is not None and (pilot_evidence.get("runtime") != shared_runtime or
                                       pilot_evidence.get("code_hashes") != identity["code_hashes"]):
        raise ValueError("pilot runtime identity differs (initial model/tokenizer/checker/code)")
    write_json(initial_record, runtime, resume=resume)
    base_dir = out / "base"
    if base_dir.exists():
        _verify(base_dir)
        base = _read(base_dir / "evaluations.json")
        base_policy = [torch.as_tensor(p, device=protocol["micro_update"]["device"]) if p is not None else None
                       for p in _load_policy(base_dir / "policy.npz", len(base))]
        floor = _read(base_dir / "floor.json")
    else:
        with tempfile.TemporaryDirectory(dir=out, prefix=".base-") as work:
            path = Path(work) / "base"
            path.mkdir()
            with progress.phase("base/evaluation"):
                base, base_policy = evaluate(student, data["probes"], protocol, progress=progress, phase="base")
            floor = {}
            if stage == "pilot":
                restore()
                with progress.phase("repeat/check"):
                    repeated, repeated_policy = evaluate(student, data["probes"], protocol, progress=progress, phase="repeat")
                restore()
                with progress.phase("zero/delta_capture_save") as timing:
                    save_delta(model, state["trainables"], path / "delta_zero.npz")
                    timing["bytes"] = (path / "delta_zero.npz").stat().st_size
                with progress.phase("zero/delta_apply"):
                    apply_delta(model, load_delta(path / "delta_zero.npz"))
                with progress.phase("zero/check"):
                    zero, zero_policy = evaluate(student, data["probes"], protocol, progress=progress, phase="zero")
                with progress.phase("zero/kl"):
                    zero_kl = _policy_kl(base_policy, zero_policy)
                floor = {"repeat": _compare(base, repeated), "zero": _compare(base, zero),
                         "zero_delta_equal": base == zero and all(v in {None, 0} for v in zero_kl["per_probe"]),
                         "zero_policy_kl": zero_kl, "parameter_floor": 0.0}
                with progress.phase("repeat/kl"):
                    floor["repeat_policy_kl"] = _policy_kl(base_policy, repeated_policy)
                floor["repeat"]["exact_equal"] = floor["repeat"]["exact_equal"] and all(
                    v in {None, 0} for v in floor["repeat_policy_kl"]["per_probe"])
                write_json(path / "repeat_evaluations.json", repeated)
                write_json(path / "zero_evaluations.json", zero)
            with progress.phase("base/artifact_save"):
                write_json(path / "evaluations.json", base)
                write_json(path / "floor.json", floor)
                _save_policy(path / "policy.npz", base_policy)
                _seal(path, {})
                path.rename(base_dir)
    records, updates = [], []
    source_root = out / "sources"
    source_root.mkdir(exist_ok=True)
    for source in sources:
        sid = source["source_id"]
        restore(f"source/{sid}/initial_restore")
        directory = source_root / _slug(sid)
        if not directory.exists():
            with tempfile.TemporaryDirectory(dir=out, prefix=".source-") as work:
                path = Path(work) / "source"
                path.mkdir()
                run_id = "source-" + content_hash({"identity": identity, "source": sid})[:24]
                cfg = {**protocol["micro_update"], "init_state_path": str(init), "source_id": sid,
                       "strict_init_state": strict_init_state,
                       "run_id": run_id, "tokenizer": student.tokenizer, "batch_size": len(source["resolved_rows"]),
                       "accumulation": 1, "eval_config": protocol["evaluation"],
                       "data_manifest_hash": content_hash(data["input_hashes"]), "output_dir": None}
                cfg.update(resident_state=resident, progress=progress, profile=progress.profile)
                if source["role"] in {"noop", "no_op"}:
                    cfg["steps"] = 0
                try:
                    with progress.phase(f"source/{sid}/micro_update"), _context(protocol):
                        update = micro.run_micro_update(lambda: model, source["resolved_rows"], cfg)
                    # A GPU run records exact replay plus the saved delta digest.
                    # Avoid copying/hashing all updated trainables just for a
                    # redundant diagnostic. CPU runs can hash in-place cheaply.
                    updated_hash = (tensor_state_hash(micro._trainables(model))
                                    if torch.device(cfg["device"]).type == "cpu" else None)
                    delta_path = path / f"delta_{_slug(sid)}.npz"
                    with progress.phase(f"source/{sid}/delta_save") as timing:
                        delta = save_delta(None, None, delta_path, delta=update.delta)
                        timing["bytes"] = delta_path.stat().st_size
                    with progress.phase(f"source/{sid}/probe_evaluation"):
                        after, policy = evaluate(student, data["probes"], protocol, progress=progress, phase=f"source/{sid}")
                    with progress.phase(f"source/{sid}/policy_kl"):
                        kl = _policy_kl(base_policy, policy, device=cfg["device"])
                    with progress.phase(f"source/{sid}/delta_replay_check"):
                        equal = _verify_delta_replay(model, resident.state["trainables"], load_delta(delta_path))
                    abstention = ([int(not u["called"]) - int(not b["called"]) for b, u in zip(base, after)]
                                  if all(b["called"] is not None and u["called"] is not None for b, u in zip(base, after)) else None)
                    summary = {"source_id": sid, "role": source["role"], "run_id": run_id,
                               "unit_type": source["unit_type"], "initial_trainables_hash": base_hash,
                               "updated_trainables_hash": updated_hash, "reload_equal": equal,
                               "delta_payload_hash": delta.manifest["payload_hash"],
                               "noop_equal": base == after if source["role"] in {"noop", "no_op"} else None,
                               "update_norm": delta.manifest["update_norm"], "kl": kl,
                               "abstention_shift": abs(float(np.mean(abstention))) if abstention is not None else None,
                               "score_change": _compare(base, after)["max_abs_score_change"],
                               "steps": cfg["steps"], "effective_loss_tokens": update.manifest["effective_loss_tokens"]}
                    summary["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
                    paired = _outcomes(source, run_id, data["probes"], base, after, protocol, student.checker_version)
                    with progress.phase(f"source/{sid}/artifact_save"):
                        write_json(path / "update.manifest.json", update.manifest)
                        write_json(path / "summary.json", summary)
                        write_json(path / "evaluations.json", after)
                        write_responses(path / "responses.jsonl", data["probe_records"], paired,
                                        {"identity": identity, "source_id": sid, "eval_seeds": protocol["eval_seeds"]})
                        _seal(path, {"source_id": sid})
                        path.rename(directory)
                finally:
                    restore(f"source/{sid}/final_restore")
        _verify(directory)
        _, paired, _ = read_responses(directory / "responses.jsonl")
        records.extend(paired)
        updates.append(_read(directory / "summary.json"))
    restore()
    # State content (not torch.save container bytes or account-specific paths)
    # is the cross-shard identity of the single common base.
    shared = {"identity": identity, "runtime": shared_runtime,
              "base_hash": content_hash(base), "base_policy_hash": content_hash((base_dir / "policy.npz").read_bytes()),
              "eval_seeds": protocol["eval_seeds"], "frozen_config_hash": content_hash(protocol)}
    with progress.phase("response_matrix"):
        matrix = _matrix(out, data["probe_records"], records, [s["source_id"] for s in sources], protocol, shared, resume)
    if stage == "pilot":
        assessment = _assess(protocol, base, records, updates, floor)
        if protocol.get("smoke"):
            assessment.update(passed=False, **{"pass": False}, status="smoke-complete")
        assessment.update(runtime=shared_runtime, code_hashes=identity["code_hashes"])
        assessment["artifact_hashes"] = {str(p.relative_to(out)): content_hash(p.read_bytes()) for p in sorted(out.rglob("*"))
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(out).parts)
            and p.name not in {"pilot_pass.json", "result.json", "complete.json"}}
        write_json(out / "pilot_pass.json", assessment, resume=resume)
    result = {"stage": stage, "status": "complete", "exit_code": 0, "updates": updates,
              "response_diagnostics": matrix.diagnostics, "new_teacher_tokens": 0,
              "shared": shared, "shard": list(partition)}
    write_json(out / "result.json", result, resume=resume)
    # Ignore abandoned transactional staging directories after an external kill.
    artifacts = {str(p.relative_to(out)): content_hash(p.read_bytes()) for p in sorted(out.rglob("*"))
                 if p.is_file() and not any(part.startswith(".") for part in p.relative_to(out).parts) and p != out / "complete.json"}
    write_json(out / "complete.json", {"status": "complete", "exit_code": 0, "artifact_hashes": artifacts}, resume=resume)
    return result


def merge(shards, output_dir, *, resume=False, dry_run=False):
    """Verify complete source coverage and common content before assembling M."""
    shards = [Path(p) for p in shards]
    records, probes, shared, partitions, sources = [], None, None, [], {}
    for directory in shards:
        _verify(directory)
        result = _read(directory / "result.json")
        if result["stage"] != "collect":
            raise ValueError("only collection shards can be merged")
        if shared is not None and result["shared"] != shared:
            raise ValueError("shard shared content hash mismatch (config/data/base/checker/tokenizer/code)")
        shared = result["shared"]
        partitions.append(result["shard"])
        pp, rr, bundle = read_responses(directory / "responses.jsonl")
        if bundle["provenance"] != shared or (probes is not None and probes != pp):
            raise ValueError("shard response provenance/probes mismatch")
        probes = pp
        for update in result["updates"]:
            sid = update["source_id"]
            if sid in sources:
                raise ValueError("duplicate source in shards")
            sources[sid] = update
        records.extend(rr)
    if not shared or {p[1] for p in partitions} != {len(shards)} or {p[0] for p in partitions} != set(range(len(shards))):
        raise ValueError("missing or duplicate shard partition")
    ids = shared["identity"]["source_ids"]
    if set(sources) != set(ids):
        raise ValueError("incomplete shard source coverage")
    order = {sid: i for i, sid in enumerate(ids)}
    probe_order = {p.probe_id: i for i, p in enumerate(probes)}
    records.sort(key=lambda r: (order[r.source_id], probe_order[r.probe_id], r.eval_seed))
    if len(records) != len(ids) * len(probes) * len(shared["eval_seeds"]):
        raise ValueError("incomplete shard response coverage")
    if dry_run:
        result = {"stage": "merge", "sources": len(ids), "probes": len(probes), "number_of_models": 0,
                  "max_rollouts": 0, "new_teacher_tokens": 0, "gpu_hours": 0}
        print(json.dumps(result, sort_keys=True))
        return result
    out = Path(output_dir)
    if out.resolve() in {p.resolve() for p in shards}:
        raise ValueError("merge output must differ from input shard directories")
    if (out / "complete.json").exists():
        if not resume:
            raise FileExistsError("merge exists; use --resume")
        _verify(out)
    matrix = _matrix(out, probes, records, ids, {"eval_seeds": shared["eval_seeds"]}, shared, resume)
    result = {"stage": "merge", "status": "complete", "shared": shared,
              "updates": [sources[sid] for sid in ids], "response_diagnostics": matrix.diagnostics,
              "shard_hashes": [content_hash((p / "complete.json").read_bytes()) for p in shards], "new_teacher_tokens": 0}
    write_json(out / "result.json", result, resume=resume)
    if not (out / "complete.json").exists():
        _seal(out, {})
    return result


def _frozen_protocol(config):
    from tools.behavior_atom_experiment import frozen_protocol
    protocol = frozen_protocol({**config, "protocol": frozen_settings(config.get("protocol", {}))})
    # Auto-creation must not change the protocol between pilot, collect and
    # resume. Tensor/layout hashes in runtime bind the actual initialization.
    protocol["input_artifact_hashes"].pop("init_state_path", None)
    return protocol


def cli_run(stage, config, data, input_path, output_dir=None, *, resume=False, dry_run=False, shard=None,
            strict_init_state=False, roles=None, profile=False, cpu_threads=None, max_sources=None, smoke=False):
    prepared = load_inputs(config, data, input_path)
    protocol = _frozen_protocol(config)
    run_id = config.get("run_id", f"{stage}-{content_hash(protocol)[:12]}")
    if smoke:
        run_id += "-smoke"
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    out = Path(output_dir) if output_dir else ROOT / "results/behavior_atom_v1" / run_id
    result = run(stage, config, protocol, prepared, out, resume=resume, dry_run=dry_run, shard=shard,
                 strict_init_state=strict_init_state, roles=roles, profile=profile, cpu_threads=cpu_threads,
                 max_sources=max_sources, smoke=smoke)
    if not dry_run:
        print(json.dumps({"status": result["status"], "stage": stage, "output_dir": str(out), "new_teacher_tokens": 0}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    child = sub.add_parser("merge")
    child.add_argument("--shards", nargs="+", type=Path, required=True)
    child.add_argument("--output-dir", type=Path, required=True)
    child.add_argument("--resume", action="store_true")
    child.add_argument("--dry-run", action="store_true")
    for stage in ("pilot", "collect", "init-state"):
        child = sub.add_parser(stage)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--init-state-path", type=Path, help="override protocol.micro_update.init_state_path")
        child.add_argument("--dry-run", action="store_true")
        if stage == "init-state":
            child.add_argument("--init-seed", type=int, help="override protocol.micro_update.init_seed")
            continue
        manifests = child.add_mutually_exclusive_group()
        manifests.add_argument("--manifest", type=Path)
        manifests.add_argument("--sources", type=Path)
        child.add_argument("--probes", type=Path)
        child.add_argument("--pool", type=Path)
        child.add_argument("--output-dir", type=Path)
        child.add_argument("--resume", action="store_true")
        child.add_argument("--shard", help="collect partition i/n")
        child.add_argument("--strict-init-state", action="store_true", help="fail on initial-state model settings differences")
        child.add_argument("--roles", nargs="+", help="selected roles (space/comma/+ separated); add anchor explicitly")
        child.add_argument("--profile", action="store_true", help="save per-phase seconds in manifest.json")
        child.add_argument("--cpu-threads", type=int, help="cap torch CPU threads (default: 8)")
        if stage == "pilot":
            child.add_argument("--max-sources", type=int, help="run at most N selected sources")
            child.add_argument("--smoke", action="store_true", help="timed diagnostic: one source, one step, one probe, at most 32 new tokens")
    args = parser.parse_args(argv)
    try:
        if args.stage == "merge":
            merge(args.shards, args.output_dir, resume=args.resume, dry_run=args.dry_run)
            return 0
        config_path = args.config.resolve()
        config = {**_read(config_path), "_base": str(config_path.parent)}
        micro = config.setdefault("protocol", {}).setdefault("micro_update", {})
        if args.init_state_path is not None:
            micro["init_state_path"] = str(args.init_state_path.resolve())
        if args.stage == "init-state":
            if args.init_seed is not None:
                micro["init_seed"] = args.init_seed
            protocol = _frozen_protocol(config)
            if not protocol["micro_update"].get("init_state_path"):
                raise ValueError("init-state requires --init-state-path or protocol.micro_update.init_state_path")
            result = create_initial_state(protocol, protocol["micro_update"]["init_state_path"], dry_run=args.dry_run)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.resume and args.dry_run:
            raise ValueError("--resume is inapplicable to dry-run plans")
        manifest_arg = args.sources or args.manifest
        input_path = (manifest_arg.resolve() if manifest_arg else
                      _resolve(config["data"], config_path.parent) if config.get("data") else None)
        if input_path is None:
            raise ValueError(f"{args.stage} requires --manifest or config.data")
        data = _read(input_path)
        for key in ("probes", "pool"):
            value = getattr(args, key)
            if value is not None:
                data[key] = str(value.resolve())
            elif key in config:
                data.setdefault(key, str(_resolve(config[key], config_path.parent)))
        return cli_run(args.stage, config, data, input_path, args.output_dir, resume=args.resume,
                       dry_run=args.dry_run, shard=args.shard, roles=args.roles,
                       strict_init_state=args.strict_init_state, profile=args.profile,
                       cpu_threads=args.cpu_threads, max_sources=getattr(args, "max_sources", None),
                       smoke=getattr(args, "smoke", False))
    except (ValueError, OSError, KeyError) as exc:
        print(f"{args.stage}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
