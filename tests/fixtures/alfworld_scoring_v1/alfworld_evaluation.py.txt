"""C26-D greedy full-episode ALFWorld campaigns, separate from active RTD runs.

Layout: TAG/artifacts/{binding.json,tasks/*.json,aggregate.json}, campaign.json,
and audit.jsonl. The completion receipt hashes the entire artifacts tree; it is
outside that tree to avoid a recursive hash. Task envelopes permit interrupted
campaigns to resume only complete episodes, never a sampled prefix. Historical
scores and training directories are read-only. Inference may be local or served.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import http.client
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import threading

from ...adapters import alfworld as adapter
from ..evaluation_lock import evaluation_lock
from ..hardware import checked_hardware
from ..persistence import manifest_digest, ComputeJournal, atomic_json, digest, file_hash, tree_hash
from .alfworld_identity import (
    campaign_identity, checked_expectations, guard_manifest,
)
from .alfworld_support import BoundedEnvBridge, EnvironmentUnavailable, FrozenRenderer


@dataclass(frozen=True)
class Generation:
    text: str
    token_count: int
    truncated: bool


def _checked_state(state):
    if (not isinstance(state, dict) or state.get("op") != "state" or
            type(state.get("done")) is not bool or type(state.get("won")) is not bool or
            not isinstance(state.get("observation"), str) or
            not isinstance(state.get("admissible"), list) or
            any(not isinstance(c, str) or not c.strip() for c in state["admissible"])):
        raise ValueError("environment state requires strict bool done/won and text commands")
    if state["won"] and not state["done"]:
        raise ValueError("environment success without terminal state")
    return state


def official_episode(task_id, backend, *, env_factory, identity):
    """The official loop; the two-task integration test uses this same function.

    backend.renderer follows FrozenRenderer's (request, full history) boundary;
    backend.generate receives explicit greedy and token limits every turn.
    Environment errors propagate and cannot become a zero-success episode.
    """
    if (identity.get("split") != "valid_seen" or identity.get("evaluation_temperature") != 0 or
            identity.get("max_steps") != 40 or identity.get("max_action_tokens") != 256):
        raise ValueError("official episode requires valid_seen/greedy/40 steps/256 tokens")
    env = env_factory(task_id)
    turns, history = [], []
    try:
        state = _checked_state(env._read())
        if state["done"]:
            raise ValueError("unexpected terminal reset")
        goal = adapter._goal_line(state["observation"])
        if not goal:
            raise ValueError("missing goal at task reset")
        request = dict(task_id=task_id, goal=goal)
        history.append(dict(role="user", index=0, content=state["observation"],
                            admissible=state["admissible"], done=False))
        for index in range(40):
            prompt = backend.renderer(request, history)
            reply = backend.generate(prompt, temperature=0.0, max_new_tokens=256)
            if (not isinstance(reply, Generation) or not isinstance(reply.text, str) or
                    type(reply.token_count) is not int or not 1 <= reply.token_count <= 256 or
                    type(reply.truncated) is not bool or reply.truncated and reply.token_count != 256):
                raise ValueError("invalid greedy generation/token-cap record")
            command = adapter.ALFWorldAdapter._teacher_command(reply.text, state["admissible"])
            state = _checked_state(env.step(command))
            turns.append(dict(prompt=prompt, generated_text=reply.text, command=command,
                token_count=reply.token_count, truncated=reply.truncated,
                observation=state["observation"], admissible=state["admissible"],
                done=state["done"], success=state["won"]))
            history.extend([dict(role="assistant", index=index + 1, content=command),
                dict(role="tool", index=index + 1, content=state["observation"],
                     admissible=state["admissible"], done=state["done"])])
            if state["done"]:
                break
        horizon = not state["done"] and len(turns) == 40
        return dict(task_id=task_id, split="valid_seen", identity_hash=digest(identity),
            max_steps=40, status="completed", steps=len(turns), success=state["won"],
            done=state["done"], horizon_reached=horizon,
            truncated=horizon or any(t["truncated"] for t in turns),
            generated_text=[t["generated_text"] for t in turns], turns=turns)
    finally:
        env.close()


class EvaluationEnvBridge(BoundedEnvBridge):
    """C26-B's bounded IO/cleanup, spawning only the existing adapter worker.

    The adapter hardcodes DATA. Refuse a different root rather than setting a
    misleading env variable or monkeypatching global modules. A relocated repo
    may use an envs/alfworld symlink to its explicit read-only installation.
    """
    def __init__(self, task_id, *, data_root, environment_root, timeout=None, reset_timeout=None,
                 episode_timeout=900.):
        if Path(data_root).resolve() != adapter.DATA.resolve():
            raise ValueError("explicit data root must match the existing adapter DATA")
        if (Path(environment_root) / ".venv/bin/python").absolute() != adapter.ENV_PYTHON.absolute():
            # Symlinked environment directories are supported without resolving
            # the Python executable itself out of its venv.
            if Path(environment_root).resolve() != adapter.ENV_PYTHON.parents[2].resolve():
                raise ValueError("environment root must match the existing adapter installation")
        if not (adapter.DATA / "valid_seen" / task_id / "game.tw-pddl").is_file():
            raise ValueError("missing valid_seen task")
        self._configure_rpc(task_id, timeout=timeout, reset_timeout=reset_timeout,
                            episode_timeout=episode_timeout)
        self._tmp = tempfile.TemporaryDirectory(prefix="rtd-alfworld-c26d-")
        self._stderr = open(Path(self._tmp.name) / "worker.stderr", "w+")
        env = dict(os.environ, PYTHONPATH=str(adapter.ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
            ALFWORLD_DATA=str(Path(data_root).resolve().parent), ALFRED_DATA=str(Path(data_root).resolve()),
            ALFWORLD_DATA_ROOT=str(Path(data_root).resolve().parent), CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false", PYTHONHASHSEED="0", TMPDIR=self._tmp.name,
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", BFAS_ALFWORLD_MAX_STEPS="40")
        try:
            self.process = subprocess.Popen(
                [str(adapter.ENV_PYTHON), "-u", "-m", "bfas.adapters.alfworld", "--worker",
                 "--split", "valid_seen", "--task-id", task_id], cwd=self._tmp.name, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                text=True, encoding="utf-8", bufsize=1, start_new_session=True)
            self._reader = threading.Thread(target=self._read_lines, daemon=True)
            self._reader.start()
            if self._read().get("op") != "ready":
                raise self._rpc_failure(RuntimeError("ALFWorld worker did not become ready"))
        except Exception as exc:
            error = exc if isinstance(exc, adapter.ALFWorldRPCError) else self._rpc_failure(exc)
            self.close()
            if error is exc:
                raise
            raise error from exc
        except BaseException:
            self.close()
            raise


class HFBackend:
    """Local HF greedy inference, with optional RTD round-N/lora overlay.

    PEFT deserializes explicitly on CPU (the same placement precaution used by
    RTD _flatten_adapter), then moves to the single evaluation device. A merged
    HF checkpoint loads directly. No export is written into a training run.
    """
    def __init__(self, manifest, *, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig
        paths = manifest["paths"]
        self.renderer = FrozenRenderer(paths["tokenizer_path"])
        self.tokenizer = self.renderer.adapter._tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(paths["model_path"], local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
        if paths["checkpoint"]:
            from peft import PeftModel
            path = Path(paths["checkpoint"])
            lora = path / "lora" if (path / "lora").is_dir() else path
            self.model = PeftModel.from_pretrained(self.model, lora, local_files_only=True,
                device_map="cpu", torch_device="cpu", is_trainable=False)
        self.model.to(device).eval()
        self.device = device
        self.max_context = manifest["config"]["max_context_tokens"]
        # Do not inherit sampling/repetition/min-length defaults from a model's
        # generation_config.json. EOS comes from the bound local tokenizer.
        if type(self.tokenizer.eos_token_id) is not int:
            raise ValueError("a single tokenizer EOS is required")
        self.generation_config = GenerationConfig(do_sample=False, num_beams=1,
            max_new_tokens=256, eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                          else self.tokenizer.eos_token_id),
            use_cache=True)

    def generate(self, prompt, *, temperature, max_new_tokens):
        import torch
        if temperature != 0.0 or max_new_tokens != 256:
            raise ValueError("official generation requires greedy/256 tokens")
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        n = inputs["input_ids"].shape[-1]
        if n + 256 > self.max_context:
            raise ValueError("context overflow; official prompts/actions cannot be silently shortened")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation_config)
        ids = output[0, n:].tolist()
        eos = self.tokenizer.eos_token_id
        truncated = not ids or ids[-1] != eos
        text = self.tokenizer.decode(ids if truncated else ids[:-1], skip_special_tokens=False)
        return Generation(text, len(ids), truncated)

    def close(self):
        self.model = None


class VLLMBackend:
    """Official greedy completions from a launch-bound loopback vLLM server.

    Send locally tokenized prompts and decode returned IDs locally, just as HF
    does. The launch-specific model name prevents a reused port from silently
    serving another job. Clients never load weights or inspect CUDA.
    """
    def __init__(self, manifest, *, server):
        from .alfworld_server import checked_server_identity
        self.server = checked_server_identity(server, manifest=manifest)
        self.renderer = FrozenRenderer(manifest["paths"]["tokenizer_path"])
        self.tokenizer = self.renderer.adapter._tokenizer
        self.max_context = manifest["config"]["max_context_tokens"]
        if type(self.tokenizer.eos_token_id) is not int:
            raise ValueError("a single tokenizer EOS is required")

    def generate(self, prompt, *, temperature, max_new_tokens):
        if temperature != 0.0 or max_new_tokens != 256:
            raise ValueError("official generation requires greedy/256 tokens")
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + 256 > self.max_context:
            raise ValueError("context overflow; official prompts/actions cannot be silently shortened")
        eos = self.tokenizer.eos_token_id
        payload = dict(model=self.server["served_model_name"], prompt=prompt_ids,
            temperature=0.0, max_tokens=256, n=1, stream=False, echo=False,
            use_beam_search=False, top_p=1.0, top_k=-1, min_p=0.0,
            repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
            min_tokens=0, stop=[], stop_token_ids=[eos],
            # Disable the engine's potentially different EOS; the bound local
            # EOS above remains the sole explicit stop token.
            ignore_eos=True, include_stop_str_in_output=False,
            skip_special_tokens=False, add_special_tokens=False,
            truncate_prompt_tokens=None, return_token_ids=True)
        connection = http.client.HTTPConnection("127.0.0.1", self.server["port"], timeout=600)
        try:
            connection.request("POST", "/v1/completions", body=json.dumps(payload),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read()
            if response.status != 200:
                raise ValueError(f"vLLM completion failed (HTTP {response.status}): {raw[:1000]!r}")
            result = json.loads(raw)
        finally:
            connection.close()
        if result.get("model") != self.server["served_model_name"] or len(result.get("choices", [])) != 1:
            raise ValueError("completion server/model identity mismatch")
        choice, usage = result["choices"][0], result.get("usage", {})
        ids, count = choice.get("token_ids"), usage.get("completion_tokens")
        if (not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids) or
                type(count) is not int or not 1 <= count <= 256 or len(ids) != count or
                usage.get("prompt_tokens") != len(prompt_ids)):
            raise ValueError("completion token count disagrees with returned tokens")
        reason = choice.get("finish_reason")
        if reason not in ("stop", "length"):
            raise ValueError("unexpected completion finish_reason")
        truncated = reason == "length"
        # EOS at token 256 is a stop, not truncation. Count alone is insufficient.
        # Explicit checks (also under python -O) fail closed on shortened replies.
        if (truncated and count != 256 or truncated != (ids[-1] != eos) or
                eos in ids[:-1] or choice.get("stop_reason") not in (None, eos)):
            raise ValueError("completion finish_reason, EOS and token count disagree")
        text = self.tokenizer.decode(ids if truncated else ids[:-1], skip_special_tokens=False)
        return Generation(text, count, truncated)

    def close(self):
        pass  # Each request owns its connection; the shared server outlives us.


def validate_records(expected, records, identity, *, complete=True):
    checked_expectations(expected)
    if (identity.get("benchmark") != "alfworld" or identity.get("split") != "valid_seen" or
            identity.get("expected_hash") != digest(expected) or identity.get("max_steps") != 40 or
            identity.get("max_action_tokens") != 256 or identity.get("evaluation_temperature") != 0):
        raise ValueError("wrong evaluation identity/split/step cap")
    ids = [r.get("task_id") for r in records]
    if (any(count != 1 for count in Counter(ids).values()) or set(ids) - set(expected["task_ids"]) or
            complete and set(ids) != set(expected["task_ids"])):
        raise ValueError("missing, duplicate or unexpected evaluation task IDs")
    for row in records:
        if (row.get("identity_hash") != digest(identity) or row.get("split") != "valid_seen" or
                row.get("max_steps") != 40 or row.get("status") != "completed"):
            raise ValueError("task identity/split/step cap mismatch")
        for name in ("success", "done", "truncated", "horizon_reached"):
            if type(row.get(name)) is not bool:
                raise ValueError(f"strict bool required: {name}")
        steps, turns = row.get("steps"), row.get("turns", [])
        if type(steps) is not int or not 1 <= steps <= 40 or len(turns) != steps:
            raise ValueError("invalid episode steps")
        for index, turn in enumerate(turns):
            if (any(type(turn.get(k)) is not bool for k in ("truncated", "done", "success")) or
                    type(turn.get("token_count")) is not int or not 1 <= turn["token_count"] <= 256 or
                    turn["truncated"] and turn["token_count"] != 256 or
                    any(not isinstance(turn.get(k), str) for k in ("prompt", "command", "generated_text", "observation")) or
                    turn["success"] and not turn["done"] or turn["done"] and index != steps - 1):
                raise ValueError("invalid turn/strict bool generation record")
        horizon = steps == 40 and not row["done"]
        if (row["done"] != turns[-1]["done"] or row["success"] != turns[-1]["success"] or
                not row["done"] and not horizon or row["horizon_reached"] != horizon or
                row["truncated"] != (horizon or any(t["truncated"] for t in turns)) or
                row.get("generated_text") != [t["generated_text"] for t in turns]):
            raise ValueError("incomplete or inconsistent full episode")
    return {r["task_id"]: r["success"] for r in records}


def aggregate_records(expected, records, identity):
    verdicts = validate_records(expected, records, identity)
    successes = sum(verdicts.values())
    groups = defaultdict(list)
    for tid, won in sorted(verdicts.items()):
        groups[tid.split("/", 1)[0]].append(won)
    # Resample whole game-name parent groups; retain task weighting within a
    # replicate. This is task-sampling uncertainty, not training-seed variation.
    rng, samples, population = random.Random(0), [], list(groups.values())
    for _ in range(2000):
        draw = [population[rng.randrange(len(population))] for _ in population]
        samples.append(sum(sum(g) for g in draw) / sum(len(g) for g in draw))
    samples.sort()
    return dict(complete=True, split="valid_seen", tasks=140, successes=successes,
        success_rate=successes / 140, success_rate_percent=100 * successes / 140,
        units=dict(success_rate="fraction_0_1", success_rate_percent="percent_0_100"),
        verdicts=verdicts, records_hash=digest(sorted(records, key=lambda r: r["task_id"])),
        uncertainty=dict(method="game_parent_cluster_bootstrap", level=0.95, seed=0, replicates=2000,
            parent_groups=len(groups), success_rate_interval=[samples[49], samples[1949]],
            interpretation="valid_seen development task sampling; one training seed"))


def compare_base(base, result):
    for value in (base, result):
        hard = value.get("hardware_class")
        if (not isinstance(hard, dict) or not hard or
                value.get("hardware_class_hash") != digest(hard) or
                value["identity"]["hardware_hash"] != digest(hard)):
            raise ValueError("base comparison requires an audited hardware class")
    if base["hardware_class"] != result["hardware_class"]:
        raise ValueError("base comparison hardware class differs")
    if (base["expected"] != result["expected"] or
            any(base["identity"][k] != result["identity"][k] for k in (
                "tokenizer_hash", "base_checkpoint_hash", "evaluation_harness_hash",
                "split", "max_steps", "max_action_tokens", "evaluation_temperature")) or
            base["identity"]["checkpoint"]["loading"] != "merged" or
            base["identity"]["checkpoint"]["base_checkpoint_hash"] != result["identity"]["base_checkpoint_hash"]):
        raise ValueError("base comparison requires the same base, tokenizer and scoring harness")
    before, after = base["aggregate"]["verdicts"], result["aggregate"]["verdicts"]
    return {tid: dict(before=before[tid], after=after[tid], repaired=not before[tid] and after[tid],
                     damaged=before[tid] and not after[tid]) for tid in sorted(after)}


def tag_lock_path(root, tag):
    if not isinstance(tag, str) or not tag or tag in (".", "..") or "/" in tag or "\\" in tag:
        raise ValueError("invalid ALFWorld campaign tag")
    # One repository-wide namespace even when callers choose different output
    # roots. BFCL keeps its original results/bfcl_std/.locks inode namespace.
    return Path(root) / "results/alfworld_std/.locks" / digest(tag) / ".lock"


def _records(artifacts, expected, identity, *, complete):
    if any(p.name not in {"binding.json", "tasks", "aggregate.json"} or p.is_symlink()
           for p in artifacts.iterdir()):
        raise ValueError("unexpected campaign artifact")
    records = []
    for path in sorted((artifacts / "tasks").glob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix != ".json":
            raise ValueError("unexpected task artifact")
        envelope = json.loads(path.read_text())
        row = envelope["record"]
        if envelope.get("record_hash") != digest(row) or path.stem != digest(row["task_id"]):
            raise ValueError("corrupted task artifact")
        records.append(row)
    validate_records(expected, records, identity, complete=complete)
    return records


def validate_evaluation(directory, identity, expected, *, audited_hashes=None, manifest_hash=None):
    """Validate receipt, complete tree, all records and recomputed metric units."""
    directory = Path(directory)
    result = json.loads((directory / "campaign.json").read_text())
    stored = result["identity"]
    hashes = audited_hashes or [identity["evaluation_harness_hash"]]
    if (dict(stored, evaluation_harness_hash=identity["evaluation_harness_hash"]) != identity or
            stored.get("evaluation_harness_hash") not in hashes or result["expected"] != expected):
        raise ValueError("evaluation identity mismatch")
    artifacts = directory / "artifacts"
    if result.get("artifacts_hash") != tree_hash(artifacts):
        raise ValueError("corrupted evaluation artifacts")
    binding = json.loads((artifacts / "binding.json").read_text())
    if (binding["identity"] != stored or binding["expected"] != expected or
            binding.get("manifest_hash") != result.get("manifest_hash") or
            manifest_hash is not None and binding.get("manifest_hash") != manifest_hash):
        raise ValueError("campaign artifact binding mismatch")
    hard = result.get("hardware_class")
    if not isinstance(hard, dict) or not hard or result.get("hardware_class_hash") != digest(hard) or digest(hard) != identity["hardware_hash"]:
        raise ValueError("missing or wrong hardware class")
    records = _records(artifacts, expected, stored, complete=True)
    aggregate = aggregate_records(expected, records, stored)
    if (json.loads((artifacts / "aggregate.json").read_text()) != aggregate or result.get("aggregate") != aggregate):
        raise ValueError("aggregate/units disagree with full task records")
    return result


def evaluate(root, manifest, *, output_root, tag, hardware, backend_factory=None, env_factory=None,
             supplement=None, base_evaluation=None, lock_timeout=None):
    """Prepare, resume or reuse one campaign, without writing any training file.

    Injected backends are for CPU fixtures. The default backend checks the live
    hardware class before its first allocation; reuse never constructs it.
    """
    lock = tag_lock_path(root, tag)
    directory = Path(output_root) / tag
    for name in ("data_root", "model_path", "tokenizer_path", "environment_root", "run_directory", "checkpoint"):
        value = manifest["paths"].get(name)
        if value and directory.resolve().is_relative_to(Path(value).resolve()):
            raise ValueError("campaign output must be separate from input/training directories")
    with evaluation_lock(lock, tag=f"alfworld/{tag}", timeout=lock_timeout):
        if directory.exists() and any(p.name not in {"artifacts", "campaign.json", "audit.jsonl"}
                                      or p.is_symlink() for p in directory.iterdir()):
            raise ValueError("existing directory is not an ALFWorld campaign; choose a new output")
        current, hashes = guard_manifest(root, manifest, hardware=hardware, supplement=supplement)
        identity = campaign_identity(current)
        expected = current["evaluation_harness"]["expected"]
        journal = ComputeJournal(directory / "audit.jsonl", cuda=False)
        base = _validated_base(base_evaluation) if base_evaluation is not None else None
        if base is not None:
            # Check base completeness/class/science before generating or
            # aggregating any tasks. Placeholder verdicts only check alignment.
            compare_base(base, dict(identity=identity, expected=expected,
                hardware_class=hardware["hard"], hardware_class_hash=digest(hardware["hard"]),
                aggregate=dict(verdicts={tid: False for tid in expected["task_ids"]})))
        if (directory / "campaign.json").exists():
            result = validate_evaluation(directory, identity, expected, audited_hashes=hashes,
                                         manifest_hash=manifest_digest(manifest))
            if base is not None:
                comparison = compare_base(base, result)
                if result.get("repairs_damage") != comparison:
                    raise ValueError("completed campaign base comparison differs")
            journal.append("evaluation_reuse_via_audited_identity" if
                result["identity"]["evaluation_harness_hash"] != identity["evaluation_harness_hash"] else "evaluation_reused",
                manifest_hash=manifest_digest(manifest), tag=tag,
                stored_harness_hash=result["identity"]["evaluation_harness_hash"],
                current_harness_hash=identity["evaluation_harness_hash"],
                supplement_hash=digest(supplement) if supplement else None,
                gpu_seconds=0., gpu_reserved_seconds=0.)
            return result  # retain the scored identity and immutable completion bytes
        artifacts = directory / "artifacts"
        binding = dict(identity=identity, expected=expected, manifest_hash=manifest_digest(manifest))
        binding_path = artifacts / "binding.json"
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("incomplete campaign belongs to another identity; refusing regeneration")
        else:
            if artifacts.exists() and any(artifacts.iterdir()):
                raise ValueError("unbound campaign artifacts")
            atomic_json(binding_path, binding)
        records = _records(artifacts, expected, identity, complete=False)
        done = {r["task_id"] for r in records}
        backend = None
        try:
            for tid in expected["task_ids"]:
                if tid in done:
                    continue
                if backend is None:
                    if backend_factory is None:
                        from ..hardware import hardware_identity
                        if checked_hardware(hardware_identity())["hard"] != hardware["hard"]:
                            raise ValueError("live evaluation hardware class differs")
                        backend = HFBackend(current)
                    else:
                        backend = backend_factory(current)
                factory = env_factory or (lambda t: EvaluationEnvBridge(t, data_root=manifest["paths"]["data_root"],
                    environment_root=manifest["paths"]["environment_root"]))
                record = official_episode(tid, backend, env_factory=factory, identity=identity)
                validate_records(expected, [record], identity, complete=False)
                atomic_json(artifacts / "tasks" / f"{digest(tid)}.json", dict(record=record, record_hash=digest(record)))
                records.append(record)
        finally:
            if backend is not None:
                backend.close()
        # Hash actual source/data/model again before publishing any aggregate.
        final, final_hashes = guard_manifest(root, manifest, hardware=hardware, supplement=supplement)
        if campaign_identity(final) != identity or final_hashes != hashes:
            raise ValueError("evaluation inputs changed during campaign")
        aggregate = aggregate_records(expected, records, identity)
        aggregate_path = artifacts / "aggregate.json"
        if aggregate_path.exists() and json.loads(aggregate_path.read_text()) != aggregate:
            raise ValueError("corrupted existing aggregate")
        atomic_json(aggregate_path, aggregate)
        result = dict(identity=identity, expected=expected, manifest_hash=manifest_digest(manifest), hardware_class=hardware["hard"],
            hardware_class_hash=digest(hardware["hard"]), aggregate=aggregate,
            artifacts_hash=tree_hash(artifacts))
        if base is not None:
            # Recheck the base tree after a possibly long campaign too.
            if _validated_base(base_evaluation) != base:
                raise ValueError("base evaluation changed during campaign")
            result["repairs_damage"] = compare_base(base, result)
            result["base_campaign_hash"] = file_hash(Path(base_evaluation) / "campaign.json")
        atomic_json(directory / "campaign.json", result)
        return validate_evaluation(directory, identity, expected)


def _validated_base(directory):
    saved = json.loads((Path(directory) / "campaign.json").read_text())
    return validate_evaluation(directory, saved["identity"], saved["expected"])


def comparison_anchors(root):
    """Historical references only; presence is observed, weights not re-scored."""
    root = Path(root)
    checkpoint = "results/appworld_students/alfabl_CE_conseq_s0"
    sources = ("notes/exp_log.md", "scripts/alf_valseen_hpg.slurm", "tools/bfas_eval_ckpt.py")
    provenance = {name: file_hash(root / name) if (root / name).is_file() else None for name in sources}
    return dict(version="alfworld-historical-comparison-anchors-v1", budget_matched=False,
        provenance=provenance, anchors=[
            dict(name="alfabl_CE_conseq_s0", split="valid_seen", tasks=140, successes=109,
                success_rate=109 / 140, success_rate_percent=100 * 109 / 140, reported_percent=77.86,
                checkpoint=f"{checkpoint}/adapter", checkpoint_exists=(root / checkpoint / "adapter").is_dir(),
                weights_exists=(root / checkpoint / "adapter/model.safetensors").is_file(),
                historical_script_checkpoint=f"{checkpoint}/hub_merged",
                historical_script_checkpoint_exists=(root / checkpoint / "hub_merged").is_dir(),
                recipe="85 consequential corrected-reply events, 4 epochs, 44 optimizer steps",
                provenance="exp_log.md:582,650; HPG 41098042_1; original valid_seen records not verified locally"),
            dict(name="base", model="Qwen/Qwen3.5-4B", split="valid_seen", tasks=140, successes=10,
                success_rate=10 / 140, success_rate_percent=100 * 10 / 140, reported_percent=7.14,
                provenance="exp_log.md:651; HPG 41098042_0; scripts/alf_valseen_hpg.slurm")],
        limitation="Counts inferred from logged percentages and denominator, pending original valid_seen record audit; "
                   "historical references are not budget/data/hardware-matched RTD comparisons.")
