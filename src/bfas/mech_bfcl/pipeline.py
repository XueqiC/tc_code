"""The support → diagnose → paired practice → three-layer evaluation workflow."""
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

from .common import (REGISTRY, STUDENT, append_row, digest, read_json, read_rows,
                     setup_harness, write_json)
from .exercises import OfficialValidator, VARIANTS, materialize, native_pair
from .harness import NoThinking, frames, official_run, trajectory_metrics
from .splits import inventory, stratum
from .teacher import Teacher, ledger_summary


def clean_context(frame, splits):
    info = splits["items"][frame["task_id"]]
    # Same start state, tools, and natural prefix for C/D. The current student
    # response, correctness labels, diagnosis and future observations are absent.
    keys = ("task_id", "frame_id", "messages", "functions", "snapshot", "snapshot_error", "involved_classes")
    return dict({k: copy.deepcopy(frame.get(k)) for k in keys},
                parent_id=info["parent_id"], category=info["category"])


def validate_sources(splits, entries):
    for tid, info in splits["items"].items():
        if digest(entries[tid]) != info["source_hash"]:
            raise ValueError(f"Official source changed since split freeze: {tid}")


def training_seeds(directory, splits):
    seeds = read_json(directory / "seeds.json")
    captured = {f["frame_id"]: f for f in frames(directory / "support/trajectories")}
    parents = set()
    if len(seeds) > 8:
        raise ValueError("At most eight failure seeds are allowed")
    for seed in seeds:
        ctx = seed["context"]
        if ctx["task_id"] not in splits["support"] or ctx["parent_id"] in parents:
            raise ValueError("Training seeds must be distinct support parents")
        frame = captured.get(ctx["frame_id"])
        if frame is None or digest(clean_context(frame, splits)) != digest(ctx):
            raise ValueError("Seed context does not match its captured support state")
        if seed["support_result"]["correct"]:
            raise ValueError("Failure seed came from a successful task")
        parents.add(ctx["parent_id"])
    return seeds


def select_seeds(items, captured, splits, max_seeds=8):
    by_task = {}
    for state in captured:
        by_task.setdefault(state["task_id"], []).append(state)
    failed = [i for i in items if not i["correct"]]
    priority = {"memory": 0, "multi_turn": 1, "irrelevance": 2, "live": 3, "non_live": 4}
    failed.sort(key=lambda i: (priority.get(stratum(i["category"]), 5), i["id"]))
    picked, parents, counts = [], set(), Counter()
    while failed and len(picked) < max_seeds:
        task = min(failed, key=lambda i: (counts[stratum(i["category"])],
                                         priority.get(stratum(i["category"]), 5), i["id"]))
        failed.remove(task)
        if task["parent_id"] in parents:
            continue
        states = by_task.get(task["id"], [])
        if not states:
            continue
        # Prefer a directly observed executor/format failure; otherwise the
        # failed task's final decision is a hypothesis seed, not a proven cause.
        candidates = [s for s in states if s.get("finish_reason") == "length" or
                      any("error" in str(o).lower() for o in s.get("execution", {}).get("outputs", []))]
        frame = (candidates or states)[0 if candidates else -1]
        picked.append(dict(seed_id=f"seed_{len(picked)}", context=clean_context(frame, splits),
                           failure_frame=frame, support_result=task,
                           attribution="observed task failure; teacher must propose a testable causal hypothesis"))
        parents.add(task["parent_id"])
        counts[stratum(task["category"])] += 1
    return picked


def rollout(args, splits):
    adapter, entries = inventory()
    validate_sources(splits, entries)
    check_server(args, "base")
    support = official_run(args, splits["support"], args.run_dir / "support", adapter, splits)
    captured = frames(args.run_dir / "support/trajectories")
    seeds = select_seeds(support, captured, splits)
    write_json(args.run_dir / "seeds.json", seeds)
    write_json(args.run_dir / "support_summary.json", dict(tasks=len(support),
        successes=sum(i["correct"] for i in support), seeds=len(seeds),
        headroom="little/no learning headroom" if len(seeds) < 2 else "support failures observed"))
    # This fixed, independent training-side access supplies natural contexts for
    # held-out layers 1/2. It never contributes failure seeds or diagnoses.
    official_run(args, splits["calibration"], args.run_dir / "calibration", adapter, splits)


def diagnose(args, splits):
    teacher = Teacher(args.run_dir)
    seeds = training_seeds(args.run_dir, splits)
    diagnoses = []
    # Reserve 6k of the shared 8k budget for independently held-out preparation.
    cap = 2000 // max(1, len(seeds))
    for seed in seeds:
        context = dict(context=seed["context"], observed_response=seed["failure_frame"]["response"],
                       execution=seed["failure_frame"].get("execution"),
                       checker_result=seed["support_result"], attribution=seed["attribution"])
        messages = [dict(role="system", content=(
            "Diagnose one BFCL decision. Return JSON with hypothesis, observed_behaviour, "
            "changed_condition, predicted_action, and falsifier, each a concrete short string. "
            "Refer to the observed call/arguments or no-call decision. A failed whole task does "
            "not prove this step caused it. Give a testable conditional prediction, not a skill label. "
            "Keep the complete response under 160 words.")),
            dict(role="user", content=json.dumps(context, ensure_ascii=False))]
        value, cost = teacher.ask("shared", "diagnose:" + seed["seed_id"], messages, cap)
        required = {"hypothesis", "observed_behaviour", "changed_condition", "predicted_action", "falsifier"}
        valid = isinstance(value, dict) and all(isinstance(value.get(k), str) and value[k].strip() for k in required)
        diagnoses.append(dict(seed_id=seed["seed_id"], diagnosis=value if valid else None,
                              valid=valid, cost_call_id=cost))
    write_json(args.run_dir / "diagnoses.json", diagnoses)
    write_json(args.run_dir / "teacher_cost.json", ledger_summary(teacher.path))


def check_server(args, arm):
    if not args.base_url:
        raise ValueError("--base-url is required for student inference")
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=15) as response:
        models = json.load(response)["data"]
    matched = [m for m in models if m["id"] == args.served_model]
    if len(matched) != 1:
        raise ValueError(f"Server does not advertise {args.served_model}")
    model = matched[0]
    if arm == "base":
        if args.served_model != STUDENT:
            raise ValueError("Base rollouts/probes must use the base student served as google/gemma-4-12B-it")
    else:
        artifact = (args.run_dir / arm / "training/adapter").resolve()
        if not (artifact / "adapter_config.json").exists():
            raise ValueError("No trained adapter for the requested arm")
        if Path(model.get("root", "")).resolve() != artifact:
            raise ValueError("vLLM /models root does not match this arm's trained adapter path")
    return model


def student_reply(args, exercise, tokenizer):
    from tools.bfcl_pool_render_gemma4 import template_render
    prompt = template_render(NoThinking(tokenizer), exercise["messages"], exercise["functions"])
    n = len(tokenizer.encode(prompt, add_special_tokens=False))
    if n + 2 >= 32768:
        raise ValueError("Complete local state exceeds serving context; do not truncate history")
    body = dict(model=args.served_model, prompt=prompt, temperature=0.001, seed=args.seed,
        top_k=1, max_tokens=min(4096, 32768-n-2), add_special_tokens=False, skip_special_tokens=False,
        stop_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in ("<eos>", "<turn|>", "<|tool_response>")])
    request = urllib.request.Request(args.base_url.rstrip("/") + "/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=72000) as response:
        raw = json.load(response)
    return raw["choices"][0]["text"], raw


def generation_prompt(context, diagnosis=None):
    common = (
        'Return JSON {"exercises":[{"user":"short task", "demo":{"kind":"call",'
        '"calls":[{"name":"declared_tool","arguments":{}}]},"variant":"condition"}]}.'
        ' Abstention demo is {"kind":"abstain","text":"short clarification or answer"}.'
        ' Produce 9 diverse, short exercises for this local task/interface/start state. '
        'Use the exact provided tools and actual snapshot; change only the last user request. '
        'Do not invent history, tool returns, API names, diagnosis text, end-of-task actions or EOS tokens. '
        'Use at most 4 calls per demonstration. You may include possible_answer in official BFCL '
        'AST format (parameter values are lists of alternatives) when multiple arguments are valid. '
        'Variant is condition (change a decisive condition AND the correct action), surface '
        '(surface change preserving the decision relation), or neighbour_correct (a nearby ordinary '
        'case). Include at least two of each type; no labels in student-visible text. '
        'Keep each task and demonstration short. Return only JSON, no markdown.'
    )
    payload = dict(context=context)
    if diagnosis is not None:
        payload["student_gap"] = diagnosis
        common += (' Target the supplied observed gap. The neighbour_correct candidates must check '
                   'the opposite/nearby correct behaviour; they will be verified on the frozen base student. '
                   'Condition variants must predictably reverse/change the relevant call, argument, or order.')
    else:
        common += ' Generate ordinary varied practice for this stage.'
    return [dict(role="system", content=common), dict(role="user", content=json.dumps(payload, ensure_ascii=False))]


def heldout(args, splits, teacher, validator):
    path = args.run_dir / "heldout.json"
    if path.exists():
        return read_json(path)
    calibration = frames(args.run_dir / "calibration/trajectories")
    contexts = {}
    for tid in splits["calibration"]:
        own = [f for f in calibration if f["task_id"] == tid]
        if not own:
            raise ValueError("Missing calibration rollout")
        # Retain only actual captured states, never invented continuations.
        contexts[tid] = own
    selected = {tid: [own[0]] for tid, own in contexts.items()}
    while sum(map(len, selected.values())) < 24:
        available = [t for t in contexts if len(selected[t]) < len(contexts[t])]
        if not available:
            break
        tid = min(available, key=lambda t: (len(selected[t]), t))
        selected[tid].append(contexts[tid][len(selected[tid])])
    output, audit = [], []
    # Fixed, exercise-weighted ceilings total at most 6k and are reproducible
    # on resume. The ledger additionally enforces the shared remaining budget.
    total_units = 3 * len(selected) + sum(map(len, selected.values()))
    for tid in splits["calibration"]:
        ctx = [clean_context(f, splits) for f in selected[tid]]
        prompt = [dict(role="system", content=(
            'Return JSON {"local":[{"user":"short variant","demo":{"kind":"call",'
            '"calls":[{"name":"tool","arguments":{}}]}}],"natural":[{"state_index":0,"demo":{...}}]}. '
            'Write exactly 3 short unseen local exercises derived from the first real state, '
            'and one short correct next-action demonstration per supplied natural state. '
            'For no-call use {"kind":"abstain","text":"short clarification/answer"}. '
            'Do not edit natural histories. Use the exact interface, snapshot and real past outputs. '
            'No labels, fabricated observations, reasoning, end-of-task markers or EOS. '
            'Return concise JSON only.')),
            dict(role="user", content=json.dumps(dict(states=ctx), ensure_ascii=False))]
        per_task = 6000 * (3 + len(ctx)) // total_units
        value, cost = teacher.ask("shared", "heldout:"+tid, prompt, per_task)
        if not value:
            audit.append(dict(task_id=tid, reason="teacher failure/budget", cost_call_id=cost))
            continue
        for key, layer, limit in (("local", 1, 3), ("natural", 2, len(ctx))):
            proposals = value.get(key, [])
            if not isinstance(proposals, list):
                audit.append(dict(task_id=tid, reason=f"{key} is not a list", cost_call_id=cost))
                continue
            for index, proposed in enumerate(proposals[:limit]):
                try:
                    source_index = proposed.get("state_index", index) if layer == 2 else 0
                    if type(source_index) is not int or not 0 <= source_index < len(ctx):
                        raise ValueError("Invalid calibration state index")
                    exercise = materialize(proposed, ctx[source_index], arm="heldout",
                        group=f"calibration:{tid}:{key}", index=index, call_id=cost, layer=layer)
                    exercise["validation"] = validator.validate(exercise)
                    if exercise["validation"]["valid"]:
                        output.append(exercise)
                    audit.append(dict(id=exercise["id"], validation=exercise["validation"], cost_call_id=cost))
                except (KeyError, ValueError, TypeError) as exc:
                    audit.append(dict(task_id=tid, reason=str(exc), cost_call_id=cost))
    # Natural continuation states are unique, even if a teacher repeats an index.
    unique = {}
    for row in output:
        key = (row["layer"], row["source_frame"] if row["layer"] == 2 else digest([row["messages"], row["functions"]]))
        unique.setdefault(key, row)
    output = list(unique.values())
    write_json(path, output)
    write_json(args.run_dir / "heldout_audit.json", dict(items=audit,
        counts=dict(Counter(e["layer"] for e in output)), requested=dict(local=48, natural=24),
        note="Budget and real-state availability take precedence over target counts"))
    return output


def generate(args, splits):
    from transformers import AutoTokenizer
    seeds = training_seeds(args.run_dir, splits)
    diagnoses = {d["seed_id"]: d for d in read_json(args.run_dir / "diagnoses.json")}
    if not seeds:
        raise ValueError("No support failures: report insufficient headroom; do not mine evaluation")
    if any(not diagnoses[s["seed_id"]]["valid"] for s in seeds):
        raise ValueError("Missing testable diagnoses; inspect diagnoses.json (failed generations remain charged)")
    check_server(args, "base")
    teacher, validator = Teacher(args.run_dir), OfficialValidator()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    heldout(args, splits, teacher, validator)
    output, audit = [], []
    for seed in seeds:
        diag = None if args.arm == "C" else dict(
            hypothesis=diagnoses[seed["seed_id"]]["diagnosis"],
            observed_response=seed["failure_frame"]["response"],
            observed_execution=seed["failure_frame"].get("execution"))
        prompt = generation_prompt(seed["context"], diag)
        value, cost = teacher.ask(args.arm, args.arm+":"+seed["seed_id"], prompt,
                                 24000 // len(seeds))
        if not value:
            audit.append(dict(seed_id=seed["seed_id"], reason="teacher failure/budget", cost_call_id=cost))
            continue
        proposals = value.get("exercises", [])
        if not isinstance(proposals, list):
            audit.append(dict(seed_id=seed["seed_id"], reason="exercises is not a list", cost_call_id=cost))
            continue
        for index, proposed in enumerate(proposals[:12]):
            try:
                row = materialize(proposed, seed["context"], arm=args.arm,
                    group=args.arm+":"+seed["seed_id"], index=index, call_id=cost)
                row["validation"] = validator.validate(row)
                if args.arm == "D" and row["variant"] not in VARIANTS:
                    row["validation"] = dict(valid=False, reason="Missing D variant type")
                if row["validation"]["valid"]:
                    native_pair(tokenizer, row)  # Native format round-trip before admission.
                    if row["variant"] == "neighbour_correct":
                        probe_path = args.run_dir / args.arm / "neighbour_probes" / (digest(row)+".json")
                        if probe_path.exists():
                            probe = read_json(probe_path)
                        else:
                            response, raw = student_reply(args, row, tokenizer)
                            probe = dict(response=response, raw=raw, score=validator.score(row, response))
                            write_json(probe_path, probe)
                        row["base_neighbour_check"] = probe["score"]
                        if not probe["score"]["correct"]:
                            row["validation"] = dict(valid=False, reason="Base was not right on neighbour candidate")
                    if row["validation"]["valid"]:
                        output.append(row)
                audit.append(row)
            except (KeyError, ValueError, TypeError) as exc:
                audit.append(dict(seed_id=seed["seed_id"], index=index, cost_call_id=cost, reason=str(exc)))
    # Exact duplicates do not purchase extra exposure or independent evidence.
    unique = {}
    for row in output:
        unique.setdefault(digest([row["messages"], row["functions"], row["demo"]]), row)
    output = list(unique.values())
    coverage = {s["seed_id"]: dict(Counter(e["variant"] for e in output
                if e["generation_group"] == args.arm+":"+s["seed_id"])) for s in seeds}
    missing = [seed for seed, counts in coverage.items() if args.arm == "D" and not VARIANTS <= counts.keys()]
    write_json(args.run_dir / args.arm / "exercises.json", output)
    write_json(args.run_dir / args.arm / "generation_audit.json", audit)
    global_coverage = Counter(e["variant"] for e in output)
    surface_actions = {digest(e["demo"]) for e in output if e["variant"] == "surface"}
    condition_changes = any(digest(e["demo"]) not in surface_actions for e in output if e["variant"] == "condition")
    ready = bool(output) and (args.arm == "C" or (VARIANTS <= global_coverage.keys() and condition_changes))
    write_json(args.run_dir / args.arm / "generation.json", dict(arm=args.arm, count=len(output),
        target=[48, 96], coverage=coverage, missing_variant_groups=missing,
        ready=ready, global_coverage=dict(global_coverage), condition_action_changes=condition_changes,
        contexts_hash=digest([s["context"] for s in seeds]),
        below_target=len(output)<48, reason="Budget first; no repetition to fill counts"))
    write_json(args.run_dir / "teacher_cost.json", ledger_summary(teacher.path))


def evaluate(args, splits):
    from transformers import AutoTokenizer
    adapter, entries = inventory()
    validate_sources(splits, entries)
    server = check_server(args, args.arm)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    validator = OfficialValidator()
    exercises = read_json(args.run_dir / "heldout.json")
    if {e["layer"] for e in exercises} != {1, 2}:
        raise ValueError("Held-out sets must contain both local and natural continuation items")
    destination = args.run_dir / "evaluation" / args.arm / args.repeat
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(arm=args.arm, split_hash=digest(splits), heldout_hash=digest(exercises),
                       server=server, temperature=0.001, top_k=1, seed=splits["seed"])
    if (destination / "protocol.json").exists() and read_json(destination / "protocol.json") != fingerprint:
        raise ValueError("Evaluation protocol/artifact changed during resume")
    write_json(destination / "protocol.json", fingerprint)
    existing = {r["id"]: r for r in read_rows(destination / "local.jsonl")}
    for exercise in exercises:
        if exercise["id"] in existing:
            continue
        if exercise["task_id"] not in splits["calibration"]:
            raise ValueError("Local evaluation source is not a calibration task")
        response, raw = student_reply(args, exercise, tokenizer)
        score = validator.score(exercise, response)
        row = dict(id=exercise["id"], parent_id=exercise["parent_id"],
            category=exercise["category"], layer=exercise["layer"], correct=score.pop("correct"),
            response=response, raw=raw, metrics=score, generation_group=exercise["generation_group"])
        append_row(destination / "local.jsonl", row)
    full = official_run(args, splits["evaluation"], destination / "full", adapter, splits)
    rows = read_rows(destination / "local.jsonl") + full
    write_json(destination / "items.json", rows)
    from .stats import refresh_pairs
    refresh_pairs(args.run_dir)
