"""Pre-registered calibration-parent confirmation, separate from heldout layers."""
import json
import math
import random
import re
import urllib.request

from .common import BUDGETS, digest, read_json, write_json
from .exercises import correct_call_set, materialize, native_pair
from .generation import condition_side, relation
from .harness import frames, restore_multiturn_context
from .teacher import ledger_summary


def isolation(directory, splits):
    seeds = read_json(directory / "seeds.json")
    sources = [s["context"] for s in seeds]
    banks = {}
    for arm in ("C", "D"):
        path = directory / arm / "exercises.json"
        if path.exists():
            banks[arm] = read_json(path)
            sources.extend(banks[arm])
    # Reserve against the whole support split, including not-yet-generated
    # exercises. A later C/D generation cannot contaminate this reservation.
    parents = {splits["items"][t]["parent_id"] for t in splits["support"]}
    parents.update(s["parent_id"] for s in sources)
    tasks = set(splits["support"]) | {s["task_id"] for s in sources}
    return parents, tasks


def build(args, splits, teacher, validator, tokenizer):
    from .pipeline import clean_context
    directory = args.run_dir
    parents, tasks = isolation(directory, splits)
    contexts = []
    captured = frames(directory / "calibration/trajectories")
    by_task = {}
    for frame in captured:
        by_task.setdefault(frame["task_id"], frame)
    # Selection precedes all authoring and student scoring; no adapter-dependent
    # filtering, item replacements, or selection on baseline correctness.
    candidates = sorted(splits["calibration"])
    random.Random(splits["seed"]).shuffle(candidates)
    used = set()
    for tid in candidates:
        parent = splits["items"][tid]["parent_id"]
        if tid in tasks or parent in parents or parent in used or tid not in by_task:
            continue
        context = clean_context(by_task[tid], splits)
        contexts.append(restore_multiturn_context(context, by_task[tid]))
        used.add(parent)
        if 3 * len(contexts) >= args.target:
            break
    if not contexts:
        raise ValueError("No independent calibration parents available for confirmation")
    plan_path = directory / "confirmation_plan.json"
    selection = dict(version=1, target=args.target, split_hash=digest(splits),
        excluded_training_parents=sorted(parents), excluded_training_tasks=sorted(tasks),
        contexts=contexts, roles=["anchor", "condition", "surface"],
        pairs=[dict(left="anchor", right="condition", relation="change"),
               dict(left="anchor", right="surface", relation="invariant")],
        rules={c["task_id"]: relation(c) for c in contexts})
    if plan_path.exists():
        plan = read_json(plan_path)
        if any(plan[k] != v for k, v in selection.items()):
            raise ValueError("Confirmation preregistration changed; use a new run directory")
    else:
        if list((directory / "evaluation").glob("*/main/confirmation.json")):
            raise ValueError("Confirmation must be pre-registered before it is evaluated")
        cost = ledger_summary(teacher.path)["shared"]
        if cost["unresolved"]:
            raise ValueError("Unresolved shared teacher usage")
        remaining = max(0, BUDGETS["shared"] - cost["output_tokens"])
        plan = dict(selection, output_tokens_per_parent=remaining // len(contexts),
                    shared_output_tokens_before=cost["output_tokens"])
        write_json(plan_path, plan)
    path = directory / "confirmation.json"
    if path.exists():
        value = read_json(path)
        if value["plan_hash"] != digest(plan):
            raise ValueError("Confirmation set does not match preregistration")
        validate_set(value, directory, splits)
        return value
    output, pairs, audit = [], [], []
    for context_index, context in enumerate(contexts):
        tid = context["task_id"]
        rule = relation(context)
        # Roles and IDs are fixed even when another parent's authoring fails.
        slot = context_index * 3
        roles = plan["roles"][:max(0, min(3, args.target-slot))]
        if not roles:
            break
        prompt = [dict(role="system", content=(
            'Return a JSON object with anchor, condition, surface, each an exercise object '
            'with user (short natural request), demo ({"kind":"call","calls":['
            '{"name":"declared_tool","arguments":{}}]} or {"kind":"abstain","text":"short answer"}). '
            'Use only supplied tools, actual state and natural history; edit only the final request. '
            'The condition request changes ONE decisive fact/requirement from anchor so the '
            'correct call set changes AND switches sides of condition_relation. The surface request '
            'only paraphrases anchor; its correct call set MUST stay identical. '
            'Include condition_side (positive|negative) and condition_evidence in each object. '
            'If a member is impossible in this exact state, return null for that member. '
            'Use at most four calls per demonstration. No drill, exercise, test, diagnosis, '
            'contrast-pair cues, fabricated observations, or control tokens in user requests. '
            'Return concise JSON only.')),
            dict(role="user", content=json.dumps(dict(context=context, condition_relation=rule, roles=roles), ensure_ascii=False))]
        value, cost = teacher.ask("shared", "confirm:"+tid, prompt, plan["output_tokens_per_parent"])
        members = {}
        for role in roles:
            try:
                if not isinstance(value, dict) or not isinstance(value.get(role), dict):
                    raise ValueError("Unavailable member or teacher failure/budget")
                proposal = dict(value[role], variant="condition" if role == "condition" else "surface")
                if re.search(r"\b(drill|exercise|test case|diagnosis|contrast.pair|condition.side)\b", proposal["user"], re.I):
                    raise ValueError("Confirmation cue leaked into student task")
                row = materialize(proposal, context, arm="confirm", group="confirm:"+tid,
                                  index=role, call_id=cost, layer="confirm")
                side = condition_side(row, rule)
                evidence = proposal.get("condition_evidence")
                if side is None or side != proposal.get("condition_side") or not isinstance(evidence, str) or not evidence.strip():
                    raise ValueError("Missing or inconsistent confirmation condition evidence")
                row.update(condition_side=side, condition_evidence=proposal["condition_evidence"],
                           confirmation_role=role)
                row["validation"] = validator.validate(row)
                if not row["validation"]["valid"]:
                    audit.append(dict(id=row["id"], validation=row["validation"], cost_call_id=cost))
                    continue
                native_pair(tokenizer, row)
                members[role] = row
                output.append(row)
            except (ValueError, KeyError, TypeError) as exc:
                audit.append(dict(task_id=tid, role=role, reason=str(exc), cost_call_id=cost))
        for spec in plan["pairs"]:
            left, right = members.get(spec["left"]), members.get(spec["right"])
            valid, reason = False, "One or both members unavailable"
            if left and right:
                same = correct_call_set(left) == correct_call_set(right)
                different_task = left["messages"] != right["messages"]
                valid = different_task and (same if spec["relation"] == "invariant" else (
                    not same and left["condition_side"] != right["condition_side"]))
                reason = None if valid else "Authored pair failed the required decision relation"
            pairs.append(dict(id="confirm:"+tid+":"+spec["relation"], parent_id=context["parent_id"],
                left=left["id"] if left else None, right=right["id"] if right else None,
                relation=spec["relation"], valid=bool(valid), reason=reason))
    result = dict(plan_hash=digest(plan), items=output, pairs=pairs,
                  requested=args.target, count=len(output), below_target=len(output)<args.target,
                  semantic_limit="Teacher-authored task semantics; pair relations check normalized target calls and sides")
    validate_set(result, directory, splits)
    write_json(path, result)
    write_json(directory / "confirmation_audit.json", audit)
    write_json(directory / "teacher_cost.json", ledger_summary(teacher.path))
    return result


def validate_set(value, directory, splits):
    parents, tasks = isolation(directory, splits)
    plan = read_json(directory / "confirmation_plan.json")
    if value["plan_hash"] != digest(plan) or plan["split_hash"] != digest(splits):
        raise ValueError("Confirmation set does not match preregistration")
    contexts = {c["task_id"]: c for c in plan["contexts"]}
    ids = set()
    for row in value["items"]:
        if (row["id"] in ids or row["task_id"] not in splits["calibration"] or row["task_id"] in tasks
                or row["parent_id"] in parents or row["parent_id"] != splits["items"][row["task_id"]]["parent_id"]
                or row["layer"] != "confirm" or not row["validation"]["valid"]):
            raise ValueError("Confirmation source isolation/identity/validation failed")
        ids.add(row["id"])
        context = contexts.get(row["task_id"])
        if context is None or row["source_hash"] != digest(context):
            raise ValueError("Confirmation source differs from preregistered context")
    by_id = {r["id"]: r for r in value["items"]}
    pair_ids = set()
    for pair in value["pairs"]:
        if pair["id"] in pair_ids:
            raise ValueError("Duplicate confirmation pair")
        pair_ids.add(pair["id"])
        if pair["valid"] and (pair["left"] not in ids or pair["right"] not in ids):
            raise ValueError("Valid confirmation pair has missing members")
        if pair["valid"]:
            left, right = by_id[pair["left"]], by_id[pair["right"]]
            same = correct_call_set(left) == correct_call_set(right)
            if (left["parent_id"] != right["parent_id"] or left["parent_id"] != pair["parent_id"]
                    or left["source_hash"] != right["source_hash"] or left["messages"] == right["messages"]
                    or pair["relation"] not in ("change", "invariant")
                    or (same if pair["relation"] == "change" else not same)
                    or pair["relation"] == "change" and left["condition_side"] == right["condition_side"]):
                raise ValueError("Invalid confirmation pair relation/provenance")


def target_probability(args, row, tokenizer):
    """Teacher-force the native target via prompt logprobs, never sampled logits.

    This is the probability of the particular authored token sequence, not
    probability mass summed over semantically equivalent calls. Unsupported
    servers produce an explicit unavailable diagnostic and keep correctness.
    """
    prompt, target = native_pair(tokenizer, row)
    p = tokenizer.encode(prompt, add_special_tokens=False)
    y = tokenizer.encode(target, add_special_tokens=False)
    if not p or not y or len(p)+len(y)+1 > 32768:
        return dict(available=False, reason="Target probability context unavailable")
    body = dict(model=args.served_model, prompt=p+y, max_tokens=1, temperature=0.001,
                seed=args.seed, top_k=1, echo=True, logprobs=1, add_special_tokens=False)
    request = urllib.request.Request(args.base_url.rstrip("/")+"/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=72000) as response:
            raw = json.load(response)
        values = raw["choices"][0]["logprobs"]["token_logprobs"][len(p):len(p)+len(y)]
        if len(values) != len(y) or any(v is None or not math.isfinite(v) or v > 0 for v in values):
            raise ValueError("Missing/nonfinite teacher-forced target logprobs")
        logp = sum(values)
        return dict(available=True, target_tokens=len(y), log_probability=logp,
            probability=math.exp(logp), mean_token_log_probability=logp/len(y), raw=raw,
            diagnostic_only=True, kind="authored call sequence" if row["demo"].get("calls") else "authored no-call text")
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        return dict(available=False, reason=f"Target logprobs unavailable: {type(exc).__name__}: {exc}", diagnostic_only=True)


def metrics(rows, specification, base=None):
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {r["id"] for r in specification["items"]}:
        raise ValueError("Incomplete confirmation evaluation")
    pairs, paired_ids = [], set()
    for spec in specification["pairs"]:
        if spec["valid"]:
            paired_ids.update((spec["left"], spec["right"]))
            pairs.append(dict(spec, both_correct=by_id[spec["left"]]["correct"] and by_id[spec["right"]]["correct"]))
    deltas = []
    if base is not None:
        before = {r["id"]: r for r in base}
        if before.keys() != by_id.keys():
            raise ValueError("Confirmation probability comparison requires identical items")
        for row in rows:
            a, b = before[row["id"]].get("target_probability", {}), row.get("target_probability", {})
            available = a.get("available", False) and b.get("available", False)
            deltas.append(dict(id=row["id"], available=available, diagnostic_only=True,
                probability_delta=b["probability"]-a["probability"] if available else None,
                log_probability_delta=b["log_probability"]-a["log_probability"] if available else None))
    return dict(items=len(rows), correct=sum(r["correct"] for r in rows),
        accuracy=sum(r["correct"] for r in rows)/len(rows) if rows else None,
        valid_pairs=len(pairs), both_sides_correct_rate=sum(p["both_correct"] for p in pairs)/len(pairs) if pairs else None,
        pairs=pairs, invalid_pairs=[p for p in specification["pairs"] if not p["valid"]],
        unpaired_items=[dict(id=r["id"], correct=r["correct"]) for r in rows if r["id"] not in paired_ids],
        per_item=[dict(id=r["id"], parent_id=r["parent_id"], correct=r["correct"]) for r in rows],
        target_probability_deltas=deltas)
