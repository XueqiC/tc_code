"""R2 frozen-subset provenance and two-sided decision coverage."""
import copy
from collections import Counter
from pathlib import Path
import shutil

from .common import append_row, digest, read_json, read_rows, write_json
from .exercises import VARIANTS, correct_call_set
from .teacher import ledger_summary


def prepare_r2(args, splits):
    source, destination = args.round1_run_dir.resolve(), args.run_dir.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("R2 requires a separate run directory outside the read-only R1 run")
    protocol = read_json(source / "protocol.json")
    if protocol["split_hash"] != digest(splits):
        raise ValueError("R1 and R2 must use the identical frozen split")
    costs = ledger_summary(source / "teacher_ledger.jsonl")
    if any(v["unresolved"] for v in costs.values()):
        raise ValueError("Resolve R1 teacher usage before preparing R2")
    names = ["seeds.json", "diagnoses.json", "heldout.json", "heldout_audit.json",
             "support", "calibration"]
    files = [p for name in names for p in (
        sorted((source / name).rglob("*")) if (source / name).is_dir() else [source / name]) if p.is_file()]
    manifest = dict(version=2, round1_run_dir=str(source), split_hash=digest(splits),
        input_hashes={str(p.relative_to(source)): digest(p.read_bytes().hex()) for p in files},
        round1_banks={a: dict(path=str(source / a / "exercises.json"),
                             hash=digest(read_json(source / a / "exercises.json"))) for a in ("C", "D")},
        round1_teacher_cost=costs,
        note="Shared R1 diagnosis/held-out spending is carried forward; arm caps cover new R2 purchases")
    path = destination / "round2.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("R1 preparation inputs changed after R2 was frozen")
        for relative, hashed in manifest["input_hashes"].items():
            if digest((destination / relative).read_bytes().hex()) != hashed:
                raise ValueError("Prepared R1 input changed: " + relative)
        return
    if any((destination / n).exists() for n in names + ["teacher_ledger.jsonl", "C", "D", "training_plan.json"]):
        raise ValueError("prepare-r2 requires a fresh run directory")
    for file in files:
        target = destination / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    # Preserve original call IDs and exact raw envelopes, sharing the same 8k
    # diagnosis/held-out/confirmation allowance rather than resetting it.
    ledger = read_rows(source / "teacher_ledger.jsonl")
    shared_ids = {r["call_id"] for r in ledger if r.get("bucket") == "shared"}
    for row in ledger:
        if row["call_id"] in shared_ids:
            append_row(destination / "teacher_ledger.jsonl", row)
    for call_id in shared_ids:
        for file in (source / "teacher_raw").glob(call_id + ".*"):
            target = destination / "teacher_raw" / file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    write_json(path, manifest)


def relation(context):
    """Shared interface-derived predicates; no observed failure is disclosed to C."""
    names = {f["name"] for f in context["functions"]}
    if any(n.startswith("archival_memory_") for n in names):
        return dict(kind="archival", positive="archival retrieval/search is needed",
                    negative="core memory or visible observations suffice")
    if "parallel" in context["category"]:
        return dict(kind="parallel", positive="several distinct calls are genuinely needed",
                    negative="only one call is necessary")
    if "lockDoors" in names:
        return dict(kind="prerequisite", positive="authorized door-lock prerequisite must be performed",
                    negative="door-lock prerequisite is unnecessary or unauthorized")
    return dict(kind="action", positive="a further tool action is required",
                negative="the request is complete or needs clarification; no further tool action")


def condition_side(row, rule):
    calls = row["demo"].get("calls", [])
    if rule["kind"] == "archival":
        archival = {"archival_memory_retrieve", "archival_memory_key_search", "archival_memory_search",
                    "archival_memory_list_keys"}
        if any(c["name"] in archival for c in calls):
            return "positive"
        if all(c["name"].startswith("core_memory_") for c in calls):
            return "negative"
        return None
    if rule["kind"] == "parallel":
        return "positive" if len(correct_call_set(row)) > 1 else "negative" if len(calls) == 1 else None
    if rule["kind"] == "prerequisite":
        return "positive" if any(c["name"] == "lockDoors" and c["arguments"].get("unlock") is False
                                 for c in calls) else "negative"
    return "positive" if calls else "negative"


def seed_id(row):
    return row["generation_group"].split(":", 1)[1]


def decision_key(row):
    return seed_id(row), row["variant"], correct_call_set(row)


def coverage(rows, seeds):
    result = {}
    for seed in seeds:
        sid, rule = seed["seed_id"], relation(seed["context"])
        directions = {}
        for direction in sorted(VARIANTS):
            own = [r for r in rows if seed_id(r) == sid and r["variant"] == direction]
            counts = Counter(condition_side(r, rule) for r in own)
            directions[direction] = dict(positive=counts["positive"], negative=counts["negative"],
                missing_sides=[s for s in ("positive", "negative") if not counts[s]])
        result[sid] = dict(relation=rule, directions=directions,
                          complete=all(not d["missing_sides"] for d in directions.values()))
    return result


def frozen_subset(args, seeds):
    source = getattr(args, "extend_from", None)
    if source is None:
        if (args.run_dir / "round2.json").exists():
            raise ValueError("R2 generation requires --extend-from for both arms")
        return []
    source = Path(source).resolve()
    if args.run_dir.resolve() == source.parent.parent or args.run_dir.resolve() in source.parents:
        raise ValueError("The frozen R1 pool must be external to the new R2 run directory")
    bank = read_json(source)
    if len(bank) > args.target_exercises:
        raise ValueError("Target cannot discard any frozen round-1 exercises")
    contexts = {s["seed_id"]: s["context"] for s in seeds}
    ids = set()
    for row in bank:
        sid = seed_id(row)
        if (row["id"] in ids or row["arm"] != args.arm or row["layer"] != 0 or sid not in contexts
                or row["task_id"] != contexts[sid]["task_id"]
                or row["parent_id"] != contexts[sid]["parent_id"]
                or row["source_frame"] != contexts[sid]["frame_id"]
                or row["source_hash"] != digest(contexts[sid])
                or row["functions"] != contexts[sid]["functions"]
                or not row.get("validation", {}).get("valid")):
            raise ValueError("Invalid frozen round-1 subset identity/validation")
        ids.add(row["id"])
    manifest = dict(path=str(source), hash=digest(bank), exercise_ids=[r["id"] for r in bank])
    round2 = args.run_dir / "round2.json"
    if round2.exists() and read_json(round2)["round1_banks"][args.arm] != dict(path=str(source), hash=digest(bank)):
        raise ValueError("Extension source differs from the prepared R1 pool")
    path = args.run_dir / args.arm / "extension.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Frozen round-1 exercise subset changed")
    write_json(path, manifest)
    return [dict(copy.deepcopy(row), origin="round1", round1_exercise_id=row["id"]) for row in bank]
