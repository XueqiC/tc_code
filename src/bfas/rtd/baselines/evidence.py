"""Read-only final-ledger admission and same-pool export in the real C25 formats.

Never call Ledger.resume/ComputeJournal on a source: those repair torn files.
Only already revealed packages are unsealed. Repeated paid queries retain their
empirical multiplicity; aliases within a query/state/text do not add mass.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re

from ..broker import SealedReplayBroker
from ..hardware import bound_hardware
from ..identity import verified_checkpoint
from ..ledger import Ledger
from ..persistence import atomic_json, digest, file_hash
from ..transport import Behavior, FullState


def read_chain(path, *, teacher=False):
    raw = Path(path).read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"missing/torn journal (read-only, no recovery): {path}")
    events = [json.loads(line) for line in raw.splitlines()]
    key = "event_hash" if teacher else "hash"
    for i, event in enumerate(events):
        if (event["sequence"] != i or event.get("previous_hash") != (events[i-1][key] if i else None)
                or event[key] != digest({k: v for k, v in event.items() if k != key})):
            raise ValueError(f"journal hash/sequence mismatch: {path}")
    return events


def readonly_ledger(path, budget):
    ledger = Ledger(budget)  # no path: transition validation has no filesystem effects
    for event in read_chain(path, teacher=True):
        q, kind = event["query_id"], event["kind"]
        if kind == "reserve":
            ledger.reserve(q, event["cap"])
        elif kind == "release":
            if q not in ledger.reservations:
                raise ValueError("release without reservation")
            ledger.release(q)
        elif kind == "authorize":
            ledger.authorize(event["budget"])
        elif kind == "reveal":
            ledger.settle(q, event["cost"], confidence=event["confidence"], usage=event["usage"],
                          dependencies=event["dependencies"])
        else:
            raise ValueError("unknown teacher event")
        if len(ledger.events) != event["sequence"] + 1:
            raise ValueError("duplicate teacher transition")
        ledger.events[-1] = event
    return ledger


def extract_feedback(steps, events, parents, config):
    """Bind successful reference/actual groups, including explicit reused actual.

    Failed/repeated attempts remain in source compute accounting, but only the
    last complete, hash-bound group at a committed window supplies its quota.
    No rewards or old actions are exported as baseline on-policy training data.
    """
    by_seq = {e["sequence"]: e for e in events}
    groups = []
    for row in steps:
        if not row["decision"]:
            continue
        r, step = row["round"], row["step"]
        trunc = row.get("truncation", {})
        reused = trunc.get("actual_feedback_reused")
        if type(reused) is not bool:
            raise ValueError("missing explicit feedback reuse record")
        matches = lambda e: e.get("round") == r and e.get("step") == step
        window_groups = []
        for role in ("reference_feedback", "actual_feedback"):
            if role == "actual_feedback" and reused:
                reuse_events = [e for e in events if matches(e) and e["kind"] == "feedback_reused"]
                if not reuse_events or reuse_events[-1]["parameter_hash"] != row["actual_hash"]:
                    raise ValueError("missing/mismatched actual feedback reuse hash")
                window_groups.append(dict(window_groups[0], role=role, reused=True))
                continue
            returns = [e for e in events if matches(e) and e["kind"] == "return_gradient" and e["role"] == role]
            if not returns:
                raise ValueError("missing feedback group")
            result = returns[-1]
            if role == "actual_feedback" or reused:
                if result["parameter_hash"] != row["actual_hash"]:
                    raise ValueError("feedback gradient does not bind committed policy")
            ends = []
            for e in events[:result["sequence"]]:
                if e["kind"] != "compute_end" or e["operation"] != role or e["status"] != "complete":
                    continue
                begin = by_seq[e["begin_sequence"]]
                if matches(begin):
                    ends.append((begin, e))
            if not ends:
                raise ValueError("feedback lacks a completed compute scope")
            begin, end = ends[-1]
            rollouts = [e for e in events[begin["sequence"]:end["sequence"]]
                        if matches(e) and e["kind"] == "feedback_rollout" and e["role"] == role]
            counts = Counter(e["parent_hash"] for e in rollouts)
            expected = trunc["reference" if role == "reference_feedback" else "actual"]["rollouts"]
            if len(rollouts) != expected or result["metadata"]["rollouts"] != expected or not counts:
                raise ValueError("feedback rollout quota is incomplete")
            singles = multis = 0
            for parent, count in counts.items():
                if parent not in parents or parents[parent]["fold"] == (r-1) % 2:
                    raise ValueError("feedback parent outside the clean rotating fold")
                multi = parents[parent]["category"].startswith("multi_turn")
                singles += not multi
                multis += multi
                wanted = config["rollouts_multi_turn" if multi else "rollouts_per_meta_task"]
                if count != wanted:
                    raise ValueError("feedback per-parent count differs from RTD config")
            legal = [p for p in parents.values() if p["runnable"] and p["fold"] != (r-1)%2]
            available_multi = sum(p["category"].startswith("multi_turn") for p in legal)
            if (singles != min(config["meta_tasks_per_feedback"], len(legal)-available_multi)
                    or multis != min(config["meta_tasks_multi_turn"], available_multi)):
                raise ValueError("feedback task quota differs from RTD config/fold availability")
            policies = {e["rollout"]["policy_id"] for e in rollouts}
            if len(policies) != 1:
                raise ValueError("mixed sampling policies in source feedback group")
            tokens = sum(len(a["prompt_ids"]) + len(a["action_ids"])
                         for e in rollouts for a in e["rollout"]["actions"])
            window_groups.append(dict(round=r, step=step, slot_offset=(step-1)*8, role=role,
                reused=False, tasks=[dict(parent_hash=p, rollout_count=c) for p, c in counts.items()],
                source_parameter_hash=result["parameter_hash"], source_return_sequence=result["sequence"],
                score_input_tokens=tokens))
        if window_groups[0]["tasks"] != window_groups[1]["tasks"]:
            raise ValueError("reference/actual feedback parents differ")
        groups.extend(window_groups)
    if len(groups) != 24:
        raise ValueError("twelve complete reference/actual windows required")
    return groups


def action_ids(tokenizer, text):
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not ids or ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    if tokenizer.eos_token_id in ids[:-1]:
        raise ValueError("teacher action contains an internal termination token")
    return ids


def prepare(source_run, bank, out, *, root, expected_hardware_hash, tokenizer=None, support=None):
    """Export a FINISHED real RTD run, without opening any source for writing.

    ``tokenizer``/``support`` injection is for synthetic offline tests. Production
    uses the source's local tokenizer and BFCLSupport's public initial states.
    V-T source lengths are refreshed from each baseline's own frozen policy at
    round start; only legal teacher costs and source-run budget targets are
    computed here. There is no claim that independently refreshed texts match.
    """
    source_run, bank, root, out = map(Path, (source_run, bank, root, out))
    if any(out.resolve().is_relative_to(p.resolve()) for p in (source_run, bank)):
        raise ValueError("prepare output must be outside the read-only source run and bank")
    if out.exists():
        raise FileExistsError(f"prepare output already exists: {out}")
    if not expected_hardware_hash:
        raise ValueError("declare the common source/training hardware class hash")
    names = ["manifest.json", "teacher.jsonl", "trajectory.json", "audit.json", "compute.jsonl",
             *[f"round-{r}/checkpoint.json" for r in (1, 2, 3)]]
    hashes = {name: file_hash(source_run/name) for name in names}
    manifest = json.loads((source_run/"manifest.json").read_text())
    trajectory = json.loads((source_run/"trajectory.json").read_text())
    audit = json.loads((source_run/"audit.json").read_text())
    if manifest.get("smoke") or manifest.get("arm") not in {"R0", "R1"}:
        raise ValueError("completed full R0/R1 source required")
    if digest(manifest["config"]) != manifest["config_hash"]:
        raise ValueError("source effective config hash mismatch")
    hardware = bound_hardware(root, manifest)
    if digest(hardware["hard"]) != expected_hardware_hash:
        raise ValueError("source hardware class mismatch")
    from ..cli import data_identity
    if data_identity(root, manifest["config"], bank) != manifest["data_hash"]:
        raise ValueError("bank/support/official data differs from source manifest")
    ledger = readonly_ledger(source_run/"teacher.jsonl", manifest["budget_ceilings"][0])
    if ledger.reservations:
        raise ValueError("pending reservations in final ledger")
    steps = trajectory["steps"]
    expected_steps = [(r, s) for r in (1, 2, 3) for s in range(1, 13)]
    if [(e["round"], e["step"]) for e in steps] != expected_steps:
        raise ValueError("incomplete committed schedule: need three rounds / 36 steps")
    for e in steps:
        if (e["decision"] != (e["step"] in (1, 4, 7, 10)) or e["inner_fold"] != (e["round"]-1) % 2
                or not e["audit_passed"]):
            raise ValueError("invalid committed window/fold audit")
    checkpoints = [verified_checkpoint(source_run, manifest, r) for r in (1, 2, 3)]
    if trajectory["checkpoints"] != checkpoints:
        raise ValueError("trajectory/checkpoint mismatch")
    final = checkpoints[-1]
    if (set(final["owned"]) != ledger.owned_ids or len(final["owned"]) != len(ledger.owned_ids)
            or final["actual_spend"] != ledger.spent or steps[-1]["actual_spend"] != ledger.spent
            or final["authorized_budget"] != ledger.budget or steps[-1]["actual_hash"] != final["parameter_hash"]):
        raise ValueError("final ledger/trajectory/checkpoint owned, spend or policy mismatch")
    if (audit.get("passed") is not True or audit.get("steps") != 36 or audit.get("decision_windows") != 12
            or audit.get("packages") != len(ledger.owned_ids) or audit.get("actual_spend") != ledger.spent
            or audit.get("authorized_budget") != ledger.budget):
        raise ValueError("missing final completion audit")
    for document in (audit, trajectory, final):
        if any(document.get(key) for key in ("pending", "unmerged", "reservations")):
            raise ValueError("pending/unmerged final evidence")
    from ..experiment import BFCLSupport
    support = support if support is not None else BFCLSupport(root, manifest["config"])
    parents = {h: dict(official_id=tid, fold=int(h, 16) % 2,
                      category=support.categories.get(tid, "unavailable"), runnable=h in support.states)
               for h, tid in support.parents.items()}
    compute = read_chain(source_run/"compute.jsonl")
    feedback = extract_feedback(steps, compute, parents, manifest["config"])
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest["model_path"], local_files_only=True, trust_remote_code=False)
    bank_hashes = {name: file_hash(bank/name) for name in ("public/requests.json", "sealed/integrity.json")}
    public = {raw["spec"]["query_id"]: raw for raw in json.loads((bank/"public/requests.json").read_text())}
    broker = SealedReplayBroker(bank, ledger, inner_parent_hashes=set(parents))
    states = {s.state_hash: dict(state=asdict(s), teachers=[]) for s in support.states.values()}
    charges, reveals = [], [e for e in ledger.events if e["kind"] == "reveal"]
    for event in reveals:
        q = event["query_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", q):
            raise ValueError("invalid query hash")
        package = broker.acquire(q)  # in-memory ledger already owns q: no reserve/reveal write
        record = public[q]
        if record["spec"]["L"] != 1:
            raise ValueError("C27 replay-L1 requires recorded L=1 requests")
        if record["parent_hash"] not in support.states:
            raise ValueError("purchased teacher parent has no supported full-state feedback adapter")
        charges.append(dict(query_id=q, dependencies=list(package.dependencies), cost=package.cost,
                            confidence=package.cost_confidence, usage=package.usage,
                            provenance=package.provenance, payload_hash=file_hash(bank/"sealed"/(q+".json"))))
        seen = set()
        for teacher in package.behaviors:
            s, text = teacher.state, teacher.text
            h = s.state_hash
            if h in states:
                FullState(**states[h]["state"]).assert_matches(s)
            item = states.setdefault(h, dict(state=asdict(s), teachers=[]))
            if (h, text) not in seen:
                item["teachers"].append(dict(query_id=q, text=text, teacher_hash=digest(text),
                    action_tokens=len(action_ids(tokenizer, text))))
                seen.add((h, text))
    for item in states.values():
        item["prompt_tokens"] = len(tokenizer.encode(item["state"]["prompt"], add_special_tokens=False))
        if item["prompt_tokens"] <= 0:
            raise ValueError("missing tokenizer cost directory")
    update_tokens, exposure = [], []
    for r in (1, 2, 3):
        costs = Counter()
        for e in steps:
            if e["round"] != r:
                continue
            for branch in ("old_exposure", "new_exposure"):
                for key in ("source_action_tokens", "teacher_action_tokens", "prompt_tokens", "teacher_slots"):
                    value = e.get(branch, {}).get(key)
                    if type(value) is not int or value < 0:
                        raise ValueError("missing/invalid final token cost directory")
                    costs[key] += value
        update_tokens.append(sum(costs[k] for k in ("source_action_tokens", "teacher_action_tokens", "prompt_tokens")))
        exposure.append(dict(costs))
    feedback_tokens = [sum(g["score_input_tokens"] for g in feedback if g["round"] == r and not g["reused"])
                       for r in (1, 2, 3)]
    begins = {e["sequence"]: e for e in compute if e["kind"] == "compute_begin"}
    outer = [e for e in compute if e["kind"] == "compute_end" and
             begins[e["begin_sequence"]].get("parent_sequence") is None]
    evidence = dict(version="c27-evidence-v1", source_run=str(source_run.resolve()), source_manifest=manifest,
        source_file_hashes=hashes, bank_path=str(bank.resolve()), bank_file_hashes=bank_hashes,
        hardware=hardware, hardware_class_hash=expected_hardware_hash,
        ledger_events=ledger.events, ledger_event_digest=digest(ledger.events),
        ledger_last_sequence=ledger.events[-1]["sequence"], ledger_last_hash=ledger.events[-1]["event_hash"],
        owned=sorted(ledger.owned_ids), charges=charges, parents=parents, states=states, feedback=feedback,
        budgets=dict(teacher_tokens=ledger.spent, authorized_cap=ledger.budget, slots=288,
            slots_per_round=[96, 96, 96], update_tokens=update_tokens,
            update_target_basis="committed RTD old+new full-input exposure, excluding virtual/pilot/VJP",
            pg_reserved_tokens=feedback_tokens, source_exposure=exposure,
            source_compute_outer_gpu_seconds=sum(e["gpu_seconds"] for e in outer),
            source_compute_failed_gpu_seconds=sum(e["gpu_seconds"] for e in outer if e["status"] != "complete")),
        source_policy="each trainer refreshes two independent sources per legal state each round",
        checkpoint_meaning="final purchased collection available from theta0; training progress only")
    if any(file_hash(source_run/name) != value for name, value in hashes.items()):
        raise ValueError("source changed during prepare; no export published")
    if any(file_hash(bank/name) != value for name, value in bank_hashes.items()):
        raise ValueError("bank changed during prepare")
    if data_identity(root, manifest["config"], bank) != manifest["data_hash"]:
        raise ValueError("data changed during prepare")
    atomic_json(out, evidence)
    return evidence


def load_evidence(path, expected_hash):
    if file_hash(path) != expected_hash:
        raise ValueError("prepared evidence hash mismatch")
    evidence = json.loads(Path(path).read_text())
    if evidence.get("version") != "c27-evidence-v1":
        raise ValueError("unsupported evidence export")
    return evidence
