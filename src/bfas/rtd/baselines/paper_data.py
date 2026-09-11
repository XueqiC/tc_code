"""Sealed-pool purchase and deployment text for the four paper adaptations.

The cost reader is privileged replay infrastructure. Methods only receive the
positive rows of purchased packages, never unpurchased responses or metadata.
"""
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import random
import re

from ..bank_build import validate_state_certificate
from ..persistence import digest, file_hash

STUDENT = "google/gemma-4-12B-it"
TEACHER = "gpt-5.6-luna"
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


def frozen_purchase(package_ids, cost_of, denominator, fraction):
    """Seed-zero prefix, actual charges, half-up cap; never skip a blocker.

    Includes unavailable/failed attempts. Actual costs are inspected by the
    sealed replay accountant, not treated as free acquisition features.
    """
    fraction = Decimal(str(fraction))
    if not fraction.is_finite() or not 0 < fraction <= 1:
        raise ValueError("budget fraction must be in (0, 1]")
    if type(denominator) is not int or denominator <= 0:
        raise ValueError("positive usable cost basis required")
    order = sorted(package_ids)
    if len(set(order)) != len(order):
        raise ValueError("duplicate package IDs")
    random.Random(0).shuffle(order)
    exact = fraction * denominator
    cap = int(exact.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    charges, spent, blocked = [], 0, None
    for qid in order:
        cost = cost_of(qid)
        if type(cost) is not int or cost < 0:
            raise ValueError("nonnegative integer recorded cost required")
        if spent + cost > cap:
            blocked = dict(package_id=qid, tokens=cost)
            break
        charges.append(dict(package_id=qid, tokens=cost))
        spent += cost
    return dict(purchase_seed=0, purchase_order=order, purchase_order_hash=digest(order),
        purchase_rule="sorted query IDs; random.Random(0).shuffle; stop before first overflow",
        budget_fraction=str(fraction), usable_cost_basis=denominator,
        B_exact=str(exact), B=cap, rounding="positive_integer_half_up",
        purchased_package_ids=[r["package_id"] for r in charges], charges=charges,
        teacher_tokens_charged=spent, remaining_tokens=cap-spent, blocked_next=blocked,
        cost_scope="sealed cached-content access including purchased unavailable attempts",
        new_teacher_calls=0, new_teacher_tokens=0)


@dataclass(frozen=True)
class TeacherRow:
    package_id: str
    task_id: str
    parent_hash: str
    index: int
    prompt: str
    target: str
    benchmark: str
    final_step: bool = False


def deployed_row(row):
    """Preserve the current Gemma handler's exact prompt/continuation boundary.

    Unlike the historical Qwen baseline rendering in the appendix, Gemma's
    current template includes EMPTY_THOUGHT in its generation prompt. It stays
    masked, exactly as in the Luna RTD banks and official Gemma handlers.
    """
    return row


def load_purchased(bank, benchmark, fraction):
    bank = Path(bank)
    if not bank.is_absolute():
        raise ValueError("--bank must be an absolute path")
    cert = validate_state_certificate(bank, benchmark=benchmark, student=STUDENT)
    inventory = json.loads((bank / "public/requests.json").read_text())
    requests = {r["spec"]["query_id"]: r for r in inventory}
    if len(requests) != len(inventory):
        raise ValueError("duplicate inventory ID")
    integrity = json.loads((bank / "sealed/integrity.json").read_text())

    def payload(qid):
        if not re.fullmatch(r"[0-9a-f]{64}", qid):
            raise ValueError("invalid sealed package ID")
        path = bank / "sealed" / (qid + ".json")
        value = json.loads(path.read_text())
        from ...cc_pairs import digest as payload_digest
        if payload_digest(value) != integrity[qid]:
            raise ValueError("sealed package integrity mismatch")
        return value

    purchase = frozen_purchase(requests, lambda q: payload(q)["cost"],
                               cert["core"]["budget_denominator"], fraction)
    rows = []
    for charge in purchase["charges"]:
        qid = charge["package_id"]
        record, p = requests[qid], payload(qid)
        provenance = p.get("provenance", {})
        historical = p.get("historical_response", {})
        if not isinstance(historical, dict):
            historical = {}
        teacher = provenance.get("teacher", historical.get("teacher", ""))
        if teacher not in {TEACHER, "openai/" + TEACHER, "openai/" + TEACHER + "-FC"}:
            raise ValueError("purchased attempt is not from the Luna teacher")
        if provenance.get("rendering_student", STUDENT) != STUDENT:
            raise ValueError("purchased package uses another student rendering")
        if p.get("cost_confidence") not in {"exact", "estimated"}:
            raise ValueError("unknown cost confidence")
        # A bank exclusion stays excluded even if its historical demo succeeded.
        usable = record["unavailable_reason"] is None
        charge.update(cost_confidence=p["cost_confidence"], teacher=teacher,
            usable=usable, unavailable_reason=record["unavailable_reason"],
            package_digest=integrity[qid], task_id=provenance.get("task_id"),
            attempt_index=provenance.get("attempt_index", historical.get("attempt_index")))
        if not usable:
            continue
        if not set(record["dependencies"]) <= set(purchase["purchased_package_ids"]):
            raise ValueError("purchased teacher record lacks an owned prefix dependency")
        behaviors = p.get("behaviors", [])
        if not behaviors:
            raise ValueError("usable package has no teacher trajectory")
        if historical.get("verified") is False or p.get("success") is False:
            raise ValueError("unverified positive package")
        for index, behavior in enumerate(behaviors):
            state = behavior["state"]
            if state["parent_hash"] != record["parent_hash"]:
                raise ValueError("teacher row outside package parent")
            prompt, target = state["prompt"], behavior["text"]
            if not prompt.startswith("<bos>") or "<|turn>model\n" not in prompt:
                raise ValueError("bank lacks native Gemma chat rendering")
            if not target.strip():
                raise ValueError("empty teacher target")
            rows.append(deployed_row(TeacherRow(qid, provenance["task_id"],
                state["parent_hash"], index, prompt, target, benchmark,
                final_step=index == len(behaviors)-1)))
    purchase.update(bank=str(bank.resolve()), certificate_sha256=file_hash(bank/"public/cap_certificate.json"),
        certificate_core=cert["core"], positive_rows=len(rows),
        purchased_usable_packages=sum(c["usable"] for c in purchase["charges"]),
        rendering="exact native RTD bank prompt/target boundary; template prefill and observations masked")
    return purchase, rows


def select_smartad(rows, nll):
    """One purchased verified trajectory per exact task, scored at base student.

    nll(row) returns (total NLL, generated token count). Pool all generated
    tokens so differing trajectory lengths do not bias the selection.
    """
    packages = {}
    for row in rows:
        packages.setdefault((row.task_id, row.package_id), []).append(row)
    selected, scores = {}, {}
    for (task, qid), trajectory in sorted(packages.items()):
        values = [nll(row) for row in trajectory]
        score = sum(v[0] for v in values) / sum(v[1] for v in values)
        if not math.isfinite(score):
            raise ValueError("nonfinite base-student NLL")
        scores[qid] = score
        if task not in selected or (score, qid) < selected[task]:
            selected[task] = (score, qid)
    ids = {v[1] for v in selected.values()}
    return [row for row in rows if row.package_id in ids], dict(
        base_student_mean_nll=scores, selected_package_ids=sorted(ids),
        selection="minimum base-student generated-token mean NLL per task; package ID tie break")


def first_thought(rows):
    """Deterministic retrospective FTP, only on the first turn of each episode."""
    trajectories = {}
    for row in rows:
        trajectories.setdefault(row.package_id, []).append(row)
    result = []
    for trajectory in trajectories.values():
        first = min(trajectory, key=lambda r: r.index)
        actions = [r.target.removeprefix(EMPTY_THOUGHT) for r in trajectory]
        # Extract rather than infer hidden teacher reasoning. Limit the derived
        # summary to 40 words; no generated labels or extra model/teacher call.
        summary = " ".join(" ".join(actions).split()[:40])
        summary = re.sub(r"<[^>]*>", "", summary)
        thought = "THOUGHT: The recorded solution starts with: " + summary + "\n"
        for row in trajectory:
            if row is first:
                suffix = row.target.removeprefix(EMPTY_THOUGHT)
                if row.benchmark == "alfworld" and not re.search(r"(?im)^ACTION:", suffix):
                    suffix = "ACTION: " + suffix
                row = replace(row, target=thought + suffix)
            result.append(row)
    return result
