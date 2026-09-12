"""Frozen task selection with explicit parent and API-family isolation."""
from collections import Counter, defaultdict
import random
import re

from .common import ROOT, digest, read_json, setup_harness, write_json


def category(task_id):
    return task_id.rsplit("_", 1)[0]


def stratum(cat):
    if cat.startswith("memory"):
        return "memory"
    if cat.startswith("multi_turn"):
        return "multi_turn"
    if "irrelevance" in cat:
        return "irrelevance"
    if cat.startswith("live"):
        return "live"
    if cat.startswith("web_search"):
        return "web_search"
    return "non_live"


def parent_id(entry):
    tid = entry["id"]
    if tid.startswith("memory_"):
        # The write chain is shared by all questions in a persona/scenario,
        # including equivalent questions evaluated with another backend.
        scenario = entry.get("scenario") or (tid.split("-")[1] if "-" in tid else tid)
        return "memory_scenario:" + str(scenario)
    if entry.get("initial_config"):
        return "state:" + digest(entry["initial_config"])
    return tid


def families(entry):
    if entry.get("involved_classes"):
        return {"executor:" + c for c in entry["involved_classes"]}
    # Namespace families for dotted APIs; normalize numbered live API names.
    return {"api:" + re.sub(r"_\d+(?=_|$)", "", f["name"]).split(".")[0]
            for f in entry.get("function", [])}


def inventory():
    setup_harness()
    from bfas.adapters.bfcl import BFCLAdapter
    adapter = BFCLAdapter()
    entries, _ = adapter._load_entries()
    if not all(any(t.startswith("memory_" + b + "_") for t in entries)
               for b in ("kv", "vector", "rec_sum")):
        raise RuntimeError("Official memory expansion failed; refusing a memory-free split")
    return adapter, {t: e for t, e in entries.items() if t not in adapter._prereq_ids}


def stratified(ids, entries, count, rng, weights):
    bins = defaultdict(list)
    for tid in sorted(ids):
        bins[category(tid)].append(tid)
    for values in bins.values():
        rng.shuffle(values)
    picked, group_counts, cat_counts = [], Counter(), Counter()
    while len(picked) < count:
        available = [c for c, b in bins.items() if b]
        if not available:
            raise ValueError(f"Only {len(picked)} eligible tasks for requested {count}")
        cat = min(available, key=lambda c: (
            group_counts[stratum(c)] / weights.get(stratum(c), 1), cat_counts[c], c))
        picked.append(bins[cat].pop())
        group_counts[stratum(cat)] += 1
        cat_counts[cat] += 1
    return picked


def make_splits(entries, source, seed=0, support_n=24, calibration_n=16, evaluation_n=256):
    rng = random.Random(seed)
    source_ids = set(source.get("support", [])) | set(source.get("demand", [])) | set(source.get("calibration", []))
    if source_ids - entries.keys():
        raise ValueError(f"Source IDs missing from official inventory: {sorted(source_ids - entries.keys())}")
    # Reserve one memory executor family and web search for full-task transfer.
    # Otherwise the 50-source pool touches every memory API, leaving no
    # possible family-disjoint memory evaluation. This is fixed before rollout.
    candidates = {t for t in source_ids if not t.startswith(("memory_rec_sum_", "web_search_"))}
    weights = dict(memory=4, multi_turn=5, irrelevance=4, live=3, non_live=2)
    # Keep complete parent groups on one side of support/calibration. Search
    # only seeded task allocations, never scores or student behaviour.
    for _ in range(2000):
        reserved = {parent_id(entries[rng.choice(sorted(t for t in candidates
                    if stratum(category(t)) == s))]) for s in weights
                    if any(stratum(category(t)) == s for t in candidates)}
        support_pool = {t for t in candidates if parent_id(entries[t]) not in reserved}
        if len(support_pool) < support_n:
            continue
        support = stratified(support_pool, entries, support_n, rng, weights)
        parents = {parent_id(entries[t]) for t in support}
        remaining = {t for t in candidates - set(support) if parent_id(entries[t]) not in parents}
        if len(remaining) < calibration_n:
            continue
        calibration = stratified(remaining, entries, calibration_n, rng, weights)
        if all(any(stratum(category(t)) == s for t in calibration)
               for s in ("memory", "multi_turn", "irrelevance", "live", "non_live")):
            break
    else:
        raise ValueError("Cannot form parent-disjoint support/calibration with required coverage")
    used = support + calibration
    excluded_parents = {parent_id(entries[t]) for t in used}
    excluded_families = set().union(*(families(entries[t]) for t in used))
    eligible = {t for t, e in entries.items() if t not in source_ids
                and parent_id(e) not in excluded_parents and not families(e) & excluded_families}
    evaluation = stratified(eligible, entries, evaluation_n, rng,
                            dict(memory=4, multi_turn=6, irrelevance=5, live=6, non_live=4, web_search=1))
    groups = dict(support=sorted(support), calibration=sorted(calibration), evaluation=sorted(evaluation))
    if evaluation_n >= 24 and {stratum(category(t)) for t in evaluation} != {
            "memory", "multi_turn", "irrelevance", "live", "non_live", "web_search"}:
        raise ValueError("Function-family exclusion left an evaluation stratum empty")
    return dict(version=1, seed=seed, **groups, source_sha256=digest(source),
                label="BFCL mechanism-validation subset; not official Overall",
                source_pool_ids=sorted(source_ids),
                family_rule="executor class; stateless normalized API namespace/name",
                parent_rule="memory persona across backends; initial-config hash; otherwise official task ID",
                reserved_evaluation_families=["MemoryAPI_rec_sum", "WebSearchAPI"],
                excluded_families=sorted(excluded_families),
                items={t: dict(category=category(t), parent_id=parent_id(entries[t]),
                               families=sorted(families(entries[t])), source_hash=digest(entries[t]))
                       for t in sorted(set(used + evaluation))},
                counts={g: dict(Counter(category(t) for t in ids)) for g, ids in groups.items()},
                eligible_evaluation_counts=dict(Counter(category(t) for t in eligible)),
                historical_base_context=dict(Overall=45.6, NL=82, Live=80, MT=53, Memory=30, Irrel=75))


def run(args):
    _, entries = inventory()
    result = make_splits(entries, read_json(args.source_split), args.seed)
    if args.splits.exists() and read_json(args.splits) != result:
        raise ValueError("Frozen split already exists with different contents; choose a new --splits path")
    write_json(args.splits, result)
    print(args.splits)
