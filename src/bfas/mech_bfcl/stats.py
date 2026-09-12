"""Strict pairing and paired parent-cluster intervals; no variant pseudoreplication."""
from collections import Counter, defaultdict
import random
import re

from .common import read_json, training_directory, variant_name, write_json
from .teacher import ledger_summary


def pair(left, right):
    def index(rows):
        result = {}
        for row in rows:
            key = (row["layer"], row["id"])
            if key in result:
                raise ValueError(f"Duplicate paired item {key}")
            result[key] = row
        return result
    a, b = index(left), index(right)
    if a.keys() != b.keys():
        raise ValueError("Incomplete pairing: models did not evaluate identical item sets")
    output = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if x["parent_id"] != y["parent_id"] or x["category"] != y["category"]:
            raise ValueError("Paired parent/category mismatch")
        if type(x["correct"]) is not bool or type(y["correct"]) is not bool:
            raise ValueError("Unscored results cannot be treated as failures")
        output.append(dict(id=key[1], layer=key[0], parent_id=x["parent_id"], category=x["category"],
            left_correct=x["correct"], right_correct=y["correct"],
            outcome=f"{int(x['correct'])}{int(y['correct'])}",
            delta=int(y["correct"])-int(x["correct"]),
            left_metrics=x.get("metrics", {}), right_metrics=y.get("metrics", {})))
    return output


def paired_interval(paired, seed=0, samples=4000):
    groups = defaultdict(list)
    for row in paired:
        groups[row["parent_id"]].append(row["delta"])
    if not groups:
        raise ValueError("No paired data")
    means = [sum(v)/len(v) for _, v in sorted(groups.items())]
    estimate = sum(means)/len(means)
    if len(means) < 2:
        interval = None  # One parent is not enough to estimate task uncertainty.
    else:
        rng = random.Random(seed)
        boot = sorted(sum(rng.choices(means, k=len(means)))/len(means) for _ in range(samples))
        interval = [boot[int(.025*(samples-1))], boot[int(.975*(samples-1))]]
    leave_one_out = [(sum(means)-v)/(len(means)-1) for v in means] if len(means)>1 else []
    return dict(delta=estimate, ci95=interval, independent_parents=len(means), items=len(paired),
        item_weighted_delta=sum(r["delta"] for r in paired)/len(paired),
        outcomes=dict(Counter(r["outcome"] for r in paired if "outcome" in r)),
        mean_absolute_item_delta=sum(abs(r["delta"]) for r in paired)/len(paired),
        method="equal-parent mean; paired parent-cluster percentile bootstrap",
        positive_parents=sum(m>0 for m in means), negative_parents=sum(m<0 for m in means),
        leave_one_parent_out_min=min(leave_one_out) if leave_one_out else None)


def evaluation_variants(directory, splits):
    """Discover completed main evaluations; the unsuffixed arm is the split seed."""
    variants = {}
    for path in (directory / "evaluation").glob("*/main/items.json"):
        name = path.parent.parent.name
        match = re.fullmatch(r"(base|C|D)(?:-s(-?\d+))?", name)
        if match is None:
            continue
        arm, suffix = match.groups()
        seed = None if arm == "base" else splits["seed"] if suffix is None else int(suffix)
        if name != variant_name(arm, seed, splits["seed"]):
            raise ValueError(f"Noncanonical evaluation variant directory: {name}")
        variants[name] = (arm, seed)
    return dict(sorted(variants.items(), key=lambda item: (
        item[0] not in ("base", "C", "D"), ("base", "C", "D").index(item[1][0]), item[1][1] or 0)))


def pool_seed_pairs(left, right):
    """Average correctness per question over matched seeds before parent pairing."""
    if not left or left.keys() != right.keys():
        raise ValueError("Seed pooling requires identical nonempty training seed sets")
    seeds = sorted(left)
    reference = left[seeds[0]]
    paired = []
    for seed in seeds:
        pair(reference, left[seed])  # Validate IDs, parents, categories and boolean scores across seeds.
        paired.append(pair(left[seed], right[seed]))
    rows = []
    for question in zip(*paired):
        first = question[0]
        a = sum(r["left_correct"] for r in question) / len(seeds)
        b = sum(r["right_correct"] for r in question) / len(seeds)
        rows.append(dict(id=first["id"], layer=first["layer"], parent_id=first["parent_id"],
                         category=first["category"], left_mean_correct=a, right_mean_correct=b,
                         delta=b-a, train_seeds=seeds))
    return rows


def refresh_pairs(directory, splits):
    from .pipeline import layer3_selection
    ids, scope = layer3_selection(splits)
    models, protocols = {}, []
    variants = evaluation_variants(directory, splits)
    for name, (arm, seed) in variants.items():
        path = directory / "evaluation" / name / "main/items.json"
        models[name] = read_json(path)
        full = [r for r in models[name] if r["layer"] == 3]
        if len(full) != len(ids) or {r["id"] for r in full} != set(ids):
            raise ValueError(f"{name}: layer-3 items do not match the evaluation selection")
        if any(r.get("evaluation_scope") != scope for r in full):
            raise ValueError(f"{name}: layer-3 evaluation scope metadata mismatch")
        pair(next(iter(models.values())), models[name])
        if path.with_name("protocol.json").exists():
            protocol = read_json(path.with_name("protocol.json"))
            if protocol.get("train_seed", None if arm == "base" else splits["seed"]) != seed:
                raise ValueError(f"{name}: evaluation training seed mismatch")
            protocols.append({k: protocol[k] for k in
                              ("split_hash", "heldout_hash", "temperature", "top_k", "seed")})
    if protocols and any(p != protocols[0] for p in protocols):
        raise ValueError("Evaluations must share the same split, held-out sets and decoding settings")
    summary = {}

    def record(name, rows, seeds=None):
        write_json(directory / "paired" / (name+".json"), rows)
        summary[name] = {str(layer): paired_interval([r for r in rows if r["layer"] == layer])
                         for layer in (1, 2, 3)}
        summary[name]["3"]["evaluation_scope"] = scope
        if seeds is not None:
            for stat in summary[name].values():
                stat.update(train_seeds=seeds, method="mean per question over training seeds; " + stat["method"])

    by_arm = {arm: {seed: models[name] for name, (a, seed) in variants.items() if a == arm}
              for arm in ("C", "D")}
    for name, (arm, seed) in variants.items():
        if arm == "base":
            continue
        label = arm if seed == splits["seed"] else f"{arm}_s{seed}"
        if "base" in models:
            record("base_vs_" + label, pair(models["base"], models[name]))
        if seed != splits["seed"] and arm in models:
            record(f"{arm}_s{splits['seed']}_vs_{arm}_s{seed}", pair(models[arm], models[name]))
    common_seeds = sorted(by_arm["C"].keys() & by_arm["D"].keys())
    for seed in common_seeds:
        name = "C_vs_D" if seed == splits["seed"] else f"C_s{seed}_vs_D_s{seed}"
        record(name, pair(by_arm["C"][seed], by_arm["D"][seed]))
    if len(common_seeds) > 1:
        record("C_vs_D_pooled", pool_seed_pairs(
            {s: by_arm["C"][s] for s in common_seeds}, {s: by_arm["D"][s] for s in common_seeds}),
            seeds=common_seeds)
    write_json(directory / "paired/intervals.json", summary)
    return models, summary


def report(args, splits):
    from .pipeline import layer3_selection
    models, intervals = refresh_pairs(args.run_dir, splits)
    if not {"base", "C", "D"} <= models.keys():
        raise ValueError("Report requires complete base/C/D evaluations at all three layers")
    rows = []
    _, scope = layer3_selection(splits)
    cost = ledger_summary(args.run_dir / "teacher_ledger.jsonl")
    lines = ["# BFCL first-round mechanism validation", "",
             "Student: google/gemma-4-12B-it; teacher: official OpenAI gpt-5.6-luna, requested Flex.",
             "These are mechanism-validation subsets, not BFCL official Overall. Historical base context: "
             "Overall 45.6; NL 82 / Live 80 / MT 53 / Memory 30 / Irrel 75.", "",
             f"Layer 3 covers {scope['evaluated_questions']} of the {scope['subset_questions']} subset questions "
             "for every arm and paired comparison. The frozen split and split hash are unchanged.",
             f"Excluded categories: {', '.join(scope['excluded_categories'])}. {scope['exclusion_reason']}",
             f"Excluded IDs: {', '.join(scope['excluded_ids']) or 'none'}.", "",
             "| Arm | Local (%) | Natural (%) | Full tasks (%) | Generated output tokens | Passes | Learning rate | Total supervised tokens | Optimizer steps | Train seconds | Train seed |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    breakdown = {}
    variants = evaluation_variants(args.run_dir, splits)
    for name, items in models.items():
        arm, seed = variants[name]
        accuracy = {str(layer): sum(i["correct"] for i in items if i["layer"] == layer) /
                    sum(i["layer"] == layer for i in items) for layer in (1, 2, 3)}
        metrics = read_json(training_directory(args.run_dir, arm, seed, splits["seed"]) / "metrics.json") if arm != "base" else {}
        if arm != "base" and metrics.get("train_seed", splits["seed"]) != seed:
            raise ValueError(f"{name}: training metrics seed mismatch")
        training = dict(accuracy=accuracy, **metrics)
        rows.append(dict(training, arm=arm, train_seed=seed, variant=name))
        lines.append(f"| {name} | " + " | ".join(f"{100*accuracy[str(l)]:.2f}" for l in (1,2,3)) +
            f" | {cost.get(arm, {}).get('output_tokens', 0)} | {metrics.get('passes', '—')} | "
            f"{metrics.get('learning_rate', '—')} | {metrics.get('supervised_tokens', 0)} | "
            f"{metrics.get('optimizer_steps', 0)} | {metrics.get('wall_seconds', 0):.1f} | {seed if seed is not None else '—'} |")
        by_category = {}
        for layer in (1, 2, 3):
            for category in sorted({i["category"] for i in items if i["layer"] == layer}):
                group = [i for i in items if i["layer"] == layer and i["category"] == category]
                totals = {key: sum(i.get("metrics", {}).get(key, 0) for i in group) for key in
                          ("tool_calls", "illegal_actions", "repeated_actions", "early_stops", "truncated_generations")}
                by_category[f"{layer}:{category}"] = dict(n=len(group), correct=sum(i["correct"] for i in group), **totals)
        breakdown[name] = by_category
    pooled_seeds = intervals.get("C_vs_D_pooled", {}).get("3", {}).get("train_seeds", [])
    unpooled = {arm: sorted(seed for a, seed in variants.values() if a == arm and seed not in pooled_seeds)
               for arm in ("C", "D")}
    lines += ["", f"C vs D pooled training seeds: {pooled_seeds or 'unavailable (requires two matched seeds)'}. "
              f"Seeds outside the pool: {unpooled}.",
              "Seed repeats reuse each arm's generated bank; generation cost shown per variant is shared. "
              "Pooling averages correctness per question over the same seeds in C and D, then pairs by parent. "
              "Intervals measure parent uncertainty conditional on these training seeds, not uncertainty over new seeds. "
              "Within-arm seed comparisons measure training noise; mean absolute item delta is the fraction of correctness flips."]
    lines += ["", f"Shared diagnosis/held-out preparation: {cost['shared']['output_tokens']} output tokens. "
              "All attempts, discarded generations, and reasoning remain in the ledger. "
              "USD cost is not inferred from an unverified price; exact raw token usage is retained.", "",
              "| Pair (right − left) | Layer | Items | Parent delta (pp) | Paired 95% interval (pp) | Parents | Catches / regressions | Mean absolute item delta (pp) |",
              "| --- | --- | ---: | ---: | --- | ---: | --- | ---: |"]
    for name, layers in intervals.items():
        for layer, stat in layers.items():
            ci = "unavailable" if stat["ci95"] is None else f"[{100*stat['ci95'][0]:.2f}, {100*stat['ci95'][1]:.2f}]"
            outcomes = f"{stat['outcomes'].get('01',0)} / {stat['outcomes'].get('10',0)}" if stat["outcomes"] else "—"
            lines.append(f"| {name} | {layer} | {stat['items']} | {100*stat['delta']:.2f} | {ci} | {stat['independent_parents']} | "
                         f"{outcomes} | {100*stat['mean_absolute_item_delta']:.2f} |")
    comparison = "C_vs_D_pooled" if pooled_seeds else "C_vs_D"
    cd = intervals[comparison]
    # Use the same seeds for base improvements and for the targeting comparison.
    def base_improvement(arm):
        seeds = pooled_seeds or [splits["seed"]]
        labels = [arm if s == splits["seed"] else f"{arm}_s{s}" for s in seeds]
        return sum(intervals["base_vs_" + label]["3"]["delta"] for label in labels) / len(seeds)

    positive = lambda stat: stat["ci95"] is not None and stat["ci95"][0] > 0
    if positive(cd["1"]) and positive(cd["3"]) and (cd["3"]["leave_one_parent_out_min"] or 0) > 0:
        finding = "D improves local and full tasks across parents: add full-trajectory A and original-position B controls next."
    elif cd["1"]["delta"] > 0 and cd["3"]["delta"] <= 0:
        finding = "D improves local practice only: inspect natural triggering, context differences, and subsequent execution."
    elif all(base_improvement(a) > 0 for a in ("C", "D")) and not positive(cd['3']):
        finding = "Both arms improve; targeting advantage remains unestablished. Investigate ordinary practice/diversity."
    elif cd["3"]["negative_parents"] and cd["1"]["delta"] > 0:
        finding = "Possible targeted repair with regressions: inspect condition coverage and retention."
    else:
        finding = "No clear targeting signal: inspect supervision, exposure, and uncertainty before interpreting the mechanism."
    repeat = args.run_dir / "evaluation/base/repeat/items.json"
    stability = paired_interval(pair(models["base"], read_json(repeat))) if repeat.exists() else None
    lines += ["", f"Interpretation uses {comparison}.", "", finding, "", "| Section-nine pattern | Next experiment |", "| --- | --- |",
              "| D improves local and full tasks across parents | Add A full trajectories and B original-position guidance |",
              "| D improves local only | Inspect natural triggering and continuation |",
              "| C and D improve similarly | Practice/diversity supported; targeting not established |",
              "| Target repair with other regressions | Inspect one-sided coverage and retention |",
              "| Neither arm learns | Check effective supervision/exposure |", "",
              "Intervals weight independent parents equally; variants are not independent trials. "
              "Natural/local scores check call decisions and available executor consistency; "
              "teacher-authored semantics and unexecutable tools retain explicit validation limits.", "",
              "Base repeat: " + (str(stability["outcomes"]) if stability else "not run; determinism unverified.")]
    write_json(args.run_dir / "report.json", dict(mechanism_table=rows, intervals=intervals,
        layer_3_scope=scope, category_breakdown=breakdown, teacher_cost=cost,
        pooled_training_seeds=pooled_seeds, unpooled_training_seeds=unpooled,
        base_stability=stability, interpretation_comparison=comparison, interpretation=finding))
    (args.run_dir / "report.md").write_text("\n".join(lines)+"\n")
    print(args.run_dir / "report.md")
