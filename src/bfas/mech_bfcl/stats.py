"""Strict pairing and paired parent-cluster intervals; no variant pseudoreplication."""
from collections import Counter, defaultdict
from itertools import combinations
import random

from .common import read_json, write_json
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
        interval = None  # One seed is not enough to estimate task uncertainty.
    else:
        rng = random.Random(seed)
        boot = sorted(sum(rng.choices(means, k=len(means)))/len(means) for _ in range(samples))
        interval = [boot[int(.025*(samples-1))], boot[int(.975*(samples-1))]]
    leave_one_out = [(sum(means)-v)/(len(means)-1) for v in means] if len(means)>1 else []
    return dict(delta=estimate, ci95=interval, independent_parents=len(means), items=len(paired),
        item_weighted_delta=sum(r["delta"] for r in paired)/len(paired),
        outcomes=dict(Counter(r["outcome"] for r in paired)),
        method="equal-parent mean; paired parent-cluster percentile bootstrap",
        positive_parents=sum(m>0 for m in means), negative_parents=sum(m<0 for m in means),
        leave_one_parent_out_min=min(leave_one_out) if leave_one_out else None)


def refresh_pairs(directory, splits):
    from .pipeline import layer3_selection
    ids, scope = layer3_selection(splits)
    models = {}
    for arm in ("base", "C", "D"):
        path = directory / "evaluation" / arm / "main/items.json"
        if path.exists():
            models[arm] = read_json(path)
            full = [r for r in models[arm] if r["layer"] == 3]
            if len(full) != len(ids) or {r["id"] for r in full} != set(ids):
                raise ValueError(f"{arm}: layer-3 items do not match the evaluation selection")
            if any(r.get("evaluation_scope") != scope for r in full):
                raise ValueError(f"{arm}: layer-3 evaluation scope metadata mismatch")
    summary = {}
    for a, b in combinations(models, 2):
        rows = pair(models[a], models[b])
        name = a + "_vs_" + b
        write_json(directory / "paired" / (name+".json"), rows)
        summary[name] = {str(layer): paired_interval([r for r in rows if r["layer"] == layer])
                         for layer in (1, 2, 3)}
        summary[name]["3"]["evaluation_scope"] = scope
    write_json(directory / "paired/intervals.json", summary)
    return models, summary


def report(args, splits):
    from .pipeline import layer3_selection
    models, intervals = refresh_pairs(args.run_dir, splits)
    if set(models) != {"base", "C", "D"}:
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
             "| Arm | Local (%) | Natural (%) | Full tasks (%) | Generated output tokens | Supervised tokens | Steps | Train seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    breakdown = {}
    for arm, items in models.items():
        accuracy = {str(layer): sum(i["correct"] for i in items if i["layer"] == layer) /
                    sum(i["layer"] == layer for i in items) for layer in (1, 2, 3)}
        metrics = read_json(args.run_dir / arm / "training/metrics.json") if arm != "base" else {}
        training = dict(accuracy=accuracy, **metrics)
        rows.append(dict(arm=arm, **training))
        lines.append(f"| {arm} | " + " | ".join(f"{100*accuracy[str(l)]:.2f}" for l in (1,2,3)) +
            f" | {cost.get(arm, {}).get('output_tokens', 0)} | {metrics.get('supervised_tokens', 0)} | "
            f"{metrics.get('optimizer_steps', 0)} | {metrics.get('wall_seconds', 0):.1f} |")
        by_category = {}
        for layer in (1, 2, 3):
            for category in sorted({i["category"] for i in items if i["layer"] == layer}):
                group = [i for i in items if i["layer"] == layer and i["category"] == category]
                totals = {key: sum(i.get("metrics", {}).get(key, 0) for i in group) for key in
                          ("tool_calls", "illegal_actions", "repeated_actions", "early_stops", "truncated_generations")}
                by_category[f"{layer}:{category}"] = dict(n=len(group), correct=sum(i["correct"] for i in group), **totals)
        breakdown[arm] = by_category
    lines += ["", f"Shared diagnosis/held-out preparation: {cost['shared']['output_tokens']} output tokens. "
              "All attempts, discarded generations, and reasoning remain in the ledger. "
              "USD cost is not inferred from an unverified price; exact raw token usage is retained.", "",
              "| Pair (right − left) | Layer | Items | Parent delta (pp) | Paired 95% interval (pp) | Parents | Catches / regressions |",
              "| --- | --- | ---: | ---: | --- | ---: | --- |"]
    for name, layers in intervals.items():
        for layer, stat in layers.items():
            ci = "unavailable" if stat["ci95"] is None else f"[{100*stat['ci95'][0]:.2f}, {100*stat['ci95'][1]:.2f}]"
            lines.append(f"| {name} | {layer} | {stat['items']} | {100*stat['delta']:.2f} | {ci} | {stat['independent_parents']} | "
                         f"{stat['outcomes'].get('01',0)} / {stat['outcomes'].get('10',0)} |")
    cd = intervals["C_vs_D"]
    positive = lambda stat: stat["ci95"] is not None and stat["ci95"][0] > 0
    if positive(cd["1"]) and positive(cd["3"]) and (cd["3"]["leave_one_parent_out_min"] or 0) > 0:
        finding = "D improves local and full tasks across parents: add full-trajectory A and original-position B controls next."
    elif cd["1"]["delta"] > 0 and cd["3"]["delta"] <= 0:
        finding = "D improves local practice only: inspect natural triggering, context differences, and subsequent execution."
    elif all(intervals['base_vs_'+a]['3']['delta'] > 0 for a in ('C', 'D')) and not positive(cd['3']):
        finding = "Both arms improve; targeting advantage remains unestablished. Investigate ordinary practice/diversity."
    elif cd["3"]["negative_parents"] and cd["1"]["delta"] > 0:
        finding = "Possible targeted repair with regressions: inspect condition coverage and retention."
    else:
        finding = "No clear targeting signal: inspect supervision, exposure, and uncertainty before interpreting the mechanism."
    repeat = args.run_dir / "evaluation/base/repeat/items.json"
    stability = paired_interval(pair(models["base"], read_json(repeat))) if repeat.exists() else None
    lines += ["", finding, "", "| Section-nine pattern | Next experiment |", "| --- | --- |",
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
        base_stability=stability, interpretation=finding))
    (args.run_dir / "report.md").write_text("\n".join(lines)+"\n")
    print(args.run_dir / "report.md")
