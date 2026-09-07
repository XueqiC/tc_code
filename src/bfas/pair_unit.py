"""Opt-in condition-contrast training. The legacy DDPO path is untouched.

See docs/cc_pair_unit.md for the producer contract and exposure accounting.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .arms import (pair_side_rows, pair_unit_loss, pair_unit_margins,
                   same_seed_pair_counts, shuffle_pair_sides)


def configuration(environ=None):
    env = os.environ if environ is None else environ

    def number(name, default, cast=float, minimum=0, strict=True):
        value = cast(env.get(name, default))
        if not math.isfinite(value) or (value <= minimum if strict else value < minimum):
            raise ValueError(f"invalid {name}: {value}")
        return value

    mode = env.get("AW_DISTILL", "pair_unit").strip().lower()
    if mode not in {"ddpo", "pair_unit"}:
        raise ValueError("CC training requires AW_DISTILL=ddpo or pair_unit")
    pairing = env.get("AW_PAIR_PAIRING", "correct")
    reduction = env.get("AW_PAIR_REDUCTION", "worst")
    if pairing not in {"correct", "shuffled"} or reduction not in {"worst", "sum"}:
        raise ValueError("pairing must be correct/shuffled; reduction must be worst/sum")
    if mode == "ddpo" and pairing != "correct":
        raise ValueError("flattened arm A uses the original sides")
    for flag in ("AW_DDPO_REF_FREE", "AW_DDPO_SPAN_ONLY"):
        if env.get(flag, "0") != "0":
            raise ValueError(f"CC requires {flag}=0 (base-centred full-response margins)")
    return dict(mode=mode, pairing=pairing, reduction=reduction,
                gamma=number("AW_PAIR_GAMMA", 1.0, strict=False),
                beta=number("AW_DDPO_BETA", 0.1),
                learning_rate=number("AW_DDPO_LR", 5e-6),
                epochs=number("AW_DDPO_EPOCHS", 1, int),
                steps=number("AW_PAIR_STEPS", 0, int, strict=False),
                units_per_step=number("AW_PAIR_BATCH_UNITS", 4, int),
                shuffle_seed=int(env.get("AW_PAIR_SHUFFLE_SEED", "0")))


def _read(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def load_pairs(path, split_path=None):
    """Read CC records (or selected candidate IDs) and require a train split.

    Canonical record: {pair_id, type, seed_function, sides: [{prompt, y_plus,
    y_minus?}, {...}]}. Seed function may instead be set on each side.
    """
    path = Path(path)
    data = _read(path)
    if isinstance(data, dict):
        for key in ("pairs", "confused_pairs", "pair_ids", "confused_pair_ids"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list) or not data:
        raise ValueError("CC pair file must contain a nonempty list of pairs or pair IDs")
    if all(isinstance(item, str) for item in data):
        candidates = _read(path.with_name("candidates.jsonl"))
        by_id = {item["pair_id"]: item for item in candidates}
        if len(by_id) != len(candidates) or len(set(data)) != len(data):
            raise ValueError("duplicate candidate or selected pair IDs")
        missing = set(data) - by_id.keys()
        if missing:
            raise ValueError(f"unknown candidate IDs: {sorted(missing)}")
        data = [by_id[item] for item in data]
    train_ids = None
    if split_path is not None:
        split = _read(split_path)
        train_ids = split.get("train_pair_ids", split.get("train"))
        if isinstance(train_ids, dict):
            train_ids = train_ids.get("pair_ids")
        if not isinstance(train_ids, list) or not all(isinstance(i, str) for i in train_ids):
            raise ValueError("split.json requires train_pair_ids (or train / train.pair_ids)")
        if len(set(train_ids)) != len(train_ids):
            raise ValueError("duplicate train pair IDs")
        train_ids = set(train_ids)
        for key in ("validation", "val", "test", "confirm", "heldout", "certification",
                    "calibration", "heldout_seed", "within_seed"):
            other = split.get(key + "_pair_ids", split.get(key, []))
            if isinstance(other, dict):
                other = other.get("pair_ids", [])
            if isinstance(other, list) and train_ids.intersection(other):
                raise ValueError(f"train/{key} pair ID overlap")
    pairs, seen = [], set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each CC pair must be an object")
        pair_id = item.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise ValueError("pair IDs must be nonempty and unique")
        seen.add(pair_id)
        if train_ids is not None:
            if pair_id not in train_ids:
                continue
            if item.get("split", "train") != "train":
                raise ValueError(f"{pair_id}: split metadata disagrees with train IDs")
        elif item.get("split") != "train":
            if "split" not in item:
                raise ValueError("explicit train split required (split.json or per-pair split)")
            continue
        kind = item.get("type", item.get("pair_type"))
        if kind is None or str(kind) == "":
            raise ValueError(f"{pair_id}: missing type")
        sides = item.get("sides")
        if sides is None and "s1" in item and "s2" in item:
            sides = [item["s1"], item["s2"]]
        if not isinstance(sides, list) or len(sides) != 2:
            raise ValueError(f"{pair_id}: exactly two sides required")
        normalized = []
        for index, side in enumerate(sides):
            if not isinstance(side, dict):
                raise ValueError(f"{pair_id}: side must be an object")
            side = dict(side)
            for key in ("prompt", "y_plus"):
                if not isinstance(side.get(key), str) or not side[key]:
                    raise ValueError(f"{pair_id}: nonempty {key} required")
            if side.get("y_minus") is not None and not isinstance(side["y_minus"], str):
                raise ValueError(f"{pair_id}: y_minus must be text or null")
            seed = side.get("seed_function", item.get("seed_function"))
            if not isinstance(seed, str) or not seed:
                raise ValueError(f"{pair_id}: seed_function required for each side")
            if side.get("_anchor") or side.get("training_excluded") or item.get("training_excluded"):
                raise ValueError(f"{pair_id}: excluded/anchor rows cannot be CC event sides")
            if "type" in side and str(side["type"]) != str(kind):
                raise ValueError(f"{pair_id}: side type disagrees with pair type")
            side.update(seed_function=seed)
            side.setdefault("side_id", f"{pair_id}:{index}")
            normalized.append(side)
        if normalized[0]["seed_function"] != normalized[1]["seed_function"]:
            raise ValueError(f"{pair_id}: original condition pair must share a seed function")
        pairs.append({**item, "pair_id": pair_id, "type": str(kind), "sides": normalized})
    if not pairs:
        raise ValueError("no selected CC pairs in the training split")
    return pairs


def load_anchors(path, trainer):
    if not path or str(path).lower() == "none":
        return []
    rows = trainer.load_pool(Path(path))
    anchors = [r for r in rows if r.get("_anchor")]
    if not anchors or len(trainer.ddpo_rows(anchors)) != len(anchors):
        raise ValueError("anchor pool must contain _anchor rows with nonempty _rejected")
    return anchors


def step_schedule(unit_count, config, rng):
    """Complete passes only: every arm consumes the identical side multiset."""
    width = config["units_per_step"]
    per_epoch = math.ceil(unit_count / width)
    steps = config["steps"] or config["epochs"] * per_epoch
    if not per_epoch or steps % per_epoch:
        raise ValueError(f"AW_PAIR_STEPS must be a multiple of {per_epoch} (one complete pass)")
    for _ in range(steps // per_epoch):
        order = rng.permutation(unit_count)
        for start in range(0, unit_count, width):
            yield order[start:start + width].tolist()


def categorical_kl(policy_logits, base_logits):
    """Exact forward KL(pi || base), summed over full-vocabulary distributions."""
    policy = policy_logits.float().log_softmax(-1)
    base = base_logits.float().log_softmax(-1)
    return (policy.exp() * (policy - base)).sum(-1)


@torch.no_grad()
def response_kl(model, ids, labels):
    """Full-vocab conditional KL on response prefixes, not a sampled log ratio."""
    mask = labels[:, 1:] != -100
    was_training = model.training
    model.eval()
    try:
        policy = model(input_ids=ids).logits[:, :-1][mask].detach()
        with model.disable_adapter():
            base = model(input_ids=ids).logits[:, :-1][mask].detach()
        total = 0.0
        for start in range(0, policy.shape[0], 32):
            total += float(categorical_kl(policy[start:start + 32], base[start:start + 32]).sum())
        return total, int(mask.sum())
    finally:
        model.train(was_training)


def train(model, tokenizer, pairs, anchors, *, seed, config, trace_path, trainer):
    """Joint events plus the legacy logistic anchors and optional dual rule.

    Each pair carries two row-units of mass; singleton anchors keep one, exactly
    as flattened DDPO. Arm A sums independent logistic side losses in each chunk.
    """
    rows = pair_side_rows(pairs)
    all_rows = rows + anchors
    trainer.ddpo_prepare_weights(all_rows)
    device = next(model.parameters()).device
    was_use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    reference_tokens = Counter()

    def encoded(row):
        for branch in (row, trainer.rejected_row(row)) if row.get("_rejected") else (row,):
            yield trainer.encode(tokenizer, branch, device=device)

    # Also caches positive-only rows, which legacy DDPO intentionally filters out.
    with torch.no_grad(), model.disable_adapter():
        for row in all_rows:
            logps = []
            for ids, labels in encoded(row):
                logps.append(float(trainer.completion_log_prob(model, ids, labels)))
                reference_tokens["input_tokens"] += ids.numel()
                reference_tokens["response_tokens"] += int((labels[:, 1:] != -100).sum())
            row["_ddpo_ref_chosen_logp"] = logps[0]
            row["_ddpo_ref_rejected_logp"] = logps[1] if len(logps) > 1 else 0.0

    rng = np.random.default_rng(seed)
    adaptive = os.environ.get("AW_ANCHOR_ADAPTIVE", "0") == "1"
    probes_by_side = {"call": [], "abstain": []}
    anchors_by_side = {"call": [], "abstain": []}
    lam = {"call": 0.0, "abstain": 0.0}
    units = [rows[i:i + 2] for i in range(0, len(rows), 2)]
    if adaptive:
        n_probe = int(os.environ.get("AW_ANCHOR_PROBES", "12"))
        # Same ordering/selection and call-vs-abstain test as train_ddpo.
        for i in rng.permutation(len(all_rows)):
            row = all_rows[int(i)]
            if not row.get("_anchor"):
                continue
            side = "call" if "<tool_call>" in str(row.get("response", "")) else "abstain"
            (probes_by_side if len(probes_by_side[side]) < n_probe else anchors_by_side)[side].append(row)
    else:
        units.extend([row] for row in anchors)
    probe_every = int(os.environ.get("AW_ANCHOR_PROBE_EVERY", "8"))
    eps = float(os.environ.get("AW_ANCHOR_EPS", "1.0"))
    eta = float(os.environ.get("AW_ANCHOR_ETA", "0.5"))
    lam_max = float(os.environ.get("AW_ANCHOR_LAM_MAX", "4.0"))
    # Validate the step boundary before any update.
    per_epoch = math.ceil(len(units) / config["units_per_step"])
    if config["steps"] and config["steps"] % per_epoch:
        raise ValueError(f"AW_PAIR_STEPS must be a multiple of {per_epoch} (one complete pass)")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config["learning_rate"])
    totals = Counter()
    adaptive_trace = []
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    try:
        with trace_path.open("w") as trace:
            for step, indices in enumerate(step_schedule(len(units), config, rng), 1):
                chunk = [units[i] for i in indices]
                mass = sum(len(unit) for unit in chunk)
                counts = Counter()
                records, kl_sum, kl_tokens, loss_value = [], 0.0, 0, 0.0
                optimizer.zero_grad(set_to_none=True)

                def margin(row):
                    logps = []
                    for ids, labels in encoded(row):
                        logps.append(trainer.completion_log_prob(model, ids, labels))
                        ntok = int((labels[:, 1:] != -100).sum())
                        counts["train_input_tokens"] += ids.numel()
                        counts["train_response_tokens"] += ntok
                        counts["train_prompt_tokens"] += ids.numel() - ntok
                        if row.get("_anchor"):
                            counts["anchor_input_tokens"] += ids.numel()
                            counts["anchor_response_tokens"] += ntok
                    reject = logps[1] if len(logps) > 1 else logps[0].new_zeros(())
                    return pair_unit_margins(logps[0], row["_ddpo_ref_chosen_logp"],
                                             reject, row["_ddpo_ref_rejected_logp"],
                                             beta=config["beta"])

                def update(unit, anchor_scale=None):
                    nonlocal kl_sum, kl_tokens, loss_value
                    margins = torch.stack([margin(row) for row in unit])
                    weights = margins.new_tensor([trainer._ddpo_event_weight(r) for r in unit])
                    if anchor_scale is not None:
                        loss = -F.logsigmoid(margins[0]) * anchor_scale
                    elif unit[0].get("_anchor") or config["mode"] == "ddpo":
                        loss = (-F.logsigmoid(margins) * weights).sum() / mass
                    else:
                        # Weight each side's hinge before the worst/sum reduction.
                        # Unit weights (the standard campaign) give exactly L_pair.
                        joint = pair_unit_loss(margins, gamma=config["gamma"],
                                               reduction=config["reduction"], side_weights=weights)
                        loss = joint * 2 / mass
                    loss_value += float(loss.detach())
                    values = margins.detach().tolist()
                    loss.backward()
                    records.append(dict(pair_id=unit[0].get("_cc_pair_id"),
                                        side_ids=[r.get("side_id", r["task_id"]) for r in unit],
                                        margins=values, anchor=bool(unit[0].get("_anchor"))))
                    # Measurements use the same pre-update policy as these margins.
                    for row in unit:
                        for ids, labels in encoded(row):
                            value, n = response_kl(model, ids, labels)
                            kl_sum += value
                            kl_tokens += n
                            counts["diagnostic_input_tokens"] += 2 * ids.numel()

                for unit in chunk:
                    update(unit)
                if adaptive:
                    for side, lam_k in lam.items():
                        pool = anchors_by_side[side]
                        if lam_k <= 0 or not pool:
                            continue
                        for _ in range(min(2, math.ceil(lam_k))):
                            update([pool[int(rng.integers(len(pool)))]], lam_k / mass)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if adaptive and step % probe_every == 0:
                    drift = {}
                    model.eval()
                    with torch.no_grad():
                        for side, probes in probes_by_side.items():
                            if not probes:
                                continue
                            deltas = []
                            for row in probes:
                                ids, labels = trainer.encode(tokenizer, row, device=device)
                                deltas.append(float(trainer.completion_log_prob(model, ids, labels)) -
                                              row["_ddpo_ref_chosen_logp"])
                                counts["probe_input_tokens"] += ids.numel()
                            drift[side] = sum(deltas) / len(deltas)
                            lam[side] = min(lam_max, max(0.0, lam[side] + eta * (-eps - drift[side])))
                    model.train()
                    adaptive_trace.append(dict(step=step, drift=drift, **{"lambda": dict(lam)}))
                totals.update(counts)
                entry = dict(step=step, loss=loss_value, pairs=records,
                             kl_to_base=kl_sum / max(kl_tokens, 1), kl_sum=kl_sum,
                             kl_response_tokens=kl_tokens,
                             kl_definition="KL(policy||base), full vocabulary, observed response prefixes, pre-update",
                             tokens=dict(counts), cumulative_tokens=dict(totals),
                             reference_tokens=dict(reference_tokens), anchor_lambda=dict(lam))
                trace.write(json.dumps(entry, allow_nan=False) + "\n")
                trace.flush()
                print(f"[train][cc] step={step} loss={loss_value:.6f} kl={entry['kl_to_base']:.6f} "
                      f"response_tokens={totals['train_response_tokens']}", flush=True)
        if adaptive and os.environ.get("AW_ANCHOR_TRACE"):
            Path(os.environ["AW_ANCHOR_TRACE"]).write_text(json.dumps(adaptive_trace, indent=1))
    finally:
        del optimizer
        model.config.use_cache = was_use_cache
    return dict(optimizer_steps=step, tokens=dict(totals), reference_tokens=dict(reference_tokens))


def run(args, trainer):
    """Entry from appworld_train only when the new mode/data path is enabled."""
    import hashlib
    import platform

    config = configuration()
    if args.selection != "full":
        raise ValueError("CC training requires --selection full to preserve complete pairs")
    if os.environ.get("AW_TEACHER", ""):
        raise ValueError("CC sides cannot be independently teacher-filtered")
    source = Path(os.environ.get("AW_CC_PAIRS_PATH", "data/cc_pairs_v1/confused_pairs.json"))
    split = Path(os.environ.get("AW_CC_SPLIT_PATH", str(source.with_name("split.json"))))
    pairs = load_pairs(source, split)
    if config["pairing"] == "shuffled":
        pairs = shuffle_pair_sides(pairs, seed=config["shuffle_seed"])
    anchor_path = os.environ.get("AW_CC_ANCHOR_PATH", "data/bfcl_sft/anchors_single_base.jsonl")
    anchors = load_anchors(anchor_path, trainer)
    trainer.seed_everything(args.seed)
    tokenizer = trainer.load_tokenizer(args.student)
    trainer.log_prompt_truncation(tokenizer, pair_side_rows(pairs) + anchors)
    output_dir = trainer.OUTPUT_ROOT / args.tag
    if output_dir.exists():
        raise FileExistsError(f"CC output already exists; use a new tag: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = pair_side_rows(pairs) + anchors
    # Hash only the canonical training row multiset: pairing and provenance differ in C.
    exposure = sorted(json.dumps({k: r.get(k) for k in
                                 ("prompt", "messages", "response", "_rejected", "_anchor")},
                                sort_keys=True) for r in rows)
    manifest = dict(config=config, seed=args.seed, student=args.student,
                    hostname=platform.node(), pair_source=str(source), split_source=str(split),
                    anchor_source=anchor_path, anchor_adaptive=os.environ.get("AW_ANCHOR_ADAPTIVE", "0"),
                    retention_environment={k: v for k, v in os.environ.items()
                                           if k.startswith(("AW_ANCHOR_", "AW_DDPO_WEIGHT", "AW_DDPO_TAU", "AW_DDPO_PERM"))},
                    prompt_truncation=trainer.prompt_truncation_config(),
                    max_response_tokens=trainer.MAX_RESPONSE_TOKENS,
                    exposure_sha256=hashlib.sha256("\n".join(exposure).encode()).hexdigest(),
                    pairs=pairs, anchor_rows=anchors)
    if config["pairing"] == "shuffled":
        manifest["shuffle_same_seed_by_type"] = same_seed_pair_counts(pairs)
    manifest_path = output_dir / "cc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    model = trainer.build_lora_model(args.student, args.seed)
    result = train(model, tokenizer, pairs, anchors, seed=args.seed, config=config,
                   trace_path=output_dir / "cc_steps.jsonl", trainer=trainer)
    merged = model.merge_and_unload()
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir()
    # bfcl_std_campaign.sh expects a single model.safetensors at this stage.
    merged.save_pretrained(adapter_dir, safe_serialization=True, max_shard_size="50GB")
    tokenizer.save_pretrained(adapter_dir)
    manifest.update(result)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[train][cc] saved={output_dir} steps={result['optimizer_steps']}", flush=True)
