"""One-pass BF16 LoRA with exact full-vocabulary reference preservation.

The API teacher supplies a demonstration, not logits in Gemma's vocabulary.
The target is q=.5*one_hot(demo_token)+.5*p_frozen_base over the FULL vocabulary.
Only position chunks are projected; no top-k or vocabulary truncation is used.
"""
from collections import Counter
import os
import random
import subprocess
import time

from .common import STUDENT, digest, read_json, write_json
from .exercises import native_pair

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def mixture_loss(logits, reference_logits, targets):
    import torch.nn.functional as F
    logp = F.log_softmax(logits.float(), dim=-1)
    q = F.softmax(reference_logits.detach().float(), dim=-1)
    teacher = -logp.gather(-1, targets[..., None]).squeeze(-1)
    return (0.5 * teacher - 0.5 * (q * logp).sum(-1)).sum()


def chunked_hidden_gradient(hidden, reference_hidden, head, targets, *, chunk_size=32, softcap=None):
    """Release each vocab-sized graph before projecting the next positions.

    Accumulate dL/dhidden and backprop through the decoder only once. The
    output head is frozen (as are all base weights); LoRA lives in the decoder.
    """
    import torch
    if chunk_size <= 0 or hidden.shape != reference_hidden.shape or len(targets) != len(hidden):
        raise ValueError("Invalid position chunk dimensions")
    gradient, total = torch.zeros_like(hidden), 0.0

    def project(h):
        logits = head(h)
        return logits if softcap is None else (logits / softcap).tanh() * softcap

    for start in range(0, len(targets), chunk_size):
        end = min(start + chunk_size, len(targets))
        leaf = hidden[start:end].detach().requires_grad_(True)
        with torch.no_grad():
            ref = project(reference_hidden[start:end])
        logits = project(leaf)
        loss = mixture_loss(logits, ref, targets[start:end])
        (grad,) = torch.autograd.grad(loss, leaf)
        gradient[start:end] = grad
        total += float(loss.detach())
        del loss, logits, ref, grad, leaf
    return gradient, total


def encode_rows(rows, tokenizer, max_context=8192):
    valid, skipped = [], []
    for row in sorted(rows, key=lambda r: r["id"]):
        if not row["validation"]["valid"]:
            continue
        prompt, target = native_pair(tokenizer, row)
        p = tokenizer.encode(prompt, add_special_tokens=False)
        y = tokenizer.encode(target, add_special_tokens=False)
        if not p or not y or len(p) + len(y) > max_context:
            skipped.append(dict(id=row["id"], input_tokens=len(p), supervised_tokens=len(y),
                                reason="complete context exceeds 8k or empty"))
            continue
        valid.append(dict(id=row["id"], prompt_ids=p, target_ids=y,
                          context_bucket=4096 if len(p)+len(y) <= 4096 else 8192))
    return valid, skipped


def token_schedule(rows, cap, tokens_per_step=512):
    """Exact token exposure; a row can cross updates, each token is used once."""
    if cap <= 0 or tokens_per_step <= 0:
        raise ValueError("Positive supervision budget and accumulation size required")
    steps, current, filled, spent = [], [], 0, 0
    for i, row in enumerate(rows):
        start = 0
        while start < len(row["target_ids"]) and spent < cap:
            count = min(len(row["target_ids"])-start, tokens_per_step-filled, cap-spent)
            current.append(dict(row=i, start=start, end=start+count))
            start += count
            filled += count
            spent += count
            if filled == tokens_per_step:
                steps.append(current)
                current, filled = [], 0
        if spent == cap:
            break
    if current:
        steps.append(current)
    if spent != cap:
        raise ValueError("Insufficient one-pass supervised tokens")
    return steps


def frozen_hidden(model, ids, indices):
    import torch
    was_training = model.training
    model.eval()
    try:
        with model.disable_adapter(), torch.no_grad():
            return model.get_base_model().model(input_ids=ids, use_cache=False).last_hidden_state[0, indices].detach()
    finally:
        model.train(was_training)


def train(args, splits):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    if os.environ.get("MECH_SERVING_PID"):
        raise RuntimeError("Stop the vLLM process and unset MECH_SERVING_PID before training")
    target_dir = args.run_dir / args.arm / "training"
    if target_dir.exists():
        raise ValueError("Training output already exists; one-pass runs cannot be resumed/repeated in place")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    banks = {arm: read_json(args.run_dir / arm / "exercises.json") for arm in ("C", "D")}
    preparation = {arm: read_json(args.run_dir / arm / "generation.json") for arm in banks}
    if preparation["C"]["contexts_hash"] != preparation["D"]["contexts_hash"]:
        raise ValueError("C/D do not share the same task/interface/start-state contexts")
    for arm in banks:
        if not preparation[arm]["ready"]:
            raise ValueError(f"{arm} generation failed coverage/validation; inspect generation.json")
        if any(e["task_id"] not in splits["support"] or e["arm"] != arm or e["layer"] != 0 for e in banks[arm]):
            raise ValueError("Training bank contains the wrong arm or non-support descendants")
    encoded, excluded = {}, {}
    for arm in banks:
        encoded[arm], excluded[arm] = encode_rows(banks[arm], tokenizer)
        random.Random(splits["seed"]).shuffle(encoded[arm])
    cap = min(16000, *(sum(len(r["target_ids"]) for r in encoded[a]) for a in encoded))
    schedules = {a: token_schedule(encoded[a], cap, args.tokens_per_step) for a in encoded}
    plan = dict(common_supervised_cap=cap, bank_hashes={a: digest(banks[a]) for a in banks},
                tokenizer=args.tokenizer, seed=splits["seed"], tokens_per_step=args.tokens_per_step,
                encoded_hashes={a: digest(encoded[a]) for a in encoded}, excluded=excluded,
                schedules=schedules)
    plan_path = args.run_dir / "training_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("C/D exposure plan changed after it was frozen")
    write_json(plan_path, plan)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires the time-shared A100; CPU tests do not load Gemma weights")
    occupancy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True)
    if occupancy.stdout.strip():
        raise RuntimeError("GPU compute processes are still resident; stop serving and wait before training")
    start_time = time.monotonic()
    torch.manual_seed(splits["seed"])
    torch.cuda.manual_seed_all(splits["seed"])
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa", device_map={"": 0})
    actual = {n: m for n, m in model.named_modules()
              if n.rsplit(".", 1)[-1] in TARGET_MODULES and isinstance(m, torch.nn.Linear)}
    found = {n.rsplit(".", 1)[-1] for n in actual}
    if found != set(TARGET_MODULES):
        raise ValueError(f"Missing expected Gemma attention/MLP linear modules: {set(TARGET_MODULES)-found}")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=list(actual), bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    base = model.get_base_model()
    # Match Gemma4ForCausalLM.forward, including final logit softcapping, without
    # allocating its full [context, vocabulary] output tensor.
    if not hasattr(base, "model") or not hasattr(base, "lm_head"):
        raise ValueError("Expected Gemma4ForCausalLM.model/lm_head backbone interface")
    if any(p.requires_grad for p in base.lm_head.parameters()):
        raise ValueError("Chunked decoder VJP requires a frozen lm_head")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.0)
    rows, steps = encoded[args.arm], schedules[args.arm]
    log, input_tokens, supervised = [], 0, 0
    write_json(target_dir / "config.json", dict(student=STUDENT, model_path=args.model_path,
        model_commit=getattr(base.config, "_commit_hash", None), lora_rank=16, lora_alpha=32,
        lora_dropout=0, target_modules=sorted(actual), dtype="bfloat16", micro_batch=1,
        learning_rate=1e-5, gradient_checkpointing=True, passes=1,
        teacher_target="one-hot demonstration", reference_target="full vocabulary frozen adapter-disabled base",
        mixture=[0.5, 0.5], thinking=False, position_chunk=args.position_chunk))
    for step, segments in enumerate(steps):
        optimizer.zero_grad(set_to_none=True)
        step_tokens = sum(s["end"]-s["start"] for s in segments)
        step_loss = 0.0
        step_start = time.monotonic()
        for segment in segments:
            row = rows[segment["row"]]
            p, y = row["prompt_ids"], row["target_ids"]
            left, right = segment["start"], segment["end"]
            # The final target token need not be forwarded to predict itself.
            ids = torch.tensor([p + y[:right-1]], device="cuda")
            indices = slice(len(p)-1+left, len(p)-1+right)
            reference = frozen_hidden(model, ids, indices)
            hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, indices]
            target = torch.tensor(y[left:right], device="cuda")
            gradient, loss = chunked_hidden_gradient(hidden, reference, base.lm_head, target,
                chunk_size=args.position_chunk, softcap=getattr(base.config, "final_logit_softcapping", None))
            if not torch.isfinite(gradient).all() or not torch.isfinite(torch.tensor(loss)):
                raise RuntimeError("Nonfinite distillation loss/gradient; preserve artifacts and inspect")
            if step == 0 and segment is segments[0]:
                if not torch.allclose(hidden.detach(), reference, rtol=0.01, atol=0.01):
                    raise RuntimeError("Initial adapter/base hidden states disagree before the first update")
                write_json(target_dir / "compatibility.json", dict(initial_reference_match=True,
                    full_vocabulary=base.lm_head.weight.shape[0], position_chunk=args.position_chunk,
                    no_second_backbone=True, no_terminal_supervision=True))
            hidden.backward(gradient / step_tokens)
            input_tokens += ids.numel()
            supervised += right-left
            step_loss += loss
            del ids, reference, hidden, target, gradient
        optimizer.step()
        log.append(dict(step=step+1, supervised_tokens=step_tokens, loss=step_loss/step_tokens,
                        wall_seconds=time.monotonic()-step_start))
        write_json(target_dir / "steps.json", log)
    model.save_pretrained(target_dir / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(target_dir / "adapter")
    write_json(target_dir / "metrics.json", dict(input_tokens=input_tokens,
        unique_input_tokens=sum(len(r["prompt_ids"])+len(r["target_ids"]) for r in rows),
        supervised_tokens=supervised, optimizer_steps=len(log), wall_seconds=time.monotonic()-start_time,
        max_allocated_bytes=torch.cuda.max_memory_allocated(), training_plan_hash=digest(plan),
        data_examples=len({s["row"] for step in steps for s in step})))
