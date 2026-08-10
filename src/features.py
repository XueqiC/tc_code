"""Reusable per-sample LoRA gradient feature extraction."""

import hashlib
import math
from collections import Counter, defaultdict

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def extract_features(model_name: str, rows: list[dict], proj_dim=64,
                     max_prompt_tok=640, max_resp_tok=512,
                     device="cuda") -> np.ndarray:
    """Extract normalized, fixed-projection LoRA-B gradients for each row."""
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = "lora_B" in name

    lora_b = [(name, param) for name, param in model.named_parameters()
              if "lora_B" in name]
    print(f"{len(lora_b)} lora_B modules")
    projs = {}
    for name, param in lora_b:
        generator = torch.Generator(device=device).manual_seed(stable_seed(name))
        projs[name] = torch.randn(
            param.numel(), proj_dim, generator=generator, device=device,
            dtype=torch.float32
        ) / math.sqrt(proj_dim)

    domain_totals = Counter(row.get("domain") for row in rows)
    domain_seen = defaultdict(int)
    feats = []
    for row in rows:
        msgs = [{"role": "user", "content": row["prompt"]}]
        prompt_txt = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[:max_prompt_tok]
        resp_ids = tok(row["response"], add_special_tokens=False)["input_ids"]
        resp_ids = resp_ids[:max_resp_tok] + [tok.eos_token_id]
        input_ids = torch.tensor([prompt_ids + resp_ids], device=device)
        labels = torch.tensor(
            [[-100] * len(prompt_ids) + resp_ids], device=device
        )

        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()

        vec = torch.cat([
            projs[name].T @ param.grad.detach().flatten().float()
            for name, param in lora_b
        ])
        vec = vec / (vec.norm() + 1e-8)
        feats.append(vec.cpu().numpy())

        domain = row.get("domain")
        domain_seen[domain] += 1
        if domain is not None and domain_seen[domain] % 40 == 0:
            print(f"{domain}: {domain_seen[domain]}/{domain_totals[domain]} "
                  f"loss={out.loss.item():.3f}", flush=True)

    return np.stack(feats)
