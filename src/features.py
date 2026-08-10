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


def _prepare_extractor(model_name, proj_dim, device):
    """Create the shared tokenizer, LoRA model, and fixed JL projections."""
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
    return tok, model, lora_b, projs


def _project_gradients(lora_b, projs):
    """JL-project and concatenate the current LoRA-B gradients."""
    vec = torch.cat([
        projs[name].T @ param.grad.detach().flatten().float()
        for name, param in lora_b
    ])
    return vec / (vec.norm() + 1e-8)


def _response_blocks(response: str, max_blocks=12) -> list[str]:
    """Return stripped non-empty response lines, merging cap overflow."""
    blocks = [line.strip() for line in response.splitlines() if line.strip()]
    if len(blocks) > max_blocks:
        blocks = blocks[:max_blocks - 1] + ["\n".join(blocks[max_blocks - 1:])]
    return blocks


def _block_token_spans(tok, blocks, max_resp_tok):
    """Tokenize growing response prefixes and return their length deltas."""
    prefix = ""
    previous_end = 0
    response_ids = []
    spans = []
    for block in blocks:
        prefix = f"{prefix}\n{block}" if prefix else block
        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        response_ids = prefix_ids[:max_resp_tok]
        end = len(response_ids)
        spans.append((previous_end, end))
        previous_end = end
    return response_ids, spans


def extract_features(model_name: str, rows: list[dict], proj_dim=64,
                     max_prompt_tok=640, max_resp_tok=512,
                     device="cuda") -> np.ndarray:
    """Extract normalized, fixed-projection LoRA-B gradients for each row."""
    tok, model, lora_b, projs = _prepare_extractor(
        model_name, proj_dim, device
    )

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

        vec = _project_gradients(lora_b, projs)
        feats.append(vec.cpu().numpy())

        domain = row.get("domain")
        domain_seen[domain] += 1
        if domain is not None and domain_seen[domain] % 40 == 0:
            print(f"{domain}: {domain_seen[domain]}/{domain_totals[domain]} "
                  f"loss={out.loss.item():.3f}", flush=True)

    return np.stack(feats)


def extract_features_stepwise(model_name: str, rows: list[dict], proj_dim=64,
                              max_prompt_tok=640, max_resp_tok=512,
                              device="cuda") -> tuple[np.ndarray, np.ndarray,
                                                        np.ndarray]:
    """Extract one normalized LoRA-B gradient feature per response block."""
    tok, model, lora_b, projs = _prepare_extractor(
        model_name, proj_dim, device
    )

    feats = []
    row_ids = []
    block_indices = []
    for row_index, row in enumerate(rows):
        blocks = _response_blocks(row["response"])
        response_ids, spans = _block_token_spans(tok, blocks, max_resp_tok)

        msgs = [{"role": "user", "content": row["prompt"]}]
        prompt_txt = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[:max_prompt_tok]

        full_response_ids = response_ids + [tok.eos_token_id]
        input_ids = torch.tensor(
            [prompt_ids + full_response_ids], device=device
        )
        response_offset = len(prompt_ids)

        for block_index, (start, end) in enumerate(spans):
            if end <= start:
                continue
            labels = torch.full_like(input_ids, -100)
            labels[0, response_offset + start:response_offset + end] = input_ids[
                0, response_offset + start:response_offset + end
            ]

            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()

            vec = _project_gradients(lora_b, projs)
            feats.append(vec.cpu().numpy())
            row_ids.append(row_index)
            block_indices.append(block_index)

        if (row_index + 1) % 40 == 0:
            print(
                f"rows: {row_index + 1}/{len(rows)} "
                f"step features: {len(feats)}",
                flush=True,
            )

    feature_dim = len(lora_b) * proj_dim
    X_steps = (np.stack(feats).astype(np.float32, copy=False) if feats else
               np.empty((0, feature_dim), dtype=np.float32))
    return (X_steps, np.asarray(row_ids, dtype=np.int64),
            np.asarray(block_indices, dtype=np.int64))
