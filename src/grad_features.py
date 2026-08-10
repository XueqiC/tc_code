"""Per-sample gradient features (LESS-style) for boundary estimation.

For each (prompt, response): loss on response tokens under the base model with
fresh LoRA adapters -> backward -> collect grads of lora_B weights only.
At init B=0, so dL/dB = G @ A^T is a random projection of the full-weight
gradient G by the fixed random A — the LESS trick. Each module's B-grad is
further JL-projected to PROJ_DIM dims with a per-module fixed-seed Gaussian,
then all modules are concatenated and L2-normalized.

Output: results/pilot/features_grad.npz  (X: [N, D], domain: [N] str, idx: [N])
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DOMAINS = ["gsm8k-code", "gsm8k-cot", "pandas", "sql", "alpaca"]
PROJ_DIM = 64
MAX_PROMPT_TOK = 640
MAX_RESP_TOK = 512
SEED = 0


def stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def main():
    torch.manual_seed(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad = "lora_B" in n

    lora_b = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
    print(f"{len(lora_b)} lora_B modules")
    projs = {}
    for n, p in lora_b:
        g = torch.Generator(device=dev).manual_seed(stable_seed(n))
        projs[n] = torch.randn(p.numel(), PROJ_DIM, generator=g, device=dev,
                               dtype=torch.float32) / math.sqrt(PROJ_DIM)

    feats, domains, idxs = [], [], []
    for dom in DOMAINS:
        rows = [json.loads(l) for l in open(ROOT / "data" / "pilot" / f"{dom}.jsonl")]
        for i, r in enumerate(rows):
            msgs = [{"role": "user", "content": r["prompt"]}]
            prompt_txt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                 tokenize=False)
            prompt_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
            prompt_ids = prompt_ids[:MAX_PROMPT_TOK]
            resp_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
            resp_ids = resp_ids[:MAX_RESP_TOK] + [tok.eos_token_id]
            input_ids = torch.tensor([prompt_ids + resp_ids], device=dev)
            labels = torch.tensor([[-100] * len(prompt_ids) + resp_ids], device=dev)

            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()

            vec = torch.cat([
                projs[n].T @ p.grad.detach().flatten().float() for n, p in lora_b
            ])
            vec = vec / (vec.norm() + 1e-8)
            feats.append(vec.cpu().numpy())
            domains.append(dom)
            idxs.append(i)
            if (i + 1) % 40 == 0:
                print(f"{dom}: {i + 1}/{len(rows)} loss={out.loss.item():.3f}",
                      flush=True)

    out_dir = ROOT / "results" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "features_grad.npz",
                        X=np.stack(feats), domain=np.array(domains),
                        idx=np.array(idxs))
    print("saved", out_dir / "features_grad.npz")


if __name__ == "__main__":
    main()
