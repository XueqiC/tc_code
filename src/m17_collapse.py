"""M17: instrumented collapse study at 4B — does demand space show precursors?

Two sequential runs on Qwen3.5-4B (gold-90 diet, 6 epochs = 66 steps):
lr=2e-4 (the unstable sweep setting) vs lr=1e-4 (scaled). At checkpoints,
measure: exec success, teacher-forced loss (in/out domain sample), total
LoRA grad norm on train batch, and demand decomposition of probe features
against a 4B-native spec subspace (fit at step 0 from the 50 spec rows):
in-subspace vs out-of-subspace demand energy. Collapse precursor candidates:
out-subspace demand rising, total norm inflection, in-domain loss reversal.
"""
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from boundary import fit_subspace
from distill_pilot import encode, split
import verifier
from features import _prepare_extractor, _project_gradients
from m2_v3 import lora_modules, sync_ref_weights

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "m17"
STUDENT = "Qwen/Qwen3.5-4B"
CHECKPOINTS = (0, 2, 5, 11, 22, 33, 44, 66)
SEED, ACCUM, EPOCHS = 2, 8, 6   # seed2 was a collapsing seed in the sweep


def probe_feats(ref_model, tok, rows, dev, ref_lora_b, projs, normalize=True):
    feats, raw_norms = [], []
    ref_model.train()
    for r in rows:
        ids, labels = encode(tok, r, dev)
        ref_model.zero_grad(set_to_none=True)
        ref_model(input_ids=ids, labels=labels).loss.backward()
        v = torch.cat([projs[n].T @ p.grad.detach().flatten().float()
                       for n, p in ref_lora_b])
        raw_norms.append(float(v.norm()))
        feats.append((v / (v.norm() + 1e-8)).cpu().numpy())
    ref_model.zero_grad(set_to_none=True)
    return np.stack(feats).astype(np.float64), np.array(raw_norms)


@torch.no_grad()
def exec_rate(model, tok, rows, golds, dev):
    model.eval()
    ok = 0
    for r, g in zip(rows, golds):
        msgs = [{"role": "user", "content": r["prompt"]}]
        ptxt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(ptxt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        gen = model.generate(ids, max_new_tokens=320, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        s = text.find("def solution")
        pred = verifier.run_solution(text[s:]) if s >= 0 else None
        ok += int(pred is not None and g is not None and abs(pred - g) < 1e-4)
    model.train()
    return ok / len(rows)


def run(lr, tok, ref_model, ref_lora_b, projs, U, test, golds, cot_probe, dev):
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to(dev)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    t_mods, r_mods = lora_modules(model), lora_modules(ref_model)

    train, _ = split("gsm8k-code")
    rec = []

    def measure(step):
        sync_ref_weights(t_mods, r_mods)
        F, raw = probe_feats(ref_model, tok, test, dev, ref_lora_b, projs)
        proj = np.linalg.norm(F @ U, axis=1)
        Fc, _ = probe_feats(ref_model, tok, cot_probe, dev, ref_lora_b, projs)
        rec.append({
            "step": step, "lr": lr,
            "exec": exec_rate(model, tok, test, golds, dev),
            "in_sub_energy": float(np.mean(proj**2)),
            "raw_demand_norm": float(np.mean(raw)),
            "cot_in_sub_energy": float(np.mean(np.linalg.norm(Fc @ U, axis=1)**2)),
        })
        r = rec[-1]
        print(f"lr={lr:g} step={step:2d} exec={r['exec']:.3f} "
              f"in_sub={r['in_sub_energy']:.3f} raw_norm={r['raw_demand_norm']:.2f} "
              f"cot_in_sub={r['cot_in_sub_energy']:.3f}", flush=True)

    measure(0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(SEED)
    step = 0
    model.train()
    for _ep in range(EPOCHS):
        order = rng.permutation(len(train))
        full = order[:(len(order)//ACCUM)*ACCUM]
        for j, i in enumerate(full, 1):
            ids, labels = encode(tok, train[i], dev)
            (model(input_ids=ids, labels=labels).loss / ACCUM).backward()
            if j % ACCUM == 0:
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step in CHECKPOINTS:
                    measure(step)
    del model, opt
    torch.cuda.empty_cache()
    return rec


def main():
    dev = "cuda"
    tok, ref_model, ref_lora_b, projs = _prepare_extractor(STUDENT, 64, dev)
    _, test = split("gsm8k-code")
    _, cot_probe = split("gsm8k-cot")
    golds = [verifier.run_solution(r["response"]) for r in test]
    spec, _ = split("gsm8k-code")
    F0, _ = probe_feats(ref_model, tok, spec[:35], dev, ref_lora_b, projs)
    from boundary import unit
    U = fit_subspace(unit(F0))
    print(f"spec subspace rank={U.shape[1]}", flush=True)

    all_rec = []
    for lr in [2e-4, 1e-4]:
        all_rec += run(lr, tok, ref_model, ref_lora_b, projs, U, test, golds,
                       cot_probe[:15], dev)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(all_rec, open(OUT / "records.json", "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
