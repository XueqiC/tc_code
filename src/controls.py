"""Two sanity controls for pilot-boundary-v1's perfect grad results.

C1 dim-match : grad features randomly projected 12544 -> 768 (same dim as BGE),
               rerun hard-pair + cross-domain eval. Rules out "high-dim
               near-orthogonality gives AUROC 1.0 for free".
C2 template  : gsm8k-code responses stripped of the fixed template
               (def solution(): header, indentation, return line -> bare
               assignments + final expression), gradients re-extracted for that
               domain, hard pair re-evaluated. Rules out "the subspace just
               recognizes template tokens".
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from boundary import (DOMAINS, N_CAL, N_FIT, N_TEST, ALPHA, fit_subspace, score,
                      split_domain, unit)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pilot"


def hard_eval(X, dom, tag, seed=0):
    from sklearn.metrics import roc_auc_score
    X = unit(X)
    idx_by_dom = {d: np.where(dom == d)[0] for d in DOMAINS}
    splits = {d: split_domain(idx_by_dom, d, np.random.default_rng(seed + hash(d) % 1000))
              for d in DOMAINS}
    fit, cal, test_in = splits["gsm8k-code"]
    U = fit_subspace(X[fit])
    s_cal = score(U, X[cal])
    thr = np.sort(s_cal)[max(int(np.floor(ALPHA * (len(s_cal) + 1))) - 1, 0)]
    s_in = score(U, X[test_in])
    hard = splits["gsm8k-cot"][2]
    s_hard = score(U, X[hard])
    out_ids = np.concatenate([splits[d][2] for d in DOMAINS if d != "gsm8k-code"])
    s_out = score(U, X[out_ids])
    yh = np.r_[np.ones_like(s_in), np.zeros_like(s_hard)]
    yo = np.r_[np.ones_like(s_in), np.zeros_like(s_out)]
    print(f"[{tag}] r={U.shape[1]} "
          f"hard-AUROC={roc_auc_score(yh, np.r_[s_in, s_hard]):.3f} "
          f"hard-reject={float((s_hard < thr).mean()):.2f} "
          f"cross-AUROC={roc_auc_score(yo, np.r_[s_in, s_out]):.3f} "
          f"coverage={float((s_in >= thr).mean()):.2f} "
          f"false-accept={float((s_out >= thr).mean()):.2f}")


def strip_template(code: str) -> str:
    lines = [l for l in code.splitlines() if not l.strip().startswith("def solution")]
    out = []
    for l in lines:
        l = l.strip()
        if l.startswith("return "):
            l = l[len("return "):]
        out.append(l)
    return "\n".join(out)


def extract_stripped():
    """Re-extract grad features with stripped gsm8k-code responses (GPU)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from grad_features import (MODEL, PROJ_DIM, MAX_PROMPT_TOK, MAX_RESP_TOK,
                               stable_seed)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad = "lora_B" in n
    lora_b = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
    projs = {}
    for n, p in lora_b:
        g = torch.Generator(device=dev).manual_seed(stable_seed(n))
        projs[n] = torch.randn(p.numel(), PROJ_DIM, generator=g, device=dev,
                               dtype=torch.float32) / math.sqrt(PROJ_DIM)

    rows = [json.loads(l) for l in open(ROOT / "data" / "pilot" / "gsm8k-code.jsonl")]
    feats = []
    for i, r in enumerate(rows):
        resp = strip_template(r["response"])
        msgs = [{"role": "user", "content": r["prompt"]}]
        ptxt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        pids = tok(ptxt, add_special_tokens=False)["input_ids"][:MAX_PROMPT_TOK]
        rids = tok(resp, add_special_tokens=False)["input_ids"][:MAX_RESP_TOK] + [tok.eos_token_id]
        input_ids = torch.tensor([pids + rids], device=dev)
        labels = torch.tensor([[-100] * len(pids) + rids], device=dev)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        vec = torch.cat([projs[n].T @ p.grad.detach().flatten().float()
                         for n, p in lora_b])
        feats.append((vec / (vec.norm() + 1e-8)).cpu().numpy())
        if (i + 1) % 40 == 0:
            print(f"stripped: {i + 1}/{len(rows)}", flush=True)
    np.savez_compressed(OUT / "features_grad_stripped.npz", X=np.stack(feats))


def main():
    d = np.load(OUT / "features_grad.npz", allow_pickle=True)
    X, dom = d["X"].astype(np.float64), d["domain"].astype(str)

    # C1: dim-matched random projection to 768
    rng = np.random.default_rng(0)
    P = rng.standard_normal((X.shape[1], 768)) / np.sqrt(768)
    hard_eval(X, dom, "grad-12544 (orig)")
    hard_eval(X @ P, dom, "grad-768 (dim-matched)")

    # C2: template-stripped gsm8k-code gradients
    f = OUT / "features_grad_stripped.npz"
    if not f.exists():
        extract_stripped()
    Xs = np.load(f)["X"].astype(np.float64)
    X2 = X.copy()
    X2[dom == "gsm8k-code"] = Xs
    hard_eval(X2, dom, "grad-stripped-template")
    hard_eval(X2 @ P, dom, "grad-stripped-768")


if __name__ == "__main__":
    main()
