"""M2/M3 v3: atom absorption dynamics via a two-model probe protocol.

Fix for the r16-vs-r8 basis mismatch and peft merge-state fragility:
a REFERENCE extractor model (features._prepare_extractor: base + fresh r8
zero-init LoRA, dictionary-matched JL basis) is kept alongside the TRAINING
model (r16). At each checkpoint the training model's effective weights
W_eff = W0 + scale * B @ A are copied into the reference model's base
weights; probe features of the 30 test rows are then extracted on the
reference model in the dictionary's exact basis.
"""
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
from transformers import AutoModelForCausalLM

import verifier
from boundary import unit
from distill_pilot import MODEL, encode, split
from features import _prepare_extractor, _project_gradients

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "m2_m3"
CHECKPOINTS = (0, 2, 5, 11, 22, 33, 44, 66)
N_ATOMS, ALPHA = 128, 0.05
SEED, LR, ACCUM, EPOCHS = 0, 2e-4, 8, 6


def fit_dictionary():
    d = np.load(ROOT / "results" / "pilot" / "features_gradsteps.npz",
                allow_pickle=True)
    Xs = unit(d["X_steps"].astype(np.float64))
    dico = MiniBatchDictionaryLearning(
        n_components=N_ATOMS, alpha=ALPHA, batch_size=256, max_iter=150,
        fit_algorithm="lars", transform_algorithm="lasso_lars",
        transform_alpha=ALPHA, random_state=0, n_jobs=4).fit(Xs)
    return dico.components_


def lora_modules(peft_model):
    out = {}
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "base_layer") and hasattr(mod, "lora_B"):
            out[name] = mod
    return out


def sync_ref_weights(train_mods, ref_mods):
    with torch.no_grad():
        for name, tm in train_mods.items():
            A = tm.lora_A["default"].weight
            B = tm.lora_B["default"].weight
            scale = tm.scaling["default"]
            w_eff = tm.base_layer.weight + scale * (B @ A).to(tm.base_layer.weight.dtype)
            ref_mods[name].base_layer.weight.copy_(w_eff)


def probe_features(ref_model, tok, rows, dev, ref_lora_b, projs):
    feats = []
    ref_model.train()
    for r in rows:
        ids, labels = encode(tok, r, dev)
        ref_model.zero_grad(set_to_none=True)
        ref_model(input_ids=ids, labels=labels).loss.backward()
        feats.append(_project_gradients(ref_lora_b, projs).cpu().numpy())
    ref_model.zero_grad(set_to_none=True)
    return np.stack(feats).astype(np.float64)


@torch.no_grad()
def exec_success(model, tok, rows, golds, dev):
    model.eval()
    out = []
    for r, gold in zip(rows, golds):
        msgs = [{"role": "user", "content": r["prompt"]}]
        ptxt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(ptxt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        gen = model.generate(ids, max_new_tokens=320, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        s = text.find("def solution")
        pred = verifier.run_solution(text[s:]) if s >= 0 else None
        out.append(pred is not None and gold is not None and abs(pred - gold) < 1e-4)
    model.train()
    return np.array(out)


def main():
    dev = "cuda"
    D = fit_dictionary()
    coder = SparseCoder(D, transform_algorithm="lasso_lars", transform_alpha=ALPHA)

    tok, ref_model, ref_lora_b, projs = _prepare_extractor(MODEL, 64, dev)
    torch.manual_seed(SEED)
    train_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).to(dev)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    train_model = get_peft_model(train_model, lora)
    t_mods, r_mods = lora_modules(train_model), lora_modules(ref_model)
    assert set(t_mods) == set(r_mods), "module name mismatch between models"

    train, test = split("gsm8k-code")
    golds = [verifier.run_solution(r["response"]) for r in test]

    codes_by_ckpt, mean_feats, succ_by_ckpt = {}, {}, {}

    def measure(step):
        sync_ref_weights(t_mods, r_mods)
        F = probe_features(ref_model, tok, test, dev, ref_lora_b, projs)
        C = np.abs(coder.transform(F))
        active = float((C > 1e-8).sum(1).mean())
        if step == 0:
            assert active > 1.0, f"probe coding degenerate (active={active})"
        codes_by_ckpt[step] = C
        mean_feats[step] = F.mean(0)
        succ_by_ckpt[step] = exec_success(train_model, tok, test, golds, dev)
        print(f"checkpoint step={step:2d} exec={succ_by_ckpt[step].mean():.3f} "
              f"active_atoms/row={active:.1f} mass/row={C.sum(1).mean():.3f}",
              flush=True)

    measure(0)
    opt = torch.optim.AdamW([p for p in train_model.parameters() if p.requires_grad], lr=LR)
    rng = np.random.default_rng(SEED)
    step = 0
    train_model.train()
    for _ep in range(EPOCHS):
        order = rng.permutation(len(train))
        full = order[:(len(order) // ACCUM) * ACCUM]
        for j, i in enumerate(full, 1):
            ids, labels = encode(tok, train[i], dev)
            (train_model(input_ids=ids, labels=labels).loss / ACCUM).backward()
            if j % ACCUM == 0:
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step in CHECKPOINTS:
                    measure(step)
    opt.zero_grad(set_to_none=True)

    steps = sorted(codes_by_ckpt)
    mean_codes = np.stack([codes_by_ckpt[s].mean(0) for s in steps])
    deltas = []
    for a, b in zip(steps[:-1], steps[1:]):
        d = mean_feats[b] - mean_feats[a]
        d = d / (np.linalg.norm(d) + 1e-12)
        dc = np.abs(coder.transform(d[None, :]))[0]
        top5 = float(np.sort(dc)[-5:].sum() / (dc.sum() + 1e-12))
        deltas.append({"interval": f"{a}->{b}", "top5_mass_fraction": top5,
                       "total_mass": float(dc.sum())})
        print(f"delta {a}->{b}: top5_fraction={top5:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "records_v3.npz",
                        steps=np.array(steps), mean_codes=mean_codes,
                        success=np.stack([succ_by_ckpt[s] for s in steps]),
                        codes_ckpt0=codes_by_ckpt[0],
                        codes_final=codes_by_ckpt[steps[-1]])
    json.dump({"deltas": deltas,
               "exec_by_ckpt": {str(s): float(succ_by_ckpt[s].mean()) for s in steps}},
              open(OUT / "summary_v3.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = np.argsort(-mean_codes[0])[:30]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor="white")
    im = axes[0].imshow(mean_codes[:, top].T, aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(steps)), steps)
    axes[0].set_xlabel("optimizer step"); axes[0].set_ylabel("atom (top-30 by initial demand)")
    axes[0].set_title("Atom demand across training")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].plot(steps, [succ_by_ckpt[s].mean() for s in steps], color="#eb6834",
                 marker="o", lw=2)
    axes[1].set_ylim(-0.03, 1.03); axes[1].set_xlabel("optimizer step")
    axes[1].set_title("Exec success")
    axes[2].bar(range(len(deltas)), [d["top5_mass_fraction"] for d in deltas],
                color="#2a78d6")
    axes[2].set_xticks(range(len(deltas)), [d["interval"] for d in deltas], rotation=45)
    axes[2].set_ylim(0, 1); axes[2].set_title("Delta concentration (top-5 atom fraction)")
    for ax in axes:
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figs" / "m2_m3.png", dpi=150, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
