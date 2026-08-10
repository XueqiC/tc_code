"""Per-sample gradient features (LESS-style) for boundary estimation.

For each (prompt, response): loss on response tokens under the base model with
fresh LoRA adapters -> backward -> collect grads of lora_B weights only.
At init B=0, so dL/dB = G @ A^T is a random projection of the full-weight
gradient G by the fixed random A — the LESS trick. Each module's B-grad is
further JL-projected to PROJ_DIM dims with a per-module fixed-seed Gaussian,
then all modules are concatenated and L2-normalized.

Output: results/pilot/features_grad.npz  (X: [N, D], domain: [N] str, idx: [N])
"""
import json
import os
from pathlib import Path

import numpy as np

from features import extract_features, stable_seed as _stable_seed

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("TC_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TAG = os.environ.get("TC_TAG", "")  # suffix for the output file, e.g. "_qwen35-4b"
DOMAINS = ["gsm8k-code", "gsm8k-cot", "pandas", "sql", "alpaca"]
PROJ_DIM = 64
MAX_PROMPT_TOK = 640
MAX_RESP_TOK = 512
SEED = 0


def stable_seed(name: str) -> int:
    return _stable_seed(name)


def main():
    rows, domains, idxs = [], [], []
    for dom in DOMAINS:
        domain_rows = [json.loads(l) for l in open(
            ROOT / "data" / "pilot" / f"{dom}.jsonl"
        )]
        rows.extend(domain_rows)
        domains.extend([dom] * len(domain_rows))
        idxs.extend(range(len(domain_rows)))

    feats = extract_features(MODEL, rows, PROJ_DIM, MAX_PROMPT_TOK,
                             MAX_RESP_TOK, "cuda")

    out_dir = ROOT / "results" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"features_grad{TAG}.npz",
                        X=feats, domain=np.array(domains),
                        idx=np.array(idxs))
    print("saved", out_dir / f"features_grad{TAG}.npz")


if __name__ == "__main__":
    main()
