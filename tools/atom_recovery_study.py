#!/usr/bin/env python3
"""Few-shot atom recovery study: how much of the FULL-training-set
capability-atom dictionary does a k-query support set recover?

Reference dictionary D* <- all task training rows (response-view gradient
features). Few-shot dictionary D_k <- k-subsets. Metrics per k (mean over
--trials random subsets):
  recovery  : per reference atom, max |cos| to any D_k atom (weighted by
              reference demand), i.e. how much of the needed vocabulary
              the few-shot dictionary can express;
  demand_l1 : L1 distance between the reference demand vector and the
              k-subset demand mapped through atom matching;
Sample-complexity anchor: dictionary identifiability theory (e.g. Arora
et al., 2014) gives k = O(m log m) for m incoherent atoms; the empirical
curve is read against m = the number of demanded reference atoms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from evol_extend_pool import load_support_rows  # noqa: E402


def fit_dict(feat, seed):
    from sklearn.decomposition import MiniBatchDictionaryLearning
    d = MiniBatchDictionaryLearning(
        n_components=int(min(64, max(8, len(feat) // 2))),
        alpha=0.05, transform_algorithm="lasso_lars", transform_alpha=0.05,
        random_state=seed, max_iter=100, batch_size=16,
    )
    d.fit(feat)
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ks", default="10,25,50,100,200")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", default="results/atom_recovery_study.json")
    parser.add_argument("--reference", choices=("train", "pool"),
                        default="train")
    parser.add_argument("--grow", default="",
                        help="comma list: also test 50 support + N random "
                             "on-task rows (in-loop refit approximation)")
    args = parser.parse_args()

    import numpy as np
    import features as _features

    # full task training rows (the support resolver without the k cap
    # returns the task's full train split minus the test rows)
    import importlib
    import e3_tier1 as e3
    import yaml
    with (ROOT / "configs" / "e3_tier1.yaml").open() as fh:
        cfg = yaml.safe_load(fh)
    cfg["task_boundary"]["domain"] = "gsm8k-code"
    cfg["task_boundary"]["filter"] = None
    cfg["task_boundary"]["spec_rows"] = 10**6  # take everything eligible
    cfg["task_boundary"]["fit_rows"] = 10**6 - 1
    cfg["task_boundary"]["calibration_rows"] = 1
    try:
        all_rows, _, _ = e3.resolve_task_rows(cfg)
    except Exception:
        all_rows = load_support_rows(limit=None)
    if args.reference == "pool":
        # widest on-task reference: every gsm8k-code trace in the
        # candidate pool (all teachers), deduplicated by prompt
        full_cfg = {"task_boundary": {"domain": "gsm8k-code",
                                      "filter": None}}
        try:
            pool_rows = e3.load_candidate_pool(cfg)
        except TypeError:
            pool_rows = e3.load_candidate_pool()
        seen, merged = set(), []
        for row in list(all_rows) + [r for r in pool_rows
                                     if r.get("domain") == "gsm8k-code"]:
            if row["prompt"] in seen:
                continue
            seen.add(row["prompt"])
            merged.append({"prompt": row["prompt"],
                           "response": row["response"]})
        all_rows = merged
    print(f"[study] reference rows: {len(all_rows)}", flush=True)

    feat = _features.extract_features(
        args.student, all_rows, proj_dim=64,
        max_prompt_tok=1024, max_resp_tok=512, device=args.device,
    ).astype(np.float64)

    ref = fit_dict(feat, 0)
    ref_codes = np.abs(ref.transform(feat))
    ref_demand = ref_codes.mean(axis=0)
    ref_demand = ref_demand / max(ref_demand.sum(), 1e-12)
    demanded = ref_demand > 1e-3
    m_atoms = int(demanded.sum())
    refc = ref.components_ / np.maximum(
        np.linalg.norm(ref.components_, axis=1, keepdims=True), 1e-12)
    print(f"[study] reference atoms: {ref.n_components}, demanded: {m_atoms}",
          flush=True)

    results = {"n_rows": len(all_rows), "ref_atoms": int(ref.n_components),
               "demanded_atoms": m_atoms, "per_k": {}}
    rng = np.random.default_rng(0)
    for k in [int(x) for x in args.ks.split(",") if x]:
        if k > len(all_rows):
            continue
        recs, dls = [], []
        for t in range(args.trials):
            idx = rng.choice(len(all_rows), size=k, replace=False)
            dk = fit_dict(feat[idx], t)
            dkc = dk.components_ / np.maximum(
                np.linalg.norm(dk.components_, axis=1, keepdims=True), 1e-12)
            sim = np.abs(refc @ dkc.T)          # (ref_atoms, k_atoms)
            best = sim.max(axis=1)              # recovery per ref atom
            rec = float((best * ref_demand).sum())  # demand-weighted
            # demand transported onto reference vocabulary
            k_codes = np.abs(dk.transform(feat[idx]))
            k_demand = k_codes.mean(axis=0)
            k_demand = k_demand / max(k_demand.sum(), 1e-12)
            match = sim.argmax(axis=1)
            k_on_ref = np.array([k_demand[match[a]] for a in
                                 range(len(ref_demand))])
            k_on_ref = k_on_ref / max(k_on_ref.sum(), 1e-12)
            dl = float(np.abs(k_on_ref - ref_demand).sum())
            recs.append(rec)
            dls.append(dl)
        results["per_k"][k] = {
            "recovery_mean": float(np.mean(recs)),
            "recovery_std": float(np.std(recs)),
            "demand_l1_mean": float(np.mean(dls)),
        }
        print(f"[study] k={k}: demand-weighted recovery "
              f"{np.mean(recs):.3f}±{np.std(recs):.3f}, "
              f"demand L1 gap {np.mean(dls):.3f}", flush=True)

    for n_extra in [int(x) for x in args.grow.split(",") if x]:
        recs = []
        for t in range(args.trials):
            sup = rng.choice(len(all_rows), size=min(50, len(all_rows)),
                             replace=False)
            rest = np.setdiff1d(np.arange(len(all_rows)), sup)
            extra = rng.choice(rest, size=min(n_extra, len(rest)),
                               replace=False)
            idx = np.concatenate([sup, extra])
            dk = fit_dict(feat[idx], t)
            dkc = dk.components_ / np.maximum(
                np.linalg.norm(dk.components_, axis=1, keepdims=True), 1e-12)
            best = np.abs(refc @ dkc.T).max(axis=1)
            recs.append(float((best * ref_demand).sum()))
        results["per_k"][f"50+{n_extra}"] = {
            "recovery_mean": float(np.mean(recs)),
            "recovery_std": float(np.std(recs)),
        }
        print(f"[study] 50 support + {n_extra} bought: recovery "
              f"{np.mean(recs):.3f}±{np.std(recs):.3f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[study] saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
