#!/usr/bin/env python3
"""Empirical coupling study: how do in-task (gsm8k-code) and off-task
(sql-join) capabilities share the capability-atom vocabulary?

Fits one JOINT dictionary on response-view gradient features of both
tasks' rows, then reads each atom's mass split between the tasks. The
sharing spectrum quantifies coupling; combined with the per-atom budget
quota it explains WHY quota+gate acquisition cannot leak: purchases are
capped at the task-demand of each atom, and atoms demanded by the other
task only carry demand there, so no money flows to them.

Outputs results/coupling_study.json and a console summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import features as _features  # noqa: E402
from distill_pilot import split  # noqa: E402


def pair_stats(grad, n_g, tag, out_all):
    from sklearn.decomposition import MiniBatchDictionaryLearning
    dico = MiniBatchDictionaryLearning(
        n_components=64, alpha=0.05, transform_algorithm="lasso_lars",
        transform_alpha=0.05, random_state=0, max_iter=200, batch_size=16,
    )
    dico.fit(grad)
    codes = np.abs(dico.transform(grad))
    mass_g = codes[:n_g].sum(axis=0)
    mass_j = codes[n_g:].sum(axis=0)
    dem_g = mass_g / max(mass_g.sum(), 1e-12)
    dem_j = mass_j / max(mass_j.sum(), 1e-12)
    kappa = float(np.minimum(dem_g, dem_j).sum())
    shared_mask = (mass_g > 1e-6) & (mass_j > 1e-6)
    out_all[tag] = {
        "kappa": kappa,
        "gsm_only": int(((mass_g > 1e-6) & ~shared_mask).sum()),
        "other_only": int(((mass_j > 1e-6) & ~shared_mask).sum()),
        "shared": int(shared_mask.sum()),
        "gsm_demand_on_shared": float(dem_g[shared_mask].sum()),
        "other_demand_on_shared": float(dem_j[shared_mask].sum()),
    }
    print(tag, json.dumps(out_all[tag]))


def main() -> int:
    import os
    view = os.environ.get("COUP_VIEW", "resp")
    gsm_rows, _ = split("gsm8k-code")
    gsm = gsm_rows[:60]
    pairs = {}
    for dom in ("sql", "pandas", "gsm8k-cot"):
        rows_d, _ = split(dom)
        if dom == "sql":
            rows_d = [r for r in rows_d
                      if "join" in (r["prompt"] + r["response"]).casefold()]
        pairs[dom] = rows_d[:60]
    out_all = {}
    for dom, other in pairs.items():
        rows = gsm + other
        if view == "prompt":
            rows = [{"prompt": "", "response": r["prompt"]} for r in rows]
        grad = _features.extract_features(
            "Qwen/Qwen3.5-2B", rows, proj_dim=64,
            max_prompt_tok=1024, max_resp_tok=512, device="cuda",
        ).astype(np.float64)
        grad = grad / np.maximum(
            np.linalg.norm(grad, axis=1, keepdims=True), 1e-12
        )
        pair_stats(grad, len(gsm), f"{view}:{dom}", out_all)
    (ROOT / "results" / f"coupling_pairs_{view}.json").write_text(
        json.dumps(out_all, indent=2)
    )
    return 0


def _legacy():

    from sklearn.decomposition import MiniBatchDictionaryLearning
    dico = MiniBatchDictionaryLearning(
        n_components=64, alpha=0.05, transform_algorithm="lasso_lars",
        transform_alpha=0.05, random_state=0, max_iter=200, batch_size=16,
    )
    dico.fit(grad)
    codes = np.abs(dico.transform(grad))
    n_g = len(gsm)
    mass_g = codes[:n_g].sum(axis=0)
    mass_j = codes[n_g:].sum(axis=0)
    total = mass_g + mass_j
    active = total > 1e-9
    share = np.zeros_like(total)
    share[active] = np.minimum(mass_g[active], mass_j[active]) / (
        np.maximum(mass_g[active], mass_j[active])
    )
    # classify atoms
    gsm_only = int(((mass_g > 1e-6) & (mass_j <= 1e-6)).sum())
    join_only = int(((mass_j > 1e-6) & (mass_g <= 1e-6)).sum())
    shared = int(((mass_g > 1e-6) & (mass_j > 1e-6)).sum())
    # how much of each task's demand sits on shared atoms
    dem_g = mass_g / max(mass_g.sum(), 1e-12)
    dem_j = mass_j / max(mass_j.sum(), 1e-12)
    shared_mask = (mass_g > 1e-6) & (mass_j > 1e-6)
    gsm_demand_on_shared = float(dem_g[shared_mask].sum())
    join_demand_on_shared = float(dem_j[shared_mask].sum())
    # strong sharing = atoms where the smaller task mass is >25% of larger
    strong = shared_mask & (share > 0.25)
    out = {
        "n_atoms_active": int(active.sum()),
        "gsm_only": gsm_only,
        "join_only": join_only,
        "shared": shared,
        "strongly_shared": int(strong.sum()),
        "gsm_demand_on_shared_atoms": gsm_demand_on_shared,
        "join_demand_on_shared_atoms": join_demand_on_shared,
        "gsm_demand_on_strong": float(dem_g[strong].sum()),
        "join_demand_on_strong": float(dem_j[strong].sum()),
        "share_spectrum_deciles": [
            float(x) for x in np.percentile(share[shared_mask], range(0, 101, 10))
        ] if shared else [],
    }
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "coupling_study.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
