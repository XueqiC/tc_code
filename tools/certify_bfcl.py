#!/usr/bin/env python3
"""Block IV certification for the BFCL teacher-consistent arm (crcd_r3_union_t_s0).

Three certificates (src/bfas/certify.py), no tuning on held-out data:

  1. representation coverage of the K=32 r3t PCA basis on held-out fingerprints
     (bfcl_r2_v1, bfcl_r3_v1 minus calibration rows; strict = state_hash not in the
     r3t fit set; group-aware cross-fit on the r3t fingerprints; ALFWorld on the BFCL
     basis), each with bootstrap CIs and random / self-PCA references;
  2. atom-wise residual certificate from the primal-dual traces
     (results/analysis/mirror_bf3_{rho1,rhoJ0,k1}_trace.jsonl): per atom J_k, rho_k,
     r_k, n_k, Hoeffding+Bonferroni LCB[J_k] and UCB[r_k], lambda_k trajectories;
  3. held-out risk on the official single-turn BFCL categories (exact
     Clopper–Pearson, McNemar paired against base, Bonferroni-simultaneous category
     UCBs, category-level risk–coverage curve) and on the 60 never-trained calibration
     prompts (data/bfcl_sft/calibration_ids.json) where per-prompt outcomes exist.

Usage
  # one-off, needs the BFCL harness venv (renders prompts -> sha1 hash map)
  PYTHONPATH=src:tools envs/bfcl-venv/bin/python tools/certify_bfcl.py hashes
  # everything else: plain numpy/scipy
  .venv/bin/python tools/certify_bfcl.py run [--student crcd_r3_union_t_s0] [--base base_hpg]

Outputs results/analysis/certify_bfcl_hashmap.json and results/analysis/certify_bfcl_r3t.json.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import certify  # noqa: E402

BFCL_DATA = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
SINGLE_TURN = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
               "parallel_multiple", "irrelevance", "live_simple", "live_multiple", "live_parallel",
               "live_parallel_multiple", "live_irrelevance", "live_relevance"]
HASHMAP = ROOT / "results/analysis/certify_bfcl_hashmap.json"
OUT = ROOT / "results/analysis/certify_bfcl_r3t.json"


def _seed_task(task_id: str) -> str:
    """'gen_live_parallel_multiple_10-9-0_1' -> 'live_parallel_multiple_10-9-0' (fingerprint._base_task_id)."""
    t = re.sub(r"^(gen|genmt|oos)_", "", task_id)
    return re.sub(r"_\d+$", "", t)


def _is_generated(task_id: str) -> bool:
    return task_id.startswith(("gen_", "genmt_", "oos_"))


# ==============================================================================================
# hashes: render prompts exactly as the miner does and map sha1(prompt)[:16] -> task
# ==============================================================================================
def cmd_hashes(args) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    from bfas.adapters.bfcl import BFCLAdapter
    from bfcl_eval.utils import populate_test_cases_with_predefined_functions

    ad = BFCLAdapter()
    entries, cats = ad._load_entries()

    def h(prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]

    def render(question, functions):
        turn = question[0] if question and isinstance(question[0], list) else question
        msgs = [dict(m) for m in turn if isinstance(m, dict)]
        return ad._render(msgs, functions)

    official = {}
    for tid, entry in entries.items():
        cat = cats[tid]
        if cat not in SINGLE_TURN:
            continue
        pop = populate_test_cases_with_predefined_functions([json.loads(json.dumps(entry))])[0]
        official[h(render(pop["question"], pop.get("function") or []))] = {"id": tid, "category": cat}
    generated = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "data/bfcl_sft/gen_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if "question" not in row or "function" not in row:
                continue
            try:
                key = h(render(row["question"], row["function"]))
            except Exception:
                continue
            generated[key].append({"id": row.get("id"), "file": Path(f).name,
                                   "category": row.get("seed_category") or row.get("category"),
                                   "seed_task": _seed_task(str(row.get("id"))),
                                   "oos": row.get("out_of_scope")})
    cal = json.load(open(ROOT / "data/bfcl_sft/calibration_ids.json"))
    hashes = cal["hashes"]
    out = {
        "n_official_single_turn_rendered": len(official),
        "n_generated_hashes": len(generated),
        "calibration": {k: {"official": official.get(k), "generated": generated.get(k)} for k in hashes},
        "official_hash_to_id": official,
    }
    n_off = sum(1 for k in hashes if k in official)
    n_gen = sum(1 for k in hashes if k in generated)
    out["calibration_summary"] = {"n": len(hashes), "n_official": n_off, "n_generated": n_gen,
                                  "n_unmapped": len(hashes) - len(set(hashes) & (set(official) | set(generated)))}
    HASHMAP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(HASHMAP, "w"), indent=1)
    print(f"[certify] hash map -> {HASHMAP}: {out['calibration_summary']}")


# ==============================================================================================
# 1. representation coverage
# ==============================================================================================
def coverage_certificate(n_boot: int, seed: int) -> dict:
    atoms = np.load(ROOT / "data/atoms/bfcl_r3t_K32.npz")
    U = np.asarray(atoms["U"], dtype=np.float64)
    K = U.shape[1]
    fit_hashes = set(map(str, atoms["fingerprint_state_hash"]))
    out = {"basis": "data/atoms/bfcl_r3t_K32.npz", "K": int(K), "d": int(U.shape[0]),
           "method": str(atoms["method"]), "n_fit_rows": int(len(atoms["fingerprint_state_hash"])),
           "n_fit_unique_state_hash": len(fit_hashes),
           "definition": "per-event c_i = ||P_U psi_i||^2 / ||psi_i||^2 (psi L2-normalised, Fisher-whitened); "
                         "set = 1 - min_Z ||Psi - U Z||_F^2 / ||Psi||_F^2; CI = percentile bootstrap over events",
           "sets": {}}

    def evaluate(name: str, Psi: np.ndarray, note: str, groups=None) -> dict:
        c = certify.coverage_per_event(U, Psi)
        rec = {"n": int(len(Psi)), "note": note,
               "per_event_mean": certify.bootstrap_ci(c, n_boot=n_boot, seed=seed),
               "per_event_median": float(np.nanmedian(c)) if len(c) else float("nan"),
               "per_event_q10_q90": ([float(np.nanquantile(c, 0.1)), float(np.nanquantile(c, 0.9))]
                                     if len(c) else [float("nan")] * 2),
               "set": certify.coverage_set(U, Psi),
               "reference_random_basis": certify.random_basis_coverage(Psi, K, n_rep=20, seed=seed),
               "reference_self_pca_in_sample": certify.self_pca_coverage(Psi, K) if len(Psi) >= K else None}
        if len(Psi) >= 2 * K and groups is not None:
            cf = certify.crossfit_coverage(Psi, K, groups=groups, n_folds=5, seed=seed)
            rec["reference_self_pca_crossfit"] = {
                "per_event_mean": certify.bootstrap_ci(cf["per_event"], n_boot=n_boot, seed=seed),
                "fold_set_mean": float(np.mean(cf["fold_set"])), "n_groups": cf["n_groups"]}
        return rec

    for tag in ["bfcl_r2_v1", "bfcl_r3_v1"]:
        d = np.load(ROOT / f"data/fingerprints/{tag}.npz")
        psi = np.asarray(d["psi"], dtype=np.float64)
        hashes = np.array([str(h) for h in d["state_hash"]])
        cal = set(int(i) for i in d["calibration_idx"])
        keep = np.array([i not in cal for i in range(len(psi))])
        strict = keep & np.array([h not in fit_hashes for h in hashes])
        out["sets"][f"{tag}_spec"] = evaluate(
            tag, psi[keep], f"all rows minus {len(cal)} calibration rows; "
            f"{int((keep & ~strict).sum())} of {int(keep.sum())} rows share a state_hash with the r3t fit set (in-sample)",
            groups=hashes[keep])
        out["sets"][f"{tag}_strict_heldout"] = evaluate(
            tag, psi[strict], "rows minus calibration rows and minus any state_hash present in the r3t fit set",
            groups=hashes[strict])
    # cross-fit on the fit set itself (state_hash groups never split across folds)
    d = np.load(ROOT / "data/fingerprints/bfcl_r3t_v1.npz")
    psi = np.asarray(d["psi"], dtype=np.float64)
    hashes = np.array([str(h) for h in d["state_hash"]])
    cal = set(int(i) for i in d["calibration_idx"])
    keep = np.array([i not in cal for i in range(len(psi))])
    cf = certify.crossfit_coverage(psi[keep], K, groups=hashes[keep], n_folds=5, seed=seed)
    out["sets"]["bfcl_r3t_v1_crossfit"] = {
        "n": int(keep.sum()), "note": "5-fold group cross-fit on the r3t fingerprints themselves "
        "(PCA-32 refit on 4/5 of the state_hash groups, scored on the held-out fifth); "
        "the honest held-out coverage estimate for a basis of this size fitted on this pool",
        "per_event_mean": certify.bootstrap_ci(cf["per_event"], n_boot=n_boot, seed=seed),
        "fold_set_mean": float(np.mean(cf["fold_set"])), "fold_sets": cf["fold_set"],
        "n_groups": cf["n_groups"],
        "in_sample_set": certify.coverage_set(U, psi[keep]),
        "in_sample_per_event_mean": float(np.nanmean(certify.coverage_per_event(U, psi[keep]))),
        "reference_random_basis": certify.random_basis_coverage(psi[keep], K, n_rep=20, seed=seed)}
    # cross-benchmark
    d = np.load(ROOT / "data/fingerprints/alfworld_v1_v1.npz")
    psi = np.asarray(d["psi"], dtype=np.float64)
    out["sets"]["alfworld_v1_on_bfcl_basis"] = evaluate(
        "alfworld", psi, "ALFWorld Fisher-whitened fingerprints projected on the BFCL r3t basis "
        "(cross-benchmark; different Fisher, expected low)", groups=np.array([str(h) for h in d["state_hash"]]))
    return out


# ==============================================================================================
# 2. atom-wise residual certificate from the primal-dual traces
# ==============================================================================================
def atom_certificate_from_trace(path: Path, delta: float, min_n_eff: float, ucb_threshold: float,
                                n_boot: int, seed: int) -> dict:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    cfg = recs[0]
    assert cfg["kind"] == "config"
    steps = [r for r in recs if r["kind"] == "step"]
    duals = [r for r in recs if r["kind"] == "dual"]
    summ = [r for r in recs if r["kind"] == "summary"]
    summ = summ[0] if summ else {}
    K = int(cfg["K"])
    rho = cfg["rho"] if isinstance(cfg.get("rho"), list) else (duals[-1]["rho"] if duals else [1.0] * K)
    rho = np.asarray(rho, dtype=np.float64)
    eps = float(cfg.get("eps", 0.0))
    kappa = float(cfg.get("kappa", 1.0))
    # last (and first) visit per pool row: the trainer visits each row once per pass
    last, first = {}, {}
    seed_of_row = {}
    for r in steps:
        for e in r["events"]:
            row = int(e["row"])
            last[row] = (e["zbar"], float(e["J_value"]), r["step"])
            first.setdefault(row, (e["zbar"], float(e["J_value"]), r["step"]))
            seed_of_row[row] = _seed_task(e["event_id"].split("#")[0])
    rows = sorted(last)
    Z = np.array([last[i][0] for i in rows], dtype=np.float64)
    v = np.array([last[i][1] for i in rows], dtype=np.float64)
    v0 = np.array([first[i][1] for i in rows], dtype=np.float64)
    groups = np.array([seed_of_row[i] for i in rows])
    lam_final = np.asarray(steps[-1]["lambda"], dtype=np.float64)
    cert = certify.atom_certificate(Z, v, rho, delta=delta, min_n_eff=min_n_eff, lambda_final=lam_final)
    # descriptive (not a guarantee): seed-task cluster bootstrap 5% quantile of J_k
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    boots = np.empty((n_boot, K))
    for b in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([idx_of[uniq[p]] for p in pick])
        Zb, vb = Z[sel], v[sel]
        den = Zb.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots[b] = np.where(den > 0, (Zb * vb[:, None]).sum(axis=0) / den, np.nan)
    cluster_q = np.nanquantile(boots, delta, axis=0)
    lam_traj = np.array([r["lambda"] for r in steps], dtype=np.float64)
    lam_summary = certify.lambda_trajectory_summary(lam_traj, cap=kappa)
    per_atom = []
    for c in cert:
        d = c.as_dict()
        d.update({"eps": eps, "satisfied_point": bool(c.J >= c.rho - eps) if not np.isnan(c.J) else None,
                  "satisfied_certified": bool(c.lcb_J >= c.rho - eps) if not np.isnan(c.lcb_J) else None,
                  "J_first_visit": float(np.sum(Z[:, c.k] * v0) / Z[:, c.k].sum()) if Z[:, c.k].sum() > 0 else None,
                  "J_trainer_window_final": (float(summ["J_final"][c.k]) if summ else None),
                  "J0_trainer": (float(summ["J0"][c.k]) if summ else None),
                  "cluster_bootstrap_q05_J": float(cluster_q[c.k]),
                  "lambda": lam_summary[c.k]})
        per_atom.append(d)
    flagged = [d["k"] for d in per_atom if d["resolved"] and d["UCB_r"] > ucb_threshold]
    unresolved = [d["k"] for d in per_atom if not d["resolved"]]
    return {
        "trace": str(path.relative_to(ROOT)), "tag": cfg.get("tag"), "K": K, "n_rows": len(rows),
        "n_visits": sum(len(r["events"]) for r in steps), "steps": len(steps),
        "n_seed_task_clusters": int(len(uniq)),
        "rho": ("uniform %.3g" % rho[0]) if np.allclose(rho, rho[0]) else "per-atom (J0 + %s)" % cfg.get("rho_offsets"),
        "eps": eps, "kappa_cap": kappa, "j_mode": cfg.get("j_mode"), "delta": delta,
        "per_test_level": certify.bonferroni(delta, K), "min_n_eff": min_n_eff,
        "ucb_threshold": ucb_threshold, "flagged_atoms_UCB_r_gt_threshold": flagged,
        "n_flagged": len(flagged), "unresolved_atoms": unresolved,
        "n_certified_satisfied": sum(1 for d in per_atom if d["satisfied_certified"]),
        "n_point_satisfied": sum(1 for d in per_atom if d["satisfied_point"]),
        "summary_stats": {
            "J_mean": float(np.nanmean([d["J"] for d in per_atom])),
            "J_first_visit_mean": float(np.nanmean([d["J_first_visit"] for d in per_atom if d["J_first_visit"] is not None])),
            "r_mean": float(np.nanmean([d["r"] for d in per_atom])),
            "UCB_r_mean": float(np.nanmean([d["UCB_r"] for d in per_atom])),
            "UCB_r_max": float(np.nanmax([d["UCB_r"] for d in per_atom])),
            "half_width_median": float(np.nanmedian([d["half_width"] for d in per_atom])),
            "n_eff_median": float(np.nanmedian([d["n_eff"] for d in per_atom])),
            "n_eff_min": float(np.nanmin([d["n_eff"] for d in per_atom])),
            "n_weighted_min": float(np.nanmin([d["n_weighted"] for d in per_atom])),
            "lambda_final_at_cap": int(sum(1 for d in per_atom if d["lambda"]["final"] >= kappa - 1e-9)),
        },
        "atoms": per_atom,
    }


# ==============================================================================================
# 3. held-out risk on official single-turn categories and calibration prompts
# ==============================================================================================
def official_ids() -> dict[str, list[str]]:
    ids = {}
    for cat in SINGLE_TURN:
        ids[cat] = [json.loads(l)["id"] for l in open(BFCL_DATA / f"BFCL_v4_{cat}.json", encoding="utf-8") if l.strip()]
    return ids


def per_id_correct(tag: str, ids: dict[str, list[str]]) -> dict[str, dict[str, bool]]:
    """correct[cat][id] from results/bfcl_std/<tag>/scoredir: id in the official test file
    and not listed as a failure in the category score file (total_count is asserted to
    equal the number of official ids, i.e. every id was evaluated)."""
    out = {}
    for cat, cat_ids in ids.items():
        files = glob.glob(str(ROOT / f"results/bfcl_std/{tag}/scoredir/*/BFCL_v4_{cat}_score.json"))
        if not files:
            raise FileNotFoundError(f"missing score file for {tag}/{cat}")
        lines = [json.loads(l) for l in open(files[0], encoding="utf-8") if l.strip()]
        head, fails = lines[0], lines[1:]
        fail_ids = {r["id"] for r in fails}
        if head["total_count"] != len(cat_ids):
            raise ValueError(f"{tag}/{cat}: total_count {head['total_count']} != {len(cat_ids)} official ids")
        if not fail_ids <= set(cat_ids):
            raise ValueError(f"{tag}/{cat}: failure ids outside the official id list")
        if head["correct_count"] != len(cat_ids) - len(fail_ids):
            raise ValueError(f"{tag}/{cat}: correct_count inconsistent with the failure list")
        out[cat] = {i: (i not in fail_ids) for i in cat_ids}
    return out


def contamination(pool_path: Path) -> dict:
    rows = [json.loads(l) for l in open(pool_path, encoding="utf-8") if l.strip()]
    tids = sorted({r["task_id"] for r in rows})
    exact = sorted(t for t in tids if not _is_generated(t))
    derived = sorted({_seed_task(t) for t in tids if _is_generated(t)})
    return {"pool": str(pool_path.relative_to(ROOT)), "n_rows": len(rows), "n_task_ids": len(tids),
            "official_ids_trained_exact": exact,
            "official_seed_ids_of_generated_rows": derived}


def risk_certificate(student: str, base: str, alpha: float, taus, n_boot: int, seed: int) -> dict:
    ids = official_ids()
    cor_s = per_id_correct(student, ids)
    cor_b = per_id_correct(base, ids)
    cont = contamination(ROOT / "data/bfcl_sft/pool_events_pref_v3t.jsonl")
    exact = set(cont["official_ids_trained_exact"])
    derived = set(cont["official_seed_ids_of_generated_rows"]) | exact
    hm = json.load(open(HASHMAP)) if HASHMAP.exists() else None

    def table(cor, exclude: set[str]) -> dict:
        per = {c: (sum(1 for i, ok in d.items() if (i not in exclude) and not ok),
                   sum(1 for i in d if i not in exclude)) for c, d in cor.items()}
        return certify.category_risk_table(per, alpha=alpha, simultaneous=True)

    def pooled(cor, exclude: set[str]) -> dict:
        n = sum(1 for d in cor.values() for i in d if i not in exclude)
        f = sum(1 for d in cor.values() for i, ok in d.items() if (i not in exclude) and not ok)
        lo, hi = certify.clopper_pearson(f, n, alpha)
        _, ucb1 = certify.clopper_pearson(f, n, alpha, side="upper")
        return {"n": n, "n_fail": f, "risk": f / n, "ci95": [lo, hi], "UCB_one_sided": ucb1}

    def paired(exclude: set[str], cats=None) -> dict:
        b = c = both = neither = n = 0
        for cat in (cats or SINGLE_TURN):
            for i in ids[cat]:
                if i in exclude:
                    continue
                s, bb = cor_s[cat][i], cor_b[cat][i]
                n += 1
                if s and not bb:
                    b += 1
                elif bb and not s:
                    c += 1
                elif s and bb:
                    both += 1
                else:
                    neither += 1
        mc = certify.mcnemar_exact(b, c)
        mc_mid = certify.mcnemar_exact(b, c, mid_p=True)
        ci = certify.paired_difference_ci(n, b, c, alpha)
        return {"n": n, "student_only_correct": b, "base_only_correct": c, "both": both, "neither": neither,
                "acc_student": (b + both) / n if n else float("nan"), "acc_base": (c + both) / n if n else float("nan"),
                "diff_student_minus_base": ci["diff"], "diff_wald_ci95": [ci["lo"], ci["hi"]],
                "mcnemar_exact_p": mc["p_value"], "mcnemar_mid_p": mc_mid["p_value"]}

    out = {"student": student, "base": base, "alpha": alpha,
           "categories": SINGLE_TURN, "n_official_single_turn": sum(len(v) for v in ids.values()),
           "contamination": cont,
           "note": ("Official single-turn tasks were never trained on directly except the "
                    f"{len(exact)} 'exact' ids (support-demand tasks with verified teacher demos in the "
                    "r3t pool); generated training tasks were seeded from the "
                    f"{len(derived) - len(exact)} additional 'derived' official ids. Tables are given for "
                    "all ids, excluding exact, and excluding exact+derived."),
           "official": {}}
    for name, excl in [("all", set()), ("excl_exact", exact), ("excl_exact_and_derived", derived)]:
        ts, tb = table(cor_s, excl), table(cor_b, excl)
        realised_b = None
        out["official"][name] = {
            "n_excluded": len(excl & {i for d in ids.values() for i in d}),
            "pooled_student": pooled(cor_s, excl), "pooled_base": pooled(cor_b, excl),
            "paired_student_vs_base": paired(excl),
            "paired_by_category": {c: paired(excl, [c]) for c in SINGLE_TURN},
            "category_table_student": ts, "category_table_base": tb,
            "risk_coverage_student": certify.risk_coverage_curve(ts, taus),
            "risk_coverage_base": certify.risk_coverage_curve(tb, taus),
        }
    # ---- calibration prompts
    cal = json.load(open(ROOT / "data/bfcl_sft/calibration_ids.json"))
    hashes = cal["hashes"]
    cal_out = {"file": "data/bfcl_sft/calibration_ids.json", "n": len(hashes), "rule": cal["rule"]}
    if hm is None:
        cal_out["status"] = "hash map missing: run `tools/certify_bfcl.py hashes` under envs/bfcl-venv"
    else:
        m = hm["calibration"]
        n_off = sum(1 for k in hashes if m[k]["official"])
        gen_rows = {k: m[k]["generated"][0] for k in hashes if m[k]["generated"]}
        cats = Counter(g["category"] for g in gen_rows.values())
        seeds = {k: g["seed_task"] for k, g in gen_rows.items()}
        cal_out.update({
            "n_mapped_to_official_prompt": n_off, "n_mapped_to_generated_prompt": len(gen_rows),
            "generated_categories": dict(cats),
            "generated_files": dict(Counter(g["file"] for g in gen_rows.values())),
            "n_seed_official_ids": len(set(seeds.values())),
            "n_seed_ids_in_trained_exact": sum(1 for s in set(seeds.values()) if s in exact),
            "n_seed_ids_in_trained_derived": sum(1 for s in set(seeds.values()) if s in derived),
            "official_per_id_outcomes_available": n_off > 0,
        })
        # base outcomes on the calibration prompts from the gate-v2 pass-rate files (k=8 samples)
        rates_b = json.load(open(ROOT / "data/bfcl_sft/gate_v2/rates_base.json"))
        rb = np.array([rates_b[k] for k in hashes if k in rates_b])
        ks = sorted({round(1 / x) for x in np.diff(np.unique(np.concatenate([rb, [0, 1]]))) if x > 0})
        k_samples = int(max(ks)) if ks else None
        prompt_fail_b = int(np.sum(rb < 0.5))
        prompt_any_fail_b = int(np.sum(rb < 1.0))
        cal_out["base_from_gate_v2"] = {
            "source": "data/bfcl_sft/gate_v2/rates_base.json (student = base Qwen3.5-4B, k samples per prompt)",
            "k_samples_inferred": k_samples, "n_prompts": int(len(rb)),
            "mean_pass_rate": float(rb.mean()),
            "prompt_level_error_rate_majority_fail": prompt_fail_b / len(rb),
            "prompt_level_error_rate_any_fail": prompt_any_fail_b / len(rb),
            "clopper_pearson_95_upper_majority_fail": certify.clopper_pearson(prompt_fail_b, len(rb), alpha, side="upper")[1],
            "clopper_pearson_95_two_sided_majority_fail": list(certify.clopper_pearson(prompt_fail_b, len(rb), alpha)),
            "sample_level_error_rate": float(1 - rb.mean()),
            "sample_level_note": "k samples of one prompt are not independent trials; the guarantee is at prompt level",
        }
        # realised-vs-nominal for base: calibration prompts all fall in the generated categories above
        for name in out["official"]:
            tb = out["official"][name]["category_table_base"]
            realised = {c: (int(np.sum([rates_b[k] < 0.5 for k in hashes if gen_rows[k]["category"] == c])),
                           int(sum(1 for k in hashes if gen_rows[k]["category"] == c))) for c in cats}
            out["official"][name]["risk_coverage_base_realised_on_calibration"] = certify.risk_coverage_curve(
                tb, taus, realised=realised)
        cal_out["student_status"] = (
            f"NOT CERTIFIABLE YET: no rollouts of {student} on the 60 calibration prompts exist "
            "(they are generated simple_python tasks, not official ids, so the official score files "
            "contain no outcome for them). Needed: k-sample rollouts of the checkpoint on exactly these "
            "prompts (e.g. tools/bfcl_event_mine_single.py --rates-key prompt against the served ckpt, "
            "or a dedicated CPU-free GPU job) -> rates_<student>.json; then this tool fills in the "
            "student Clopper–Pearson bound, the paired McNemar vs base and realised-vs-nominal.")
        rates_s_path = ROOT / f"data/bfcl_sft/gate_v2/rates_{student}.json"
        if rates_s_path.exists():
            rates_s = json.load(open(rates_s_path))
            rs = np.array([rates_s[k] for k in hashes if k in rates_s])
            fs = int(np.sum(rs < 0.5))
            b_ = int(np.sum((rs >= 0.5) & (rb < 0.5)))
            c_ = int(np.sum((rs < 0.5) & (rb >= 0.5)))
            cal_out["student_status"] = f"rates file found: {rates_s_path.relative_to(ROOT)}"
            cal_out["student"] = {
                "n_prompts": int(len(rs)), "mean_pass_rate": float(rs.mean()),
                "prompt_level_error_rate_majority_fail": fs / len(rs),
                "clopper_pearson_95_upper_majority_fail": certify.clopper_pearson(fs, len(rs), alpha, side="upper")[1],
                "paired_vs_base": {"student_only_correct": b_, "base_only_correct": c_,
                                   "mcnemar_exact_p": certify.mcnemar_exact(b_, c_)["p_value"],
                                   "diff_wald_ci95": certify.paired_difference_ci(len(rs), b_, c_, alpha)}}
    out["calibration"] = cal_out
    return out



# ==============================================================================================
# 3'. stratified official calibration set (v2) and its certificate
# ==============================================================================================
CALIB_V2 = ROOT / "data/bfcl_sft/calibration_ids_official_v2.json"


def used_official_ids() -> dict:
    """Every official id touched by any training/mining artefact: task_id / _traj (split on '#')
    of data/bfcl_sft/pool_*.jsonl and events*.jsonl rows, their _seed_task / seed_task, the seed
    task of every generated id (gen*/genmt*/oos* prefix stripped), the seed_task of every
    data/bfcl_sft/gen*.jsonl row, and the demand + calibration ids of configs/bfcl_support_split.json."""
    used, sources = set(), Counter()
    files = sorted(glob.glob(str(ROOT / "data/bfcl_sft/pool_*.jsonl")) + glob.glob(str(ROOT / "data/bfcl_sft/events*.jsonl"))
                   + glob.glob(str(ROOT / "data/bfcl_sft/gen*.jsonl")))
    for f in files:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cands = []
            for k in ("task_id", "_traj", "id"):
                v = r.get(k)
                if isinstance(v, str):
                    cands.append(v.split("#")[0])
            for k in ("_seed_task", "seed_task"):
                v = r.get(k)
                if isinstance(v, str):
                    cands.append(v.split("#")[0])
            for t in cands:
                used.add(t)
                used.add(_seed_task(t))
                sources[Path(f).name] += 1
    split = json.load(open(ROOT / "configs/bfcl_support_split.json"))
    for k in ("demand", "calibration", "seed"):
        for t in (split.get(k) if isinstance(split.get(k), list) else []):
            used.add(t)
            used.add(_seed_task(t))
    return {"used": used, "n_files": len(files)}


def build_calibration_v2(seed: int = 0, per_cat: int = 5) -> dict:
    ids = official_ids()
    used = used_official_ids()["used"]
    rng = np.random.default_rng(seed)
    per_category, chosen = {}, []
    for cat in SINGLE_TURN:
        free = sorted(i for i in ids[cat] if i not in used)
        take = min(per_cat, len(free))
        pick = sorted(rng.choice(free, size=take, replace=False).tolist()) if take else []
        per_category[cat] = {"n_official": len(ids[cat]), "n_unused": len(free), "n_selected": take, "ids": pick}
        chosen += pick
    out = {"note": ("Stratified held-out calibration set over the official BFCL single-turn categories: "
                    "ids never used as task_id/_traj/seed in any data/bfcl_sft pool_*/events*/gen* row "
                    "(generated ids mapped to their seed official id) nor in configs/bfcl_support_split.json "
                    f"(seed/demand/calibration); {per_cat} ids per category sampled with numpy default_rng({seed}) "
                    "over the sorted unused ids (fewer if a category has fewer). Reserved for certification: "
                    "never train, mine or select hyper-parameters on them. Built 2026-09-04 by tools/certify_bfcl.py."),
           "seed": seed, "per_category": per_category, "ids": chosen, "n": len(chosen),
           "categories": SINGLE_TURN, "n_official_used_excluded": len(used & {i for v in ids.values() for i in v})}
    json.dump(out, open(CALIB_V2, "w"), indent=1)
    return out


def calibration_v2_certificate(student: str, base: str, alpha: float, taus) -> dict:
    cal = json.load(open(CALIB_V2))
    cal_ids = set(cal["ids"])
    ids = official_ids()
    cor_s, cor_b = per_id_correct(student, ids), per_id_correct(base, ids)
    cat_of = {i: c for c, v in ids.items() for i in v}

    def cp(cor):
        n = len(cal_ids)
        f = sum(1 for i in cal_ids if not cor[cat_of[i]][i])
        lo, hi = certify.clopper_pearson(f, n, alpha)
        return {"n": n, "n_fail": f, "error_rate": f / n, "ci95": [lo, hi],
                "UCB_one_sided_95": certify.clopper_pearson(f, n, alpha, side="upper")[1],
                "per_category_fail": {c: sum(1 for i in cal["per_category"][c]["ids"] if not cor[c][i]) for c in SINGLE_TURN}}

    b = sum(1 for i in cal_ids if cor_s[cat_of[i]][i] and not cor_b[cat_of[i]][i])
    c = sum(1 for i in cal_ids if cor_b[cat_of[i]][i] and not cor_s[cat_of[i]][i])
    n = len(cal_ids)
    mc = certify.mcnemar_exact(b, c)
    ci = certify.paired_difference_ci(n, b, c, alpha)
    paired = {"n": n, "student_only_correct": b, "base_only_correct": c,
              "diff_student_minus_base": ci["diff"], "diff_wald_ci95": [ci["lo"], ci["hi"]],
              "mcnemar_exact_p": mc["p_value"], "mcnemar_mid_p": certify.mcnemar_exact(b, c, mid_p=True)["p_value"]}
    # certificate built on official ids minus the calibration ids; realised on the calibration ids
    out = {"file": str(CALIB_V2.relative_to(ROOT)), "n": n, "seed": cal["seed"],
           "composition": {c: cal["per_category"][c]["n_selected"] for c in SINGLE_TURN},
           "n_unused_per_category": {c: cal["per_category"][c]["n_unused"] for c in SINGLE_TURN},
           "student": cp(cor_s), "base": cp(cor_b), "paired_student_vs_base": paired,
           "certificate_fit_set": "official single-turn ids minus the calibration ids (category UCBs at alpha/13, Bonferroni)"}
    for name, cor in (("student", cor_s), ("base", cor_b)):
        per = {cat: (sum(1 for i, ok in d.items() if i not in cal_ids and not ok),
                     sum(1 for i in d if i not in cal_ids)) for cat, d in cor.items()}
        tab = certify.category_risk_table(per, alpha=alpha, simultaneous=True)
        realised = {cat: (sum(1 for i in cal["per_category"][cat]["ids"] if not cor[cat][i]),
                          cal["per_category"][cat]["n_selected"]) for cat in SINGLE_TURN}
        out[f"risk_coverage_{name}"] = certify.risk_coverage_curve(tab, taus, realised=realised)
        out[f"category_UCB_{name}"] = {cat: tab[cat]["UCB"] for cat in SINGLE_TURN}
    return out


# ==============================================================================================
def cmd_run(args) -> None:
    taus = [0.1, 0.2, 0.3, 0.4, 0.5]
    res = {"created": "2026-09-04", "student": args.student, "base": args.base,
           "alpha": args.alpha, "delta_atoms": args.delta,
           "guarantees": {
               "coverage": "descriptive with percentile-bootstrap 95% CIs over events; references: random basis, in-sample self-PCA, group cross-fit",
               "atoms": "Hoeffding one-sided LCB on the zbar-weighted mean of [0,1] per-event best-target mass, Bonferroni over K atoms "
                        "(family level delta); assumes independent per-event values and fixed weights — see caveats",
               "risk": "exact Clopper–Pearson binomial bounds (one-sided UCB at alpha/C per category => simultaneous at alpha); "
                       "exact McNemar for the paired student-base comparison; risk–coverage: deploy categories with UCB <= tau",
           }}
    print("[certify] coverage ...", flush=True)
    res["coverage"] = coverage_certificate(args.n_boot, args.seed)
    print("[certify] atoms ...", flush=True)
    res["atoms"] = {}
    for tag in args.traces:
        p = ROOT / f"results/analysis/mirror_bf3_{tag}_trace.jsonl"
        if not p.exists():
            res["atoms"][tag] = {"status": f"missing {p.relative_to(ROOT)}"}
            continue
        res["atoms"][tag] = atom_certificate_from_trace(p, args.delta, args.min_n_eff, args.ucb_threshold,
                                                        args.n_boot, args.seed)
    print("[certify] risk ...", flush=True)
    res["risk"] = risk_certificate(args.student, args.base, args.alpha, taus, args.n_boot, args.seed)
    if not CALIB_V2.exists() or args.rebuild_calib_v2:
        build_calibration_v2(seed=0, per_cat=5)
    res["calibration_official_v2"] = calibration_v2_certificate(args.student, args.base, args.alpha, [0.2, 0.3, 0.4])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=_json_default)
    print(f"[certify] -> {OUT}")
    _print_summary(res)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


def _print_summary(res: dict) -> None:
    print("\n== coverage ==")
    for k, v in res["coverage"]["sets"].items():
        pe = v["per_event_mean"]
        rr = v.get("reference_random_basis", {}).get("per_event_mean")
        print(f"  {k:32s} n={v['n']:4d} per-event {pe['point']:.3f} [{pe['lo']:.3f},{pe['hi']:.3f}] "
              f"set {v.get('set', v.get('in_sample_set', float('nan'))):.3f} random {rr:.3f}")
    print("== atoms ==")
    for tag, a in res["atoms"].items():
        if "status" in a:
            print(f"  {tag}: {a['status']}")
            continue
        s = a["summary_stats"]
        print(f"  {tag:6s} K={a['K']} rows={a['n_rows']} J={s['J_mean']:.3f} (first visit {s['J_first_visit_mean']:.3f}) "
              f"r={s['r_mean']:.3f} UCB_r mean {s['UCB_r_mean']:.3f} max {s['UCB_r_max']:.3f} "
              f"half-width med {s['half_width_median']:.3f} n_eff med {s['n_eff_median']:.1f} "
              f"flagged {a['n_flagged']}/{a['K']} certified-satisfied {a['n_certified_satisfied']} unresolved {len(a['unresolved_atoms'])}")
    print("== risk ==")
    for name, o in res["risk"]["official"].items():
        p = o["paired_student_vs_base"]
        print(f"  {name:24s} n={p['n']} acc S {p['acc_student']:.4f} B {p['acc_base']:.4f} diff {p['diff_student_minus_base']:+.4f} "
              f"[{p['diff_wald_ci95'][0]:+.4f},{p['diff_wald_ci95'][1]:+.4f}] McNemar p={p['mcnemar_exact_p']:.3g} "
              f"(b={p['student_only_correct']}, c={p['base_only_correct']})")
        for r in o["risk_coverage_student"]:
            print(f"     tau={r['tau']:.1f} student coverage {r['coverage']:.3f} certified {r['certified_risk']:.3f} "
                  f"empirical {r['empirical_risk_in_sample']:.3f} deployed {len(r['deployed'])}")
    c = res["risk"]["calibration"]
    print(f"  calibration: {c.get('n')} prompts, official={c.get('n_mapped_to_official_prompt')} "
          f"generated={c.get('n_mapped_to_generated_prompt')} cats={c.get('generated_categories')}")
    if "base_from_gate_v2" in c:
        b = c["base_from_gate_v2"]
        print(f"     base prompt-level error {b['prompt_level_error_rate_majority_fail']:.3f} "
              f"CP95 upper {b['clopper_pearson_95_upper_majority_fail']:.3f} (k={b['k_samples_inferred']})")
    print(f"     student: {c.get('student_status', '')[:120]}")
    v2 = res.get("calibration_official_v2")
    if v2:
        print(f"== calibration_official_v2 == n={v2['n']} composition={v2['composition']}")
        for who in ("student", "base"):
            x = v2[who]
            print(f"  {who}: fail {x['n_fail']}/{x['n']} = {x['error_rate']:.3f} CI95 [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}] UCB {x['UCB_one_sided_95']:.3f}")
        p = v2["paired_student_vs_base"]
        print(f"  paired: b={p['student_only_correct']} c={p['base_only_correct']} diff {p['diff_student_minus_base']:+.3f} McNemar p={p['mcnemar_exact_p']:.3g}")
        for who in ("student", "base"):
            for r in v2[f"risk_coverage_{who}"]:
                print(f"  {who} tau={r['tau']:.1f} deployed {len(r['deployed'])} cov {r['coverage']:.3f} certified {r['certified_risk'] if r['n_deployed'] else float('nan'):.3f} "
                      f"realised {r['realised_fail']}/{r['realised_n']} = {r['realised_risk']:.3f} CI {np.round(r['realised_ci95'],3).tolist()} within={r['realised_within_nominal']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hashes")
    r = sub.add_parser("run")
    r.add_argument("--student", default="crcd_r3_union_t_s0")
    r.add_argument("--base", default="base_hpg", help="results/bfcl_std/<tag> of the reference model")
    r.add_argument("--traces", nargs="*", default=["rho1", "rhoJ0", "k1"])
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--delta", type=float, default=0.05, help="family level for the atom certificate")
    r.add_argument("--min-n-eff", type=float, default=5.0)
    r.add_argument("--ucb-threshold", type=float, default=0.1)
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--rebuild-calib-v2", action="store_true", help="re-sample data/bfcl_sft/calibration_ids_official_v2.json")
    sub.add_parser("calib-v2", help="build data/bfcl_sft/calibration_ids_official_v2.json only")
    args = ap.parse_args(argv)
    if args.cmd == "hashes":
        cmd_hashes(args)
    elif args.cmd == "calib-v2":
        out = build_calibration_v2(seed=0, per_cat=5)
        print(f"[certify] -> {CALIB_V2}: n={out['n']} " + str({c: v['n_selected'] for c, v in out['per_category'].items()}))
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
