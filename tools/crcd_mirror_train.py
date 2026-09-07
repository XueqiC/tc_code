#!/usr/bin/env python3
"""CRCD capability-constrained mirror distillation trainer (spec §5, Block III).

Reuses ``src/appworld_train.py`` for pool loading, tokenisation (``encode``),
sequence scoring (``completion_log_prob``) and the LoRA student
(``build_lora_model``); the mirror-target math lives in ``src/bfas/mirror.py``.

Pipeline
  1. rows -> candidate sets ``Y_i = {y^T (response), y^S (_rejected), extras}``
     with utilities from ``_event_u_plus`` / ``_event_u_minus``;
  2. cache length-normalised frozen-base scores ``log pi_0(y|s)/|y|`` once per
     pool (``--base-cache``);
  3. atom loadings ``Z`` (``--loadings`` npz with keys ``Z``, ``state_hash``
     aligned to event ids, or ``--k-random K``); ``W = |Z| / column sums``;
  4. ``rho_k`` from ``--rho`` or ``--rho-offsets`` (``rho_k = J_k(pi_0) + off_k``);
  5. primal loop: ``L = sum_i KL(q_i || pi_theta) + gamma_probe KL_tok(pi_theta||pi_0)
     + gamma_perp ||(I-UU^+) sketch(F^{1/2} dtheta_B)||^2``;
  6. dual update every ``--dual-every`` steps from the window estimate of ``J_k``;
  7. save the student like appworld_train (``results/appworld_students/<tag>/adapter``).

Every CLI value may be given in a JSON ``--config``; explicit CLI flags win.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# BFCL/AppWorld transcripts are long: match the campaign scripts' training caps
# unless the caller set them explicitly.
os.environ.setdefault("AW_MAX_PROMPT_TOKENS", "4096")
os.environ.setdefault("AW_TRUNCATE_SIDE", "tail")
os.environ.setdefault("AW_GRAD_CKPT", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import appworld_train as awt  # noqa: E402
from bfas import mirror  # noqa: E402


DEFAULTS: dict[str, Any] = {
    "pool": "data/bfcl_sft/pool_events_pref_v2.jsonl",
    "rows": 0,  # 0 = all
    "row_order": "first",  # first | random
    "student": "Qwen/Qwen3.5-4B",
    "seed": 0,
    "tag": "mirror_dev",
    "loadings": None,
    "k_random": 1,
    "loading_seed": 0,
    "probe": "data/bfcl_sft/anchors_single_base.jsonl",
    "probe_rows": 4,
    "steps": 6,
    "grad_accum": awt.GRADIENT_ACCUMULATION,
    "lr": awt.LEARNING_RATE,
    "dual_every": 2,
    "eta": 0.25,
    "eta_dual": 1.0,
    "lambda_init": 0.5,
    "rho": None,
    "rho_offsets": None,
    "eps": 0.0,
    "kappa": 5.0,
    "d": None,
    "j_mode": "best_mass",
    "utility_mode": "auto",
    "gamma_probe": 0.1,
    "gamma_perp": 1e-3,
    "perp_mode": "decoupled",  # decoupled (AdamW-style proximal step) | coupled (into Adam)
    "fisher": None,  # path to torch dict {param_name: diag} or None -> F = 1
    "U": None,  # path to .npy (sketch_dim x K) or None -> penalty on full displacement
    "sketch_dim": 256,
    "sketch_seed": 0,
    "trace": "results/analysis/mirror_trace.jsonl",
    "base_cache": "results/analysis/mirror_base_scores.json",
    "save": "merged",  # merged | adapter | none
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file with any of the flags below")
    for key, default in DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(flag, type=lambda s: s.lower() in {"1", "true", "yes"}, default=None)
        elif isinstance(default, int):
            p.add_argument(flag, type=int, default=None)
        elif isinstance(default, float):
            p.add_argument(flag, type=float, default=None)
        else:
            p.add_argument(flag, default=None)
    args = p.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            loaded = json.load(fh)
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg.update(loaded)
    for key in DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return argparse.Namespace(**cfg)


def _vector(value: Any, K: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [float(v) for v in value.split(",")]
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, K)
    if arr.size != K:
        raise ValueError(f"{name} needs {K} entries, got {arr.size}")
    return arr


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def event_key(ev: mirror.Event) -> str:
    """Unique event key ``_traj|sha1(prompt)|sha1(y^S)`` (== tools/atoms_loadings.event_key_for_event).
    ``_traj`` alone is not unique in re-mined pools (v3t: 80 distinct _traj over 302 rows). 2026-09-04 (T7)."""
    y_s = ev.candidates[ev.index_of(mirror.STUDENT)].text
    return f"{ev.event_id}|{_sha(ev.prompt)}|{_sha(y_s)}"


def load_loadings(cfg: argparse.Namespace, events: list[mirror.Event]) -> tuple[np.ndarray, str]:
    n = len(events)
    if cfg.loadings:
        data = np.load(cfg.loadings, allow_pickle=True)
        Z_all = np.asarray(data["Z"], dtype=np.float64)
        ids = [str(h) for h in data["state_hash"]]
        lookup = {h: i for i, h in enumerate(ids)}
        by_key = "event_key" in data.files  # npz keyed by the unique event key, not by _traj
        Z = np.zeros((n, Z_all.shape[1]))
        missing = []
        for i, ev in enumerate(events):
            j = lookup.get(event_key(ev) if by_key else ev.event_id)
            if j is None:
                missing.append(ev.event_id)
            else:
                Z[i] = Z_all[j]
        if missing:
            print(f"[mirror][warn] {len(missing)}/{n} events without loadings "
                  f"(zero rows): {missing[:5]}", flush=True)
        return Z, f"file:{cfg.loadings}"
    K = int(cfg.k_random)
    if K == 1:
        return np.ones((n, 1)), "global:K=1,zbar=1"
    rng = np.random.default_rng(int(cfg.loading_seed))
    return rng.random((n, K)), f"random:K={K},seed={cfg.loading_seed}"


def load_probes(cfg: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    if not cfg.probe or int(cfg.probe_rows) <= 0:
        return []
    probes = []
    with open(cfg.probe, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("prompt") or not row.get("response"):
                continue
            input_ids, labels = awt.encode(tokenizer, row)
            # KL is evaluated at positions whose *next* token is a response token
            mask = (labels[:, 1:] != -100)
            probes.append({"id": mirror.event_id_for_row(row), "input_ids": input_ids,
                           "mask": mask, "n_tokens": int(mask.sum())})
            if len(probes) >= int(cfg.probe_rows):
                break
    return probes


def cache_key(cfg: argparse.Namespace, ev: mirror.Event, cand: mirror.Candidate) -> str:
    cap, side = awt.prompt_truncation_config()
    return f"{cfg.student}|cap{cap}{side}|{ev.event_id}|{_sha(ev.prompt)}|{cand.key()}"  # prompt hash: _traj is not unique


def score_candidate(model: Any, tokenizer: Any, ev: mirror.Event,
                    cand: mirror.Candidate) -> tuple[torch.Tensor, int]:
    input_ids, labels = awt.encode(tokenizer, {"prompt": ev.prompt, "response": cand.text,
                                               "messages": None})
    length = int((labels != -100).sum())
    return awt.completion_log_prob(model, input_ids, labels), length


def cache_base_scores(cfg, model, tokenizer, events) -> int:
    path = ROOT / cfg.base_cache
    cache: dict[str, Any] = {}
    if path.is_file():
        cache = json.loads(path.read_text(encoding="utf-8"))
    todo = [(ev, c) for ev in events for c in ev.candidates
            if cache_key(cfg, ev, c) not in cache]
    if todo:
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            for k, (ev, c) in enumerate(todo, start=1):
                logp, length = score_candidate(model, tokenizer, ev, c)
                cache[cache_key(cfg, ev, c)] = {"logp0": float(logp.item()), "length": length}
                if k % 20 == 0 or k == len(todo):
                    print(f"[mirror][base] scored {k}/{len(todo)}", flush=True)
        model.train()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    for ev in events:
        for c in ev.candidates:
            hit = cache[cache_key(cfg, ev, c)]
            c.logp0, c.length = hit["logp0"], hit["length"]
    return len(todo)


def probe_kl(model, probe) -> torch.Tensor:
    ids, mask = probe["input_ids"], probe["mask"]
    logits_theta = model(input_ids=ids).logits[:, :-1, :]
    with torch.no_grad(), model.disable_adapter():
        logits_base = model(input_ids=ids).logits[:, :-1, :]
    pos = mask[0].nonzero().squeeze(-1)
    return mirror.token_kl_theta_base(
        logits_theta[:, pos, :], logits_base[:, pos, :],
        torch.ones(1, pos.numel(), dtype=torch.bool, device=ids.device))


def main(argv=None) -> None:
    cfg = parse_args(argv)
    t0 = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("mirror training needs a CUDA device")
    awt.seed_everything(int(cfg.seed))

    rows = awt.load_pool(ROOT / cfg.pool)
    if int(cfg.rows) > 0:
        if cfg.row_order == "random":
            order = np.random.default_rng(int(cfg.seed)).permutation(len(rows))[: int(cfg.rows)]
            rows = [rows[int(i)] for i in sorted(order)]
        else:
            rows = rows[: int(cfg.rows)]
    events = [mirror.candidates_from_row(r, cfg.utility_mode) for r in rows]
    n = len(events)
    Z, z_source = load_loadings(cfg, events)
    K = Z.shape[1]
    W = mirror.normalized_loadings(Z)
    print(f"[mirror] events={n} atoms={K} loadings={z_source}", flush=True)

    tokenizer = awt.load_tokenizer(cfg.student)
    model = awt.build_lora_model(cfg.student, int(cfg.seed))
    model.config.use_cache = False
    new_scores = cache_base_scores(cfg, model, tokenizer, events)
    print(f"[mirror] base scores: {new_scores} newly computed", flush=True)

    # J under pi_0 and the constraint levels
    p0 = [mirror.restricted_policy(ev.scores0()) for ev in events]
    J0 = mirror.atom_J_from_events(Z, [mirror.event_J_value(p, ev, cfg.j_mode)
                                       for p, ev in zip(p0, events)])
    rho = _vector(cfg.rho, K, "rho")
    if rho is None:
        offsets = _vector(cfg.rho_offsets, K, "rho_offsets")
        if offsets is None:
            raise ValueError("give --rho or --rho-offsets")
        rho = np.clip(J0 + offsets, 0.0, 1.0)
    dual = mirror.DualState(
        lam=_vector(cfg.lambda_init, K, "lambda_init"), rho=rho,
        eps=_vector(cfg.eps, K, "eps"), kappa=float(cfg.kappa),
        d=_vector(cfg.d, K, "d"), eta_dual=float(cfg.eta_dual))
    print(f"[mirror] J(pi_0)={np.round(J0, 4).tolist()} rho={np.round(rho, 4).tolist()} "
          f"lambda_init={dual.lam.tolist()}", flush=True)

    probes = load_probes(cfg, tokenizer)
    lora_b = [(name, p) for name, p in model.named_parameters()
              if p.requires_grad and "lora_B" in name]
    fisher: Any = 1.0
    if cfg.fisher:
        fisher = torch.load(cfg.fisher, map_location="cpu")
    U = np.load(cfg.U) if cfg.U else None
    penalty = mirror.DisplacementPenalty(lora_b, fisher=fisher, U=U,
                                         dim=int(cfg.sketch_dim), seed=int(cfg.sketch_seed))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(cfg.lr))
    trace_path = ROOT / cfg.trace
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace = trace_path.open("w", encoding="utf-8")

    def emit(record: dict[str, Any]) -> None:
        trace.write(json.dumps(mirror.as_list(record)) + "\n")
        trace.flush()

    emit({"kind": "config", **{k: v for k, v in vars(cfg).items()},
          "n_events": n, "K": K, "loadings_source": z_source,
          "event_ids": [ev.event_id for ev in events],
          "utility_modes": [ev.meta["utility_mode"] for ev in events],
          "Z": Z, "W": W, "J0": J0, "rho": rho, "n_probes": len(probes),
          "n_lora_B_params": sum(p.numel() for _, p in lora_b),
          "definitions": {
              "J_k": "sum_i w_ki * p_theta(y_i* | s_i; Y_i), w_ki = |z_ki|/sum_j |z_kj| over "
                     "the events seen since the last dual update; y_i* = argmax_y Q_i(y) "
                     "(ties -> teacher); p_theta = softmax of length-normalised scores on Y_i"
                     if cfg.j_mode == "best_mass" else
                     "sum_i w_ki * sum_y p_theta(y|s_i;Y_i) Q_i(y) (expected utility)",
              "rho_k": "target level of J_k; from --rho or J_k(pi_0) + rho_offsets",
              "dual": "lambda_k <- clip(lambda_k + eta_dual (rho_k - eps_k - J_k), 0, kappa d_k); "
                      "xi_k = [rho_k - eps_k - J_k]_+ where the clamp binds",
              "sketch": "count-sketch on LoRA-B displacement, seed/dim per config",
          }})

    rng = np.random.default_rng(int(cfg.seed))
    order: list[int] = []
    est = mirror.JEstimator(K)
    J_prev = J0.copy()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    accum = int(cfg.grad_accum)
    for step in range(1, int(cfg.steps) + 1):
        step_t = time.time()
        if len(order) < accum:
            order.extend(int(i) for i in rng.permutation(n))
        chunk, order = order[:accum], order[accum:]
        lam = dual.lam.copy()
        Lambda = mirror.capability_pressure(W, lam)
        event_records = []
        cmd_total = 0.0
        for i in chunk:
            ev = events[i]
            scores_theta = torch.stack([
                logp / length for logp, length in
                (score_candidate(model, tokenizer, ev, c) for c in ev.candidates)])
            q = mirror.mirror_target(ev.scores0(), ev.utilities(), float(Lambda[i]), float(cfg.eta))
            loss_i = mirror.cmd_loss(q, scores_theta)
            (loss_i / accum).backward()
            cmd_total += float(loss_i.item())
            p_theta = torch.softmax(scores_theta.detach().double(), dim=0).cpu().numpy()
            j_val = mirror.event_J_value(p_theta, ev, cfg.j_mode)
            est.add(Z[i], j_val)
            owners = ev.owners()
            ti, si = owners.index(mirror.TEACHER), owners.index(mirror.STUDENT)
            stats = mirror.target_stats(q, ev)
            event_records.append({
                "event_id": ev.event_id, "row": i, "Lambda": float(Lambda[i]),
                "zbar": Z[i], "w": W[i],
                "target_entropy": stats["entropy"],
                "teacher_target_mass": stats["teacher_mass"],
                "student_target_mass": stats["student_mass"],
                "best_target_mass": stats["best_mass"],
                "log_odds_TS_numeric": float(np.log(q[ti] / q[si])),
                "log_odds_TS_analytic": mirror.log_odds(
                    ev.scores0(), ev.utilities(), float(Lambda[i]), float(cfg.eta), ti, si),
                "score0": ev.scores0(), "utility": ev.utilities(),
                "score_theta": scores_theta.detach().double().cpu().numpy(),
                "p_theta_teacher": float(p_theta[ti]), "p_theta_student": float(p_theta[si]),
                "p0_teacher": float(p0[i][ti]), "p0_student": float(p0[i][si]),
                "J_value": j_val, "cmd_loss": float(loss_i.item()),
                "constraint_satisfied": (J_prev >= dual.rho - dual.eps),
            })
        kl_val = float("nan")
        if probes and float(cfg.gamma_probe) >= 0:
            probe = probes[(step - 1) % len(probes)]
            kl = probe_kl(model, probe)
            if float(cfg.gamma_probe) > 0:
                (float(cfg.gamma_probe) * kl).backward()
            kl_val = float(kl.item())
        pen = penalty()
        pen_val = float(pen.item())
        perp_grads = None
        if float(cfg.gamma_perp) > 0:
            if cfg.perp_mode == "coupled":
                (float(cfg.gamma_perp) * pen).backward()
            else:
                # Decoupled (AdamW-style) proximal step: the penalty is a quadratic
                # in a 256-dim sketch of ~12M coordinates; fed through Adam's
                # per-coordinate normalisation its tiny, sign-coherent gradient
                # becomes a coherent +-lr move on every coordinate of a bucket
                # (sketch step ~ coords/bucket * lr) and oscillates.  Applying
                # lr * gamma_perp * grad directly keeps the sketch-space map
                # contractive (observed 2026-09-04 smoke: perp 0.11 -> 1.8e3).
                perp_grads = torch.autograd.grad(pen, [p for _, p in lora_b],
                                                 allow_unused=True)
        grad_norm = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in model.parameters()
                                  if p.grad is not None))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        perp_step_norm = 0.0
        if perp_grads is not None:
            with torch.no_grad():
                for (_, p), g in zip(lora_b, perp_grads):
                    if g is None:
                        continue
                    upd = g.to(p.dtype) * (float(cfg.lr) * float(cfg.gamma_perp))
                    perp_step_norm += float(upd.float().pow(2).sum())
                    p.sub_(upd)
            perp_step_norm = math.sqrt(perp_step_norm)
        pen_after = float(penalty().item())
        step_rec = {
            "kind": "step", "step": step, "lambda": lam, "Lambda_mean": float(Lambda[chunk].mean()),
            "cmd_loss_mean": cmd_total / len(chunk), "probe_kl": kl_val,
            "probe_id": probes[(step - 1) % len(probes)]["id"] if probes else None,
            "perp_penalty": pen_val, "perp_penalty_after_step": pen_after,
            "perp_step_norm": perp_step_norm, "grad_norm": grad_norm,
            "teacher_target_mass_mean": float(np.mean([r["teacher_target_mass"] for r in event_records])),
            "student_target_mass_mean": float(np.mean([r["student_target_mass"] for r in event_records])),
            "target_entropy_mean": float(np.mean([r["target_entropy"] for r in event_records])),
            "log_odds_max_abs_err": float(max(abs(r["log_odds_TS_numeric"] - r["log_odds_TS_analytic"])
                                              for r in event_records)),
            "J_window": est.value(), "window_events": est.count,
            "sec": time.time() - step_t,
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "events": event_records,
        }
        emit(step_rec)
        print(f"[mirror] step {step}/{cfg.steps} cmd={step_rec['cmd_loss_mean']:.4f} "
              f"probe_kl={kl_val:.4g} perp={pen_val:.3g}->{pen_after:.3g} lambda={np.round(lam, 3).tolist()} "
              f"T_mass={step_rec['teacher_target_mass_mean']:.3f} "
              f"J_win={np.round(est.value(), 3).tolist()} {step_rec['sec']:.1f}s "
              f"peak={step_rec['gpu_peak_gb']:.1f}GB", flush=True)
        if step % int(cfg.dual_every) == 0:
            J = est.value(fallback=J_prev)
            rec = dual.update(J)
            J_prev = J
            emit({"kind": "dual", "step": step, "rho": dual.rho, "eps": dual.eps,
                  "cap": dual.cap, "window_events": est.count, **rec})
            print(f"[mirror][dual] step {step} J={np.round(J, 4).tolist()} "
                  f"r={np.round(rec['r'], 4).tolist()} xi={np.round(rec['xi'], 4).tolist()} "
                  f"lambda {np.round(rec['lambda_before'], 3).tolist()} -> "
                  f"{np.round(rec['lambda_after'], 3).tolist()}", flush=True)
            est.reset()

    # final mirror targets for every event under the final duals (no model needed)
    Lambda_f = mirror.capability_pressure(W, dual.lam)
    finals = []
    for i, ev in enumerate(events):
        q = mirror.mirror_target(ev.scores0(), ev.utilities(), float(Lambda_f[i]), float(cfg.eta))
        st = mirror.target_stats(q, ev)
        finals.append({"event_id": ev.event_id, "Lambda": float(Lambda_f[i]),
                       "teacher_target_mass": st["teacher_mass"],
                       "student_target_mass": st["student_mass"],
                       "target_entropy": st["entropy"]})
    lam_v = np.array([f["Lambda"] for f in finals])
    tm_v = np.array([f["teacher_target_mass"] for f in finals])
    corr = float(np.corrcoef(lam_v, tm_v)[0, 1]) if lam_v.std() > 0 and tm_v.std() > 0 else float("nan")
    hi = lam_v >= np.median(lam_v)
    summary = {
        "tag": cfg.tag, "n_events": n, "K": K, "steps": int(cfg.steps),
        "J0": J0, "rho": dual.rho, "lambda_final": dual.lam, "J_final": J_prev,
        "r_final": dual.residual(J_prev), "xi_final": dual.slack(J_prev),
        "Lambda_vs_teacher_mass_corr": corr,
        "teacher_mass_high_Lambda_half": float(tm_v[hi].mean()),
        "teacher_mass_low_Lambda_half": float(tm_v[~hi].mean()) if (~hi).any() else float("nan"),
        "final_targets": finals,
        "runtime_sec": time.time() - t0,
        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    emit({"kind": "summary", **summary})
    trace.close()
    summary_path = trace_path.with_name(trace_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(mirror.as_list(summary), indent=1), encoding="utf-8")

    output_dir = awt.OUTPUT_ROOT / cfg.tag
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save == "merged":
        merged = model.merge_and_unload()
        adapter_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
    elif cfg.save == "adapter":
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
    (output_dir / "mirror_manifest.json").write_text(json.dumps(mirror.as_list({
        "config": vars(cfg), "save": cfg.save, "trace": str(trace_path),
        "summary": {k: v for k, v in summary.items() if k != "final_targets"},
    }), indent=1), encoding="utf-8")
    print(f"SUMMARY mirror tag={cfg.tag} events={n} K={K} steps={cfg.steps} "
          f"lambda={np.round(dual.lam, 3).tolist()} J={np.round(J_prev, 3).tolist()} "
          f"corr(Lambda,T_mass)={corr:.3f} runtime={summary['runtime_sec']:.0f}s "
          f"peak={summary['gpu_peak_gb']:.1f}GB saved={output_dir} ({cfg.save})", flush=True)


if __name__ == "__main__":
    main()
