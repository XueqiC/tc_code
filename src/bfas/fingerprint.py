"""Fisher-whitened teacher-student differential gradient fingerprints (spec §3.2).

All gradients are taken through the STUDENT at the fixed base checkpoint
theta_0 (Qwen/Qwen3.5-4B) with a fresh LoRA (r=8, same target modules as
src/features.py); only the LoRA-B parameters carry gradient, as in features.py.
Because B is initialised to zero, dL/dB = (A x) delta^T is a fixed random sketch
of the full weight gradient, so the fresh-LoRA gradient is a deterministic
function of theta_0 and the LoRA init seed.

Per event e_i = (s_i, y_i^S, y_i^T, O_i):

    g_i^S = grad_theta (1/|y^S|) log pi_theta(y_i^S | s_i) |_{theta_0}   (length-normalised)
    g_i^T = grad_theta (1/|y^T|) log pi_theta(y_i^T | s_i) |_{theta_0}
    F     = diag mean_i (g_i^S)^2                (empirical Fisher on target-task
                                                  student rollouts)
    psi~  = P^T F^{-1/2} (g_i^T - g_i^S)         (P: fixed random projection,
                                                  seed 0, to 256 dims)
    psi   = psi~ / (|psi~|_2 + eps)

Baselines kept next to psi with the same P and dimensionality:
    sketch_diff_raw = unit(P^T (g^T - g^S))      (unwhitened differential)
    sketch_teacher  = unit(P^T g^T)              (teacher-only gradient)

Every array is tagged with a ``fingerprint_version`` string built from the
model id, LoRA config, tokenizer hash, truncation limits, projection seed/dim,
normalisation and Fisher id; ``assert_compatible`` / ``load_fingerprints``
refuse to mix incompatible versions.

Notation follows spec §2: ownership superscripts S/T, never y+/y-.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

FINGERPRINT_SCHEMA = "fpv1"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
# identical to src/features.py / src/appworld_train.py LORA_TARGET_MODULES
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")


def stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


class FingerprintVersionError(ValueError):
    """Raised when fingerprints of incompatible versions would be mixed."""


@dataclass
class FingerprintConfig:
    model_id: str = DEFAULT_MODEL
    lora_r: int = 8
    lora_alpha: int = 16
    lora_seed: int = 0
    target_modules: tuple = LORA_TARGET_MODULES
    proj_dim: int = 256
    proj_seed: int = 0
    max_prompt_tok: int = 2048          # prompt keeps its TAIL (decision-adjacent context)
    max_resp_tok: int = 1024
    length_normalize: bool = True
    fisher_damping_rel: float = 1e-3    # damping = rel * mean(F)
    eps: float = 1e-8
    grad_ckpt: bool = True
    dtype: str = "bfloat16"

    def lora_tag(self) -> str:
        return f"r{self.lora_r}a{self.lora_alpha}s{self.lora_seed}:" + ",".join(self.target_modules)


# --------------------------------------------------------------------------
# pure numerics (device-agnostic, unit-tested without a model)
# --------------------------------------------------------------------------

def unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm() + eps)


def fisher_damping(fisher: torch.Tensor, rel: float) -> float:
    return float(rel) * float(fisher.mean())


def whiten(diff: torch.Tensor, fisher: torch.Tensor, damping: float = 0.0) -> torch.Tensor:
    """F^{-1/2} diff for a diagonal Fisher (elementwise), with additive damping."""
    return diff / torch.sqrt(fisher + damping)


def make_projection(shapes: Sequence[tuple[str, int]], proj_dim: int, seed: int,
                    device: str | torch.device = "cpu",
                    dtype: torch.dtype = torch.float32) -> list[torch.Tensor]:
    """Fixed Gaussian JL projection of the concatenated parameter vector.

    One (numel, proj_dim) block per module, seeded by (seed, module name) so the
    result is independent of device and of which other modules are present."""
    blocks = []
    device = torch.device(device)
    for name, numel in shapes:
        gen = torch.Generator(device=device).manual_seed(stable_seed(f"fp-proj:{seed}:{name}"))
        blocks.append(torch.randn(numel, proj_dim, generator=gen, device=device, dtype=dtype)
                      / math.sqrt(proj_dim))
    return blocks


def project(vec: torch.Tensor, blocks: Sequence[torch.Tensor]) -> torch.Tensor:
    """P^T vec for the block-structured projection built by make_projection."""
    out = None
    off = 0
    for blk in blocks:
        n = blk.shape[0]
        part = blk.T @ vec[off:off + n].to(blk.dtype)
        out = part if out is None else out + part
        off += n
    assert off == vec.numel(), f"projection covers {off} dims, vector has {vec.numel()}"
    return out


# --------------------------------------------------------------------------
# continuation log-probability and its gradient
# --------------------------------------------------------------------------

def _backbone_and_head(model: Any):
    """(backbone, lm_head) of a (possibly PEFT-wrapped) causal LM, or (None, None)."""
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(hf, "model", None)
    head = getattr(hf, "lm_head", None)
    if backbone is None or head is None or not callable(head):
        return None, None
    return backbone, head


def continuation_logprob(model: Any, input_ids: torch.Tensor, n_prompt: int,
                         length_normalize: bool = True, use_backbone: bool = True
                         ) -> tuple[torch.Tensor, int]:
    """log pi(y | s) for the continuation tokens input_ids[:, n_prompt:].

    Only the logits of the continuation positions are materialised (the
    backbone's hidden states are sliced before the LM head), which keeps the
    (T x vocab) logits from dominating memory on long prompts."""
    T = input_ids.shape[1]
    n = T - n_prompt
    if n <= 0:
        raise ValueError("continuation is empty")
    backbone, head = _backbone_and_head(model) if use_backbone else (None, None)
    if backbone is not None:
        hidden = backbone(input_ids=input_ids).last_hidden_state[:, n_prompt - 1:T - 1]
        logits = head(hidden)
    else:
        logits = model(input_ids=input_ids).logits[:, n_prompt - 1:T - 1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = input_ids[:, n_prompt:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    total = tok_lp.sum()
    return (total / n if length_normalize else total), n


def grad_logprob(model: Any, params: Sequence[tuple[str, torch.nn.Parameter]],
                 input_ids: torch.Tensor, n_prompt: int, length_normalize: bool = True,
                 use_backbone: bool = True) -> tuple[torch.Tensor, float, int]:
    """Flattened fp32 gradient of the (length-normalised) continuation log-prob
    w.r.t. ``params`` (in the given order).  Returns (grad, logp, n_tokens)."""
    logp, n = continuation_logprob(model, input_ids, n_prompt, length_normalize, use_backbone)
    grads = torch.autograd.grad(logp, [p for _, p in params], allow_unused=True)
    flat = torch.cat([
        (g if g is not None else torch.zeros_like(p)).detach().reshape(-1).float()
        for g, (_, p) in zip(grads, params)
    ])
    return flat, float(logp.detach()), n


# --------------------------------------------------------------------------
# tokenisation of an event
# --------------------------------------------------------------------------

def encode_event(tok: Any, state_text: str, continuation: str, cfg: FingerprintConfig
                 ) -> tuple[list[int], list[int], dict]:
    """(prompt_ids, continuation_ids, info).  The prompt is already rendered
    with the deployment chat template, so it is tokenised verbatim; it keeps
    its last ``max_prompt_tok`` tokens.  The continuation is capped at
    ``max_resp_tok`` tokens and terminated with EOS (<|im_end|>)."""
    p_all = tok(state_text, add_special_tokens=False)["input_ids"]
    p = p_all[-cfg.max_prompt_tok:] if len(p_all) > cfg.max_prompt_tok else p_all
    r_all = tok(continuation, add_special_tokens=False)["input_ids"]
    r = r_all[:cfg.max_resp_tok]
    eos = tok.eos_token_id
    if eos is not None:
        r = r + [eos]
    info = {"prompt_tokens": len(p_all), "prompt_dropped": len(p_all) - len(p),
            "resp_tokens": len(r_all), "resp_dropped": len(r_all) - min(len(r_all), cfg.max_resp_tok)}
    return p, r, info


# --------------------------------------------------------------------------
# model loading (mirrors src/features.py::_prepare_extractor)
# --------------------------------------------------------------------------

def tokenizer_hash(tok: Any) -> str:
    payload = json.dumps({"vocab": tok.get_vocab(),
                          "special": {k: str(v) for k, v in sorted(tok.special_tokens_map.items())},
                          "eos": tok.eos_token_id}, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()


def load_student(cfg: FingerprintConfig, device: str = "cuda"):
    """Base checkpoint + fresh LoRA; only lora_B requires grad.
    Returns (tokenizer, model, [(name, lora_B param), ...])."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=getattr(torch, cfg.dtype)).to(device)
    torch.manual_seed(cfg.lora_seed)
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=list(cfg.target_modules))
    model = get_peft_model(model, lora)
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()  # dropout is 0; train() only matters for checkpointing
    for name, p in model.named_parameters():
        p.requires_grad = "lora_B" in name
    lora_b = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
    return tok, model, lora_b


# --------------------------------------------------------------------------
# versioning
# --------------------------------------------------------------------------

def build_version(cfg: FingerprintConfig, tok_hash: str, fisher_id: str | None) -> str:
    parts = [FINGERPRINT_SCHEMA, f"model={cfg.model_id}", f"lora={cfg.lora_tag()}",
             f"tok={tok_hash[:12]}", f"prompt<={cfg.max_prompt_tok}", f"resp<={cfg.max_resp_tok}",
             f"proj=s{cfg.proj_seed}d{cfg.proj_dim}",
             "norm=len" if cfg.length_normalize else "norm=sum",
             f"fisher={fisher_id if fisher_id else 'none'}"]
    return "|".join(parts)


def parse_version(version: str) -> dict[str, str]:
    parts = version.split("|")
    out = {"schema": parts[0]}
    for p in parts[1:]:
        if "<=" in p:
            k, v = p.split("<=", 1)
        else:
            k, _, v = p.partition("=")
        out[k] = v
    return out


def assert_compatible(a: str, b: str, ignore: Iterable[str] = ()) -> None:
    pa, pb = parse_version(a), parse_version(b)
    ignore = set(ignore)
    diffs = [k for k in sorted(set(pa) | set(pb)) if k not in ignore and pa.get(k) != pb.get(k)]
    if diffs:
        detail = ", ".join(f"{k}: {pa.get(k)!r} vs {pb.get(k)!r}" for k in diffs)
        raise FingerprintVersionError(f"incompatible fingerprint versions ({detail})")


def fisher_id_of(fisher: np.ndarray, set_name: str, damping_rel: float) -> str:
    h = hashlib.sha1(np.ascontiguousarray(fisher, dtype=np.float32).tobytes()).hexdigest()[:12]
    return f"{set_name}:{h}:damp{damping_rel:g}"


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

@dataclass
class FingerprintResult:
    set_name: str
    psi: np.ndarray                    # (N, proj_dim) whitened differential, unit
    sketch_diff_raw: np.ndarray        # (N, proj_dim) unwhitened differential, unit
    sketch_teacher: np.ndarray         # (N, proj_dim) teacher-only, unit
    state_hash: list[str]
    fisher: np.ndarray                 # (D,) diagonal empirical Fisher
    fisher_id: str
    fisher_n: int
    fisher_damping: float
    fingerprint_version: str
    baseline_version: str
    config: dict
    param_layout: list[tuple[str, int]]
    per_event: list[dict] = field(default_factory=list)
    seconds_per_event: float = 0.0


def _event_fields(ev: Any) -> tuple[str, str, str, str]:
    if isinstance(ev, dict):
        return ev["state_text"], ev["student_continuation"], ev["teacher_continuation"], ev["state_hash"]
    return ev.state_text, ev.student_continuation, ev.teacher_continuation, ev.state_hash


def fingerprints(model_id: str, events: Sequence[Any], *, set_name: str,
                 cfg: FingerprintConfig | None = None, device: str = "cuda",
                 fisher: np.ndarray | None = None, fisher_id: str | None = None,
                 log=print) -> FingerprintResult:
    """Compute psi (and the two baselines) for ``events`` (UnifiedEvent or dicts
    with state_text / student_continuation / teacher_continuation / state_hash).

    Pass 1 computes g^S, g^T per event, accumulates the diagonal Fisher from
    g^S, emits the two unwhitened sketches and caches g^T - g^S on the CPU.
    Pass 2 whitens the cached differentials with the finished Fisher and
    projects them.  If ``fisher`` is given it is used instead of re-estimating."""
    cfg = cfg or FingerprintConfig()
    if model_id != cfg.model_id:
        cfg = FingerprintConfig(**{**asdict(cfg), "model_id": model_id})
    tok, model, lora_b = load_student(cfg, device)
    layout = [(n, p.numel()) for n, p in lora_b]
    D = sum(n for _, n in layout)
    log(f"[fp] {len(lora_b)} lora_B tensors, D={D:,}; projecting to {cfg.proj_dim} (seed {cfg.proj_seed})")
    blocks = make_projection(layout, cfg.proj_dim, cfg.proj_seed, device)
    tok_hash = tokenizer_hash(tok)

    N = len(events)
    diffs = torch.empty((N, D), dtype=torch.float32, device="cpu")
    fisher_acc = torch.zeros(D, dtype=torch.float32, device=device)
    sk_raw = np.zeros((N, cfg.proj_dim), dtype=np.float32)
    sk_teacher = np.zeros((N, cfg.proj_dim), dtype=np.float32)
    hashes, per_event = [], []
    t0 = time.time()
    for i, ev in enumerate(events):
        state, y_s, y_t, h = _event_fields(ev)
        p_ids, s_ids, info_s = encode_event(tok, state, y_s, cfg)
        _, t_ids, info_t = encode_event(tok, state, y_t, cfg)
        n_p = len(p_ids)
        ids_s = torch.tensor([p_ids + s_ids], device=device)
        ids_t = torch.tensor([p_ids + t_ids], device=device)
        g_s, lp_s, n_s = grad_logprob(model, lora_b, ids_s, n_p, cfg.length_normalize)
        g_t, lp_t, n_t = grad_logprob(model, lora_b, ids_t, n_p, cfg.length_normalize)
        fisher_acc += g_s * g_s
        diff = g_t - g_s
        sk_teacher[i] = unit(project(g_t, blocks), cfg.eps).cpu().numpy()
        sk_raw[i] = unit(project(diff, blocks), cfg.eps).cpu().numpy()
        diffs[i].copy_(diff.cpu())
        hashes.append(h)
        per_event.append({"state_hash": h, "n_tok_S": n_s, "n_tok_T": n_t,
                          "logp_S": lp_s, "logp_T": lp_t,
                          "norm_gS": float(g_s.norm()), "norm_gT": float(g_t.norm()),
                          "norm_diff": float(diff.norm()),
                          "prompt_tokens": info_s["prompt_tokens"],
                          "prompt_dropped": info_s["prompt_dropped"],
                          "resp_dropped_S": info_s["resp_dropped"],
                          "resp_dropped_T": info_t["resp_dropped"]})
        del g_s, g_t, diff
        if (i + 1) % 20 == 0 or i + 1 == N:
            el = time.time() - t0
            log(f"[fp] pass1 {i + 1}/{N}  {el / (i + 1):.2f}s/event  "
                f"gpu_max={torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if device.startswith("cuda")
                else f"[fp] pass1 {i + 1}/{N}  {el / (i + 1):.2f}s/event")
    pass1 = time.time() - t0

    if fisher is None:
        fisher_t = fisher_acc / max(N, 1)
        fisher_n = N
        fisher_np = fisher_t.cpu().numpy()
        fisher_id = fisher_id_of(fisher_np, set_name, cfg.fisher_damping_rel)
    else:
        fisher_np = np.asarray(fisher, dtype=np.float32)
        assert fisher_np.shape == (D,), f"fisher has {fisher_np.shape}, expected ({D},)"
        fisher_t = torch.from_numpy(fisher_np).to(device)
        fisher_n = -1
        fisher_id = fisher_id or fisher_id_of(fisher_np, "external", cfg.fisher_damping_rel)
    damping = fisher_damping(fisher_t, cfg.fisher_damping_rel)
    log(f"[fp] fisher: mean={float(fisher_t.mean()):.3e} min={float(fisher_t.min()):.3e} "
        f"max={float(fisher_t.max()):.3e} zero_frac={float((fisher_t == 0).float().mean()):.4f} "
        f"damping={damping:.3e} id={fisher_id}")

    psi = np.zeros((N, cfg.proj_dim), dtype=np.float32)
    t1 = time.time()
    for i in range(N):
        d = diffs[i].to(device, non_blocking=True)
        w = whiten(d, fisher_t, damping)
        per_event[i]["norm_wdiff"] = float(w.norm())
        psi[i] = unit(project(w, blocks), cfg.eps).cpu().numpy()
    pass2 = time.time() - t1
    total = pass1 + pass2
    log(f"[fp] pass2 whiten+project {pass2:.1f}s; total {total:.1f}s = {total / max(N, 1):.2f}s/event")

    version = build_version(cfg, tok_hash, fisher_id)
    baseline_version = build_version(cfg, tok_hash, None)
    del model, blocks, diffs
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return FingerprintResult(
        set_name=set_name, psi=psi, sketch_diff_raw=sk_raw, sketch_teacher=sk_teacher,
        state_hash=hashes, fisher=fisher_np, fisher_id=fisher_id, fisher_n=fisher_n,
        fisher_damping=damping, fingerprint_version=version, baseline_version=baseline_version,
        config=asdict(cfg), param_layout=layout, per_event=per_event,
        seconds_per_event=total / max(N, 1))


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save_fingerprints(res: FingerprintResult, out_dir: str | Path, tag: str = "v1"
                      ) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{res.set_name}_{tag}.npz"
    index = out_dir / f"{res.set_name}_{tag}.index.jsonl"
    fisher = out_dir / f"fisher_{res.set_name}_{tag}.npz"
    pe = res.per_event
    np.savez(npz, psi=res.psi, sketch_diff_raw=res.sketch_diff_raw,
             sketch_teacher=res.sketch_teacher, state_hash=np.array(res.state_hash),
             fingerprint_version=res.fingerprint_version, baseline_version=res.baseline_version,
             fisher_id=res.fisher_id, fisher_path=str(fisher), config=json.dumps(res.config),
             param_layout=json.dumps(res.param_layout),
             logp_S=np.array([e["logp_S"] for e in pe], dtype=np.float32),
             logp_T=np.array([e["logp_T"] for e in pe], dtype=np.float32),
             n_tok_S=np.array([e["n_tok_S"] for e in pe], dtype=np.int32),
             n_tok_T=np.array([e["n_tok_T"] for e in pe], dtype=np.int32),
             norm_gS=np.array([e["norm_gS"] for e in pe], dtype=np.float32),
             norm_gT=np.array([e["norm_gT"] for e in pe], dtype=np.float32),
             norm_diff=np.array([e["norm_diff"] for e in pe], dtype=np.float32),
             norm_wdiff=np.array([e["norm_wdiff"] for e in pe], dtype=np.float32),
             prompt_dropped=np.array([e["prompt_dropped"] for e in pe], dtype=np.int32))
    np.savez(fisher, fisher=res.fisher, fisher_id=res.fisher_id, fisher_n=res.fisher_n,
             damping=res.fisher_damping, param_layout=json.dumps(res.param_layout),
             fingerprint_version=res.fingerprint_version)
    with open(index, "w", encoding="utf-8") as f:
        for i, e in enumerate(pe):
            f.write(json.dumps({"row": i, **e}) + "\n")
    return {"npz": npz, "index": index, "fisher": fisher}


def load_fingerprints(path: str | Path, expect_version: str | None = None) -> dict[str, Any]:
    """Load an npz written by save_fingerprints; refuses a version mismatch."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    for k in ("fingerprint_version", "baseline_version", "fisher_id", "fisher_path", "config", "param_layout"):
        if k in out:
            out[k] = str(out[k])
    out["state_hash"] = [str(h) for h in out["state_hash"]]
    if expect_version is not None:
        assert_compatible(out["fingerprint_version"], expect_version)
    return out


def merge_fingerprint_sets(paths: Sequence[str | Path], key: str = "psi") -> tuple[np.ndarray, list[str], str]:
    """Concatenate several sets; every set must share the fingerprint version
    (Fisher included).  Use ``key='sketch_diff_raw'`` / ``'sketch_teacher'`` with
    ``baseline_version`` semantics via load_fingerprints if needed."""
    mats, hashes, version = [], [], None
    for p in paths:
        z = load_fingerprints(p)
        v = z["fingerprint_version"] if key == "psi" else z["baseline_version"]
        if version is None:
            version = v
        else:
            assert_compatible(version, v)
        mats.append(z[key])
        hashes.extend(z["state_hash"])
    return np.concatenate(mats, 0), hashes, version


def _base_task_id(task_id: str) -> str:
    """Strip generator prefixes (gen_/genmt_/oos_) and a trailing _<n> variant
    suffix: 'gen_live_parallel_multiple_10-9-0_1' -> 'live_parallel_multiple_10-9-0'."""
    t = re.sub(r"^(gen|genmt|oos)_", "", task_id)
    return re.sub(r"_\d+$", "", t)


def calibration_indices(events: Sequence[Any], calibration_ids: Iterable[str]
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(exact_idx, derived_idx): event rows whose task_id (or provenance.seed_task)
    is one of the calibration ids, and rows whose task id is a generated/OOS
    variant seeded from a calibration id (superset of exact)."""
    ids = set(calibration_ids)
    exact, derived = [], []
    for i, ev in enumerate(events):
        tid = ev["task_id"] if isinstance(ev, dict) else ev.task_id
        prov = (ev.get("provenance", {}) if isinstance(ev, dict) else ev.provenance) or {}
        seed = prov.get("seed_task")
        if tid in ids or seed in ids:
            exact.append(i)
            derived.append(i)
        elif _base_task_id(tid) in ids or (seed and _base_task_id(seed) in ids):
            derived.append(i)
    return np.asarray(exact, dtype=np.int64), np.asarray(derived, dtype=np.int64)


def annotate_calibration(npz_path: str | Path, events: Sequence[Any],
                         calibration_ids: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Re-save an existing npz with 'calibration_idx' (exact) and
    'calibration_derived_idx' arrays; the event order must match state_hash."""
    z = np.load(npz_path, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    hashes = [str(h) for h in data["state_hash"]]
    ev_hashes = [e["state_hash"] if isinstance(e, dict) else e.state_hash for e in events]
    if hashes != ev_hashes:
        raise ValueError("event order does not match the npz state_hash order")
    exact, derived = calibration_indices(events, calibration_ids)
    data["calibration_idx"] = exact
    data["calibration_derived_idx"] = derived
    np.savez(npz_path, **data)
    return exact, derived


def stamp_events(events: Sequence[Any], res: FingerprintResult, npz_path: str | Path) -> None:
    """Write fingerprint_version / fingerprint_path into UnifiedEvent records."""
    for ev, h in zip(events, res.state_hash):
        assert ev.state_hash == h
        ev.fingerprint_version = res.fingerprint_version
        ev.fingerprint_path = str(npz_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Fisher-whitened differential gradient fingerprints (spec §3.2)")
    ap.add_argument("--events", required=True, help="unified events jsonl (tools/events_convert.py)")
    ap.add_argument("--set", dest="set_name", required=True)
    ap.add_argument("--out-dir", default="data/fingerprints")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-prompt-tok", type=int, default=2048)
    ap.add_argument("--max-resp-tok", type=int, default=1024)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--fisher-damping-rel", type=float, default=1e-3)
    ap.add_argument("--fisher", default=None, help="reuse fisher_<set>_<tag>.npz instead of re-estimating")
    ap.add_argument("--stamp", action="store_true", help="write fingerprint_version/path back into the events file")
    ap.add_argument("--calibration-json", default=None,
                    help="configs/bfcl_support_split.json; its 'calibration' ids are recorded as calibration_idx")
    ap.add_argument("--annotate-only", action="store_true",
                    help="only add calibration_idx to the existing npz (no GPU work)")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bfas.events_schema import read_events, write_events

    events = read_events(args.events)
    if args.limit:
        events = events[: args.limit]
    calib_ids: list[str] = []
    if args.calibration_json:
        with open(args.calibration_json) as f:
            calib_ids = list(json.load(f)["calibration"])
    if args.annotate_only:
        npz = Path(args.out_dir) / f"{args.set_name}_{args.tag}.npz"
        exact, derived = annotate_calibration(npz, events, calib_ids)
        print(f"[fp] {npz}: calibration_idx={exact.tolist()} calibration_derived_idx={derived.tolist()}", flush=True)
        return 0
    cfg = FingerprintConfig(model_id=args.model, max_prompt_tok=args.max_prompt_tok,
                            max_resp_tok=args.max_resp_tok, proj_dim=args.proj_dim,
                            proj_seed=args.proj_seed, fisher_damping_rel=args.fisher_damping_rel)
    fisher = fisher_id = None
    if args.fisher:
        z = np.load(args.fisher, allow_pickle=False)
        fisher, fisher_id = z["fisher"], str(z["fisher_id"])
    print(f"[fp] {args.set_name}: {len(events)} events from {args.events}", flush=True)
    res = fingerprints(args.model, events, set_name=args.set_name, cfg=cfg, device=args.device,
                       fisher=fisher, fisher_id=fisher_id, log=lambda s: print(s, flush=True))
    paths = save_fingerprints(res, args.out_dir, args.tag)
    print(f"[fp] version: {res.fingerprint_version}")
    print(f"[fp] wrote {paths['npz']} ({res.psi.shape}), {paths['index']}, {paths['fisher']}; "
          f"{res.seconds_per_event:.2f}s/event", flush=True)
    if args.calibration_json:
        exact, derived = annotate_calibration(paths["npz"], events, calib_ids)
        print(f"[fp] calibration_idx={exact.tolist()} calibration_derived_idx={derived.tolist()}", flush=True)
    if args.stamp:
        stamp_events(events, res, paths["npz"])
        if not args.limit:
            write_events(args.events, events)
            print(f"[fp] stamped {len(events)} events in {args.events}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
