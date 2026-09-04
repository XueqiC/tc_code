"""Numerical tests for src/bfas/fingerprint.py (CPU, toy model; no checkpoint)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import fingerprint as fp  # noqa: E402

torch.manual_seed(0)


class _Out:
    def __init__(self, x):
        self.last_hidden_state = x
        self.logits = x


class ContextFreeLM(torch.nn.Module):
    """logits_t depends on token t only (no mixing across positions), so the
    per-token score of a repeated continuation is exactly repeated."""

    def __init__(self, vocab=40, dim=12):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.model = _Backbone(self.emb, dim)
        self.lm_head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids):
        return _Out(self.lm_head(self.model(input_ids=input_ids).last_hidden_state))


class _Backbone(torch.nn.Module):
    def __init__(self, emb, dim):
        super().__init__()
        self.emb = emb
        self.mix = torch.nn.Linear(dim, dim)

    def forward(self, input_ids):
        return _Out(torch.tanh(self.mix(self.emb(input_ids))))


class ContextualLM(torch.nn.Module):
    """Causal cumulative-mean mixing: the log-prob really depends on context."""

    def __init__(self, vocab=40, dim=12):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids):
        h = self.emb(input_ids)
        h = torch.cumsum(h, 1) / torch.arange(1, h.shape[1] + 1, device=h.device)[None, :, None]
        return _Out(self.head(torch.tanh(h)))


def _params(m):
    return [(n, p) for n, p in m.named_parameters()]


def _ids(prompt, resp):
    return torch.tensor([prompt + resp]), len(prompt)


PROMPT = [3, 7, 11, 2, 5]
# RESP ends with the prompt's last token so that RESP + RESP repeats exactly
# the same (previous token -> target) pairs; the doubling test is then exact
# for the context-free model.
RESP = [8, 1, 4, 9, 6, 5]


def test_length_normalisation_doubling_does_not_scale_gradient():
    m = ContextFreeLM()
    ids1, n_p = _ids(PROMPT, RESP)
    ids2, _ = _ids(PROMPT, RESP + RESP)          # doubled continuation
    g1, lp1, n1 = fp.grad_logprob(m, _params(m), ids1, n_p, length_normalize=True)
    g2, lp2, n2 = fp.grad_logprob(m, _params(m), ids2, n_p, length_normalize=True)
    assert n2 == 2 * n1
    assert torch.allclose(g1, g2, atol=1e-6)
    assert abs(lp1 - lp2) < 1e-6
    # without normalisation the same doubling doubles the gradient
    s1, _, _ = fp.grad_logprob(m, _params(m), ids1, n_p, length_normalize=False)
    s2, _, _ = fp.grad_logprob(m, _params(m), ids2, n_p, length_normalize=False)
    assert torch.allclose(2 * s1, s2, atol=1e-6)
    assert torch.allclose(s1, n1 * g1, atol=1e-6)


def test_backbone_slice_matches_full_logits_route():
    m = ContextualLM()
    ids, n_p = _ids(PROMPT, RESP)
    # ContextualLM has no .model/.lm_head -> full-logits route
    lp_full, n = fp.continuation_logprob(m, ids, n_p, use_backbone=True)
    lp_ref, _ = fp.continuation_logprob(m, ids, n_p, use_backbone=False)
    assert torch.allclose(lp_full, lp_ref)
    # manual reference: mean log p(y_t | prefix) over the continuation
    logits = m(ids).logits[0]
    lps = torch.log_softmax(logits.float(), -1)
    ref = torch.stack([lps[n_p - 1 + t, ids[0, n_p + t]] for t in range(n)]).mean()
    assert torch.allclose(lp_full, ref)
    # ContextFreeLM has .model/.lm_head -> sliced route must agree with full logits
    m2 = ContextFreeLM()
    a, _ = fp.continuation_logprob(m2, ids, n_p, use_backbone=True)
    b, _ = fp.continuation_logprob(m2, ids, n_p, use_backbone=False)
    assert torch.allclose(a, b)


def test_whitening_identity_when_fisher_is_one():
    d = torch.randn(1000)
    assert torch.equal(fp.whiten(d, torch.ones(1000), 0.0), d)
    # with damping the direction is unchanged after unit normalisation
    w = fp.whiten(d, torch.ones(1000), 1e-3)
    assert torch.allclose(fp.unit(w), fp.unit(d), atol=1e-6)
    # a non-trivial diagonal really rescales coordinates by F^{-1/2}
    F = torch.tensor([4.0, 0.25])
    assert torch.allclose(fp.whiten(torch.tensor([2.0, 2.0]), F, 0.0), torch.tensor([1.0, 4.0]))
    assert fp.fisher_damping(torch.tensor([1.0, 3.0]), 1e-3) == pytest.approx(2e-3)


def test_projection_deterministic_and_seed_sensitive():
    layout = [("a.lora_B", 300), ("b.lora_B", 500)]
    P1 = fp.make_projection(layout, 16, seed=0)
    P2 = fp.make_projection(layout, 16, seed=0)
    P3 = fp.make_projection(layout, 16, seed=1)
    assert all(torch.equal(x, y) for x, y in zip(P1, P2))
    assert not torch.equal(P1[0], P3[0])
    v = torch.randn(800)
    assert torch.equal(fp.project(v, P1), fp.project(v, P2))
    # block structure == one big matrix applied to the concatenated vector
    big = torch.cat(P1, 0)
    assert torch.allclose(fp.project(v, P1), big.T @ v, atol=1e-5)
    # per-module seeding: adding a module leaves existing blocks unchanged
    P4 = fp.make_projection(layout + [("c.lora_B", 7)], 16, seed=0)
    assert torch.equal(P4[0], P1[0]) and torch.equal(P4[1], P1[1])
    # JL scaling: E|P^T v|^2 = |v|^2
    vs = torch.randn(64, 800)
    ratio = torch.stack([fp.project(x, P1).norm() ** 2 / x.norm() ** 2 for x in vs]).mean()
    assert 0.6 < float(ratio) < 1.4
    with pytest.raises(AssertionError):
        fp.project(torch.randn(801), P1)


def test_version_string_round_trip_and_mismatch():
    cfg = fp.FingerprintConfig()
    v = fp.build_version(cfg, "abcdef0123456789", "bfcl_r2:deadbeef0123:damp0.001")
    parsed = fp.parse_version(v)
    assert parsed["schema"] == "fpv1"
    assert parsed["model"] == "Qwen/Qwen3.5-4B"
    assert parsed["tok"] == "abcdef012345"
    assert parsed["prompt"] == "2048" and parsed["resp"] == "1024"
    assert parsed["proj"] == "s0d256" and parsed["norm"] == "len"
    assert parsed["fisher"] == "bfcl_r2:deadbeef0123:damp0.001"
    assert parsed["lora"].startswith("r8a16s0:q_proj,")
    fp.assert_compatible(v, v)
    # tokenizer change
    with pytest.raises(fp.FingerprintVersionError, match="tok"):
        fp.assert_compatible(v, fp.build_version(cfg, "ffff0000ffff0000", "bfcl_r2:deadbeef0123:damp0.001"))
    # projection seed change
    cfg2 = fp.FingerprintConfig(proj_seed=1)
    with pytest.raises(fp.FingerprintVersionError, match="proj"):
        fp.assert_compatible(v, fp.build_version(cfg2, "abcdef0123456789", "bfcl_r2:deadbeef0123:damp0.001"))
    # different Fisher -> incompatible unless explicitly ignored
    v_other_fisher = fp.build_version(cfg, "abcdef0123456789", "alf:0123456789ab:damp0.001")
    with pytest.raises(fp.FingerprintVersionError, match="fisher"):
        fp.assert_compatible(v, v_other_fisher)
    fp.assert_compatible(v, v_other_fisher, ignore={"fisher"})
    # model change
    cfg3 = fp.FingerprintConfig(model_id="Qwen/Qwen3.5-2B")
    with pytest.raises(fp.FingerprintVersionError, match="model"):
        fp.assert_compatible(v, fp.build_version(cfg3, "abcdef0123456789", "bfcl_r2:deadbeef0123:damp0.001"))


def test_save_load_refuses_version_mismatch(tmp_path):
    cfg = fp.FingerprintConfig(proj_dim=8)
    N, D = 3, 20
    fisher = np.abs(np.random.rand(D)).astype(np.float32)
    fid = fp.fisher_id_of(fisher, "toy", cfg.fisher_damping_rel)
    v = fp.build_version(cfg, "0" * 40, fid)
    res = fp.FingerprintResult(
        set_name="toy", psi=np.zeros((N, 8), np.float32), sketch_diff_raw=np.zeros((N, 8), np.float32),
        sketch_teacher=np.zeros((N, 8), np.float32), state_hash=["h0", "h1", "h2"], fisher=fisher,
        fisher_id=fid, fisher_n=N, fisher_damping=1e-3, fingerprint_version=v,
        baseline_version=fp.build_version(cfg, "0" * 40, None), config={"proj_dim": 8},
        param_layout=[("x", D)],
        per_event=[{"state_hash": f"h{i}", "n_tok_S": 1, "n_tok_T": 1, "logp_S": -1.0, "logp_T": -1.0,
                    "norm_gS": 1.0, "norm_gT": 1.0, "norm_diff": 1.0, "norm_wdiff": 1.0,
                    "prompt_tokens": 5, "prompt_dropped": 0, "resp_dropped_S": 0, "resp_dropped_T": 0}
                   for i in range(N)])
    paths = fp.save_fingerprints(res, tmp_path, "t")
    z = fp.load_fingerprints(paths["npz"], expect_version=v)
    assert z["state_hash"] == ["h0", "h1", "h2"] and z["psi"].shape == (N, 8)
    assert z["fingerprint_version"] == v
    other = fp.build_version(fp.FingerprintConfig(proj_dim=8, proj_seed=1), "0" * 40, fid)
    with pytest.raises(fp.FingerprintVersionError):
        fp.load_fingerprints(paths["npz"], expect_version=other)
    # merging a set with a different Fisher is refused
    res2 = fp.FingerprintResult(**{**res.__dict__, "set_name": "toy2",
                                   "fingerprint_version": fp.build_version(cfg, "0" * 40, "toy2:abc:damp0.001")})
    p2 = fp.save_fingerprints(res2, tmp_path, "t")
    with pytest.raises(fp.FingerprintVersionError, match="fisher"):
        fp.merge_fingerprint_sets([paths["npz"], p2["npz"]])
    X, hs, ver = fp.merge_fingerprint_sets([paths["npz"], p2["npz"]], key="sketch_teacher")
    assert X.shape == (2 * N, 8) and len(hs) == 2 * N


def test_encode_event_keeps_prompt_tail_and_appends_eos():
    class Tok:
        eos_token_id = 99

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [int(x) for x in text.split()]}

    cfg = fp.FingerprintConfig(max_prompt_tok=3, max_resp_tok=2)
    p, r, info = fp.encode_event(Tok(), "1 2 3 4 5", "7 8 9", cfg)
    assert p == [3, 4, 5] and r == [7, 8, 99]
    assert info == {"prompt_tokens": 5, "prompt_dropped": 2, "resp_tokens": 3, "resp_dropped": 1}


def test_end_to_end_on_toy_model_whitening_changes_direction():
    """fingerprints() pipeline pieces on the toy model: Fisher from student
    continuations, whitened vs raw sketches differ, both unit-norm."""
    torch.manual_seed(1)
    m = ContextualLM()
    params = _params(m)
    layout = [(n, p.numel()) for n, p in params]
    P = fp.make_projection(layout, 16, 0)
    events = [([3, 7, 11], [8, 1, 4], [8, 2, 4, 5]), ([2, 5, 9, 1], [6, 6], [1, 4, 4])]
    gs, gt = [], []
    for prompt, ys, yt in events:
        ids_s, n_p = _ids(prompt, ys)
        ids_t, _ = _ids(prompt, yt)
        gs.append(fp.grad_logprob(m, params, ids_s, n_p)[0])
        gt.append(fp.grad_logprob(m, params, ids_t, n_p)[0])
    F = torch.stack([g * g for g in gs]).mean(0)
    damp = fp.fisher_damping(F, 1e-3)
    for s, t in zip(gs, gt):
        raw = fp.unit(fp.project(t - s, P))
        psi = fp.unit(fp.project(fp.whiten(t - s, F, damp), P))
        assert abs(float(raw.norm()) - 1) < 1e-5 and abs(float(psi.norm()) - 1) < 1e-5
        assert not torch.allclose(raw, psi, atol=1e-3)


@pytest.mark.skipif(os.environ.get("FP_INTEGRATION") != "1" or not torch.cuda.is_available(),
                    reason="set FP_INTEGRATION=1 with a GPU to run the real-model smoke test")
def test_real_model_smoke():
    events = [{"state_text": "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n",
               "student_continuation": "hi", "teacher_continuation": "Hello there!",
               "state_hash": "a"},
              {"state_text": "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n",
               "student_continuation": "5", "teacher_continuation": "4", "state_hash": "b"}]
    model = os.environ.get("FP_MODEL", "Qwen/Qwen3.5-0.8B")
    cfg = fp.FingerprintConfig(model_id=model, proj_dim=32)
    res = fp.fingerprints(model, events, set_name="smoke", cfg=cfg)
    assert res.psi.shape == (2, 32)
    assert np.allclose(np.linalg.norm(res.psi, axis=1), 1, atol=1e-4)
    res2 = fp.fingerprints(model, events, set_name="smoke", cfg=cfg)
    assert res.fingerprint_version == res2.fingerprint_version
    assert np.allclose(res.psi, res2.psi, atol=1e-3)
