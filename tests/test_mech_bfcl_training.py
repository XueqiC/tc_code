import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl.training import chunked_hidden_gradient, frozen_hidden, mixture_loss, token_schedule


@pytest.mark.parametrize("chunk", [1, 3, 20])
@pytest.mark.parametrize("softcap", [None, 2.0])
def test_full_vocab_chunked_value_and_gradient_equal_dense(chunk, softcap):
    torch.manual_seed(19)
    head = torch.nn.Linear(5, 17, bias=False)
    head.requires_grad_(False)
    seen = []
    hook = head.register_forward_hook(lambda m, a, o: seen.append(a[0].shape[0]))
    hidden = torch.randn(11, 5, requires_grad=True)
    ref = torch.randn_like(hidden)
    targets = torch.randint(0,17,(11,))
    def project(h):
        x = head(h)
        return x if softcap is None else (x/softcap).tanh()*softcap
    dense = mixture_loss(project(hidden), project(ref), targets)
    expected, = torch.autograd.grad(dense, hidden)
    seen.clear()
    gradient, loss = chunked_hidden_gradient(hidden, ref, head, targets, chunk_size=chunk, softcap=softcap)
    assert max(seen) <= chunk
    torch.testing.assert_close(gradient, expected)
    assert loss == pytest.approx(dense.item(), rel=1e-6)
    assert head.weight.grad is None
    hook.remove()


def test_non_topk_reference_tail_affects_loss():
    logits = torch.tensor([[4., 0., -2., -6.]], requires_grad=True)
    a = torch.tensor([[8., 1., 0., -1.]])
    b = torch.tensor([[8., 1., -1., 0.]])
    assert mixture_loss(logits,a,torch.tensor([0])) != mixture_loss(logits,b,torch.tensor([0]))


def test_reference_detached_and_explicit_half_mixture():
    live = torch.tensor([[.3, -.5]], requires_grad=True)
    ref = torch.tensor([[.2, .4]], requires_grad=True)
    target = torch.tensor([1])
    loss = mixture_loss(live, ref, target)
    logp = live.log_softmax(-1)
    expected = -.5*logp[0,1] - .5*(ref.detach().softmax(-1)*logp).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert ref.grad is None


def test_token_accumulation_exact_one_pass_partial_last_row():
    rows = [dict(target_ids=list(range(n))) for n in (3,7,2)]
    steps = token_schedule(rows, 11, 4)
    assert [sum(s['end']-s['start'] for s in b) for b in steps] == [4,4,3]
    consumed = [(s['row'], p) for b in steps for s in b for p in range(s['start'],s['end'])]
    assert len(consumed) == len(set(consumed)) == 11
    assert consumed[-1] == (2,0)
    with pytest.raises(ValueError, match="Insufficient"):
        token_schedule(rows,13,4)


def test_reference_uses_same_backbone_disabled_adapter_and_restores_mode():
    from contextlib import contextmanager
    from types import SimpleNamespace
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor(2.),requires_grad=False)
            self.adapter=torch.nn.Parameter(torch.tensor(3.))
            self.enabled=True
            self.calls=[]
        def get_base_model(self):
            return SimpleNamespace(model=self)
        @contextmanager
        def disable_adapter(self):
            self.enabled=False
            try:
                yield
            finally:
                self.enabled=True
        def forward(self,input_ids,use_cache):
            self.calls.append((self.enabled,self.training,torch.is_grad_enabled()))
            out=input_ids.float()[...,None]*(self.weight+(self.adapter if self.enabled else 0))
            return SimpleNamespace(last_hidden_state=out)
    model=Model().train()
    ids=torch.tensor([[1,2,3]])
    reference=frozen_hidden(model,ids,slice(1,3))
    torch.testing.assert_close(reference,torch.tensor([[4.],[6.]]))
    assert not reference.requires_grad
    assert model.calls == [(False,False,False)]
    assert model.training and model.enabled
    assert model.adapter.item() == 3 and model.weight.item() == 2
