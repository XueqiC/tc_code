"""Frozen initial-student projection and train-only standardized gate features."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .selector import PublicFeatures

PROJECTION_DIM = 32
GATE_DIM = 35


@dataclass(frozen=True)
class FeatureRow:
    parent_hash: str
    state_hash: str
    initial_snapshot_id: str
    features: PublicFeatures

    def tensor(self):
        f = self.features
        if f.projection is None or f.source_logprob is None or f.source_length is None:
            raise ValueError("deferred public placeholders cannot be used for training")
        return torch.tensor((*f.projection, f.source_logprob, f.source_length), dtype=torch.float64)


class FrozenProjection:
    """Random map of theta0 hidden states; source policy refresh does not change it."""
    def __init__(self, hidden_dim: int, *, initial_snapshot_id: str, seed: int = 0):
        if hidden_dim < 1 or not initial_snapshot_id:
            raise ValueError("initial student identity and hidden dimension required")
        self.initial_snapshot_id, self.seed, self.hidden_dim = initial_snapshot_id, seed, hidden_dim
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self._matrix = torch.randn(hidden_dim, PROJECTION_DIM, generator=generator, dtype=torch.float64) / math.sqrt(PROJECTION_DIM)

    def __call__(self, hidden: torch.Tensor, *, snapshot_id: str):
        if snapshot_id != self.initial_snapshot_id or hidden.shape != (self.hidden_dim,):
            raise ValueError("projection must use the frozen INITIAL student's state representation")
        values = hidden.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite hidden state")
        return tuple((values @ self._matrix).tolist())


@dataclass(frozen=True)
class FrozenStandardizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    inner_parent_hashes: frozenset[str]
    allowed_state_hashes: frozenset[str]
    initial_snapshot_id: str
    training_state_hashes: tuple[str, ...]

    @classmethod
    def fit(cls, rows, *, inner_parent_hashes, allowed_state_hashes, initial_snapshot_id):
        rows = tuple(rows)
        if not rows:
            raise ValueError("train-side rows required; defer features when inner data are empty")
        parents, states = frozenset(inner_parent_hashes), frozenset(allowed_state_hashes)
        for row in rows:
            if row.parent_hash not in parents or row.state_hash not in states or row.initial_snapshot_id != initial_snapshot_id:
                raise ValueError("feature statistics require legal inner states and initial-student features")
        values = torch.stack([r.tensor() for r in rows]).detach()
        mean = values.mean(0)
        scale = values.std(0, correction=0)
        scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
        return cls(tuple(mean.tolist()), tuple(scale.tolist()), parents, states,
                   initial_snapshot_id, tuple(r.state_hash for r in rows))

    def transform(self, row: FeatureRow, *, device=None, dtype=torch.float64):
        if (row.parent_hash not in self.inner_parent_hashes or row.state_hash not in self.allowed_state_hashes
                or row.initial_snapshot_id != self.initial_snapshot_id):
            raise ValueError("feature row outside the frozen inner scope")
        values = (row.tensor() - torch.tensor(self.mean, dtype=torch.float64)) / torch.tensor(self.scale, dtype=torch.float64)
        # Standardize the 34 varying features; an intercept must remain one.
        return torch.cat((values, values.new_ones(1))).detach().to(device=device, dtype=dtype)


def linear_sigmoid_gate(phi: torch.Tensor, chi: torch.Tensor):
    if phi.shape != (GATE_DIM,) or chi.shape[-1] != GATE_DIM:
        raise ValueError("chi=[32 projection, source logprob, source length, 1]")
    if not torch.isfinite(phi).all() or not torch.isfinite(chi).all():
        raise ValueError("finite gate parameters/features required")
    return torch.sigmoid(chi.detach() @ phi)
