"""Content-addressed CPU values. Frozen dataclasses alone do not freeze tensors."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from ...behavior.deltas import tensor_state_hash
from ..persistence import digest


@dataclass(frozen=True)
class Array:
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def of(cls, value):
        if isinstance(value, cls):
            return value
        if torch.is_tensor(value):
            value = value.detach().cpu().double().numpy()
        a = np.asarray(value, dtype='<f8')
        if not np.isfinite(a).all():
            raise ValueError('finite numeric representation required')
        return cls(tuple(a.shape), a.tobytes(order='C'))

    def __post_init__(self):
        if (type(self.shape) is not tuple or type(self.data) is not bytes
                or any(type(n) is not int or n < 0 for n in self.shape)
                or len(self.data) != int(np.prod(self.shape))*8
                or not np.isfinite(self.numpy()).all()):
            raise ValueError('invalid immutable array')

    def numpy(self):
        # A bytes-backed view cannot be made writeable, even with setflags.
        return np.frombuffer(self.data, dtype='<f8').reshape(self.shape)

    @property
    def hash(self):
        return digest((self.shape, hashlib.sha256(self.data).hexdigest()))


@dataclass(frozen=True)
class Parameters:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    values: Array

    @classmethod
    def of(cls, parameters):
        if isinstance(parameters, cls):
            return parameters
        return cls(tuple(parameters), tuple(tuple(p.shape) for p in parameters.values()),
                   tuple(str(p.dtype).split('.')[-1] for p in parameters.values()),
                   Array.of(torch.cat([p.detach().cpu().double().flatten() for p in parameters.values()])))

    def __post_init__(self):
        if (not self.names or type(self.names) is not tuple or len(set(self.names)) != len(self.names)
                or type(self.shapes) is not tuple or any(type(s) is not tuple for s in self.shapes)
                or type(self.dtypes) is not tuple or len(self.names) != len(self.shapes)
                or len(self.names) != len(self.dtypes)
                or self.values.shape != (sum(int(np.prod(s)) for s in self.shapes),)
                or any(d not in {'float64', 'float32', 'float16', 'bfloat16'} for d in self.dtypes)):
            raise ValueError('invalid ordered parameter layout')

    def tensors(self, *, device='cpu', requires_grad=True, values=None):
        values = self.values.numpy() if values is None else np.asarray(values)
        if values.shape != self.values.shape or not np.isfinite(values).all():
            raise ValueError('parameter vector/layout mismatch')
        result, offset = {}, 0
        for name, shape, dtype in zip(self.names, self.shapes, self.dtypes):
            size = int(np.prod(shape))
            result[name] = torch.tensor(values[offset:offset+size].copy(), device=device,
                dtype=getattr(torch, dtype)).reshape(shape).requires_grad_(requires_grad)
            offset += size
        return result

    def flatten(self, tensors):
        if tuple(tensors) != self.names or tuple(tuple(p.shape) for p in tensors.values()) != self.shapes:
            raise ValueError('ordered gradient layout mismatch')
        return Array.of(torch.cat([p.detach().cpu().double().flatten() for p in tensors.values()]))

    @property
    def hash(self):
        return tensor_state_hash(self.tensors(requires_grad=False))
