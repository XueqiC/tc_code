"""Memory-aware scheduling only: never change a state, sample, or reduction."""
from dataclasses import dataclass
import gc
import math

import torch


@dataclass(frozen=True)
class MemoryPolicy:
    max_batch_size: int = 2
    peak_budget_bytes: int = 38_000_000_000
    reserve_bytes: int = 2_000_000_000
    estimated_state_bytes: int = 16_000_000_000

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in
                (self.max_batch_size, self.peak_budget_bytes, self.estimated_state_bytes))
                or type(self.reserve_bytes) is not int or self.reserve_bytes < 0):
            raise ValueError('positive memory budget, state estimate and max batch required')

    @classmethod
    def from_config(cls, config):
        values = {}
        for key, default in [('memory_peak_budget_gb', 38.), ('memory_reserve_gb', 2.),
                             ('memory_state_estimate_gb', 16.)]:
            value = config.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('finite memory settings required')
            values[key] = int(value * 1_000_000_000)
        return cls(config.get('max_state_batch_size', 2), values['memory_peak_budget_gb'],
                   values['memory_reserve_gb'], values['memory_state_estimate_gb'])


def release_device_cache(device):
    device = torch.device(device)
    if device.type == 'cuda':
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def memory_batch_size(count, device, policy):
    """Use the visible logical device, counting other users and driver memory.

    State workspace is an estimate, not a CUDA allocation guarantee. An item is
    indivisible: when the estimate does not fit, run one and report that fact.
    The caller streams individual graphs even inside a scheduled batch.
    """
    if count < 1:
        raise ValueError('nonempty batch required')
    device = torch.device(device)
    details = dict(logical_device=str(device), max_batch_size=policy.max_batch_size,
                   estimated_state_bytes=policy.estimated_state_bytes,
                   peak_budget_bytes=policy.peak_budget_bytes, reserve_bytes=policy.reserve_bytes)
    size = min(count, policy.max_batch_size)
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        # Called after empty_cache: total-free also includes CUDA context and
        # non-PyTorch allocations, which memory_allocated alone would miss.
        available = max(0, min(free-policy.reserve_bytes, policy.peak_budget_bytes-(total-free)))
        size = min(size, max(1, available // policy.estimated_state_bytes))
        details.update(free_bytes=free, total_bytes=total, available_bytes=available,
                       singleton_exceeds_estimate=available < policy.estimated_state_bytes)
    return size, details | dict(batch_size=size)


def memory_batches(items, device, policy, *, record=None):
    """Stable consecutive batches; re-read free memory after each cache release."""
    items = tuple(items)
    offset = 0
    while offset < len(items):
        release_device_cache(device)
        size, metadata = memory_batch_size(len(items)-offset, device, policy)
        if record is not None:
            record(metadata | dict(offset=offset, items=len(items)))
        try:
            yield items[offset:offset+size]
        finally:
            release_device_cache(device)
        offset += size
