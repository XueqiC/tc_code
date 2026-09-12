"""Versioned BFCL/HotpotQA feedback streams, independent of source RNG and scheduling."""
from dataclasses import asdict, dataclass

import torch

from .persistence import digest


# V1 shared the durable sampling RNG across episode-major continuations.
# V2 keys every complete episode (including its first action) by task and role.
FEEDBACK_RNG_VERSION = 2


@dataclass(frozen=True)
class FeedbackRNG:
    run_seed: int
    round: int
    step: int
    feedback_role: str

    def episode_seed(self, meta_task_id, rollout_index):
        """Canonical SHA256 derivation, as for ALFWorld's episode streams."""
        if any(type(v) is not int or v < 0 for v in
               (self.run_seed, self.round, self.step, rollout_index)):
            raise ValueError('nonnegative integer feedback seed/round/step/rollout index required')
        if not isinstance(meta_task_id, str) or not meta_task_id or not isinstance(
                self.feedback_role, str) or not self.feedback_role:
            raise ValueError('nonempty feedback role and meta task id required')
        key = dict(feedback_rng_version=FEEDBACK_RNG_VERSION, **asdict(self),
                   meta_task_id=meta_task_id, rollout_index=rollout_index)
        return int(digest(key)[:16], 16) % (2**63)

    def generator(self, meta_task_id, rollout_index, *, device):
        return torch.Generator(device=device).manual_seed(self.episode_seed(meta_task_id, rollout_index))


def feedback_rng_identity(config, manifest=None):
    """Missing version in a saved manifest means legacy shared-stream V1."""
    if config.get('benchmark', 'bfcl') not in {'bfcl', 'hotpotqa'}:
        return {}
    return dict(feedback_rng_version=(FEEDBACK_RNG_VERSION if manifest is None else
                                      manifest.get('feedback_rng_version', 1)))


def guard_feedback_rng_comparison(manifests):
    versions = {m.get('feedback_rng_version', 1) for m in manifests
                if m['config'].get('benchmark', 'bfcl') in {'bfcl', 'hotpotqa'}}
    if len(versions) > 1:
        raise ValueError('feedback RNG versions differ; matched-arm comparison refused')
