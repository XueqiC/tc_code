"""RTD v1.0.6 sampling/teacher-forcing comparisons (spec 4.2), in nats/token."""
from dataclasses import asdict, dataclass
import math

import torch

from .persistence import digest


@dataclass(frozen=True)
class ScoreTolerance:
    mean_abs: float = .05
    max_abs: float = 1.
    max_abs_outlier_tokens: int = 2
    max_abs_hard: float = 8.

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in (self.mean_abs, self.max_abs, self.max_abs_hard)):
            raise ValueError('score tolerances must be finite and nonnegative')
        if type(self.max_abs_outlier_tokens) is not int or self.max_abs_outlier_tokens < 0:
            raise ValueError('max_abs_outlier_tokens must be a nonnegative integer')

    @classmethod
    def from_config(cls, config):
        # Resumes use the immutable manifest config, which may predate these fields.
        return cls(**config.get('score_consistency_tolerance', {}))


def score_diagnostic(action, token_logprobs, score, scoring_metadata, tolerance, *,
                     expected_prompt_ids=None, score_atol=None, score_rtol=None):
    """Build a JSON-safe record before enforcing either structural or numeric checks.

    No cancellation in the per-token check; EOS and every generated token count.
    The optional sequence check preserves the double-precision test oracle.
    """
    generated = action.generation_token_logprobs
    rescored = token_logprobs.detach().double().cpu()
    gen = torch.tensor(generated, dtype=torch.float64)
    errors = []
    if expected_prompt_ids is not None and tuple(expected_prompt_ids) != action.prompt_ids:
        errors.append('prompt token boundary/template mismatch')
    if len(generated) != len(action.action_ids) or len(rescored) != len(action.action_ids):
        errors.append('per-token score coverage mismatch')
    finite = bool(torch.isfinite(gen).all() and torch.isfinite(rescored).all()
                  and torch.isfinite(score.detach()))
    if not finite:
        errors.append('nonfinite token score')
    if generated and not math.isclose(sum(generated), action.generation_logprob, rel_tol=2e-7, abs_tol=1e-6):
        errors.append('generation total differs from recorded per-token sum')
    delta = (gen - rescored).abs() if len(gen) == len(rescored) and finite and len(gen) else None
    mean_abs = float(delta.mean()) if delta is not None else None
    max_abs = float(delta.max()) if delta is not None else None
    # Positions are zero-based action-token offsets, including sampled EOS.
    outlier_positions = (delta > tolerance.max_abs).nonzero().flatten().tolist() if delta is not None else None
    outlier_count = len(outlier_positions) if outlier_positions is not None else None
    outlier_fraction = outlier_count / len(gen) if delta is not None else None
    total = float(score.detach())
    sequence_abs = abs(total - action.generation_logprob) if finite else None
    within = (mean_abs is not None and mean_abs <= tolerance.mean_abs
              and outlier_count <= tolerance.max_abs_outlier_tokens and max_abs <= tolerance.max_abs_hard)
    if score_atol is not None or score_rtol is not None:
        within = within and sequence_abs <= (score_atol or 0.) + (score_rtol or 0.)*abs(action.generation_logprob)
    p, n = len(action.prompt_ids), len(action.action_ids)
    # JSON forbids NaN/Inf in the durable hash chain. Keep their identity as text.
    def numbers(values):
        return [float(v) if math.isfinite(float(v)) else str(float(v)) for v in values]
    return dict(version='rtd-score-consistency-v1', protocol_version='1.0.6', n_tokens=n,
        state_hash=digest(dict(prompt_ids=action.prompt_ids)),
        prompt_ids=action.prompt_ids, action_ids=action.action_ids, generated_text=action.text,
        expected_prompt_ids=expected_prompt_ids,
        generation_token_logprobs=numbers(gen), teacher_forced_token_logprobs=numbers(rescored),
        generation_logprob=action.generation_logprob,
        teacher_forced_logprob=total if finite else str(total),
        mean_abs_difference=mean_abs, max_abs_difference=max_abs, sequence_abs_difference=sequence_abs,
        outlier_token_count=outlier_count, outlier_positions=outlier_positions, outlier_fraction=outlier_fraction,
        tolerance=asdict(tolerance), sequence_atol=score_atol, sequence_rtol=score_rtol,
        passed=bool(within and not errors), structural_errors=errors,
        truncated=action.truncated,
        eos=dict(token_id=action.eos_token_id, action_positions=[] if action.truncated else [n-1],
                 included_in_both_scores=not action.truncated,
                 appended=False, text_decode_excludes_eos=not action.truncated),
        masks=dict(generation_prompt_attention=[1]*p, generation_last_input_attention=[1]*(p+n-1),
                   generation_prefix_attention_lengths=list(range(p, p+n)),
                   teacher_forced_attention=[1]*(p+n), teacher_forced_action=[False]*p+[True]*n,
                   scored_logit_positions=list(range(p-1, p+n-1)),
                   scored_label_positions=list(range(p, p+n))),
        policy_id=action.policy_id, backend_id=action.backend_id,
        generation_backend=action.generation_metadata, scoring_backend=scoring_metadata)


def enforce_score_diagnostic(diagnostic):
    if not diagnostic['passed']:
        raise ValueError('generation/teacher-forced likelihood differs: '
            f"mean |delta|={diagnostic['mean_abs_difference']}, "
            f"max |delta|={diagnostic['max_abs_difference']} nats/token; "
            f"outlier_token_count={diagnostic['outlier_token_count']}; "
            f"tolerance={diagnostic['tolerance']}; structural_errors={diagnostic['structural_errors']}; "
            'see score_consistency record in compute.jsonl')
    if diagnostic['outlier_token_count']:
        print('[rtd] score-consistency outlier '
              f"state_hash={diagnostic['state_hash']} n_tokens={diagnostic['n_tokens']} "
              f"outlier_token_count={diagnostic['outlier_token_count']} "
              f"max_abs_difference={diagnostic['max_abs_difference']:.6g} nats/token", flush=True)
