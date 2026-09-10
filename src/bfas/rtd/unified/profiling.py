"""Task book §5.1 machine-readable stage profiles; no CUDA initialization on CPU."""
from contextlib import contextmanager
import math

STAGES = ('student_sampling', 'teacher_replay', 'source_teacher_soft_gradients',
          'sketch_metric_gram', 'feedback_rollouts', 'qp', 'controller', 'commit', 'save')


def profile_line(stage, wall_seconds, peak_allocated_bytes, peak_reserved_bytes=0, *, status='complete'):
    if (stage not in STAGES or not math.isfinite(wall_seconds) or wall_seconds < 0
            or min(peak_allocated_bytes, peak_reserved_bytes) < 0 or status not in {'complete', 'failed'}):
        raise ValueError('invalid stage profile')
    return (f'[rtd-profile] stage={stage} wall_seconds={wall_seconds:.6f} '
            f'peak_gpu_gb={peak_allocated_bytes/1e9:.6f} '
            f'peak_reserved_gb={peak_reserved_bytes/1e9:.6f} status={status}')


@contextmanager
def profile(journal, stage, **metadata):
    operation = 'p1_'+stage
    start = len(journal.events)
    try:
        with journal.measure_phase(operation, **metadata):
            yield
    finally:
        events = [e for e in journal.events[start:]
                  if e['kind'] == 'compute_end' and e['operation'] == operation]
        if events:
            e = events[-1]
            print(profile_line(stage, e['wall_seconds'], e['peak_allocated_bytes'],
                               e['peak_reserved_bytes'], status=e['status']), flush=True)
