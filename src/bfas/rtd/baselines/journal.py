"""ComputeJournal-compatible accounting, with disjoint token roles and timing."""
from collections import Counter
from contextlib import contextmanager
import time

from ..persistence import ComputeJournal


class ResourceJournal(ComputeJournal):
    @contextmanager
    def phase(self, operation, **context):
        start, wall = time.process_time(), time.monotonic()
        outer = not self._measure_stack
        with self.measure(operation, **context):
            try:
                yield
            finally:
                self.append("baseline_cpu", operation=operation, outer=outer,
                    cpu_seconds=time.process_time()-start,
                    gpu_reserved_seconds=time.monotonic()-wall if self.cuda else 0., **context)

    def usage(self, role, *, input_tokens=0, action_tokens=0, backward=True, **context):
        if role not in {"update", "aux", "reference", "generation", "tuning"}:
            raise ValueError("unknown compute role")
        self.append("baseline_tokens", role=role, input_tokens=input_tokens, action_tokens=action_tokens,
                    forward_tokens=input_tokens, backward_tokens=input_tokens if backward else 0, **context)

    def summary(self):
        totals = Counter()
        for e in self.events:
            if e["kind"] == "baseline_tokens":
                for k in ("input_tokens", "action_tokens", "forward_tokens", "backward_tokens"):
                    totals[f"{e['role']}_{k}"] += e[k]
                totals["student_tokens"] += e["input_tokens"]
            elif e["kind"] == "baseline_step":
                for k in ("optimizer_commits", "raw_slots", "weighted_slots", "teacher_slots", "source_slots"):
                    totals[k] += e[k]
            elif e["kind"] == "feedback_rollout":
                totals["tuning_rollouts" if e.get("tuning") else "feedback_rollouts"] += 1
            elif e["kind"] == "teacher_budget":
                totals["teacher_tokens"] += e["teacher_tokens"]
                totals["new_teacher_tokens"] += e["new_teacher_tokens"]
            elif e["kind"] == "baseline_cpu" and e["outer"]:
                totals["cpu_seconds"] += e["cpu_seconds"]
                totals["gpu_reserved_seconds"] += e["gpu_reserved_seconds"]
        begins = {e["sequence"]: e for e in self.events if e["kind"] == "compute_begin"}
        ended = set()
        for e in self.events:
            if e["kind"] != "compute_end":
                continue
            ended.add(e["begin_sequence"])
            if begins[e["begin_sequence"]].get("parent_sequence") is None:
                totals["gpu_seconds"] += e["gpu_seconds"]
                totals["wall_seconds"] += e["wall_seconds"]
                if e["status"] != "complete":
                    totals["failed_gpu_seconds"] += e["gpu_seconds"]
        totals["interrupted_compute_events"] = len(set(begins)-ended)
        totals["T_update"] = totals["update_input_tokens"]
        totals["T_aux"] = totals["aux_input_tokens"] + totals["reference_input_tokens"]
        return dict(totals)
