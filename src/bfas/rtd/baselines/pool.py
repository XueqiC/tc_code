"""Parent -> deduplicated state -> two frozen sources -> paid teacher empirical item."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from ..persistence import digest
from ..transport import Behavior, FullState, SourceSample
from .evidence import action_ids


@dataclass(frozen=True)
class Slot:
    source: SourceSample
    teacher: Behavior | None
    query_id: str | None
    p: float
    prompt_tokens: int
    teacher_tokens: int

    def __post_init__(self):
        if self.teacher is not None:
            self.source.behavior.state.assert_matches(self.teacher.state)
            if not self.query_id:
                raise ValueError("teacher requires paid query identity")

    def costs(self, recipe):
        teacher = self.teacher is not None
        source = recipe == "b3_mix" or (recipe == "b2" and teacher)
        action = int(source) * self.source.length + int(teacher) * self.teacher_tokens
        prompt = (int(source) + int(teacher)) * self.prompt_tokens
        return dict(action_tokens=action, prompt_tokens=prompt, input_tokens=action + prompt,
                    source_action_tokens=int(source) * self.source.length,
                    teacher_action_tokens=int(teacher) * self.teacher_tokens,
                    source_slots=int(source), teacher_slots=int(teacher))

    def identity(self):
        return dict(query_id=self.query_id, state_hash=self.source.behavior.state.state_hash,
                    parent_hash=self.source.behavior.state.parent_hash,
                    source_snapshot=self.source.frozen_snapshot_id, source_hash=digest(asdict(self.source)),
                    teacher_hash=digest(self.teacher.text) if self.teacher else None)


def build_slots(evidence, sources, tokenizer, *, round_number, recipe, distribution="strong"):
    by_parent = defaultdict(list)
    for h, item in sorted(evidence["states"].items()):
        state = FullState(**item["state"])
        if int(state.parent_hash, 16) % 2 == (round_number - 1) % 2:
            if h != state.state_hash:
                raise ValueError("exported full state hash mismatch")
            by_parent[state.parent_hash].append((h, item, state))
    if not by_parent:
        raise ValueError("empty legal inner pool")
    slots = []
    for parent, states in sorted(by_parent.items()):
        for h, item, state in states:
            samples = sources[h]
            if len(samples) != 2:
                raise ValueError("exactly two independently sampled frozen sources per state required")
            teachers = item["teachers"] or [None]
            p = 1 / len(by_parent) / len(states) / 2 / len(teachers)
            for source in samples:
                state.assert_matches(source.behavior.state)
                for t in teachers:
                    if t is not None and t["query_id"] not in evidence["owned"]:
                        raise ValueError("unowned teacher evidence")
                    slots.append(Slot(source, Behavior(state, t["text"]) if t else None,
                        t["query_id"] if t else None, p,
                        len(tokenizer.encode(state.prompt, add_special_tokens=False)),
                        len(action_ids(tokenizer, t["text"])) if t else 0))
    teacher_mass = sum(s.p for s in slots if s.teacher is not None)
    use_pt = distribution == "strong" and recipe != "b3_mix"
    if use_pt:
        if not teacher_mass:
            raise ValueError("no legal teacher evidence in this round; pT is undefined")
        slots = [Slot(s.source, s.teacher, s.query_id, s.p / teacher_mass, s.prompt_tokens, s.teacher_tokens)
                 for s in slots if s.teacher is not None]
    parents = Counter()
    query_mass = Counter()
    for slot in slots:
        parents[slot.source.behavior.state.parent_hash] += slot.p
        if slot.query_id:
            query_mass[slot.query_id] += slot.p
    return slots, dict(distribution="pT" if use_pt else "p0", teacher_available_mass=teacher_mass,
                       parent_mass=dict(parents), query_mass=dict(query_mass))
