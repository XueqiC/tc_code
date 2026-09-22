"""Observe parser matches without changing the frozen command selector."""
from ...adapters.alfworld import ALFWorldAdapter, _ACTION_MARKER_RE


class MatchedCommand(str):
    def __new__(cls, value, matches):
        obj = super().__new__(cls, value)
        obj.matches = matches
        return obj

    def __eq__(self, other):
        equal = super().__eq__(other)
        if equal is True:
            self.matches.append(True)
        return equal

    __hash__ = str.__hash__


def parser_fields(reply, command, matches, *, teacher=True):
    return fallback_fields(reply, not matches and not isinstance(command, MatchedCommand), teacher=teacher)


def fallback_fields(reply, fallback, *, teacher=True):
    candidate_text = ALFWorldAdapter._teacher_command_text(reply) if teacher else reply
    # Keep the raw first candidate, including quotes/bullets before normalization.
    candidate = candidate_text.splitlines()[0] if candidate_text.splitlines() else ""
    fields = dict(parser_fallback=fallback, parser_candidate=candidate)
    if fallback:
        fields["parser_fallback_reason"] = ("empty_reply" if not reply.strip() else
            "no_action_marker" if teacher and not _ACTION_MARKER_RE.search(reply) else
            "empty_candidate" if not candidate.strip() else "candidate_not_admissible")
    return fields


def parse_with_diagnostics(reply, admissible, *, teacher=True):
    matches = []
    commands = [MatchedCommand(c, matches) for c in admissible]
    parse = ALFWorldAdapter._teacher_command if teacher else ALFWorldAdapter._pick_command
    command = parse(reply, commands)
    return str(command), parser_fields(reply, command, matches, teacher=teacher)


class EpisodeParserDiagnostics:
    """Observe the actual parser at the environment step boundary.

    Exact membership is observed by __eq__; case/substring matches return a
    marked admissible string. The literal fallback has neither signal, even
    when 'look' is admissible. Freeze that signal before the environment sees
    the command. Prompt text, parser branches, and executed commands are intact.
    """
    def __init__(self, env_factory):
        self.env_factory = env_factory
        self.fallbacks = []

    def __enter__(self):
        diagnostics = self

        class ObservedEnv:
            def __init__(self, task_id):
                self.env = diagnostics.env_factory(task_id)
                self.matches = []

            def state(self, state):
                self.matches = []
                if isinstance(state, dict) and isinstance(state.get("admissible"), list):
                    state = dict(state, admissible=[
                        MatchedCommand(c, self.matches) if isinstance(c, str) else c
                        for c in state["admissible"]])
                return state

            def _read(self):
                return self.state(self.env._read())

            def step(self, command):
                diagnostics.fallbacks.append(not self.matches and not isinstance(command, MatchedCommand))
                return self.state(self.env.step(command))

            def close(self):
                self.env.close()

        return self, ObservedEnv

    def __exit__(self, *exc):
        pass  # official_episode owns environment cleanup, including failures.

    def annotate(self, record):
        if len(record["turns"]) != len(self.fallbacks):
            raise ValueError("parser diagnostic/turn count mismatch")
        for turn, fallback in zip(record["turns"], self.fallbacks):
            turn.update(fallback_fields(turn["generated_text"], fallback))
        return record
