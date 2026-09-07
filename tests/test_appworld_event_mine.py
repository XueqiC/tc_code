"""Unit tests for tools/appworld_event_mine.py with a mocked environment and student
(no GPU, no AppWorld server, no teacher API)."""
from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import appworld_event_mine as m  # noqa: E402

COMPLETE = "apis.supervisor.complete_task()"
TEACHER_CODES = ["print(apis.api_docs.show_app_descriptions())", "x = apis.spotify.login(username='u', password='p')", COMPLETE]


def reply(code: str, prose: str = "Okay.") -> str:
    return f"{prose}\n\n```python\n{code}\n```"


def make_demo(task_id: str = "abc1234_1", codes=TEACHER_CODES) -> dict:
    return {
        "task_id": task_id,
        "targets": [reply(c) for c in codes],
        "codes": list(codes),
        "messages0": [{"role": "user", "content": "USER instructions\nTask: do it\n\n"}],
        "recorded_outputs": [m.observation_message(f"out-{c}") for c in codes],
        "raw": {"teacher": "gpt-5.4"},
    }


class FakeEnv:
    """Stateful fake: success iff `winning` code was executed and complete_task was called."""

    instances: list["FakeEnv"] = []

    def __init__(self, winning: str):
        self.winning = winning
        self.started = False
        self.executed: list[str] = []
        self.episodes: list[list[str]] = []
        self.starts = 0
        self.stops = 0
        FakeEnv.instances.append(self)

    def start(self, task_id, experiment_name, seed):
        assert not self.started, "start() on an env that was not stopped"
        self.started = True
        self.starts += 1
        self.executed = []
        return {"instruction": "do it", "output_directory": experiment_name}

    def execute(self, code):
        assert self.started, "execute() before start()"
        self.executed.append(code)
        return f"out-{code}"

    def completed(self):
        return any(COMPLETE in c for c in self.executed)

    def evaluate(self):
        assert self.started
        ok = self.completed() and any(m.same_action(c, self.winning) for c in self.executed)
        return {"success": ok, "passed": 2 if ok else 1, "failed": 0 if ok else 1, "num_tests": 2}

    def stop(self):
        self.episodes.append(list(self.executed))
        self.started = False
        self.stops += 1

    def close(self):
        pass


class FakeStudent:
    """A fixed policy: at the state with n assistant turns it replies `probe_codes[n]` if set,
    else the teacher's n-th code (agreement), else complete_task.  The same policy serves the
    trunk probes and the branch continuations, as the real student does."""

    def __init__(self, probe_codes: dict[int, str] | None = None, teacher_codes=TEACHER_CODES):
        self.probe_codes = probe_codes or {}
        self.teacher_codes = list(teacher_codes)
        self.calls: list[tuple[int, int]] = []   # (n_assistant_messages, seed)
        self.lock = threading.Lock()

    def sample(self, messages, seed):
        n = sum(1 for x in messages if x["role"] == "assistant")
        with self.lock:
            self.calls.append((n, seed))
        if n in self.probe_codes:
            code = self.probe_codes[n]
        elif n < len(self.teacher_codes):
            code = self.teacher_codes[n]
        else:
            code = COMPLETE
        return {"content": reply(code), "finish_reason": "stop", "completion_tokens": 5}


def run_miner(demo, winning, student, k=2, max_probes=4, **kw):
    FakeEnv.instances = []
    cfg = m.MinerConfig(k=k, max_probes=max_probes, keep_outputs=True, **kw)
    miner = m.Miner(env_factory=lambda: FakeEnv(winning), student=student,
                    renderer=m.PlainRenderer(), cfg=cfg, log=lambda s: None)
    rows = []
    stats = miner.mine_task(demo, rows.append)
    miner.close()
    return rows, stats


# --------------------------------------------------------------------------- scaffold port
def test_extract_code_matches_official_agent():
    assert m.extract_code_and_fix_content("a\n```python\nx = 1\n```\ntrailing") == ("x = 1", "a\n```python\nx = 1\n```")
    # first block wins (ignore_multiple_calls=True)
    code, fixed = m.extract_code_and_fix_content("```python\na\n```\n```python\nb\n```")
    assert code == "a" and fixed == "```python\na\n```"
    # partial block without terminator gets closed
    assert m.extract_code_and_fix_content("hi\n```python\nprint(1)") == ("print(1)", "hi\n```python\nprint(1)\n```")
    assert m.extract_code_and_fix_content("no code here") == ("", "no code here")
    assert m.assistant_message("t\n```python\nq\n```zzz") == "t\n```python\nq\n```\n\n"
    assert m.observation_message("abc") == "Output:\n```\nabc\n```\n\n"
    assert m.observation_message("abc\n") == "Output:\n```\nabc\n```\n\n"


def test_same_action_is_format_insensitive_but_semantic():
    assert m.same_action("x=apis.spotify.login(username='u')  # c", "x = apis.spotify.login(username = 'u')")
    assert m.same_action("print(a)\n\n\nprint(b)", "print(a)\nprint(b)")
    assert not m.same_action("print(apis.spotify.show_playlist_library())", "print(apis.spotify.show_song_library())")
    assert not m.same_action("f(a=1, b=2)", "f(b=2, a=1)")          # keyword order is a different call
    assert not m.same_action("x = 1", "y = 1")
    # unparsable text falls back to whitespace/comment-insensitive lines
    assert m.same_action("def (:  # bad\n  y", "def (:\ny")
    assert not m.same_action("def (:", "def [:")


def test_same_output_masks_jwt_tokens_only():
    a = '{"access_token": "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2ODQ0MTI5ODN9.ZNkc4BReJw0U4SHaOq-QIBj6q8q"}'
    b = '{"access_token": "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2ODQ0MTIwNzV9.JwdvZK5PUWolqIwiMsy8TZ2ziof"}'
    assert m.same_output(a, b)
    assert not m.same_output('{"count": 3}', '{"count": 4}')
    assert m.normalise_output("x " + a) == 'x {"access_token": "<JWT>"}'


def test_laplace_mean():
    assert m.laplace_mean([1, 1, 1], 3) == pytest.approx(0.8)
    assert m.laplace_mean([0, 0, 0], 3) == pytest.approx(0.2)
    assert m.laplace_mean([0.5, 1.0], 2) == pytest.approx(0.625)
    with pytest.raises(ValueError):
        m.laplace_mean([1], 2)


def test_api_effects_equality():
    tok = {"access_token": "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2ODQ0MTI5ODN9.ZNkc4BReJw0U4SHaOq-QIBj6q8q"}
    a = [{"method": "get", "url": "/spotify/playlists", "data": {**tok, "page_index": 0}}]
    b = [{"method": "get", "url": "/spotify/playlists", "data": {"page_index": 0, "access_token": "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2ODQ0MTIwNzV9.JwdvZK5PUWolqIwiMsy8TZ2ziof"}}]
    # textually different blocks, same calls -> agreement
    assert m.same_effects("x = apis.spotify.show_playlist_library(access_token=t)", a, "o1",
                          "print(apis.spotify.show_playlist_library(access_token=tok, page_index=0))", b, "o2")
    # same endpoint, different args -> different action
    c = [{"method": "get", "url": "/spotify/playlists", "data": {"page_index": 1}}]
    assert not m.same_effects("x", a, "o", "y", c, "o")
    # different number of calls -> different action
    assert not m.same_effects("x", a + a, "o", "y", a, "o")
    # timestamps are masked, other values are not
    assert m.normalise_api_calls([{"method": "post", "url": "/u", "data": {"at": "2023-05-18T10:00:00"}}]) == \
        m.normalise_api_calls([{"method": "post", "url": "/u", "data": {"at": "2023-05-18T11:30:00"}}])
    assert m.normalise_api_calls([{"method": "post", "url": "/u", "data": {"q": "a"}}]) != \
        m.normalise_api_calls([{"method": "post", "url": "/u", "data": {"q": "b"}}])
    # no API calls on either side: AST or identical printed output
    assert m.same_effects("print(len(xs))", [], "3\n", "n = len(xs)\nprint(n)", [], "3\n")
    assert not m.same_effects("print(len(xs))", [], "3\n", "print(len(ys))", [], "4\n")
    # one side calls an API, the other does not
    assert not m.same_effects("print(1)", [], "1", "apis.x.y()", a, "1")


class EffectEnv(FakeEnv):
    """FakeEnv whose blocks report API calls parsed from `call:<url>` tokens in the code."""

    def execute(self, code):
        out = super().execute(code)
        self.last_api_calls = [{"method": "get", "url": tokn.split("call:", 1)[1], "data": {}}
                               for tokn in code.split() if tokn.startswith("call:")]
        return out


def test_api_effects_mode_counts_equivalent_blocks_as_agreement():
    codes = ["a = call:/x/1", "b = call:/x/2", COMPLETE]
    demo = make_demo(codes=codes)
    # student block at t=0 is textually different but makes the same call; at t=1 a different call
    student = FakeStudent(probe_codes={0: "print( call:/x/1 )", 1: "q = call:/other"}, teacher_codes=codes)
    FakeEnv.instances = []
    cfg = m.MinerConfig(k=1, max_probes=4, keep_outputs=True, equality="api_effects", max_cont_steps=0)
    miner = m.Miner(env_factory=lambda: EffectEnv("b = call:/x/2"), student=student,
                    renderer=m.PlainRenderer(), cfg=cfg, log=lambda s: None)
    rows = []
    stats = miner.mine_task(demo, rows.append)
    miner.close()
    assert stats["agree_trace"][0] == 1.0 and stats["agree_trace"][1] == 0.0
    assert stats["probes"] == 1 and rows[0]["turn_index"] == 1 and rows[0]["_equality"] == "api_effects"
    # every scratch env episode was stopped (start/stop paired) and the trunk still ran the full demo
    assert all(env.starts == env.stops for env in FakeEnv.instances)
    assert any(ep == codes for env in FakeEnv.instances for ep in env.episodes)


# --------------------------------------------------------------------------- mining logic
def test_divergence_detection_and_du_sign_teacher_better():
    demo = make_demo()
    student = FakeStudent(probe_codes={1: "print('wrong')"})     # agrees at t=0, diverges at t=1
    rows, stats = run_miner(demo, winning=TEACHER_CODES[1], student=student, k=2)
    assert stats["probes"] == 1 and stats["events"] == 1 and stats["error"] is None
    row = rows[0]
    assert row["turn_index"] == 1 and row["_prefix_len"] == 1
    assert row["_event_good"] == TEACHER_CODES[1] and row["_event_bad"] == "print('wrong')"
    assert row["_event_wins"] == [2, 0]
    assert row["_event_u_plus"] == pytest.approx(0.75) and row["_event_u_minus"] == pytest.approx(0.25)
    assert row["_event_dU"] == pytest.approx(0.5) and stats["consequential"] == 1
    assert row["response"] == reply(TEACHER_CODES[1]) and row["_rejected"] == reply("print('wrong')")
    assert stats["agree_trace"][0] == 1.0 and stats["agree_trace"][1] == 0.0


def test_du_negative_when_student_action_is_better():
    """Teacher behaviour is not assumed better: dU < 0 is emitted, never reversed or dropped."""
    demo = make_demo()
    student = FakeStudent(probe_codes={1: "print('better')"})
    rows, stats = run_miner(demo, winning="print('better')", student=student, k=3)
    assert rows[0]["_event_wins"] == [0, 3]
    assert rows[0]["_event_dU"] == pytest.approx((1 / 5) - (4 / 5))
    assert rows[0]["_event_good"] == TEACHER_CODES[1]     # ownership stays teacher/student
    assert stats["consequential"] == 0 and stats["events"] == 1


def test_all_agree_yields_no_events_and_zero_episodes():
    demo = make_demo()
    student = FakeStudent()
    rows, stats = run_miner(demo, winning=TEACHER_CODES[1], student=student, k=2)
    assert rows == [] and stats["probes"] == 0 and stats["events"] == 0
    # only the trunk env was ever started
    assert sum(e.starts for e in FakeEnv.instances) == 1


@pytest.mark.parametrize("policy,positions,expected", [
    ("spread", None, [2, 5, 8, 11]),
    ("late", None, [8, 9, 10, 11]),
    ("first", (0.25, 0.5, 0.75, 1.0), [2, 5, 8, 11]),
    ("late", (1.0, 0.5, 0.0, 0.5, 0.01), [0, 5, 11]),
])
def test_fixed_selection_is_deterministic_and_emits_agreements(policy, positions, expected):
    codes = [f"step_{i}()" for i in range(11)] + [COMPLETE]
    demo = make_demo(codes=codes)
    runs = []
    for _ in range(2):
        student = FakeStudent(teacher_codes=codes)
        rows, stats = run_miner(demo, winning=codes[0], student=student, k=2,
                                probe_select=policy, probe_positions=positions, sample_seed=37)
        assert stats["error"] is None
        assert stats["probes"] == stats["agreements"] == len(expected)
        assert stats["events"] == stats["consequential"] == 0
        assert [row["turn_index"] for row in rows] == expected
        assert sorted(student.calls) == [(t, 37 + 10 * t + j) for t in expected for j in range(2)]
        # Every selected state starts fresh and executes only its own teacher prefix/action.
        episodes = [ep for env in FakeEnv.instances for ep in env.episodes]
        assert episodes == [codes[:t + 1] for t in expected]
        assert all(env.starts == env.stops for env in FakeEnv.instances)
        for i, row in enumerate(rows):
            t = row["turn_index"]
            assert row["_probe_select"] == ("explicit" if positions is not None else policy)
            assert row["_probe_pos"] == pytest.approx((t + 1) / len(codes))
            assert row["_prefix_codes"] == codes[:t] and row["_prefix_len"] == t
            assert row["_event_k"] == row["_cost"]["episodes"] == 0
            assert row["_event_wins"] == [0, 0] and row["_event_seeds"] == []
            assert row["_event_cont_steps"] == row["_event_completed"] == [[], []]
            assert row["_event_u_plus"] == row["_event_u_minus"] == 0.5
            assert row["_event_dU"] == 0 and row["_student_agree_rate"] == 1.0
            assert row["_agree_trace"] == [1.0] * (i + 1)
            assert row["response"] == row["_rejected"]
            assert row["_prefix_matches_demo"] is True
            assert json.loads(json.dumps(row)) == row
        runs.append([(row["turn_index"], row["prompt"], row["_probe_pos"]) for row in rows])
    assert runs[0] == runs[1]


@pytest.mark.parametrize("policy,positions", [
    ("spread", None), ("late", None), ("first", (0.0, 0.25, 0.5, 0.75, 1.0)),
])
def test_probe_selection_bounds_short_demos_and_budgets(policy, positions):
    for n in (0, 1, 2, 3, 5, 17):
        for budget in (0, 1, 2, 4, 30):
            steps = m.select_probe_steps(n, budget, policy, positions)
            assert steps == m.select_probe_steps(n, budget, policy, positions)
            assert steps == sorted(set(steps))
            assert len(steps) <= min(n, budget)
            assert all(0 <= t < n for t in steps)
            if positions is None and n and budget:
                assert len(steps) == min(n, budget) and steps[-1] == n - 1
    assert m.select_probe_steps(8, 2, "spread") == [3, 7]
    assert m.select_probe_steps(8, 2, "late") == [6, 7]
    assert m.select_probe_steps(8, 2, "late", (1.0, 0.25, 0.5, 0.75)) == [1, 3]


@pytest.mark.parametrize("policy,positions", [
    ("spread", None), ("late", None), ("first", (0.25, 0.5, 0.75, 1.0)),
])
def test_later_probe_state_and_seeds_ignore_earlier_probe_outcomes(policy, positions):
    codes = [f"step_{i}()" for i in range(7)] + [COMPLETE]
    demo = make_demo(codes=codes)
    selected = m.select_probe_steps(len(codes), 4, policy, positions)
    results = []
    for early_divergence in (False, True):
        overrides = {selected[-1]: "wrong_final_action()"}
        if early_divergence:
            overrides[selected[0]] = COMPLETE  # student branch terminates early
        student = FakeStudent(probe_codes=overrides, teacher_codes=codes)
        rows, stats = run_miner(demo, winning=codes[0], student=student, k=2,
                                probe_select=policy, probe_positions=positions,
                                max_cont_steps=0, sample_seed=19)
        assert stats["error"] is None and stats["probes"] == 4
        assert stats["events"] == (2 if early_divergence else 1)
        assert [row["turn_index"] for row in rows] == selected
        assert sorted(student.calls) == [(t, 19 + 10 * t + j) for t in selected for j in range(2)]
        for row in rows:
            t = row["turn_index"]
            expected_messages = list(demo["messages0"])
            for code in codes[:t]:
                expected_messages += [
                    {"role": "assistant", "content": m.assistant_message(reply(code))},
                    {"role": "user", "content": m.observation_message(f"out-{code}")},
                ]
            assert row["_render_context"]["messages"] == expected_messages
            assert row["_prefix_codes"] == codes[:t]
            assert row["_replay_output_mismatch"] == 0
        results.append(rows[-1])
    for field in ("prompt", "response", "_rejected", "_traj", "_event_seeds", "_event_wins", "_event_dU"):
        assert results[0][field] == results[1][field]
    assert results[0]["_event_seeds"] == [401019, 402019]


@pytest.mark.parametrize("equality,expected_events", [("api_effects", 0), ("code_ast", 1)])
def test_selected_state_uses_existing_equality_and_agreement_schema(equality, expected_events):
    codes = ["a = call:/x/1", "b = call:/x/2", COMPLETE]
    student = FakeStudent(probe_codes={1: "print( call:/x/2 )"}, teacher_codes=codes)
    FakeEnv.instances = []
    cfg = m.MinerConfig(k=2, max_probes=1, probe_positions=(0.5,), keep_outputs=True,
                        equality=equality, max_cont_steps=0)
    miner = m.Miner(env_factory=lambda: EffectEnv(codes[1]), student=student,
                    renderer=m.PlainRenderer(), cfg=cfg, log=lambda s: None)
    rows = []
    stats = miner.mine_task(make_demo(codes=codes), rows.append)
    miner.close()
    assert stats["error"] is None and stats["events"] == expected_events
    assert stats["probes"] == len(rows) == 1
    row = rows[0]
    assert row["turn_index"] == 1 and row["_equality"] == equality
    assert row["_event_k"] == expected_events * 2
    assert row["_cost"]["episodes"] == expected_events * 4
    assert row["_student_agree_rate"] == 1.0 - expected_events
    assert row["_event_good"] != row["_event_bad"]  # effect equality still preserves actual actions
    assert all(env.starts == env.stops for env in FakeEnv.instances)
    from bfas.events_schema import from_alfworld_row

    tok = lambda text, **_: {"input_ids": list(text.encode())}  # noqa: E731
    ev = from_alfworld_row(row, tok, source="selected.jsonl")
    ev.validate()
    assert ev.teacher_branch_outcomes[0]["k"] == expected_events * 2


def test_matched_k_branching_bookkeeping():
    demo = make_demo()
    student = FakeStudent(probe_codes={0: "print('s0')", 1: "print('s1')"})
    rows, stats = run_miner(demo, winning=TEACHER_CODES[1], student=student, k=3, max_probes=4)
    assert stats["probes"] == 2 and len(rows) == 2
    for row in rows:
        assert row["_event_k"] == 3 and len(row["_event_seeds"]) == 3
        assert row["_cost"]["episodes"] == 6
        assert [len(x) for x in row["_event_cont_steps"]] == [3, 3]
    # continuation seeds are shared between the T and S branches: every continuation seed
    # appears exactly twice (once per branch) among the continuation calls
    from collections import Counter
    cont_seeds = [seed for n, seed in student.calls if seed >= 1000]   # trunk probes use seeds < 1000
    counts = Counter(cont_seeds)
    assert counts and all(c == 2 for c in counts.values()), counts
    assert {s for row in rows for s in row["_event_seeds"]} <= set(counts)
    probe_seeds = [seed for n, seed in student.calls if seed < 1000]
    assert len(probe_seeds) == 3 * len(TEACHER_CODES)      # K samples at every trunk state


def test_reset_between_branches_invariant():
    """Every continuation runs in a freshly started env: prefix replay + forced action + student."""
    demo = make_demo()
    student = FakeStudent(probe_codes={1: "print('wrong')"})
    rows, stats = run_miner(demo, winning=TEACHER_CODES[1], student=student, k=2)
    episodes = [ep for env in FakeEnv.instances for ep in env.episodes]
    # 1 trunk episode + 2K branch episodes, every start paired with a stop
    assert len(episodes) == 1 + 4
    assert all(env.starts == env.stops for env in FakeEnv.instances)
    # the trunk and the K teacher branches all execute exactly the demo sequence (replayed from
    # a fresh start each time); the K student branches execute prefix + y^S + continuation
    t_like = [ep for ep in episodes if ep == TEACHER_CODES]
    s_branches = [ep for ep in episodes if ep[:2] == [TEACHER_CODES[0], "print('wrong')"]]
    assert len(t_like) == 3 and len(s_branches) == 2
    for ep in s_branches:
        assert ep[-1] == COMPLETE and len(ep) == 3   # prefix (1) + forced action + one student step
    assert rows[0]["_replay_output_mismatch"] == 0 and rows[0]["_prefix_matches_demo"] is True


def test_max_probes_cap_and_continuation_cap():
    demo = make_demo(codes=["a()", "b()", "c()", "d()", COMPLETE])
    student = FakeStudent(probe_codes={0: "s0()", 1: "s1()", 2: "s2()", 3: "s3()"}, teacher_codes=demo["codes"])
    rows, stats = run_miner(demo, winning="b()", student=student, k=1, max_probes=2, max_cont_steps=0)
    assert stats["probes"] == 2 and len(rows) == 2
    # with a zero continuation budget no student call happens inside a branch
    assert all(row["_event_cont_steps"] == [[0], [0]] for row in rows)


def test_event_row_field_set_and_schema_mapping():
    demo = make_demo()
    student = FakeStudent(probe_codes={1: "print('wrong')"})
    rows, _ = run_miner(demo, winning=TEACHER_CODES[1], student=student, k=2)
    row = rows[0]
    required = {
        "task_id", "teacher", "turn_index", "prompt", "response", "_rejected", "_render_context",
        "_event_u_plus", "_event_u_minus", "_event_dU", "_event_k", "_event_wins", "_event_good",
        "_event_bad", "_prefix_len", "_prefix_codes", "_traj", "_seed_category", "_teacher_tokens",
        "_student_agree_rate", "_agree_trace", "_event_pass_frac", "_cost", "_utility",
        "_replay_output_mismatch", "_prefix_matches_demo", "_student_checkpoint",
        "_probe_select", "_probe_pos",
    }
    assert required <= set(row)
    assert row["_teacher_tokens"] == 0 and row["_traj"] == "abc1234_1#p1"
    assert row["_probe_select"] == "first" and row["_probe_pos"] == pytest.approx(2 / 3)
    assert row["prompt"].endswith("<|im_start|>assistant\n<think>\n")
    assert json.loads(json.dumps(row)) == row       # JSON-serialisable
    # unified schema: the ALFWorld converter understands the field set; only `benchmark` changes
    from bfas.events_schema import from_alfworld_row

    tok = lambda text, **_: {"input_ids": list(text.encode())}  # noqa: E731
    ev = from_alfworld_row(row, tok, source="pilot.jsonl", split="train")
    ev.benchmark = "appworld"
    ev.validate()
    assert ev.teacher_continuation == row["response"] and ev.student_continuation == row["_rejected"]
    assert ev.teacher_branch_outcomes[0]["n_success"] == 2 and ev.student_branch_outcomes[0]["n_success"] == 0
    assert ev.teacher_output_tokens > 0


def test_student_request_path_thinking_modes():
    parser = m.VLLMStudent(1, "bfas-policy", 0.7, 4096, thinking="parser")
    body = parser.request_body([{"role": "user", "content": "x"}], seed=3)
    assert body["max_tokens"] == 4096 and body["seed"] == 3 and "chat_template_kwargs" not in body
    off = m.VLLMStudent(1, "bfas-policy", 0.7, None, thinking="off")
    body = off.request_body([{"role": "user", "content": "x"}], seed=3)
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and "max_tokens" not in body
    with pytest.raises(ValueError):
        m.VLLMStudent(1, "bfas-policy", 0.7, None, thinking="on")


def test_check_server_refuses_leaked_thinking(monkeypatch):
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 64)
    monkeypatch.setattr(student, "sample", lambda messages, seed: {
        "content": "Let me think.\n</think>\n\nOK", "finish_reason": "stop",
        "completion_tokens": 9, "reasoning_chars": 0})
    with pytest.raises(RuntimeError, match="reasoning-parser"):
        student.check_server()
    monkeypatch.setattr(student, "sample", lambda messages, seed: {
        "content": "OK", "finish_reason": "stop", "completion_tokens": 1, "reasoning_chars": 40})
    assert student.check_server()["content"] == "OK"


def mock_student_http(monkeypatch, respond):
    """Capture chat requests and return mocked student responses; never open a socket."""
    requests = []

    def urlopen(request, timeout):
        body = json.loads(request.data)
        requests.append(body)
        return io.BytesIO(json.dumps(respond(body)).encode())

    monkeypatch.setattr(m.urllib.request, "urlopen", urlopen)
    return requests


def student_response(content="OK", prompt_tokens=100):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5}}


CONTEXT_ERROR = ("maximum context length is 32768 tokens; requested 4096 output tokens; "
                 "prompt contains at least 28673 input tokens")


def student_http_error(detail=CONTEXT_ERROR, code=400):
    return m.urllib.error.HTTPError("http://mock/v1/chat/completions", code, "Bad request", {},
                                  io.BytesIO(json.dumps({"error": {"message": detail}}).encode()))


@pytest.mark.parametrize("prompt_tokens,context_len,cap,expected", [
    (100, 32768, 4096, 4096),
    (28673, 32768, 4096, 4031),
    (32000, 32768, 4096, 704),
    (32768, 32768, 4096, 256),
    (40000, 32768, 4096, 256),
    (28673, 32768, 1024, 1024),
    (64000, 65536, 4096, 1472),
    (100, 32768, None, None),  # preserve --max-tokens 0 (uncapped)
])
def test_student_adaptive_cap_uses_tokenizer(monkeypatch, prompt_tokens, context_len, cap, expected):
    requests = mock_student_http(monkeypatch, lambda body: student_response(prompt_tokens=prompt_tokens))
    counted = []

    def count_tokens(messages):
        counted.append(messages)
        return prompt_tokens

    student = m.VLLMStudent(1, "bfas-policy", 0.7, cap, context_len=context_len,
                            prompt_token_counter=count_tokens)
    messages = [{"role": "user", "content": "a short string with a mocked token count"}]
    result = student.sample(messages, seed=37)
    assert requests[0].get("max_tokens") == result["max_tokens"] == expected
    assert result["prompt_tokens"] == prompt_tokens
    assert requests[0]["messages"] == messages and requests[0]["seed"] == 37
    assert counted == ([messages] if cap else [])
    assert student.calls == 1 and student.completion_tokens == 5


def test_student_adaptive_cap_uses_usage_from_matching_conversation(monkeypatch):
    usages = iter([28673, 100, 31000, 28873, 31000])
    requests = mock_student_http(monkeypatch, lambda body: student_response(prompt_tokens=next(usages)))
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 4096)
    messages = [{"role": "user", "content": "task a"}]
    student.sample(messages, seed=0)
    # An unrelated task must not replace the baseline for task a.
    student.sample([{"role": "user", "content": "task b"}], seed=1)
    messages.extend([{"role": "assistant", "content": "a" * 199},
                     {"role": "user", "content": "b" * 200}])

    def unavailable_tokenizer(messages):
        raise RuntimeError("tokenizer unavailable")

    student.prompt_token_counter = unavailable_tokenizer
    student.sample(messages, seed=2)
    # A sibling continuation shares only the original prompt, not the other branch's usage.
    sibling = [messages[0], {"role": "assistant", "content": "c" * 400},
               {"role": "user", "content": "d" * 400}]
    student.sample(sibling, seed=3)
    student.sample(list(messages), seed=4)  # identical state: no newly appended text
    assert [body["max_tokens"] for body in requests] == [4096, 4096, 3931, 3831, 1704]


def test_student_adaptive_cap_without_usage_estimates_whole_prompt(monkeypatch):
    def respond(body):
        response = student_response()
        del response["usage"]
        return response

    requests = mock_student_http(monkeypatch, respond)
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 4096)
    messages = [{"role": "user", "content": "x" * (28673 * 4)}]
    for seed in range(2):
        student.sample(messages, seed=seed)
    assert [body["max_tokens"] for body in requests] == [4031, 4031]


@pytest.mark.parametrize("thinking", ["off", "parser"])
def test_renderer_counts_tokens_with_same_chat_template(thinking):
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return [1, 2, 3] if kwargs["tokenize"] else "rendered prompt"

    renderer = m.HFRenderer.__new__(m.HFRenderer)
    renderer.tokenizer = Tokenizer()
    renderer.kwargs = {"enable_thinking": False} if thinking == "off" else {}
    messages = [{"role": "user", "content": "task"}]
    assert renderer.render(messages) == "rendered prompt"
    assert renderer.count_tokens(messages) == 3
    assert calls == [(messages, {"tokenize": tokenize, "add_generation_prompt": True, **renderer.kwargs})
                     for tokenize in (False, True)]


def test_student_context_error_retries_once_at_minimum_even_with_one_try(monkeypatch):
    def respond(body):
        if len(requests) == 1:
            raise student_http_error()
        return student_response()

    requests = mock_student_http(monkeypatch, respond)
    monkeypatch.setattr(m.time, "sleep", lambda seconds: pytest.fail("context retry must not sleep"))
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 4096, retries=1)
    result = student.sample([{"role": "user", "content": "task"}], seed=19)
    assert [body["max_tokens"] for body in requests] == [4096, 256]
    assert requests[1] == {**requests[0], "max_tokens": 256}
    assert result["max_tokens"] == 256 and result["content"] == "OK"
    assert student.calls == 1


@pytest.mark.parametrize("estimated_tokens,initial_cap", [(100, 4096), (33000, 256)])
def test_student_context_retry_is_bounded(monkeypatch, estimated_tokens, initial_cap):
    def respond(body):
        raise student_http_error()

    requests = mock_student_http(monkeypatch, respond)
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 4096,
                            prompt_token_counter=lambda messages: estimated_tokens)
    with pytest.raises(m.StudentContextOverflow, match="maximum context length") as caught:
        student.sample([{"role": "user", "content": "task"}], seed=0)
    assert caught.value.max_tokens == 256
    assert [body["max_tokens"] for body in requests] == [initial_cap, 256]
    assert student.calls == 0


@pytest.mark.parametrize("code,detail", [(400, "invalid model"), (401, CONTEXT_ERROR), (404, "not found")])
def test_student_other_client_errors_are_not_retried_or_overflows(monkeypatch, code, detail):
    def respond(body):
        raise student_http_error(detail, code)

    requests = mock_student_http(monkeypatch, respond)
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 4096)
    with pytest.raises(RuntimeError, match=f"HTTP {code}") as caught:
        student.sample([{"role": "user", "content": "task"}], seed=0)
    assert not isinstance(caught.value, m.StudentContextOverflow)
    assert len(requests) == 1


def test_student_transient_errors_keep_existing_retry_policy(monkeypatch):
    def respond(body):
        if len(requests) == 1:
            raise student_http_error("temporarily unavailable", 503)
        return student_response()

    requests = mock_student_http(monkeypatch, respond)
    sleeps = []
    monkeypatch.setattr(m.time, "sleep", sleeps.append)
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 4096)
    assert student.sample([{"role": "user", "content": "task"}], seed=0)["content"] == "OK"
    assert len(requests) == 2 and requests[0] == requests[1] and sleeps == [2.0]


@pytest.mark.parametrize("policy", ["first", "spread", "late"])
@pytest.mark.parametrize("equality", ["code_ast", "api_effects"])
def test_overflow_row_continues_to_next_probe_state(monkeypatch, policy, equality):
    fake = FakeStudent(probe_codes={2: "wrong_final_action()"})

    def respond(body):
        n = sum(msg["role"] == "assistant" for msg in body["messages"])
        if n == 1:
            raise student_http_error()
        sample = fake.sample(body["messages"], body["seed"])
        return student_response(sample["content"], prompt_tokens=28673)

    requests = mock_student_http(monkeypatch, respond)
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 4096,
                            prompt_token_counter=lambda messages: 28673)
    rows, stats = run_miner(make_demo(), winning=TEACHER_CODES[0], student=student, k=2,
                            max_probes=3, probe_select=policy, equality=equality, max_cont_steps=0)
    assert stats["error"] is None and stats["events"] == stats["overflows"] == 1
    assert stats["agreements"] == (0 if policy == "first" else 1)
    assert [row["turn_index"] for row in rows] == ([1, 2] if policy == "first" else [0, 1, 2])
    overflow = next(row for row in rows if row["_event_overflow"])
    assert overflow["_student_finish_reason"] == "overflow" and overflow["_student_max_tokens"] == 256
    assert overflow["_event_k"] == overflow["_cost"]["episodes"] == overflow["_student_samples"] == 0
    assert overflow["_event_wins"] == [0, 0] and overflow["_event_cont_steps"] == [[], []]
    assert overflow["_event_u_plus"] == overflow["_event_u_minus"] == 0.5
    assert overflow["_student_agree_rate"] is None and overflow["_agree_trace"] == [1.0]
    assert overflow["_rejected"] == overflow["_event_bad"] == ""
    assert "maximum context length" in overflow["_event_errors"][0]
    assert overflow["_prefix_codes"] == TEACHER_CODES[:1]
    assert rows[-1]["_event_overflow"] is False and rows[-1]["_event_k"] == 2
    assert rows[-1]["_student_max_tokens"] == 4031 and rows[-1]["_agree_trace"] == [1.0, 0.0]
    assert all(env.starts == env.stops for env in FakeEnv.instances)
    # pool.map may cancel a pending sample after the first future raises. Every sample
    # that reached the server must nevertheless get exactly one minimum-cap retry.
    overflow_seeds = {body["seed"] for body in requests if body["seed"] in (10, 11)}
    assert 10 in overflow_seeds
    for seed in overflow_seeds:
        assert [body["max_tokens"] for body in requests if body["seed"] == seed] == [4031, 256]
    assert json.loads(json.dumps(rows)) == rows


def test_row_records_minimum_cap_after_successful_context_retry(monkeypatch):
    fake = FakeStudent()

    def respond(body):
        if body["max_tokens"] > 256:
            raise student_http_error()
        return student_response(fake.sample(body["messages"], body["seed"])["content"])

    requests = mock_student_http(monkeypatch, respond)
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 4096,
                            prompt_token_counter=lambda messages: 28673)
    rows, stats = run_miner(make_demo(), winning=TEACHER_CODES[0], student=student, k=1,
                            max_probes=3, probe_select="spread")
    assert stats["error"] is None and stats["overflows"] == 0 and stats["agreements"] == 3
    assert all(row["_student_max_tokens"] == 256 and not row["_event_overflow"] for row in rows)
    assert [body["max_tokens"] for body in requests] == [4031, 256] * 3


def test_adaptive_cap_applies_to_continuations_and_row_keeps_probe_cap(monkeypatch):
    fake = FakeStudent(probe_codes={0: "wrong_first_action()"})

    def count_tokens(messages):
        return 28673 if len(messages) == 1 else 32000

    requests = mock_student_http(monkeypatch, lambda body: student_response(
        fake.sample(body["messages"], body["seed"])["content"], count_tokens(body["messages"])))
    student = m.VLLMStudent(1, "bfas-policy", 0.7, 4096, prompt_token_counter=count_tokens)
    rows, stats = run_miner(make_demo(), winning=TEACHER_CODES[1], student=student, k=2, max_probes=1)
    assert stats["error"] is None and stats["events"] == 1
    assert rows[0]["_student_max_tokens"] == 4031
    assert rows[0]["_event_errors"] == []
    assert [body["max_tokens"] for body in requests if body["seed"] < 1000] == [4031, 4031]
    continuations = [body for body in requests if body["seed"] >= 1000]
    assert len(continuations) == 8 and all(body["max_tokens"] == 704 for body in continuations)


# --------------------------------------------------------------------------- cli / io
def test_probe_selection_cli_defaults_and_positions_override():
    parser = m.build_parser()
    args = parser.parse_args([])
    assert args.probe_select == "first" and args.probe_positions is None
    assert args.context_len == m.MinerConfig().context_len == 32768
    assert parser.parse_args(["--context-len", "65536"]).context_len == 65536
    args = parser.parse_args(["--probe-select", "late", "--probe-positions", "0.25, 0.5,0.75,1.0"])
    assert args.probe_positions == (0.25, 0.5, 0.75, 1.0)
    demo = make_demo(codes=[f"step_{i}()" for i in range(7)] + [COMPLETE])
    summary = m.plan({demo["task_id"]: demo}, [demo["task_id"]], args)
    assert summary["per_task"][0]["probe_steps"] == [1, 3, 5, 7]
    assert summary["max_probe_student_calls"] == 4 * args.k
    assert summary["max_episodes_total"] == 2 * args.k * 4


@pytest.mark.parametrize("positions", ["", "0.5,", "oops", "-0.1", "1.1", "nan", "inf", "-inf"])
def test_probe_positions_cli_rejects_invalid_fractions(positions):
    with pytest.raises(SystemExit):
        m.build_parser().parse_args([f"--probe-positions={positions}"])


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    def turn(messages, target):
        return {"prompt": "p", "target": target, "context": {"messages": messages}}

    base = [{"role": "user", "content": "USER instr\nTask: t\n\n"}]
    turns, msgs = [], list(base)
    for c in TEACHER_CODES:
        turns.append(turn(list(msgs), reply(c)))
        msgs = msgs + [{"role": "assistant", "content": reply(c) + "\n\n"},
                       {"role": "user", "content": m.observation_message("o")}]
    demos = {"schema_version": 1, "demos": {
        "aaa0000_1": {"task_id": "aaa0000_1", "turns": {"items": turns}, "worked_example": "", "raw": {}},
        "bbb0000_2": {"task_id": "bbb0000_2", "turns": {"items": turns}, "worked_example": "", "raw": {}},
        "ccc0000_3": {"task_id": "ccc0000_3", "turns": {"items": turns}, "worked_example": "", "raw": {}},
    }}
    demo_path = tmp_path / "demos.json"
    demo_path.write_text(json.dumps(demos))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"support": ["aaa0000_1", "bbb0000_2", "ccc0000_3"],
                                      "demand": ["aaa0000_1", "bbb0000_2"], "calibration": ["ccc0000_3"]}))
    return demo_path, split_path


def test_load_demos_and_split_selection(tmp_path):
    demo_path, split_path = _write_fixture(tmp_path)
    demos = m.load_demos(demo_path)
    assert demos["aaa0000_1"]["codes"] == TEACHER_CODES
    assert len(demos["aaa0000_1"]["recorded_outputs"]) == len(TEACHER_CODES) - 1
    assert m.select_task_ids(demos, split_path, "demand") == ["aaa0000_1", "bbb0000_2"]
    assert m.select_task_ids(demos, split_path, "support") == ["aaa0000_1", "bbb0000_2"]   # calibration never mined
    assert m.select_task_ids(demos, None, "all") == ["aaa0000_1", "bbb0000_2", "ccc0000_3"]


def test_resume_and_shard_selection(tmp_path, capsys):
    demo_path, split_path = _write_fixture(tmp_path)
    out = tmp_path / "events.jsonl"
    (tmp_path / "events.jsonl.done").write_text("aaa0000_1\n")
    rc = m.main(["--dry-run", "--resume", "--demos", str(demo_path), "--split-file", str(split_path),
                 "--split", "all", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "skipping 1 finished tasks" in text and "2 demos (all)" in text
    m.main(["--dry-run", "--shard", "1/2", "--demos", str(demo_path), "--split-file", str(split_path),
            "--split", "all", "--out", str(out)])
    text = capsys.readouterr().out
    assert "1 demos (all)" in text and "bbb0000_2" in text


def test_dry_run_writes_nothing(tmp_path, capsys):
    demo_path, split_path = _write_fixture(tmp_path)
    out = tmp_path / "events.jsonl"
    rc = m.main(["--dry-run", "--demos", str(demo_path), "--split-file", str(split_path),
                 "--split", "demand", "--k", "2", "--max-probes", "2", "--out", str(out), "--limit", "1"])
    assert rc == 0 and not out.exists()
    captured = capsys.readouterr().out
    assert "1 demos (demand), K=2" in captured and "max episodes=8" in captured and "dry-run" in captured
    assert "teacher_tokens=0" in captured


def test_cli_agreement_rows_do_not_consume_branch_episode_budget(tmp_path, capsys, monkeypatch):
    demo_path, split_path = _write_fixture(tmp_path)
    out = tmp_path / "events.jsonl"
    policy = FakeStudent()
    FakeEnv.instances = []
    monkeypatch.setattr(m, "SubprocessEnv", lambda: FakeEnv(TEACHER_CODES[1]))
    monkeypatch.setattr(m, "HFRenderer", lambda *args, **kw: m.PlainRenderer())
    monkeypatch.setattr(m.VLLMStudent, "check_server", lambda self: {"finish_reason": "stop", "reasoning_chars": 0})
    monkeypatch.setattr(m.VLLMStudent, "sample", lambda self, messages, seed: policy.sample(messages, seed))
    rc = m.main(["--demos", str(demo_path), "--split-file", str(split_path), "--out", str(out),
                 "--probe-select", "spread", "--max-probes", "2", "--max-episodes", "1", "--keep-outputs"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rc == 0 and len(rows) == 4  # two selected states for each of two demand tasks
    assert [row["turn_index"] for row in rows] == [1, 2, 1, 2]
    assert all(row["_probe_select"] == "spread" and row["_cost"]["episodes"] == 0 for row in rows)
    assert len(Path(str(out) + ".done").read_text().splitlines()) == 2
    assert sum(env.starts for env in FakeEnv.instances) == 4
    assert "'episodes': 0" in capsys.readouterr().out
