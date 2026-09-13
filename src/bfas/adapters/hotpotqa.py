"""HotpotQA-ReAct adapter: one shared episode loop and exact teacher accounting."""
from __future__ import annotations

from copy import deepcopy
import os
import sys
from pathlib import Path
from collections.abc import Mapping

from ..adapter import BenchmarkAdapter, Demo, Rollout, TaskRef, TeacherEpisode, Turn
from .. import hotpotqa as protocol
from ..hotpotqa_budget import TeacherSession
from ..ledger import acquire_demos
from ..protocol import SupportSplit, TEACHER_ATTEMPTS

SERVER_MODEL_NAME = "bfas-policy"


class HotpotQASupportSplit(SupportSplit):
    def as_dict(self):
        return dict(super().as_dict(), seed=0)


class HotpotQAAdapter(BenchmarkAdapter):
    name = "hotpotqa"
    server_backed = True
    served_model_name = SERVER_MODEL_NAME
    prompt_version = protocol.PROMPT_VERSION

    def __init__(self, seed=0, port=8900, *, offline=None):
        self.seed, self.port = seed, port
        self.offline = protocol.retrieval_offline(offline)
        self._tokenizer, self._loaded_policy = None, None
        self._suffix = "<|assistant|>\n"
        self._questions = {}

    def questions(self, split):
        if split not in self._questions:
            self._questions[split] = {q["_id"]: q for q in protocol.load_questions(split)}
        return self._questions[split]

    def task_pool(self):
        return [TaskRef(q["_id"], q["type"]) for q in self.questions("train").values()]

    def support_split(self):
        # Benchmark-specific, user-fixed 200/seed-0 protocol overrides 5%/seed-50.
        manifest = protocol.load_manifest("train")
        return HotpotQASupportSplit(tuple(manifest["ids"]), tuple(manifest["demand"]), tuple(manifest["calibration"]))

    def official_eval_split_disjoint(self):
        return True

    @staticmethod
    def teacher_name():
        return os.environ.get("BFAS_TEACHER", "openai/gpt-5.6-luna")

    def prepare_renderer(self, policy_ref):
        if self._tokenizer is None or self._loaded_policy != str(policy_ref):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(policy_ref), trust_remote_code=False)
            self._loaded_policy = str(policy_ref)
        self._render(protocol.build_messages(""))
        if pool := os.environ.get("BFAS_HOTPOTQA_TEACHER_POOL"):
            self._import_pool(Path(pool))

    def _render(self, messages):
        if self._tokenizer is None:
            raise RuntimeError("prepare the student renderer before HotpotQA acquisition")
        rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        plain = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not rendered.startswith(plain) or rendered == plain:
            raise RuntimeError("HotpotQA tokenizer has no identifiable generation suffix")
        self._suffix = rendered[len(plain):]
        return rendered

    def _client(self):
        return protocol.make_client(f"http://127.0.0.1:{self.port}/v1")

    def _wiki(self):
        return protocol.Wikipedia(offline=self.offline)

    def _episode(self, question, generate, temperature, demo=None):
        turns, deployment_turns = [], []
        if demo is not None and (demo.task_id != question["_id"] or
                                (demo.raw or {}).get("prompt_version") != self.prompt_version):
            raise ValueError("HotpotQA guidance task/prompt identity mismatch")

        def capture(messages, stop, temperature):
            deployment = deepcopy(messages)
            messages = deepcopy(messages)
            if demo is not None:
                messages[0]["content"] = "Expert demonstration for this task:\n" + demo.worked_example + "\n\n" + messages[0]["content"]
            rendered, deployed = self._render(messages), self._render(deployment)
            reply = generate(messages, stop, temperature)
            turns.append(Turn(rendered, reply, messages))
            deployment_turns.append(Turn(deployed, reply, deployment))
            return reply

        record = protocol.run_episode(question, self._wiki(), capture, temperature=temperature)
        if demo is not None:
            record["deployment_turns"] = deployment_turns
        return Rollout(question["_id"], record["em"] == 1.0, tuple(turns), record)

    def rollout(self, policy, task_ids, temperature, guided_demos=None):
        self.prepare_renderer(policy)
        questions = self.questions("train")
        with self._client() as client:
            generate = protocol.student_generator(client, self.served_model_name)
            return [self._episode(questions[tid], generate, temperature, (guided_demos or {}).get(tid))
                    for tid in task_ids]

    def teacher_episode(self, task_id, attempt_index, temperature, *, budget=None):
        if not 0 <= attempt_index < TEACHER_ATTEMPTS:
            raise ValueError("HotpotQA allows at most three teacher attempts")
        if task_id not in self.questions("train"):
            raise ValueError("HotpotQA teacher acquisition is restricted to the frozen train support set")
        # Fail before opening a paid session if rendering is not prepared.
        self._render(protocol.build_messages(""))
        import appworld_teacher
        session = TeacherSession(appworld_teacher.load_teacher_config(self.teacher_name()),
                                 budget=budget, task_id=task_id, attempt_index=attempt_index)
        try:
            rollout = self._episode(self.questions("train")[task_id], session.generate, temperature)
        except protocol.OfflineCacheMiss as exc:
            # The previous model request is still paid. Pool and shared gateway
            # receive an unverified episode with all usage and can continue.
            if not session.usage["prompt_tokens"]:
                raise
            rollout = Rollout(task_id, False, (), exc.record)
        demo = None
        if rollout.verified:
            demo = Demo(task_id, rollout.turns,
                        f"Question: {self.questions('train')[task_id]['question']}\n" + rollout.raw["transcript"],
                        rollout.raw)
        return TeacherEpisode(task_id, rollout.verified, demo, tuple(session.response_texts),
                              session.tokens_spent, teacher=session.config.name, usage=session.usage,
                              usage_status=session.usage_status, raw=rollout.raw)

    def teacher_demo(self, task_ids, attempts):
        if set(task_ids) - set(self.questions("train")):
            raise ValueError("HotpotQA teacher acquisition is restricted to the frozen train support set")
        return dict(acquire_demos(self.name, self, task_ids, min(attempts, TEACHER_ATTEMPTS)))

    def evaluate(self, policy_ref, out_dir):
        self.preflight_evaluation()
        self.prepare_renderer(policy_ref)
        # Standalone evaluator and adapter evaluation share decoding and records.
        if str(protocol.ROOT) not in sys.path:
            sys.path.insert(0, str(protocol.ROOT))
        from tools.hotpotqa_eval import evaluate
        return evaluate(base_url=f"http://127.0.0.1:{self.port}/v1", model=self.served_model_name,
                        start=0, n=500, out=out_dir, offline=self.offline)

    def preflight_evaluation(self):
        protocol.preflight_retrieval(offline=self.offline)

    def serving_probe(self):
        with self._client() as client:
            protocol.student_generator(client, self.served_model_name)(protocol.build_messages("What is 1 + 1?"),
                                                                       ["\nObservation 1:"], 0.0)

    def generation_suffix(self):
        return self._suffix

    def target_policy(self, category):
        return "prose_ok"

    def is_call_target(self, target, category):
        return protocol.parse_action(target) is not None

    def rerender(self, row: Mapping):
        context = row.get("_render_context")
        if not isinstance(context, list):
            raise ValueError("row lacks HotpotQA render context")
        return self._render(context)

    def release_policy(self):
        self._tokenizer, self._loaded_policy = None, None

    def _import_pool(self, directory):
        import json
        from .. import ledger
        identity = json.loads((directory / "identity.json").read_text())
        if (identity["prompt_version"] != self.prompt_version or
                identity["teacher"] != self.teacher_name() or
                identity["support"] != protocol.load_manifest("train")):
            raise ValueError("HotpotQA teacher pool identity mismatch")
        with protocol.file_lock(directory / "collection.lock"), ledger._purchase_lock(self.name):
            previous = {(r["task_id"], r["attempt_index"]): r for r in ledger.read_records(self.name)}
            for row in ledger.read_records(directory / "teacher_ledger.jsonl"):
                key = row["task_id"], row["attempt_index"]
                if row["task_id"] not in identity["support"]["ids"] or row["teacher"] != self.teacher_name():
                    raise ValueError("HotpotQA pool contains an outside task/teacher")
                if key in previous:
                    for field in ("teacher", "verified", "usage", "tokens_spent", "timestamp"):
                        if previous[key].get(field) != row.get(field):
                            raise ValueError("HotpotQA pool conflicts with an existing purchase")
                    continue
                demo = None
                if row["verified"]:
                    payload = row["demo"]
                    turns = tuple(Turn(self._render(t["context"]), t["target"], t["context"]) for t in payload["turns"])
                    demo = Demo(row["task_id"], turns, payload["worked_example"], {"prompt_version": self.prompt_version})
                ledger.append_episode(self.name, **{k: row[k] for k in
                                      ("task_id", "teacher", "attempt_index", "temperature", "verified", "tokens_spent", "timestamp")},
                                      demo=demo, usage=row.get("usage"), usage_status=row.get("usage_status"),
                                      prompt_version=self.prompt_version)
                previous[key] = row
