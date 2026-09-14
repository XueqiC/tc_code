"""Kang FTP acquisition with injected teachers only (no provider client/import).

The teacher must implement real assistant-prefix continuation and return the
existing Ledger.online_request receipt: cost, confidence, usage, output. Output
is the newly generated suffix; this wrapper restores the prefix for the agent,
as the official VLLMModel does. Completion usage includes reasoning exactly once.
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from ..ledger import Ledger, LedgerError
from ..persistence import digest
from .paper_fidelity import KANG_PREFIX, KANG_SUMMARY, require_unsealed_output

OFFICIAL_COT_SYSTEM_PROMPT = (
    "You are an expert assistant who can answer the given question accurately and provide clear reasoning.\n\n"
    "When answering questions, follow these guidelines:\n"
    "1. Provide a clear and structured reasoning first\n"
    "2. Follow up with a final answer, must in the <answer> </answer> tag. For example, <answer> xxx </answer>.\n"
    "3. The answer must be succinct and final. For math problems, return the answer using LaTeX in the \\boxed{} format.\n"
    "4. If the question requires multiple steps or facts, break down your reasoning accordingly\n"
    "5. Be precise and factual in your responses\n"
    "6. If you're unsure about something, acknowledge the uncertainty\n\n"
    "Now, please answer the following question: "
)


@dataclass(frozen=True)
class KangAcquisitionConfig:
    run_name: str
    cot_max_output_tokens: int
    agent_max_output_tokens: int
    kang_mode: str = KANG_PREFIX
    cot_system_prompt: str = OFFICIAL_COT_SYSTEM_PROMPT

    def __post_init__(self):
        if (not self.run_name or self.kang_mode not in {KANG_PREFIX, KANG_SUMMARY}
                or any(type(n) is not int or n <= 0 for n in
                       (self.cot_max_output_tokens, self.agent_max_output_tokens))):
            raise ValueError("fresh run name, explicit mode and positive request caps required")


def first_paragraph_prefix(response):
    # Literal official build_prefix_memory.py operation: no strip/word cap.
    return "Thought: " + response.split("\n\n")[0] + "\n\n"


def read_budget(ledger_path, *, budget=None):
    """Read a real RTD ledger or a sealed purchase manifest without any writes.

    A manifest is an offline replay allowance, not an online authorization.
    JSONL journals require their original initial budget; authorize events replay.
    """
    path = Path(ledger_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        if budget is None:
            raise ValueError("JSONL ledger requires its initial budget")
        ledger = Ledger.resume(budget, path, read_only=True)
        return dict(budget=ledger.budget, spent=ledger.spent, remaining=ledger.remaining,
                    reservations=sum(ledger.reservations.values()), ledger_path=str(path.resolve()),
                    scope="RTD ledger; outstanding reservations included")
    value = json.loads(path.read_text())
    charges = value["charges"]
    amounts = [c["tokens"] for c in charges]
    if (any(type(n) is not int or n < 0 for n in amounts)
            or len({c["package_id"] for c in charges}) != len(charges)
            or sum(amounts) != value["teacher_tokens_charged"]
            or value["B"] - sum(amounts) != value["remaining_tokens"]):
        raise LedgerError("purchase manifest charge totals disagree")
    return dict(budget=value["B"], spent=sum(amounts), remaining=value["remaining_tokens"],
                reservations=0, ledger_path=str(path.resolve()),
                scope="sealed replay allowance; not online authorization")


def prefix_dry_run(questions, config, *, ledger_path, budget=None):
    questions = list(questions)
    _questions(questions)
    state = read_budget(ledger_path, budget=budget)
    count = len(questions) if config.kang_mode == KANG_PREFIX else 0
    worst = count * config.cot_max_output_tokens
    return dict(method=config.kang_mode, run_name=config.run_name, dry_run=True,
        cot_requests=count, unique_questions=len(set(questions)),
        cot_max_output_tokens=config.cot_max_output_tokens,
        reserved_worst_case_output_tokens=worst, reservation_performed=False,
        fits_remaining_budget=worst <= state["remaining"],
        additional_tokens_required=max(0, worst-state["remaining"]),
        trajectory_tokens_included=False, new_teacher_calls=0, new_teacher_tokens=0, **state)


def _questions(questions):
    if any(not isinstance(q, str) or not q for q in questions):
        raise ValueError("nonempty question text required (not task IDs)")


def _text(output):
    if not isinstance(output, str):
        raise ValueError("teacher receipt output must be text")
    return output


class KangAcquisition:
    """Shared ledger for the CoT pass and every agent-model request.

    Instantiate only with an explicitly supplied teacher and ledger. No retry
    loop: uncertain usage holds its reservation for reconciliation. Reusing a
    run/request ID fails before the teacher is called.
    """
    def __init__(self, ledger, config):
        if ledger.path is not None:
            require_unsealed_output(ledger.path)
        self.ledger, self.config = ledger, config

    def _request(self, teacher, query_id, cap, *, messages, prefix=None):
        def precheck():
            if prefix is not None and getattr(teacher, "supports_assistant_prefix", False) is not True:
                raise ValueError("teacher must support true assistant-prefix continuation")
        kwargs = dict(messages=messages, max_output_tokens=cap)
        if prefix is not None:
            kwargs["prefix"] = prefix
        output = self.ledger.online_request(query_id, cap, precheck=precheck,
            request=lambda: teacher(**kwargs), validate=_text)
        return (prefix or "") + output

    def build_prefix_memory(self, questions, teacher, *, output_path=None):
        """One plain CoT request per input occurrence; last duplicate key wins.

        Entire completions are charged, including discarded later paragraphs.
        No agent tools, executed trajectory, or reference answer enter this pass.
        """
        if self.config.kang_mode != KANG_PREFIX:
            raise ValueError("action-list summary mode has no CoT prefix memory")
        questions = list(questions)
        _questions(questions)
        if output_path is not None:
            require_unsealed_output(output_path)
            if Path(output_path).exists():
                raise FileExistsError("prefix memory exists; choose a fresh run")
        memory = {}
        for index, question in enumerate(questions):
            query_id = digest([self.config.run_name, KANG_PREFIX, "cot", index, question])
            response = self._request(teacher, query_id, self.config.cot_max_output_tokens,
                messages=[dict(role="system", content=self.config.cot_system_prompt),
                          dict(role="user", content=question)])
            memory[question] = first_paragraph_prefix(response)
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                json.dump(memory, stream, ensure_ascii=False, indent=4)
        return memory

    def acquire_with_prefix(self, question, teacher, *, prefix_memory, attempt_id, purchase_trajectory):
        """Run the existing trajectory loop with a budgeted model callback.

        purchase_trajectory(question=..., model=...) owns environment execution,
        filtering and packaging. It must use model(messages=...) for each turn;
        the first returned text includes the prefix, later turns do not. It must
        preserve that full text in new trajectory targets. No cached trajectory
        can substitute for this call. A failed/invalid episode remains charged.
        """
        _questions([question])
        if not attempt_id:
            raise ValueError("unique attempt ID required")
        prefix = None
        if self.config.kang_mode == KANG_PREFIX:
            prefix = prefix_memory[question]
            if not isinstance(prefix, str) or not prefix.startswith("Thought: ") or not prefix.endswith("\n\n"):
                raise ValueError("invalid official first-thought prefix")
        turn = 0
        def model(*, messages):
            nonlocal turn
            query_id = digest([self.config.run_name, self.config.kang_mode, "agent", attempt_id, question, turn])
            response = self._request(teacher, query_id, self.config.agent_max_output_tokens,
                messages=messages, prefix=prefix if turn == 0 else None)
            turn += 1
            return response
        result = purchase_trajectory(question=question, model=model)
        if turn == 0:
            raise ValueError("trajectory purchase made no model calls; cached trajectories cannot implement FTP")
        # Carry acquisition evidence into the existing bank/row training paths.
        # Targets are the agent's actual continuations; never rewrite them here.
        if isinstance(result, dict):
            result = {**result, "provenance": {**result.get("provenance", {}),
                "acquisition_method": self.config.kang_mode,
                "first_thought_prefix": prefix, "prefix_question": question,
                "acquisition_run_name": self.config.run_name, "attempt_id": attempt_id}}
        elif isinstance(result, list):
            from .paper_data import TeacherRow
            result = [replace(row, acquisition_method=self.config.kang_mode)
                      if isinstance(row, TeacherRow) else row for row in result]
        return result


def build_prefix_memory(questions, teacher, *, ledger, config, output_path=None):
    return KangAcquisition(ledger, config).build_prefix_memory(questions, teacher, output_path=output_path)


def acquire_with_prefix(question, teacher, *, ledger, config, prefix_memory, attempt_id, purchase_trajectory):
    return KangAcquisition(ledger, config).acquire_with_prefix(question, teacher,
        prefix_memory=prefix_memory, attempt_id=attempt_id, purchase_trajectory=purchase_trajectory)
