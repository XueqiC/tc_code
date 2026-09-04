from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import appworld_train


class IntegerTokenizer:
    eos_token_id = 99

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        return "rendered-prompt"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        values = {
            "raw-prompt": [0, 1, 2, 3, 4, 5],
            "rendered-prompt": [0, 1, 2, 3, 4, 5],
            "response": [8],
        }
        return {"input_ids": values[text]}


@pytest.mark.parametrize("use_messages", [False, True])
def test_tail_prompt_truncation_keeps_suffix_gpu_free(
    use_messages: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AW_MAX_PROMPT_TOKENS", "3")
    monkeypatch.setenv("AW_TRUNCATE_SIDE", "tail")
    row: dict[str, Any] = {"prompt": "raw-prompt", "response": "response"}
    if use_messages:
        row["messages"] = [{"role": "user", "content": "ignored"}]

    input_ids, labels = appworld_train.encode(IntegerTokenizer(), row, device="cpu")

    assert input_ids.tolist() == [[3, 4, 5, 8, 99]]
    assert labels.tolist() == [[-100, -100, -100, 8, 99]]


def test_prompt_truncation_log_reports_count_cap_and_side(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AW_MAX_PROMPT_TOKENS", "3")
    monkeypatch.setenv("AW_TRUNCATE_SIDE", "tail")

    appworld_train.log_prompt_truncation(
        IntegerTokenizer(), [{"prompt": "raw-prompt"}]
    )

    assert capsys.readouterr().out == (
        "[train][trunc] rows_over_cap=1/1 cap=3 side=tail\n"
    )
