#!/usr/bin/env python3
"""Run the vendored tau2 CLI with a guard around its LiteLLM calls."""
from __future__ import annotations

import os
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    from bfas.tau2_budget import RequestBudget, register_luna_price, response_usage, service_tier

    config = os.environ.get("BFAS_TAU2_BUDGET_CONFIG")
    budget = RequestBudget(Path(config)) if config else None
    tier = budget.service_tier if budget else service_tier()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    register_luna_price(litellm, tier)
    from tau2.evaluator import evaluator_nl_assertions
    from tau2.utils import llm_utils

    original = llm_utils.completion

    def completion(**kwargs):
        if budget is not None:
            return budget.call(original, **kwargs)
        response = original(**kwargs)
        if (kwargs.get("metadata") or {}).get("bfas_purpose") == "teacher_judge":
            directory = llm_utils.llm_log_dir.get()
            simulation_id = next(part[4:] for part in directory.parts if part.startswith("sim_"))
            row = {"simulation_id": simulation_id, "usage": response_usage(response)}
            with Path(os.environ["BFAS_TAU2_JUDGE_USAGE_PATH"]).open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        return response

    llm_utils.completion = completion
    # Some pinned retail tasks require this native LLM grading step. Preserve
    # the rubric and evaluator, price its Luna calls alongside the two actors.
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = "openai/gpt-5.6-luna"
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
        "base_url": "https://api.openai.com/v1", "service_tier": tier,
        "metadata": {"bfas_purpose": "teacher_judge"},
    }
    # tau2 imports this module's generate function; its globals now use the guard.
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
